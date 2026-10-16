package testcases

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// Each profile's full-duplex gateway sends a request to the full-duplex-selected
// backend only when it carries the x-selected-model header that the Router sets
// once the body is routed, and every other request to full-duplex-default. The
// Router gives full-duplex-provider a provider chat path and credential that
// differ from the client's. Both backends are provider-mocker instances that
// report, per session, the path and a digest of the Authorization header they
// received (tools/test/services/provider-mocker).
const (
	fullDuplexModel        = "full-duplex-provider"
	fullDuplexProviderPath = "/provider/v1/chat/completions"
	fullDuplexProviderAuth = "Bearer e2e-provider-credential"
	fullDuplexClientPath   = "/v1/chat/completions"
	fullDuplexClientAuth   = "Bearer e2e-client-credential"
	fullDuplexSession      = "x-vsr-test-session-id"
	fullDuplexSelected     = "full-duplex-selected"
	fullDuplexDefault      = "full-duplex-default"
)

type fullDuplexGateway struct {
	service           pkgtestcases.ServiceConfig
	backendsNamespace string
}

func init() {
	pkgtestcases.Register("agentgateway-full-duplex-routing", pkgtestcases.TestCase{
		Description: "Verify agentgateway FullDuplexStreamed requests reach the backend x-selected-model names with the provider path and credential, and an end-of-stream rejection reaches no backend",
		Tags:        []string{"agentgateway", "gateway", "streaming", "routing"},
		Fn: fullDuplexRoutingTest(fullDuplexGateway{
			service: pkgtestcases.ServiceConfig{
				LabelSelector: "gateway.networking.k8s.io/gateway-name=agentgateway-full-duplex",
				Namespace:     "agentgateway-system",
				ServicePort:   "80",
			},
			backendsNamespace: "agentgateway-system",
		}),
	})
	pkgtestcases.Register("envoy-full-duplex-routing", pkgtestcases.TestCase{
		Description: "Verify raw Envoy FULL_DUPLEX_STREAMED requests reach the backend x-selected-model names with the provider path and credential, and an end-of-stream rejection reaches no backend",
		Tags:        []string{"streaming", "gateway", "routing"},
		Fn: fullDuplexRoutingTest(fullDuplexGateway{
			service: pkgtestcases.ServiceConfig{
				LabelSelector: "gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router-full-duplex",
				Namespace:     "envoy-gateway-system",
				ServicePort:   "80",
			},
			backendsNamespace: "default",
		}),
	})
}

type fullDuplexRequest struct {
	name    string
	chunks  []string // one chunk is sent with Content-Length, more are chunked
	delay   time.Duration
	trailer bool
	reject  bool // the Router rejects the body at end of stream
}

func fullDuplexRequests() []fullDuplexRequest {
	// "urgent" routes to a keyword decision without a response cache in both
	// profiles, so every routed request reaches a backend. Digits would trip
	// the streaming profile's PII block.
	body := func(model, n string) []string {
		return []string{
			fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":`, model),
			fmt.Sprintf(`"This is urgent, please answer request %s."}],`, n),
			`"max_tokens":16}`,
		}
	}
	return []fullDuplexRequest{
		{name: "one-write body", chunks: []string{strings.Join(body(fullDuplexModel, "one"), "")}},
		{name: "delayed body", chunks: body(fullDuplexModel, "two"), delay: 500 * time.Millisecond},
		// The backend does not report trailers, so this checks only that a
		// request with trailers is routed.
		{name: "request with trailers", chunks: body(fullDuplexModel, "three"), trailer: true},
		// A model that no provider serves is rejected only once the body is
		// complete, so the Router's immediate response replaces the held reply.
		// This case also passes without the held reply; the three above do not.
		{name: "end-of-stream rejection", chunks: body("full-duplex-unknown", "four"), reject: true},
	}
}

func fullDuplexRoutingTest(gateway fullDuplexGateway) func(context.Context, *kubernetes.Clientset, pkgtestcases.TestCaseOptions) error {
	return func(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
		// The gateway controller creates this Service after setup applies the Gateway.
		if _, err := helpers.WaitForServiceByLabelWithReadyPods(
			ctx, client, gateway.service.Namespace, gateway.service.LabelSelector,
			2*time.Minute, 2*time.Second, opts.Verbose, nil,
		); err != nil {
			return err
		}
		opts.ServiceConfig = gateway.service
		session, err := fixtures.OpenServiceSession(ctx, client, opts)
		if err != nil {
			return err
		}
		defer session.Close()
		backends := map[string]*fixtures.ServiceSession{}
		for _, name := range []string{fullDuplexSelected, fullDuplexDefault} {
			backend, err := fixtures.OpenServiceEndpointSession(ctx, client, opts, gateway.backendsNamespace, name, "8000")
			if err != nil {
				return err
			}
			defer backend.Close()
			backends[name] = backend
		}

		httpClient := session.HTTPClient(60 * time.Second)
		runID := time.Now().UnixNano()
		requests := fullDuplexRequests()
		// Every request runs, so a failure reports each request's outcome.
		var failures []error
		for i, request := range requests {
			sessionID := fmt.Sprintf("full-duplex-%d-%d", runID, i)
			if err := checkFullDuplexRequest(ctx, httpClient, session.URL(fullDuplexClientPath), backends, request, sessionID); err != nil {
				failures = append(failures, fmt.Errorf("%s: %w", request.name, err))
			}
		}
		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{"requests": len(requests), "failed": len(failures)})
		}
		return errors.Join(failures...)
	}
}

func checkFullDuplexRequest(
	ctx context.Context,
	httpClient *http.Client,
	url string,
	backends map[string]*fixtures.ServiceSession,
	request fullDuplexRequest,
	sessionID string,
) error {
	status, responseBody, err := sendFullDuplexRequest(ctx, httpClient, url, request, sessionID)
	if err != nil {
		return err
	}
	selected, err := fullDuplexObservation(ctx, backends[fullDuplexSelected], sessionID)
	if err != nil {
		return err
	}
	unrouted, err := fullDuplexObservation(ctx, backends[fullDuplexDefault], sessionID)
	if err != nil {
		return err
	}

	if request.reject {
		// The mocker would accept this body, so a 400 carrying the Router's
		// error envelope is the Router's immediate response.
		if status != http.StatusBadRequest || !strings.Contains(responseBody, `"code":"model_not_found"`) {
			return fmt.Errorf("expected the Router's HTTP 400 with code model_not_found, got %d: %s", status, responseBody)
		}
		if selected != nil || unrouted != nil {
			return fmt.Errorf("a rejected request reached a backend (%s %v, %s %v)",
				fullDuplexSelected, selected, fullDuplexDefault, unrouted)
		}
		return nil
	}
	if status != http.StatusOK {
		return fmt.Errorf("expected HTTP 200, got %d: %s", status, responseBody)
	}
	if selected == nil {
		return fmt.Errorf("%s never received the request (%s received %v)", fullDuplexSelected, fullDuplexDefault, unrouted)
	}
	if selected.Path != fullDuplexProviderPath || selected.AuthorizationSHA256 != fullDuplexDigest(fullDuplexProviderAuth) {
		return fmt.Errorf("%s received %v, want path %q with the provider credential",
			fullDuplexSelected, selected, fullDuplexProviderPath)
	}
	if unrouted != nil {
		return fmt.Errorf("%s also received the request: %v", fullDuplexDefault, unrouted)
	}
	return nil
}

func sendFullDuplexRequest(
	ctx context.Context,
	httpClient *http.Client,
	url string,
	request fullDuplexRequest,
	sessionID string,
) (int, string, error) {
	var body io.Reader = strings.NewReader(request.chunks[0])
	if len(request.chunks) > 1 {
		reader, writer := io.Pipe()
		go func() {
			for i, chunk := range request.chunks {
				if i > 0 {
					time.Sleep(request.delay)
				}
				if _, err := io.WriteString(writer, chunk); err != nil {
					return
				}
			}
			writer.Close()
		}()
		body = reader
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
	if err != nil {
		return 0, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fullDuplexClientAuth)
	req.Header.Set(fullDuplexSession, sessionID)
	if request.trailer {
		req.Trailer = http.Header{"X-E2e-Trailer": {"end"}}
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	responseBody, err := io.ReadAll(resp.Body)
	return resp.StatusCode, string(responseBody), err
}

type fullDuplexObserved struct {
	Path                string `json:"path"`
	AuthorizationSHA256 string `json:"authorization_sha256"`
}

func (o *fullDuplexObserved) String() string {
	credential := "another credential"
	switch o.AuthorizationSHA256 {
	case fullDuplexDigest(fullDuplexProviderAuth):
		credential = "the provider credential"
	case fullDuplexDigest(fullDuplexClientAuth):
		credential = "the client credential"
	}
	return fmt.Sprintf("path %q with %s", o.Path, credential)
}

// fullDuplexObservation returns what the backend received for the session, or
// nil when it received nothing.
func fullDuplexObservation(ctx context.Context, backend *fixtures.ServiceSession, sessionID string) (*fullDuplexObserved, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, backend.URL("/debug/last-request"), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set(fullDuplexSession, sessionID)
	resp, err := backend.HTTPClient(30 * time.Second).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return nil, nil
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("backend observation returned HTTP %d", resp.StatusCode)
	}
	var observed fullDuplexObserved
	if err := json.NewDecoder(resp.Body).Decode(&observed); err != nil {
		return nil, err
	}
	return &observed, nil
}

func fullDuplexDigest(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}
