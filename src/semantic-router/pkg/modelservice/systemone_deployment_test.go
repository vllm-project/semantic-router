package modelservice

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const routeAnswer = `{"answers":{"route":{"type":"choice","choice":"engineering","probabilities":{"billing":0.02,"engineering":0.98},"confidence":0.96}},"model":"vllm-sr/d3-ultra","usage":{"input_tokens":148,"output_tokens":0}}`

var routeQuestion = Question{ID: "route", Type: "choice", Instructions: "Which team should handle this ticket?", Truncate: true, Choices: []Choice{
	{Key: "billing", Description: "Payments and refunds"}, {Key: "engineering", Description: "Outages and bugs"},
}}

// systemOneConfig parses one systemone deployment, remote-d3, from YAML.
func systemOneConfig(t *testing.T, deployment string) *config.RouterConfig {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.3
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
routing: {}
global:
  model_catalog:
    deployments:
      remote-d3:
        provider: systemone
` + deployment))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func newSystemOne(t *testing.T, cfg *config.RouterConfig) *RemoteSystemOne {
	t.Helper()
	remote, err := NewRemoteSystemOne(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = remote.Close() })
	return remote
}

func TestSystemOneDeploymentSpeaksToAGatewayWithItsOwnCredentialHeader(t *testing.T) {
	type seen struct {
		path, key, authorization string
		body                     map[string]json.RawMessage
	}
	calls := make(chan seen, 4)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			http.NotFound(w, r) // This gateway publishes no models listing.
			return
		}
		var body map[string]json.RawMessage
		_ = json.NewDecoder(r.Body).Decode(&body)
		calls <- seen{r.URL.Path, r.Header.Get("Ocp-Apim-Subscription-Key"), r.Header.Get("Authorization"), body}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(routeAnswer))
	}))
	defer server.Close()
	t.Setenv("GATEWAY_TEST_KEY", "fake-subscription-key")
	remote := newSystemOne(t, systemOneConfig(t, `        endpoint: `+server.URL+`/Unified/v1/systemone
        model: vLLM-SR-d3-27B
        question_types: [choice, noul]
        auth: {header: Ocp-Apim-Subscription-Key, prefix: "", api_key_env: GATEWAY_TEST_KEY}
`))

	card, ok := remote.Card("remote-d3")
	if !ok || card.ID != "vLLM-SR-d3-27B" || !card.Answers("noul") || card.Answers("score") {
		t.Fatalf("declared card: %+v %v", card, ok)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	response, err := remote.Decide(ctx, "remote-d3", Request{State: "Our app has returned 500 errors since the 09:00 deploy.", MaxTokens: 2048, Questions: []Question{routeQuestion}})
	if err != nil {
		t.Fatal(err)
	}
	if answer := response.Answers["route"]; answer.Choice != "engineering" || answer.Probabilities["billing"] != 0.02 {
		t.Fatalf("answer: %+v", answer)
	}
	call := <-calls
	if call.path != "/Unified/v1/systemone" || call.key != "fake-subscription-key" || call.authorization != "" {
		t.Fatalf("request reached %q with key %q and authorization %q", call.path, call.key, call.authorization)
	}
	if string(call.body["model"]) != `"vLLM-SR-d3-27B"` || !strings.Contains(string(call.body["questions"]), `"criteria":{"billing":"Payments and refunds","engineering":"Outages and bugs"}`) {
		t.Fatalf("System One body: model %s questions %s", call.body["model"], call.body["questions"])
	}
	if options := string(call.body["options"]); !strings.Contains(options, "deadline_ms") || strings.Contains(options, "max_tokens") {
		t.Fatalf("options carry the deadline and no scan budget: %s", options)
	}
	time.Sleep(100 * time.Millisecond)
	if card, _ := remote.Card("remote-d3"); card.Answers("score") {
		t.Fatalf("without a listing the declared types stay: %+v", card)
	}
}

func TestSystemOneDeploymentUsesBearerAndItsModelsListing(t *testing.T) {
	keyFile := filepath.Join(t.TempDir(), "key")
	if err := os.WriteFile(keyFile, []byte("fake-bearer-key\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer fake-bearer-key" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet && r.URL.Path == "/v1/systemone/models" {
			_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"vllm-sr/d3-flash","question_types":["choice"]},{"id":"vllm-sr/d3-ultra","api":"systemone","question_types":["choice","noul","score","span"],"presets":["pii"]}]}`))
			return
		}
		_, _ = w.Write([]byte(routeAnswer))
	}))
	defer server.Close()
	remote := newSystemOne(t, systemOneConfig(t, `        endpoint: `+server.URL+`/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_file: `+keyFile+`}
`))

	if card, _ := remote.Card("remote-d3"); card.Answers("span") || !card.Answers("score") {
		t.Fatalf("before the listing answers, the System One types apply: %+v", card)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		card, _ := remote.Card("remote-d3")
		if card.Answers("span") && card.HasPreset("pii") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the listing never reached the card: %+v", card)
		}
		time.Sleep(20 * time.Millisecond)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if response, err := remote.Decide(ctx, "remote-d3", Request{State: "x", Questions: []Question{routeQuestion}}); err != nil || response.Answers["route"].Choice != "engineering" {
		t.Fatalf("bearer call: %+v %v", response, err)
	}
}

func TestSystemOneDeploymentFailuresStayUnknownWithoutTheKey(t *testing.T) {
	release := make(chan struct{})
	var status, redirects atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/elsewhere" {
			redirects.Add(1)
			return
		}
		if r.Method == http.MethodGet {
			http.NotFound(w, r)
			return
		}
		_, _ = io.Copy(io.Discard, r.Body)
		switch code := int(status.Load()); code {
		case 0:
			select {
			case <-release:
			case <-r.Context().Done():
			}
		case http.StatusFound:
			http.Redirect(w, r, "/elsewhere", http.StatusFound)
		default:
			w.WriteHeader(code)
			_, _ = w.Write([]byte(`{"error":{"code":"invalid_api_key","message":"unknown or revoked API key"}}`))
		}
	}))
	defer server.Close()
	defer close(release)
	t.Setenv("FAILING_TEST_KEY", "fake-secret-key")
	cfg := systemOneConfig(t, `        endpoint: `+server.URL+`/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: FAILING_TEST_KEY}
`)
	remote := newSystemOne(t, cfg)
	request := Request{State: "x", Questions: []Question{routeQuestion}}
	ask := func(timeout time.Duration) error {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		_, err := remote.Decide(ctx, "remote-d3", request)
		if err != nil && strings.Contains(err.Error(), "fake-secret-key") {
			t.Fatalf("an error carries the credential: %v", err)
		}
		return err
	}

	started := time.Now()
	if err := ask(200 * time.Millisecond); !errors.Is(err, context.DeadlineExceeded) || time.Since(started) > 2*time.Second {
		t.Fatalf("a slow API ends at the call's deadline: %v after %s", err, time.Since(started))
	}
	for code, want := range map[int]error{http.StatusUnauthorized: ErrRejected, http.StatusTooManyRequests: ErrOverloaded, http.StatusServiceUnavailable: ErrUnavailable, http.StatusInternalServerError: ErrFailed, http.StatusFound: ErrFailed} {
		status.Store(int64(code))
		if err := ask(2 * time.Second); !errors.Is(err, want) {
			t.Fatalf("status %d: %v, want %v", code, err, want)
		}
	}
	if redirects.Load() != 0 {
		t.Fatal("a redirect was followed with the credential")
	}
	t.Setenv("FAILING_TEST_KEY", "")
	if err := ask(2 * time.Second); !errors.Is(err, ErrFailed) || !strings.Contains(err.Error(), "FAILING_TEST_KEY is not set") {
		t.Fatalf("a missing credential is a failure that names its variable: %v", err)
	}
	if _, err := remote.Decide(context.Background(), "unknown", request); !errors.Is(err, ErrUnknownDeployment) {
		t.Fatalf("an unknown deployment: %v", err)
	}
	if _, ok := remote.Card("unknown"); ok {
		t.Fatal("an unknown deployment has no card")
	}
	server.Close()
	if err := ask(2 * time.Second); !errors.Is(err, ErrFailed) {
		t.Fatalf("an unreachable API: %v", err)
	}
}
