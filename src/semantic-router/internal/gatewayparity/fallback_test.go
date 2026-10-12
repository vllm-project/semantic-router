package gatewayparity

import (
	"context"
	"fmt"
	"io"
	"maps"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// A decision with two candidates and cross-model fallback on 5xx. Envoy mode
// falls back in the router's response phase; the native gateway runs the same
// policy in the upstream layer.
const fallbackConfig = `
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 30s
providers:
  defaults:
    model: primary-model
  models:
    - name: primary-model
      provider_model_id: primary-model
      api_format: openai
      backend_refs:
        - name: primary
          endpoint: PRIMARY
          protocol: http
          provider: vllm
    - name: fallback-model
      provider_model_id: fallback-model
      api_format: openai
      backend_refs:
        - name: secondary
          endpoint: SECONDARY
          protocol: http
          provider: vllm
routing:
  modelCards:
    - name: primary-model
    - name: fallback-model
  decisions:
    - name: default_route
      priority: 10
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: primary-model
          use_reasoning: false
        - model: fallback-model
          use_reasoning: false
      algorithm:
        type: static
  fallback:
    enabled: true
    max_attempts: 3
    retryable_status_codes: [502, 503, 504]
`

const fallbackRequest = `{"model":"vllm-sr/auto","messages":[{"role":"user","content":"Hello there."}]}`

// scriptedBackend answers every request with one status and body, and counts
// the requests it saw.
type scriptedBackend struct {
	status int
	body   string
	hits   atomic.Int32
	// seen is the header of the last request.
	seen atomic.Pointer[http.Header]
}

func (b *scriptedBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b.hits.Add(1)
	header := r.Header.Clone()
	b.seen.Store(&header)
	_, _ = io.Copy(io.Discard, r.Body)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(b.status)
	_, _ = io.WriteString(w, b.body)
}

func openAIError(message string) string {
	return `{"error":{"message":"` + message + `","type":"server_error","code":"unavailable"}}`
}

func chatCompletion(content string) string {
	return `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"fallback-model",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"` + content + `"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`
}

// fallbackGateway serves fallbackConfig over two backends. native selects who
// runs the fallback chain.
func fallbackGateway(t *testing.T, primary, secondary *scriptedBackend, native bool) *httptest.Server {
	t.Helper()
	return gatewayOver(t, fallbackConfig, map[string]http.Handler{"PRIMARY": primary, "SECONDARY": secondary}, native)
}

// gatewayOver composes the routing core, the upstream layer and the frontend
// for configYAML, whose backend placeholders name the given handlers. native
// selects who runs the fallback chain.
func gatewayOver(t *testing.T, configYAML string, backends map[string]http.Handler, native bool) *httptest.Server {
	t.Helper()
	var placeholders []string
	for placeholder, handler := range backends {
		server := httptest.NewServer(handler)
		t.Cleanup(server.Close)
		placeholders = append(placeholders, placeholder, strings.TrimPrefix(server.URL, "http://"))
	}
	configYAML = strings.NewReplacer(placeholders...).Replace(configYAML)
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte(configYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := extproc.NewOpenAIRouter(configPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = router.Close() })
	set, err := upstream.Build(router.Config, upstream.Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = set.Close(ctx)
	})
	opts := routing.DefaultOptions
	opts.ExecutesFallback = native
	handler, err := gateway.NewHandler(gateway.Options{
		Serving:  gateway.Static(gateway.Serving{Engine: routing.NewEngine(extproc.NewRouterService(router), opts), Upstream: set}),
		Listener: "http-8899",
	})
	if err != nil {
		t.Fatal(err)
	}
	frontend := httptest.NewServer(handler)
	t.Cleanup(frontend.Close)
	return frontend
}

// answer is what a client can tell about a response: status, the headers the
// router sets (content type and the x-vsr-* family, without timings) and body.
// routingLatency is the one timing, kept apart because its value varies.
type answer struct {
	status         int
	header         map[string]string
	body           string
	routingLatency string
}

func (a answer) String() string { return fmt.Sprintf("%d %v %s", a.status, a.header, a.body) }

// clientTag is a header the client sends, to see which requests carry it.
const clientTag = "X-Client-Tag"

func postChat(t *testing.T, frontend *httptest.Server) answer {
	t.Helper()
	return postChatWithHeaders(t, frontend, nil)
}

func postChatWithHeaders(t *testing.T, frontend *httptest.Server, extra map[string]string) answer {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, frontend.URL+"/v1/chat/completions", strings.NewReader(fallbackRequest))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(clientTag, "from-the-client")
	for name, value := range extra {
		req.Header.Set(name, value)
	}
	resp, err := frontend.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	header := map[string]string{}
	for name, values := range resp.Header {
		name = strings.ToLower(name)
		if name == "content-type" || (strings.HasPrefix(name, "x-vsr-") && name != "x-vsr-routing-latency-ms") {
			header[name] = strings.Join(values, ",")
		}
	}
	return answer{
		status: resp.StatusCode, header: header, body: string(body),
		routingLatency: resp.Header.Get("x-vsr-routing-latency-ms"),
	}
}

// A candidate's success reaches the client the same way in both modes, with
// the decision headers of any routed response. Envoy mode answers with an
// immediate response, which replaces the primary's response and its headers;
// the native gateway passes the candidate's response through the response
// phases.
func TestFallbackServesTheCandidateInBothModes(t *testing.T) {
	answers := map[bool]answer{}
	for _, native := range []bool{true, false} {
		primary := &scriptedBackend{status: http.StatusServiceUnavailable, body: openAIError("primary overloaded")}
		secondary := &scriptedBackend{status: http.StatusOK, body: chatCompletion("from the fallback model")}
		got := postChat(t, fallbackGateway(t, primary, secondary, native))

		if got.status != http.StatusOK || !strings.Contains(got.body, "from the fallback model") {
			t.Fatalf("native=%t: response = %s, want the fallback model's answer", native, got)
		}
		if got.header["x-vsr-selected-model"] != "fallback-model" || got.header["x-vsr-fallback-attempts"] != "2" {
			t.Fatalf("native=%t: headers = %v, want the serving model after two attempts", native, got.header)
		}
		if got.header["x-vsr-selected-decision"] != "default_route" || got.header["x-vsr-selected-algorithm"] != "static" ||
			got.routingLatency == "" {
			t.Fatalf("native=%t: headers = %v, routing latency %q, want the decision headers of a routed response",
				native, got.header, got.routingLatency)
		}
		if primary.hits.Load() != 1 || secondary.hits.Load() != 1 {
			t.Fatalf("native=%t: backend hits primary=%d secondary=%d, want one each",
				native, primary.hits.Load(), secondary.hits.Load())
		}
		// The primary's request is the client's in both modes. A native
		// candidate's request is built the same way, for its own route; Envoy
		// mode's response-phase fallback sends the candidate only the provider's
		// headers.
		primarySeen, candidateSeen := *primary.seen.Load(), *secondary.seen.Load()
		if primarySeen.Get(clientTag) != "from-the-client" || primarySeen.Get("X-Selected-Model") != "primary-model" {
			t.Fatalf("native=%t: the primary got %v, want the client's request", native, primarySeen)
		}
		wantCandidate := http.Header{}
		if native {
			wantCandidate.Set(clientTag, "from-the-client")
			wantCandidate.Set("X-Selected-Model", "fallback-model")
			wantCandidate.Set("X-Request-Id", primarySeen.Get("X-Request-Id"))
		}
		for _, name := range []string{clientTag, "X-Selected-Model", "X-Request-Id"} {
			if candidateSeen.Get(name) != wantCandidate.Get(name) {
				t.Fatalf("native=%t: the candidate got %s %q, want %q", native, name, candidateSeen.Get(name), wantCandidate.Get(name))
			}
		}
		answers[native] = got
	}
	native, envoy := answers[true], answers[false]
	if native.status != envoy.status || native.body != envoy.body || !maps.Equal(native.header, envoy.header) {
		t.Fatalf("answers differ:\nnative: %s\nenvoy:  %s", native, envoy)
	}
}

// A primary that refuses connections gets Envoy's local reply. Envoy's
// ext_proc ends processing on its own local replies, so behind Envoy the
// client gets that reply and no candidate is tried. The native gateway's
// chain falls back on it: a deliberate difference, since a primary that is
// down is what fallback is for.
func TestARefusedPrimaryFallsBackOnlyInNativeMode(t *testing.T) {
	// The port stays reserved. Closing it lets another server bind the address,
	// and a real HTTP answer is not Envoy's local reply: the response-phase
	// fallback then calls the candidate.
	refused := resetListener(t)
	configYAML := strings.Replace(fallbackConfig, "PRIMARY", refused, 1)

	for _, native := range []bool{true, false} {
		secondary := &scriptedBackend{status: http.StatusOK, body: chatCompletion("from the fallback model")}
		got := postChat(t, gatewayOver(t, configYAML, map[string]http.Handler{"SECONDARY": secondary}, native))
		if native {
			if got.status != http.StatusOK || !strings.Contains(got.body, "from the fallback model") || secondary.hits.Load() != 1 {
				t.Fatalf("native: response = %s after %d candidate calls, want the candidate's answer", got, secondary.hits.Load())
			}
			continue
		}
		if got.status != http.StatusServiceUnavailable || got.header["content-type"] != "text/plain" ||
			!strings.HasPrefix(got.body, "upstream connect error or disconnect/reset before headers") {
			t.Fatalf("envoy: response = %s, want Envoy's local reply as it is", got)
		}
		if secondary.hits.Load() != 0 {
			t.Fatalf("envoy: the candidate got %d calls, want none", secondary.hits.Load())
		}
	}
}

// resetListener holds a loopback port and resets every connection, so the
// dial fails with Envoy's local reply and no other server can claim the port.
func resetListener(t *testing.T) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			if tcp, ok := conn.(*net.TCPConn); ok {
				_ = tcp.SetLinger(0)
			}
			_ = conn.Close()
		}
	}()
	return listener.Addr().String()
}

// When every candidate fails, the client gets the primary's failure, and the
// chain still tried each candidate exactly once: the response phases did not
// fall back a second time behind the native gateway's chain.
func TestExhaustedFallbackReturnsThePrimaryFailureInBothModes(t *testing.T) {
	answers := map[bool]string{}
	for _, native := range []bool{true, false} {
		primary := &scriptedBackend{status: http.StatusServiceUnavailable, body: openAIError("primary overloaded")}
		secondary := &scriptedBackend{status: http.StatusBadGateway, body: openAIError("secondary bad gateway")}
		got := postChat(t, fallbackGateway(t, primary, secondary, native))

		if got.status != http.StatusServiceUnavailable || !strings.Contains(got.body, "primary overloaded") {
			t.Fatalf("native=%t: response = %s, want the primary's failure", native, got)
		}
		if primary.hits.Load() != 1 || secondary.hits.Load() != 1 {
			t.Fatalf("native=%t: backend hits primary=%d secondary=%d, want one each",
				native, primary.hits.Load(), secondary.hits.Load())
		}
		answers[native] = got.String()
	}
	if answers[true] != answers[false] {
		t.Fatalf("client answers differ:\nnative: %s\nenvoy:  %s", answers[true], answers[false])
	}
}
