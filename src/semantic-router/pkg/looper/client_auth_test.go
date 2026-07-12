package looper

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

const (
	looperProxyHelperEnvironment = "VLLM_SR_TEST_LOOPER_PROXY_HELPER"
	looperProxyHelperEndpoint    = "VLLM_SR_TEST_LOOPER_PROXY_ENDPOINT"
	looperProxyHelperAddress     = "VLLM_SR_TEST_LOOPER_PROXY_ADDRESS"
)

func TestClientAuthenticatorOverridesConfiguredInternalHeaders(t *testing.T) {
	authenticator, err := NewRequestAuthenticator()
	if err != nil {
		t.Fatalf("NewRequestAuthenticator() error = %v", err)
	}

	authenticated := make(chan bool, 2)
	captured := make(chan http.Header, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured <- r.Header.Clone()
		authenticated <- authenticator.Authenticate(
			r.Header.Get(headers.VSRLooperRequest),
			r.Header.Get(headers.VSRLooperSecret),
		)
		http.Error(w, "fixture response", http.StatusInternalServerError)
	}))
	defer server.Close()

	cfg := &config.LooperConfig{
		Endpoint: server.URL,
		Headers: map[string]string{
			headers.VSRLooperRequest:   "false",
			headers.VSRLooperSecret:    "configured-value-must-not-win",
			headers.VSRLooperDecision:  "configured-decision-must-not-win",
			headers.VSRLooperIteration: "999",
			headers.VSRFusionDepth:     "999",
			"x-vsr-looper-extension":   "configured-extension-must-not-leak",
		},
	}
	authenticatedClient := NewClient(cfg)
	authenticatedClient.setRequestAuthenticator(authenticator)
	if _, err := authenticatedClient.CallModel(
		context.Background(),
		&openai.ChatCompletionNewParams{},
		"model-a",
		false,
		1,
		nil,
		"",
	); err == nil {
		t.Fatal("CallModel() error = nil, want fixture status error")
	}
	if !<-authenticated {
		t.Fatal("authenticated client did not send valid runtime-owned headers")
	}
	authenticatedHeaders := <-captured
	if got := authenticatedHeaders.Get(headers.VSRLooperIteration); got != "1" {
		t.Fatalf("iteration header = %q, want 1", got)
	}
	if got := authenticatedHeaders.Get(headers.VSRLooperDecision); got != "" {
		t.Fatalf("configured decision header leaked into request: %q", got)
	}
	if got := authenticatedHeaders.Get(headers.VSRFusionDepth); got != "" {
		t.Fatalf("configured fusion-depth header leaked into request: %q", got)
	}
	if got := authenticatedHeaders.Get("x-vsr-looper-extension"); got != "" {
		t.Fatalf("configured reserved looper header leaked into request: %q", got)
	}

	plainClient := NewClient(cfg)
	if _, err := plainClient.CallModel(
		context.Background(),
		&openai.ChatCompletionNewParams{},
		"model-a",
		false,
		1,
		nil,
		"",
	); err == nil {
		t.Fatal("CallModel() error = nil, want fixture status error")
	}
	if <-authenticated {
		t.Fatal("plain NewClient unexpectedly authenticated as an internal request")
	}
	plainHeaders := <-captured
	if got := plainHeaders.Get(headers.VSRLooperSecret); got != "" {
		t.Fatalf("plain client sent configured secret header %q", got)
	}
}

func TestClientOmitsIterationHeaderForAuxiliaryCall(t *testing.T) {
	authenticator, err := NewRequestAuthenticator()
	if err != nil {
		t.Fatalf("NewRequestAuthenticator() error = %v", err)
	}

	captured := make(chan http.Header, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured <- r.Header.Clone()
		http.Error(w, "fixture response", http.StatusInternalServerError)
	}))
	defer server.Close()

	client := NewClient(&config.LooperConfig{Endpoint: server.URL})
	client.setRequestAuthenticator(authenticator)
	if _, err := client.CallModel(
		context.Background(),
		&openai.ChatCompletionNewParams{},
		"model-a",
		false,
		0,
		nil,
		"",
	); err == nil {
		t.Fatal("CallModel() error = nil, want fixture status error")
	}

	requestHeaders := <-captured
	if got := requestHeaders.Get(headers.VSRLooperIteration); got != "" {
		t.Fatalf("iteration header = %q, want omitted", got)
	}
	if !authenticator.Authenticate(
		requestHeaders.Get(headers.VSRLooperRequest),
		requestHeaders.Get(headers.VSRLooperSecret),
	) {
		t.Fatal("auxiliary call did not carry valid internal request authentication")
	}
}

func TestClientDoesNotFollowRedirectsWithInternalCredentials(t *testing.T) {
	authenticator, err := NewRequestAuthenticator()
	if err != nil {
		t.Fatalf("NewRequestAuthenticator() error = %v", err)
	}

	redirected := make(chan struct{}, 1)
	sink := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		redirected <- struct{}{}
		http.Error(w, "redirect sink reached", http.StatusInternalServerError)
	}))
	defer sink.Close()

	sourceAuthenticated := make(chan bool, 1)
	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sourceAuthenticated <- authenticator.Authenticate(
			r.Header.Get(headers.VSRLooperRequest),
			r.Header.Get(headers.VSRLooperSecret),
		)
		http.Redirect(w, r, sink.URL, http.StatusTemporaryRedirect)
	}))
	defer source.Close()

	client := NewClient(&config.LooperConfig{Endpoint: source.URL})
	client.setRequestAuthenticator(authenticator)
	if _, err := client.CallModel(
		context.Background(),
		&openai.ChatCompletionNewParams{},
		"model-a",
		false,
		1,
		nil,
		"",
	); err == nil {
		t.Fatal("CallModel() error = nil, want redirect status error")
	}
	if !<-sourceAuthenticated {
		t.Fatal("initial trusted endpoint did not receive valid internal credentials")
	}
	select {
	case <-redirected:
		t.Fatal("client followed a redirect and replayed the privileged request")
	default:
	}
}

func TestClientIgnoresAmbientProxyForPrivilegedReentry(t *testing.T) {
	if os.Getenv(looperProxyHelperEnvironment) == "1" {
		runLooperProxyHelper(t)
		return
	}

	targetHit := make(chan struct{}, 1)
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get(headers.VSRLooperSecret); len(got) != looperRequestTokenHexLength {
			t.Errorf("direct endpoint Looper credential length = %d, want %d", len(got), looperRequestTokenHexLength)
		}
		targetHit <- struct{}{}
		writeProxyTestCompletion(w)
	}))
	defer target.Close()

	proxyHit := make(chan struct{}, 1)
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		proxyHit <- struct{}{}
		writeProxyTestCompletion(w)
	}))
	defer proxy.Close()

	_, targetPort, err := net.SplitHostPort(target.Listener.Addr().String())
	if err != nil {
		t.Fatalf("split direct target address: %v", err)
	}
	endpointAddress := net.JoinHostPort("looper.internal.test", targetPort)
	endpoint := "http://" + endpointAddress + "/v1/chat/completions"

	// #nosec G204 -- os.Args[0] is the Go test binary selected by the test
	// harness; no user-controlled command or argument is executed.
	cmd := exec.Command(os.Args[0], "-test.run=^TestClientIgnoresAmbientProxyForPrivilegedReentry$")
	cmd.Env = looperProxyHelperEnv(proxy.URL, endpoint, endpointAddress, target.Listener.Addr().String())
	if output, runErr := cmd.CombinedOutput(); runErr != nil {
		t.Fatalf("proxy-isolated helper failed: %v\n%s", runErr, output)
	}

	select {
	case <-targetHit:
	default:
		t.Fatal("privileged Looper client did not reach the direct endpoint")
	}
	select {
	case <-proxyHit:
		t.Fatal("privileged Looper client sent its request through ambient HTTP_PROXY")
	default:
	}
}

func runLooperProxyHelper(t *testing.T) {
	t.Helper()

	endpoint := os.Getenv(looperProxyHelperEndpoint)
	endpointAddress := os.Getenv(looperProxyHelperEndpoint + "_HOST")
	targetAddress := os.Getenv(looperProxyHelperAddress)
	if endpoint == "" || endpointAddress == "" || targetAddress == "" {
		t.Fatal("proxy helper environment is incomplete")
	}

	client := NewClient(&config.LooperConfig{Endpoint: endpoint})
	baseTransport, ok := client.httpClient.Transport.(*http.Transport)
	if !ok || baseTransport == nil {
		t.Fatalf("Looper client transport = %T, want private *http.Transport", client.httpClient.Transport)
	}
	transport := baseTransport.Clone()
	dialer := &net.Dialer{}
	transport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
		if address == endpointAddress {
			address = targetAddress
		}
		return dialer.DialContext(ctx, network, address)
	}
	client.httpClient.Transport = transport

	authenticator, err := NewRequestAuthenticator()
	if err != nil {
		t.Fatalf("NewRequestAuthenticator() error = %v", err)
	}
	client.setRequestAuthenticator(authenticator)
	response, err := client.CallModel(
		context.Background(),
		&openai.ChatCompletionNewParams{},
		"model-a",
		false,
		1,
		nil,
		"provider-key-canary",
	)
	if err != nil {
		t.Fatalf("direct privileged request failed: %v", err)
	}
	if response.Content != "direct Looper endpoint reached" {
		t.Fatalf("response content = %q, want direct endpoint fixture", response.Content)
	}
}

func looperProxyHelperEnv(proxyURL, endpoint, endpointAddress, targetAddress string) []string {
	drop := map[string]struct{}{
		"ALL_PROXY": {}, "all_proxy": {},
		"HTTP_PROXY": {}, "http_proxy": {},
		"HTTPS_PROXY": {}, "https_proxy": {},
		"NO_PROXY": {}, "no_proxy": {},
		"REQUEST_METHOD":                    {},
		looperProxyHelperEnvironment:        {},
		looperProxyHelperEndpoint:           {},
		looperProxyHelperEndpoint + "_HOST": {},
		looperProxyHelperAddress:            {},
	}
	environment := make([]string, 0, len(os.Environ())+8)
	for _, entry := range os.Environ() {
		name, _, _ := strings.Cut(entry, "=")
		if _, excluded := drop[name]; !excluded {
			environment = append(environment, entry)
		}
	}
	return append(environment,
		looperProxyHelperEnvironment+"=1",
		looperProxyHelperEndpoint+"="+endpoint,
		looperProxyHelperEndpoint+"_HOST="+endpointAddress,
		looperProxyHelperAddress+"="+targetAddress,
		"HTTP_PROXY="+proxyURL,
		"http_proxy="+proxyURL,
		"NO_PROXY=",
		"no_proxy=",
	)
}

func writeProxyTestCompletion(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{
  "id":"chatcmpl-proxy-test",
  "object":"chat.completion",
  "created":1,
  "model":"model-a",
  "choices":[{"index":0,"message":{"role":"assistant","content":"direct Looper endpoint reached"},"finish_reason":"stop"}],
  "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
}`))
}
