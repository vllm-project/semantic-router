package embedding

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestOpenAICompatibleProviderDoesNotFollowRedirects(t *testing.T) {
	for _, crossOrigin := range []bool{false, true} {
		name := "same origin"
		if crossOrigin {
			name = "cross origin"
		}
		t.Run(name, func(t *testing.T) {
			assertOpenAIProviderDoesNotFollowRedirect(t, crossOrigin)
		})
	}
}

func TestOpenAICompatibleProviderDefaultTransportIgnoresAmbientProxy(t *testing.T) {
	t.Setenv("HTTPS_PROXY", "http://ambient-proxy.invalid:3128")
	t.Setenv(config.EmbeddingAPIKeyEnvName, "provider-secret")
	for _, tt := range []struct {
		name   string
		client *http.Client
	}{
		{name: "default client"},
		{name: "caller client without transport", client: &http.Client{}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
				BaseURL: "https://embedding.example/v1", Model: "embedding-model",
				APIKeyEnv: config.EmbeddingAPIKeyEnvName, HTTPClient: tt.client,
			})
			transport, ok := provider.client.Transport.(*http.Transport)
			if !ok {
				t.Fatalf("provider transport = %T, want *http.Transport", provider.client.Transport)
			}
			if transport.Proxy != nil {
				t.Fatal("provider default transport retained ambient proxy resolution")
			}
			if tt.client != nil && tt.client.Transport != nil {
				t.Fatal("provider mutated caller client transport")
			}
		})
	}
}

func TestOpenAICompatibleProviderPreservesExplicitCustomTransport(t *testing.T) {
	customTransport := &http.Transport{Proxy: http.ProxyFromEnvironment}
	callerClient := &http.Client{Transport: customTransport}
	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL: "https://embedding.example/v1", Model: "embedding-model",
		HTTPClient: callerClient,
	})

	if provider.client.Transport != customTransport {
		t.Fatal("provider replaced explicitly supplied custom transport")
	}
	if callerClient.Transport != customTransport {
		t.Fatal("provider mutated caller client transport")
	}
}

func assertOpenAIProviderDoesNotFollowRedirect(t *testing.T, crossOrigin bool) {
	t.Helper()
	var targetCalls atomic.Int32
	var targetAuthorization atomic.Value
	targetAuthorization.Store("")
	targetHandler := func(w http.ResponseWriter, r *http.Request) {
		targetCalls.Add(1)
		targetAuthorization.Store(r.Header.Get("Authorization"))
		writeEmbeddingResponse(t, w, [][]float64{{0.1}})
	}
	targetServer := httptest.NewTLSServer(http.HandlerFunc(targetHandler))
	defer targetServer.Close()

	var redirectLocation string
	var sourceAuthorization atomic.Value
	sourceAuthorization.Store("")
	sourceServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/redirect-target" {
			targetHandler(w, r)
			return
		}
		sourceAuthorization.Store(r.Header.Get("Authorization"))
		http.Redirect(w, r, redirectLocation, http.StatusFound)
	}))
	defer sourceServer.Close()
	redirectLocation = redirectTargetURL(sourceServer.URL, targetServer.URL, crossOrigin)

	var callerRedirectPolicyCalled atomic.Bool
	callerClient := sourceServer.Client()
	callerClient.CheckRedirect = func(*http.Request, []*http.Request) error {
		callerRedirectPolicyCalled.Store(true)
		return nil
	}
	originalTimeout := callerClient.Timeout
	t.Setenv(config.EmbeddingAPIKeyEnvName, "redirect-secret")
	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL: sourceServer.URL, Model: "embedding-model", ExpectedDimension: 1,
		APIKeyEnv: config.EmbeddingAPIKeyEnvName, HTTPClient: callerClient,
	})

	_, err := provider.Embed(context.Background(), "hello")
	assertRedirectRequestIsolation(t, redirectTestState{
		provider: provider, callerClient: callerClient, originalTimeout: originalTimeout,
		callerPolicyCalled:  &callerRedirectPolicyCalled,
		sourceAuthorization: &sourceAuthorization,
		targetAuthorization: &targetAuthorization,
		targetCalls:         &targetCalls,
	}, err)
}

func redirectTargetURL(sourceURL, targetURL string, crossOrigin bool) string {
	if crossOrigin {
		return targetURL + "/redirect-target"
	}
	return sourceURL + "/redirect-target"
}

type redirectTestState struct {
	provider            *OpenAICompatibleProvider
	callerClient        *http.Client
	originalTimeout     time.Duration
	callerPolicyCalled  *atomic.Bool
	sourceAuthorization *atomic.Value
	targetAuthorization *atomic.Value
	targetCalls         *atomic.Int32
}

func assertRedirectRequestIsolation(t *testing.T, state redirectTestState, err error) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), "status 302") {
		t.Fatalf("Embed() error = %v, want un-followed redirect status", err)
	}
	if state.provider.client == state.callerClient {
		t.Fatal("provider retained the caller's mutable HTTP client object")
	}
	if state.callerPolicyCalled.Load() {
		t.Fatal("provider used the caller's redirect policy instead of refusing redirects")
	}
	if err := state.callerClient.CheckRedirect(nil, nil); err != nil {
		t.Fatalf("caller redirect policy was mutated: %v", err)
	}
	if !state.callerPolicyCalled.Load() {
		t.Fatal("caller redirect policy was replaced on the original HTTP client")
	}
	if got := state.sourceAuthorization.Load().(string); got != "Bearer redirect-secret" {
		t.Fatalf("source Authorization = %q, want bearer token", got)
	}
	if got := state.targetCalls.Load(); got != 0 {
		t.Fatalf("redirect target received %d request(s), want 0", got)
	}
	if got := state.targetAuthorization.Load().(string); got != "" {
		t.Fatalf("redirect target received Authorization %q", got)
	}
	if state.callerClient.Timeout != state.originalTimeout {
		t.Fatalf("provider mutated caller HTTP client timeout to %s", state.callerClient.Timeout)
	}
}

func TestOpenAICompatibleProviderRequiresConfiguredAPIKeyEnv(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		writeEmbeddingResponse(t, w, [][]float64{{0.1, 0.2}})
	}))
	defer server.Close()
	t.Setenv(config.EmbeddingAPIKeyEnvName, "")

	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL: server.URL, Model: "embedding-model",
		APIKeyEnv: config.EmbeddingAPIKeyEnvName, HTTPClient: server.Client(),
	})

	_, err := provider.Embed(context.Background(), "hello")
	if err == nil || !strings.Contains(err.Error(), config.EmbeddingAPIKeyEnvName) {
		t.Fatalf("Embed() error = %v, want missing env error", err)
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("provider made %d request(s), want 0", got)
	}
}

func TestOpenAICompatibleProviderRejectsUnrelatedSecretWithoutRequest(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		calls.Add(1)
	}))
	defer server.Close()

	provider, err := NewOpenAICompatibleProvider(OpenAICompatibleConfig{
		BaseURL: server.URL, Model: "embedding-model",
		APIKeyEnv: "VLLM_SR_LOOPER_SHARED_SECRET", HTTPClient: server.Client(),
	})
	if err == nil || !strings.Contains(err.Error(), config.EmbeddingAPIKeyEnvName) {
		t.Fatalf("NewOpenAICompatibleProvider() = (%v, %v), want dedicated-env rejection", provider, err)
	}
	if provider != nil {
		t.Fatal("NewOpenAICompatibleProvider() returned a provider for unrelated secret env")
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("rejected configuration sent %d request(s), want 0", got)
	}
}

func TestNewOpenAICompatibleProviderValidatesConfig(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	cases := map[string]OpenAICompatibleConfig{
		"missing URL":         {Model: "embedding-model"},
		"missing scheme":      {BaseURL: "localhost:8000", Model: "embedding-model"},
		"unsupported scheme":  {BaseURL: "file://localhost/tmp/embeddings", Model: "embedding-model"},
		"URL userinfo":        {BaseURL: "https://user:password@localhost:8000", Model: "embedding-model"},
		"dimension mismatch":  {BaseURL: "http://localhost:8000", Dimensions: 2, ExpectedDimension: 3},
		"negative dimensions": {BaseURL: "http://localhost:8000", Model: "embedding-model", Dimensions: -1},
		"negative expected dimensions": {
			BaseURL: "http://localhost:8000", Model: "embedding-model", ExpectedDimension: -1,
		},
		"negative timeout": {BaseURL: "http://localhost:8000", Model: "embedding-model", TimeoutSeconds: -1},
		"timeout above limit": {
			BaseURL: "http://localhost:8000", Model: "embedding-model",
			TimeoutSeconds: config.EmbeddingEndpointMaxTimeoutSeconds + 1,
		},
		"extreme timeout":  {BaseURL: "http://localhost:8000", Model: "embedding-model", TimeoutSeconds: maxInt},
		"negative retries": {BaseURL: "http://localhost:8000", Model: "embedding-model", MaxRetries: -1},
		"retries above limit": {
			BaseURL: "http://localhost:8000", Model: "embedding-model",
			MaxRetries: config.EmbeddingEndpointMaxRetries + 1,
		},
		"extreme retries": {BaseURL: "http://localhost:8000", Model: "embedding-model", MaxRetries: maxInt},
		"unrelated secret env": {
			BaseURL: "https://localhost:8000", Model: "embedding-model",
			APIKeyEnv: "VLLM_SR_LOOPER_SHARED_SECRET",
		},
		"credential over HTTP": {
			BaseURL: "http://localhost:8000", Model: "embedding-model",
			APIKeyEnv: config.EmbeddingAPIKeyEnvName,
		},
		"missing model": {BaseURL: "http://localhost:8000"},
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := NewOpenAICompatibleProvider(cfg); err == nil {
				t.Fatal("NewOpenAICompatibleProvider() returned nil error")
			}
		})
	}
}

func TestEmbeddingEndpointErrorsDoNotExposeBaseURL(t *testing.T) {
	const sensitiveMarker = "DO-NOT-EXPOSE-ENDPOINT-VALUE"
	for _, baseURL := range []string{
		"https://user:" + sensitiveMarker + "@example.test/v1?token=" + sensitiveMarker,
		"http://[" + sensitiveMarker,
		"https://example.test/v1?token=" + sensitiveMarker,
		"https://example.test/v1#" + sensitiveMarker,
	} {
		_, err := NewOpenAICompatibleProvider(OpenAICompatibleConfig{
			BaseURL: baseURL, Model: "embedding-model",
		})
		if err == nil {
			t.Fatalf("NewOpenAICompatibleProvider() returned nil error")
		}
		if strings.Contains(err.Error(), sensitiveMarker) || strings.Contains(err.Error(), baseURL) {
			t.Fatalf("NewOpenAICompatibleProvider() leaked base_url content: %v", err)
		}
	}
}

func TestEmbeddingTransportErrorsDoNotExposeBaseURL(t *testing.T) {
	const sensitiveMarker = "do-not-expose-transport-url"
	baseURL := "https://" + sensitiveMarker + ".example/private-path"
	client := &http.Client{Transport: securityRoundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("dial failure mentioning " + sensitiveMarker)
	})}
	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL: baseURL, Model: "embedding-model", HTTPClient: client,
	})

	_, err := provider.Embed(context.Background(), "hello")
	if err == nil {
		t.Fatal("Embed() returned nil error")
	}
	if strings.Contains(err.Error(), sensitiveMarker) || strings.Contains(err.Error(), baseURL) {
		t.Fatalf("Embed() leaked base_url content: %v", err)
	}
}

func TestEmbeddingsEndpointRejectsQueryAndFragment(t *testing.T) {
	const sensitiveMarker = "DO-NOT-EXPOSE-URL-COMPONENT"
	for _, baseURL := range []string{
		"https://example.test/v1?api_key=" + sensitiveMarker,
		"https://example.test/v1#" + sensitiveMarker,
	} {
		_, err := embeddingsEndpoint(baseURL, false)
		if err == nil || !strings.Contains(err.Error(), "must not include query or fragment") {
			t.Fatalf("embeddingsEndpoint() error = %v, want component rejection", err)
		}
		if strings.Contains(err.Error(), sensitiveMarker) || strings.Contains(err.Error(), baseURL) {
			t.Fatalf("embeddingsEndpoint() leaked base_url content: %v", err)
		}
	}
}

func TestNewOpenAICompatibleProviderAcceptsConfiguredLimits(t *testing.T) {
	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL:        "https://embedding.example/v1",
		Model:          "embedding-model",
		TimeoutSeconds: config.EmbeddingEndpointMaxTimeoutSeconds,
		MaxRetries:     config.EmbeddingEndpointMaxRetries,
	})
	if provider.timeout != time.Duration(config.EmbeddingEndpointMaxTimeoutSeconds)*time.Second {
		t.Fatalf("provider timeout = %s, want %ds", provider.timeout, config.EmbeddingEndpointMaxTimeoutSeconds)
	}
	if provider.maxRetries != config.EmbeddingEndpointMaxRetries {
		t.Fatalf("provider maxRetries = %d, want %d", provider.maxRetries, config.EmbeddingEndpointMaxRetries)
	}
}

type securityRoundTripFunc func(*http.Request) (*http.Response, error)

func (f securityRoundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}
