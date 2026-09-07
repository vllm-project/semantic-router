package handlers

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"net/url"
	"strings"
	"testing"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

type modelDiscoveryRoundTripFunc func(*http.Request) (*http.Response, error)

func (function modelDiscoveryRoundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func TestModelDiscoveryHandlerListsAndSortsProviderModels(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer secret" {
			t.Fatalf("authorization = %q", got)
		}
		_, _ = w.Write([]byte(`{"data":[{"id":"zeta"},{"id":"alpha"},{"id":"alpha"}]}`))
	}))
	defer provider.Close()

	request := httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(
		`{"baseUrl":"`+provider.URL+`/v1","apiKey":"secret","provider":"vllm"}`,
	))
	response := httptest.NewRecorder()
	ModelDiscoveryHandler(provider.Client()).ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", response.Code, response.Body.String())
	}
	if got := response.Body.String(); !strings.Contains(got, `"models":["alpha","zeta"]`) {
		t.Fatalf("body = %s", got)
	}
}

func TestModelDiscoveryHandlerUsesAnthropicHeaders(t *testing.T) {
	registry, err := modelcatalog.BuiltIn()
	if err != nil {
		t.Fatalf("catalog.BuiltIn(): %v", err)
	}
	provider, ok := registry.Provider("anthropic")
	if !ok {
		t.Fatal("built-in Anthropic provider is missing")
	}
	request := httptest.NewRequest(http.MethodGet, "https://api.anthropic.com/v1/models", nil)
	applyModelDiscoveryHeaders(request, provider, "secret")
	if got := request.Header.Get("x-api-key"); got != "secret" {
		t.Fatalf("x-api-key = %q", got)
	}
	if got := request.Header.Get("anthropic-version"); got != "2023-06-01" {
		t.Fatalf("anthropic-version = %q", got)
	}
}

func TestModelDiscoveryHandlerRejectsInvalidBaseURL(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(
		`{"baseUrl":"file:///etc/passwd","provider":"openai"}`,
	))
	response := httptest.NewRecorder()
	ModelDiscoveryHandler(nil).ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body = %s", response.Code, response.Body.String())
	}
}

func TestModelDiscoveryHandlerDoesNotForwardCredentialsAcrossRedirects(t *testing.T) {
	targetCalled := false
	target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		targetCalled = true
	}))
	defer target.Close()
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Location", target.URL)
		w.WriteHeader(http.StatusTemporaryRedirect)
	}))
	defer provider.Close()

	request := httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(
		`{"baseUrl":"`+provider.URL+`","apiKey":"secret","provider":"vllm"}`,
	))
	response := httptest.NewRecorder()
	ModelDiscoveryHandler(provider.Client()).ServeHTTP(response, request)

	if response.Code != http.StatusBadGateway {
		t.Fatalf("status = %d body = %s", response.Code, response.Body.String())
	}
	if targetCalled {
		t.Fatal("redirect target received the credentialed discovery request")
	}
}

func TestModelDiscoveryHandlerPinsCloudProvidersToCatalogOrigin(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		called = true
	}))
	defer server.Close()

	request := httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(
		`{"baseUrl":"`+server.URL+`/v1","apiKey":"secret","provider":"openai"}`,
	))
	response := httptest.NewRecorder()
	ModelDiscoveryHandler(server.Client()).ServeHTTP(response, request)

	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "built-in provider origin") {
		t.Fatalf("status = %d body = %s", response.Code, response.Body.String())
	}
	if called {
		t.Fatal("untrusted cloud-provider origin received the credentialed request")
	}
}

func TestModelInventoryTargetUsesProviderOperationAndFixedOrigin(t *testing.T) {
	registry, err := modelcatalog.BuiltIn()
	if err != nil {
		t.Fatalf("catalog.BuiltIn(): %v", err)
	}
	provider, ok := registry.Provider("openai")
	if !ok {
		t.Fatal("built-in OpenAI provider is missing")
	}
	target, err := modelInventoryTarget("https://api.openai.com:443/v1", registry, provider)
	if err != nil {
		t.Fatalf("modelInventoryTarget(): %v", err)
	}
	if target.url != "https://api.openai.com:443/v1/models" {
		t.Fatalf("target URL = %q", target.url)
	}
	if target.policy.allowPrivate {
		t.Fatal("fixed cloud origin unexpectedly permits private addresses")
	}
}

func TestModelDiscoveryHandlerRejectsLinkLocalRuntimeTarget(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(
		`{"baseUrl":"http://169.254.169.254/latest","provider":"vllm"}`,
	))
	response := httptest.NewRecorder()
	ModelDiscoveryHandler(nil).ServeHTTP(response, request)

	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "restricted network address") {
		t.Fatalf("status = %d body = %s", response.Code, response.Body.String())
	}
}

func TestModelDiscoveryNetworkPolicySeparatesCloudAndSelfHostedTargets(t *testing.T) {
	cloud := modelDiscoveryNetworkPolicy{}
	selfHosted := modelDiscoveryNetworkPolicy{allowPrivate: true}
	for _, test := range []struct {
		name    string
		address string
		policy  modelDiscoveryNetworkPolicy
		want    bool
	}{
		{name: "cloud public", address: "8.8.8.8", policy: cloud, want: true},
		{name: "cloud loopback", address: "127.0.0.1", policy: cloud, want: false},
		{name: "cloud private", address: "10.0.0.8", policy: cloud, want: false},
		{name: "runtime loopback", address: "127.0.0.1", policy: selfHosted, want: true},
		{name: "runtime private", address: "10.0.0.8", policy: selfHosted, want: true},
		{name: "runtime ipv6 loopback", address: "::1", policy: selfHosted, want: true},
		{name: "runtime link local", address: "169.254.169.254", policy: selfHosted, want: false},
		{name: "runtime ipv6 link local", address: "fe80::1", policy: selfHosted, want: false},
		{name: "runtime carrier grade metadata", address: "100.100.100.200", policy: selfHosted, want: false},
		{name: "runtime this network alias", address: "0.0.0.1", policy: selfHosted, want: false},
		{name: "runtime azure host agent", address: "168.63.129.16", policy: selfHosted, want: false},
		{name: "runtime ipv4 reserved", address: "240.0.0.1", policy: selfHosted, want: false},
		{name: "runtime nat64 translation", address: "64:ff9b::7f00:1", policy: selfHosted, want: false},
		{name: "runtime ipv6 site local", address: "fec0::1", policy: selfHosted, want: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := modelDiscoveryAddressAllowed(netip.MustParseAddr(test.address), test.policy); got != test.want {
				t.Fatalf("modelDiscoveryAddressAllowed(%s) = %v, want %v", test.address, got, test.want)
			}
		})
	}
}

func TestModelDiscoveryDialerRejectsDNSResolvingToPrivateForCloud(t *testing.T) {
	ctx := withModelDiscoveryNetworkPolicy(context.Background(), modelDiscoveryNetworkPolicy{})
	if _, err := modelDiscoveryDialContext(ctx, "tcp", "localhost:80"); err == nil || !strings.Contains(err.Error(), "restricted network address") {
		t.Fatalf("modelDiscoveryDialContext() error = %v", err)
	}
}

func TestSecureModelDiscoveryClientCarriesDNSPolicyToTransport(t *testing.T) {
	client := secureModelDiscoveryClient(nil)
	ctx := withModelDiscoveryNetworkPolicy(context.Background(), modelDiscoveryNetworkPolicy{})
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://localhost/v1/models", nil)
	if err != nil {
		t.Fatalf("http.NewRequestWithContext(): %v", err)
	}
	response, err := client.Do(request)
	if response != nil {
		defer response.Body.Close()
	}
	if err == nil || !strings.Contains(err.Error(), "restricted network address") {
		t.Fatalf("secure discovery request error = %v", err)
	}
}

func TestSecureModelDiscoveryTransportCannotBypassPolicyWithProxyOrTLSDialer(t *testing.T) {
	base := http.DefaultTransport.(*http.Transport).Clone()
	base.Proxy = http.ProxyFromEnvironment
	base.DialTLSContext = func(context.Context, string, string) (net.Conn, error) {
		return nil, nil
	}

	secured := secureModelDiscoveryTransport(base)
	if secured.Proxy != nil || secured.DialTLS != nil || secured.DialTLSContext != nil || !secured.DisableKeepAlives {
		t.Fatal("secure transport retained a network-policy bypass")
	}
}

func TestSecureModelDiscoveryClientReplacesOpaqueTransport(t *testing.T) {
	client := secureModelDiscoveryClient(&http.Client{Transport: modelDiscoveryRoundTripFunc(func(*http.Request) (*http.Response, error) {
		t.Fatal("opaque transport bypassed the discovery network policy")
		return nil, nil
	})})
	if _, ok := client.Transport.(*http.Transport); !ok {
		t.Fatalf("secure client transport = %T, want *http.Transport", client.Transport)
	}
}

func TestModelDiscoveryNetworkPolicyRejectsUnknownProviderCategory(t *testing.T) {
	_, err := modelDiscoveryNetworkPolicyForProvider(
		mustParseModelDiscoveryURL(t, "https://models.example/v1"),
		modelcatalog.ProviderDefinition{
			ID:             "future-provider",
			Category:       "future-category",
			DefaultBaseURL: "https://models.example/v1",
		},
	)
	if err == nil || !strings.Contains(err.Error(), "network policy") {
		t.Fatalf("modelDiscoveryNetworkPolicyForProvider() error = %v", err)
	}
}

func mustParseModelDiscoveryURL(t *testing.T, raw string) *url.URL {
	t.Helper()
	parsed, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("url.Parse(%q): %v", raw, err)
	}
	return parsed
}

func TestModelDiscoveryHandlerRejectsUnknownProvider(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(
		`{"baseUrl":"https://example.com/v1","provider":"operator-invented"}`,
	))
	response := httptest.NewRecorder()
	ModelDiscoveryHandler(nil).ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body = %s", response.Code, response.Body.String())
	}
}

func TestModelDiscoveryHandlerRejectsUndeclaredProviderOperation(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(
		`{"baseUrl":"https://example.openai.azure.com/openai/deployments/example","provider":"azure-openai"}`,
	))
	response := httptest.NewRecorder()
	ModelDiscoveryHandler(nil).ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "does not declare model discovery support") {
		t.Fatalf("status = %d body = %s", response.Code, response.Body.String())
	}
}
