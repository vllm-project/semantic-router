package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// gatewayCredentialTestRouter serves one trusted gateway and one static-key
// backend under an enabled gateway credential contract.
func gatewayCredentialTestRouter(t *testing.T, extraGlobal string) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.3
providers:
  models:
    - name: gateway-model
      provider_model_id: gateway-model
      api_format: openai
      backend_refs:
        - name: gateway
          provider: openai-compatible
          endpoint: https://gateway.example/v1
          trusted_gateway: true
    - name: static-model
      provider_model_id: static-model
      api_format: openai
      backend_refs:
        - name: direct
          provider: openai-compatible
          endpoint: https://direct.example/v1
          api_key: static-secret
routing: {}
global:
  services:
    authz:
      gateway_credential:
        enabled: true
` + extraGlobal))
	if err != nil {
		t.Fatal(err)
	}
	return &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}
}

func gatewayCredentialProfile(t *testing.T, router *OpenAIRouter, endpoint string) *config.ProviderProfile {
	t.Helper()
	profile, err := router.Config.GetProviderProfileForEndpoint(endpoint)
	if err != nil {
		t.Fatal(err)
	}
	return profile
}

func requestHeadersMessage(pairs ...string) *ext_proc.ProcessingRequest_RequestHeaders {
	values := []*core.HeaderValue{
		{Key: ":method", Value: "POST"},
		{Key: ":path", Value: "/v1/chat/completions"},
		{Key: "content-type", Value: "application/json"},
	}
	for i := 0; i+1 < len(pairs); i += 2 {
		values = append(values, &core.HeaderValue{Key: pairs[i], Value: pairs[i+1]})
	}
	return &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: values}},
	}
}

func TestCaptureGatewayCredentialKeepsOnlyAnAuthenticatedValue(t *testing.T) {
	router := gatewayCredentialTestRouter(t, "")
	for _, tt := range []struct {
		name    string
		headers map[string]string
		want    string
	}{
		{
			name:    "with the trusted principal",
			headers: map[string]string{"X-Authz-User-Id": "alice", "X-Vsr-Gateway-Credential": " gw-key "},
			want:    "gw-key",
		},
		{
			name:    "without a principal the credential is dropped",
			headers: map[string]string{"x-vsr-gateway-credential": "gw-key"},
			want:    "",
		},
		{
			name:    "an empty value is nothing",
			headers: map[string]string{"x-authz-user-id": "alice", "x-vsr-gateway-credential": "  "},
			want:    "",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &RequestContext{Headers: tt.headers}
			router.captureGatewayCredential(ctx)
			if got, _ := trustedGatewayCredential(ctx); got != tt.want {
				t.Fatalf("credential = %q, want %q", got, tt.want)
			}
			if v := headerValueCI(ctx, headers.VSRGatewayCredential); v != "" {
				t.Fatalf("the header stayed on the request context: %q", v)
			}
		})
	}
}

func TestCaptureGatewayCredentialHonorsTheConfiguredHeaderAndStillRemovesTheDefault(t *testing.T) {
	router := gatewayCredentialTestRouter(t, "        header: X-Gateway-Token\n")
	ctx := &RequestContext{Headers: map[string]string{
		"x-authz-user-id":          "alice",
		"x-gateway-token":          "configured-key",
		"x-vsr-gateway-credential": "default-name-key",
	}}
	router.captureGatewayCredential(ctx)
	if got, _ := trustedGatewayCredential(ctx); got != "configured-key" {
		t.Fatalf("credential = %q, want the configured header's value", got)
	}
	for _, name := range []string{"x-gateway-token", headers.VSRGatewayCredential} {
		if v := headerValueCI(ctx, name); v != "" {
			t.Fatalf("%s stayed on the request context: %q", name, v)
		}
	}
	removal := router.gatewayCredentialHeadersForRemoval()
	if len(removal) != 2 || removal[0] != headers.VSRGatewayCredential || removal[1] != "x-gateway-token" {
		t.Fatalf("headers for removal = %v", removal)
	}
}

func TestCaptureGatewayCredentialIgnoresHeadersOnAHopAndWhenDisabled(t *testing.T) {
	hopRouter := gatewayCredentialTestRouter(t, "")
	hop := &RequestContext{
		Headers: map[string]string{"x-authz-user-id": "alice", "x-vsr-gateway-credential": "forged"},
		Hop:     &routing.Hop{Decision: "fusion", Iteration: 1},
	}
	hopRouter.captureGatewayCredential(hop)
	if got, ok := trustedGatewayCredential(hop); ok || got != "" {
		t.Fatalf("a hop believed a header: %q", got)
	}
	if v := headerValueCI(hop, headers.VSRGatewayCredential); v != "" {
		t.Fatalf("the hop kept the header: %q", v)
	}

	disabled := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{Headers: map[string]string{"x-authz-user-id": "alice", "x-vsr-gateway-credential": "gw-key"}}
	disabled.captureGatewayCredential(ctx)
	if got, ok := trustedGatewayCredential(ctx); ok || got != "" {
		t.Fatalf("a disabled contract kept a credential: %q", got)
	}
	if v := headerValueCI(ctx, headers.VSRGatewayCredential); v != "" {
		t.Fatalf("a disabled contract left the header on the request: %q", v)
	}
	if removal := disabled.gatewayCredentialHeadersForRemoval(); len(removal) != 1 || removal[0] != headers.VSRGatewayCredential {
		t.Fatalf("a disabled contract still removes the reserved header, got %v", removal)
	}
}

func TestRequestHeaderPhaseRemovesTheGatewayCredentialOnEveryContinuePath(t *testing.T) {
	router := gatewayCredentialTestRouter(t, "  router:\n    skip_processing:\n      enabled: true\n")
	for _, tt := range []struct {
		name string
		skip bool
	}{
		{name: "routed request", skip: false},
		{name: "skip-processing opt-out", skip: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			pairs := []string{"x-authz-user-id", "alice", "x-vsr-gateway-credential", "gw-key"}
			if tt.skip {
				pairs = append(pairs, headers.VSRSkipProcessing, "true")
			}
			ctx := &RequestContext{Headers: map[string]string{}}
			response, err := router.handleRequestHeaders(requestHeadersMessage(pairs...), ctx)
			if err != nil {
				t.Fatal(err)
			}
			if ctx.SkipProcessing != tt.skip {
				t.Fatalf("skip processing = %v, want %v", ctx.SkipProcessing, tt.skip)
			}
			mutation := response.GetRequestHeaders().GetResponse().GetHeaderMutation()
			if mutation == nil {
				t.Fatalf("no header mutation: %+v", response)
			}
			removed := false
			for _, name := range mutation.GetRemoveHeaders() {
				removed = removed || strings.EqualFold(name, headers.VSRGatewayCredential)
			}
			if !removed {
				t.Fatalf("remove headers = %v, want %s", mutation.GetRemoveHeaders(), headers.VSRGatewayCredential)
			}
			if got, _ := trustedGatewayCredential(ctx); got != "gw-key" {
				t.Fatalf("captured credential = %q", got)
			}
		})
	}
}

func TestTrustedGatewayDispatchPresentsTheCredentialUnderTheProviderAuthHeader(t *testing.T) {
	router := gatewayCredentialTestRouter(t, "")
	profile := gatewayCredentialProfile(t, router, "gateway-model_gateway")
	state := &routeHeaderState{profile: profile}
	ctx := &RequestContext{
		Headers:           map[string]string{"authorization": "Bearer raw-client-token", headers.UserOpenAIKey: "per-user-key"},
		gatewayCredential: "gw-key",
	}
	if response := router.appendProviderCredential(state, "gateway-model", "gateway-model_gateway", ctx); response != nil {
		t.Fatalf("trusted gateway dispatch returned %#v", response)
	}
	if len(state.setHeaders) != 1 {
		t.Fatalf("set headers = %#v, want the provider auth header once", state.setHeaders)
	}
	header := state.setHeaders[0]
	if header.GetHeader().GetKey() != "Authorization" || string(header.GetHeader().GetRawValue()) != "Bearer gw-key" {
		t.Fatalf("credential header = %#v, want Bearer gw-key", header)
	}
	if header.GetAppendAction() != core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD {
		t.Fatalf("append action = %v, want overwrite so the raw client token never travels", header.GetAppendAction())
	}
	for _, want := range []string{headers.VSRGatewayCredential, headers.UserOpenAIKey} {
		found := false
		for _, name := range state.removeHeaders {
			found = found || strings.EqualFold(name, want)
		}
		if !found {
			t.Fatalf("remove headers = %v, want %s", state.removeHeaders, want)
		}
	}
}

func TestTrustedGatewayDispatchFailsClosedWithoutTheCredential(t *testing.T) {
	router := gatewayCredentialTestRouter(t, "")
	profile := gatewayCredentialProfile(t, router, "gateway-model_gateway")
	state := &routeHeaderState{profile: profile}
	// Per-user and static keys are present and must not stand in.
	ctx := &RequestContext{Headers: map[string]string{
		headers.UserOpenAIKey: "per-user-key",
		"authorization":       "Bearer raw-client-token",
	}}
	response := router.appendProviderCredential(state, "gateway-model", "gateway-model_gateway", ctx)
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatalf("missing credential was not rejected: %#v", response)
	}
	if status := response.GetImmediateResponse().GetStatus().GetCode(); int(status) != http.StatusUnauthorized {
		t.Fatalf("status = %v, want 401", status)
	}
	body := string(response.GetImmediateResponse().GetBody())
	if !strings.Contains(body, "trusted gateway credential") || strings.Contains(body, "raw-client-token") {
		t.Fatalf("body = %q", body)
	}
	// The protocol contract encodes the typed error, so the client's reply
	// names the contract instead of the generic 401 message.
	if ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != trustedGatewayCredentialCode {
		t.Fatalf("protocol error = %+v", ctx.ImmediateProtocolError)
	}
	encoded := router.encodeImmediateResponseForClient(response, ctx).GetImmediateResponse().GetBody()
	var envelope struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(encoded, &envelope); err != nil {
		t.Fatalf("decode encoded body: %v (%s)", err, encoded)
	}
	if envelope.Error.Code != trustedGatewayCredentialCode || !strings.Contains(envelope.Error.Message, "trusted gateway credential") {
		t.Fatalf("encoded body = %s", encoded)
	}
	if len(state.setHeaders) != 0 {
		t.Fatalf("a rejected request still set headers: %#v", state.setHeaders)
	}
}

func TestStaticBackendNeverReceivesTheGatewayCredential(t *testing.T) {
	router := gatewayCredentialTestRouter(t, "")
	profile := gatewayCredentialProfile(t, router, "static-model_direct")
	state := &routeHeaderState{profile: profile}
	ctx := &RequestContext{Headers: map[string]string{}, gatewayCredential: "gw-key"}
	if response := router.appendProviderCredential(state, "static-model", "static-model_direct", ctx); response != nil {
		t.Fatalf("static dispatch returned %#v", response)
	}
	if len(state.setHeaders) != 1 || string(state.setHeaders[0].GetHeader().GetRawValue()) != "Bearer static-secret" {
		t.Fatalf("set headers = %#v, want the static key only", state.setHeaders)
	}
	found := false
	for _, name := range state.removeHeaders {
		found = found || strings.EqualFold(name, headers.VSRGatewayCredential)
	}
	if !found {
		t.Fatalf("remove headers = %v, want %s", state.removeHeaders, headers.VSRGatewayCredential)
	}
}

func TestFallbackAuthorizerResolvesTheSameTrustedGatewayContract(t *testing.T) {
	router := gatewayCredentialTestRouter(t, "")
	profile := gatewayCredentialProfile(t, router, "gateway-model_gateway")

	authorize, err := router.fallbackProviderAuthorizer(
		&RequestContext{Headers: map[string]string{"authorization": "Bearer raw-client-token"}, gatewayCredential: "gw-key"},
		profile, "gateway-model",
	)
	if err != nil {
		t.Fatal(err)
	}
	request, _ := http.NewRequest(http.MethodPost, "https://gateway.example/v1/chat/completions", nil)
	if authErr := authorize(context.Background(), request); authErr != nil {
		t.Fatal(authErr)
	}
	if got := request.Header.Get("Authorization"); got != "Bearer gw-key" {
		t.Fatalf("Authorization = %q, want Bearer gw-key", got)
	}

	_, err = router.fallbackProviderAuthorizer(&RequestContext{Headers: map[string]string{headers.UserOpenAIKey: "per-user-key"}}, profile, "gateway-model")
	if !errors.Is(err, errTrustedGatewayCredentialMissing) {
		t.Fatalf("missing credential error = %v", err)
	}
}

func TestRouterInitiatedCallsCannotSpendTheGatewayCredential(t *testing.T) {
	router := gatewayCredentialTestRouter(t, "")
	profile := gatewayCredentialProfile(t, router, "gateway-model_gateway")
	if _, err := configuredProviderAuthorizer(router.Config, profile, "gateway-model"); !errors.Is(err, errTrustedGatewayRouterInitiated) {
		t.Fatalf("configured authorizer error = %v", err)
	}
	static := gatewayCredentialProfile(t, router, "static-model_direct")
	if _, err := configuredProviderAuthorizer(router.Config, static, "static-model"); err != nil {
		t.Fatalf("static backend authorizer error = %v", err)
	}
}

func TestAHopInheritsTheClientRequestsGatewayCredentialTyped(t *testing.T) {
	router := &OpenAIRouter{}
	client := &RequestContext{gatewayCredential: "gw-key"}
	ctx := withGatewayCredential(context.Background(), client)
	hop := router.newRoutingSession(routing.WithHop(ctx, routing.Hop{Decision: "fusion", Iteration: 1}), nil)
	if got, _ := trustedGatewayCredential(hop.ctx); got != "gw-key" {
		t.Fatalf("hop credential = %q, want the client request's", got)
	}
	plain := router.newRoutingSession(ctx, nil)
	if got, ok := trustedGatewayCredential(plain.ctx); ok || got != "" {
		t.Fatalf("a client session took a credential from its context: %q", got)
	}
	if withGatewayCredential(context.Background(), &RequestContext{}) != context.Background() {
		t.Fatal("a request without a credential changed the context")
	}
}

func TestTrustedGatewayCredentialIsNotResolvedThroughTheCredentialChain(t *testing.T) {
	// The chain's per-user and static providers must not be consulted for a
	// trusted gateway: a fail-closed chain with no key would otherwise 401
	// for the wrong reason, and a fail-open one would dispatch unauthenticated.
	router := gatewayCredentialTestRouter(t, "")
	router.CredentialResolver = authz.NewCredentialResolver(authz.NewStaticConfigProvider(router.Config))
	profile := gatewayCredentialProfile(t, router, "gateway-model_gateway")
	state := &routeHeaderState{profile: profile}
	ctx := &RequestContext{Headers: map[string]string{}, gatewayCredential: "gw-key"}
	if response := router.appendProviderCredential(state, "gateway-model", "gateway-model_gateway", ctx); response != nil {
		t.Fatalf("fail-closed chain interfered with the trusted gateway: %#v", response)
	}
	if len(state.setHeaders) != 1 || string(state.setHeaders[0].GetHeader().GetRawValue()) != "Bearer gw-key" {
		t.Fatalf("set headers = %#v", state.setHeaders)
	}
}
