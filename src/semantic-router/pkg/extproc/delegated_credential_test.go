package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// delegatedCredentialTestRouter serves one delegated and one static-key
// backend under an enabled delegated credential contract.
func delegatedCredentialTestRouter(t *testing.T, extraGlobal string) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.5
providers:
  models:
    - name: gateway-model
      provider_model_id: gateway-model
      api_format: openai
      backend_refs:
        - name: gateway
          provider: openai-compatible
          endpoint: https://gateway.example/v1
          auth: {credential: delegated}
    - name: static-model
      provider_model_id: static-model
      api_format: openai
      backend_refs:
        - name: direct
          provider: openai-compatible
          endpoint: https://direct.example/v1
          api_key: static-secret
recipes:
  - name: default
global:
  services:
    authz:
      delegated_credential:
        enabled: true
` + extraGlobal))
	if err != nil {
		t.Fatal(err)
	}
	return &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}
}

func delegatedCredentialProfile(t *testing.T, router *OpenAIRouter, endpoint string) *config.ProviderProfile {
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

func TestCaptureDelegatedCredentialKeepsOnlyAnAuthenticatedValue(t *testing.T) {
	router := delegatedCredentialTestRouter(t, "")
	for _, tt := range []struct {
		name    string
		headers map[string]string
		want    string
	}{
		{
			name:    "with the trusted principal",
			headers: map[string]string{"X-Authz-User-Id": "alice", "X-Vsr-Delegated-Credential": " gw-key "},
			want:    "gw-key",
		},
		{
			name:    "without a principal the credential is dropped",
			headers: map[string]string{"x-vsr-delegated-credential": "gw-key"},
			want:    "",
		},
		{
			name:    "an empty value is nothing",
			headers: map[string]string{"x-authz-user-id": "alice", "x-vsr-delegated-credential": "  "},
			want:    "",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &RequestContext{Headers: tt.headers}
			router.captureDelegatedCredential(ctx)
			if got, _ := requestDelegatedCredential(ctx); got != tt.want {
				t.Fatalf("credential = %q, want %q", got, tt.want)
			}
			if v := headerValueCI(ctx, headers.VSRDelegatedCredential); v != "" {
				t.Fatalf("the header stayed on the request context: %q", v)
			}
		})
	}
}

func TestCaptureDelegatedCredentialHonorsTheConfiguredHeaderAndStillRemovesTheDefault(t *testing.T) {
	router := delegatedCredentialTestRouter(t, "        header: X-Gateway-Token\n")
	ctx := &RequestContext{Headers: map[string]string{
		"x-authz-user-id":            "alice",
		"x-gateway-token":            "configured-key",
		"x-vsr-delegated-credential": "default-name-key",
	}}
	router.captureDelegatedCredential(ctx)
	if got, _ := requestDelegatedCredential(ctx); got != "configured-key" {
		t.Fatalf("credential = %q, want the configured header's value", got)
	}
	for _, name := range []string{"x-gateway-token", headers.VSRDelegatedCredential} {
		if v := headerValueCI(ctx, name); v != "" {
			t.Fatalf("%s stayed on the request context: %q", name, v)
		}
	}
	removal := router.delegatedCredentialHeadersForRemoval()
	if len(removal) != 2 || removal[0] != headers.VSRDelegatedCredential || removal[1] != "x-gateway-token" {
		t.Fatalf("headers for removal = %v", removal)
	}
}

func TestCaptureDelegatedCredentialIgnoresHeadersOnAHopAndWhenDisabled(t *testing.T) {
	hopRouter := delegatedCredentialTestRouter(t, "")
	hop := &RequestContext{
		Headers: map[string]string{"x-authz-user-id": "alice", "x-vsr-delegated-credential": "forged"},
		Hop:     &routing.Hop{Decision: "fusion", Iteration: 1},
	}
	hopRouter.captureDelegatedCredential(hop)
	if got, ok := requestDelegatedCredential(hop); ok || got != "" {
		t.Fatalf("a hop believed a header: %q", got)
	}
	if v := headerValueCI(hop, headers.VSRDelegatedCredential); v != "" {
		t.Fatalf("the hop kept the header: %q", v)
	}

	disabled := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{Headers: map[string]string{"x-authz-user-id": "alice", "x-vsr-delegated-credential": "gw-key"}}
	disabled.captureDelegatedCredential(ctx)
	if got, ok := requestDelegatedCredential(ctx); ok || got != "" {
		t.Fatalf("a disabled contract kept a credential: %q", got)
	}
	if v := headerValueCI(ctx, headers.VSRDelegatedCredential); v != "" {
		t.Fatalf("a disabled contract left the header on the request: %q", v)
	}
	if removal := disabled.delegatedCredentialHeadersForRemoval(); len(removal) != 1 || removal[0] != headers.VSRDelegatedCredential {
		t.Fatalf("a disabled contract still removes the reserved header, got %v", removal)
	}
}

func TestRequestHeaderPhaseRemovesTheDelegatedCredentialOnEveryContinuePath(t *testing.T) {
	router := delegatedCredentialTestRouter(t, "  router:\n    skip_processing:\n      enabled: true\n")
	for _, tt := range []struct {
		name string
		skip bool
	}{
		{name: "routed request", skip: false},
		{name: "skip-processing opt-out", skip: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			pairs := []string{"x-authz-user-id", "alice", "x-vsr-delegated-credential", "gw-key"}
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
				removed = removed || strings.EqualFold(name, headers.VSRDelegatedCredential)
			}
			if !removed {
				t.Fatalf("remove headers = %v, want %s", mutation.GetRemoveHeaders(), headers.VSRDelegatedCredential)
			}
			if got, _ := requestDelegatedCredential(ctx); got != "gw-key" {
				t.Fatalf("captured credential = %q", got)
			}
		})
	}
}

func TestDelegatedDispatchPresentsTheCredentialUnderTheProviderAuthHeader(t *testing.T) {
	router := delegatedCredentialTestRouter(t, "")
	profile := delegatedCredentialProfile(t, router, "gateway-model_gateway")
	state := &routeHeaderState{profile: profile}
	ctx := &RequestContext{
		Headers:      map[string]string{"authorization": "Bearer raw-client-token", headers.UserOpenAIKey: "per-user-key"},
		TraceContext: principal.ForRequest(context.Background(), "gw-key"),
	}
	if response := router.appendProviderCredential(state, "gateway-model", "gateway-model_gateway", ctx); response != nil {
		t.Fatalf("delegated dispatch returned %#v", response)
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
	for _, want := range []string{headers.VSRDelegatedCredential, headers.UserOpenAIKey} {
		found := false
		for _, name := range state.removeHeaders {
			found = found || strings.EqualFold(name, want)
		}
		if !found {
			t.Fatalf("remove headers = %v, want %s", state.removeHeaders, want)
		}
	}
}

func TestDelegatedBackendOnAnotherAuthHeaderDropsTheClientsAuthorization(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.5
providers:
  models:
    - name: claude-gateway
      provider_model_id: claude-gateway
      api_format: anthropic
      backend_refs:
        - name: gateway
          provider: anthropic
          endpoint: https://gateway.example/v1
          auth: {credential: delegated}
recipes:
  - name: default
global:
  services:
    authz:
      delegated_credential:
        enabled: true
`))
	if err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}
	state := &routeHeaderState{profile: delegatedCredentialProfile(t, router, "claude-gateway_gateway")}
	ctx := &RequestContext{
		Headers:      map[string]string{"authorization": "Bearer raw-client-token"},
		TraceContext: principal.ForRequest(context.Background(), "gw-key"),
	}
	if response := router.appendProviderCredential(state, "claude-gateway", "claude-gateway_gateway", ctx); response != nil {
		t.Fatalf("delegated dispatch returned %#v", response)
	}
	if len(state.setHeaders) != 1 || !strings.EqualFold(state.setHeaders[0].GetHeader().GetKey(), "x-api-key") ||
		string(state.setHeaders[0].GetHeader().GetRawValue()) != "gw-key" {
		t.Fatalf("set headers = %#v, want x-api-key: gw-key", state.setHeaders)
	}
	removed := map[string]bool{}
	for _, name := range state.removeHeaders {
		removed[strings.ToLower(name)] = true
	}
	for _, want := range []string{"authorization", "api-key", "cookie"} {
		if !removed[want] {
			t.Fatalf("remove headers = %v, want the client's %s removed", state.removeHeaders, want)
		}
	}
	if removed["x-api-key"] {
		t.Fatalf("remove headers = %v drop the header that carries the credential", state.removeHeaders)
	}
}

func TestDelegatedDispatchFailsClosedWithoutTheCredential(t *testing.T) {
	router := delegatedCredentialTestRouter(t, "")
	profile := delegatedCredentialProfile(t, router, "gateway-model_gateway")
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
	if !strings.Contains(body, "delegated credential") || strings.Contains(body, "raw-client-token") {
		t.Fatalf("body = %q", body)
	}
	// The protocol contract encodes the typed error, so the client's reply
	// names the contract instead of the generic 401 message.
	if ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != delegatedCredentialCode {
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
	if envelope.Error.Code != delegatedCredentialCode || !strings.Contains(envelope.Error.Message, "delegated credential") {
		t.Fatalf("encoded body = %s", encoded)
	}
	if len(state.setHeaders) != 0 {
		t.Fatalf("a rejected request still set headers: %#v", state.setHeaders)
	}
}

func TestStaticBackendNeverReceivesTheDelegatedCredential(t *testing.T) {
	router := delegatedCredentialTestRouter(t, "")
	profile := delegatedCredentialProfile(t, router, "static-model_direct")
	state := &routeHeaderState{profile: profile}
	ctx := &RequestContext{Headers: map[string]string{}, delegatedCredential: "gw-key"}
	if response := router.appendProviderCredential(state, "static-model", "static-model_direct", ctx); response != nil {
		t.Fatalf("static dispatch returned %#v", response)
	}
	if len(state.setHeaders) != 1 || string(state.setHeaders[0].GetHeader().GetRawValue()) != "Bearer static-secret" {
		t.Fatalf("set headers = %#v, want the static key only", state.setHeaders)
	}
	found := false
	for _, name := range state.removeHeaders {
		found = found || strings.EqualFold(name, headers.VSRDelegatedCredential)
	}
	if !found {
		t.Fatalf("remove headers = %v, want %s", state.removeHeaders, headers.VSRDelegatedCredential)
	}
}

func TestFallbackAuthorizerResolvesTheSameDelegatedContract(t *testing.T) {
	router := delegatedCredentialTestRouter(t, "")
	profile := delegatedCredentialProfile(t, router, "gateway-model_gateway")

	authorize, err := router.fallbackProviderAuthorizer(
		&RequestContext{Headers: map[string]string{"authorization": "Bearer raw-client-token"}, delegatedCredential: "gw-key"},
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
	if !errors.Is(err, principal.ErrCredentialMissing) {
		t.Fatalf("missing credential error = %v", err)
	}
}

func TestRouterOwnCallsCannotPresentTheDelegatedCredential(t *testing.T) {
	router := delegatedCredentialTestRouter(t, "")
	profile := delegatedCredentialProfile(t, router, "gateway-model_gateway")
	if _, err := configuredProviderAuthorizer(router.Config, profile, "gateway-model", "test"); !errors.Is(err, principal.ErrRouterCall) {
		t.Fatalf("configured authorizer error = %v", err)
	}
	static := delegatedCredentialProfile(t, router, "static-model_direct")
	if _, err := configuredProviderAuthorizer(router.Config, static, "static-model", "test"); err != nil {
		t.Fatalf("static backend authorizer error = %v", err)
	}
}

func TestAHopSharesTheClientRequestsPrincipal(t *testing.T) {
	router := &OpenAIRouter{}
	client := principal.ForRequest(context.Background(), "gw-key")
	hop := router.newRoutingSession(routing.WithHop(client, routing.Hop{Decision: "fusion", Iteration: 1}), nil)
	hop.ctx.Headers["x-vsr-delegated-credential"] = "forged"
	router.captureDelegatedCredential(hop.ctx)
	if got, _ := requestDelegatedCredential(hop.ctx); got != "gw-key" {
		t.Fatalf("hop credential = %q, want the client request's", got)
	}
	if got, err := principal.Delegated(hop.ctx.TraceContext); err != nil || got != "gw-key" {
		t.Fatalf("a hop's calls act for the client request: %q, %v", got, err)
	}
	plain := router.newRoutingSession(client, nil)
	router.captureDelegatedCredential(plain.ctx)
	if got, ok := requestDelegatedCredential(plain.ctx); ok || got != "" {
		t.Fatalf("a client session took a credential from its context: %q", got)
	}
}

func TestARefusedCallAnswersTheRequest401(t *testing.T) {
	router := delegatedCredentialTestRouter(t, "")
	ctx := &RequestContext{Headers: map[string]string{}, TraceContext: principal.ForRequest(context.Background(), "")}
	if router.delegatedCredentialRefusal(ctx) != nil {
		t.Fatal("a request whose calls were all made answered 401")
	}
	if _, err := principal.Delegated(ctx.TraceContext); !errors.Is(err, principal.ErrCredentialMissing) {
		t.Fatalf("a decision-model call without the credential: %v", err)
	}
	response := router.delegatedCredentialRefusal(ctx)
	if response == nil || int(response.GetImmediateResponse().GetStatus().GetCode()) != http.StatusUnauthorized {
		t.Fatalf("a refused call answers 401, got %#v", response)
	}
	if ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != delegatedCredentialCode {
		t.Fatalf("protocol error = %+v", ctx.ImmediateProtocolError)
	}
}

func TestDelegatedCredentialIsNotResolvedThroughTheCredentialChain(t *testing.T) {
	// The chain's per-user and static providers must not be consulted for a
	// delegated backend: a fail-closed chain with no key would otherwise 401
	// for the wrong reason, and a fail-open one would dispatch unauthenticated.
	router := delegatedCredentialTestRouter(t, "")
	router.CredentialResolver = authz.NewCredentialResolver(authz.NewStaticConfigProvider(router.Config))
	profile := delegatedCredentialProfile(t, router, "gateway-model_gateway")
	state := &routeHeaderState{profile: profile}
	ctx := &RequestContext{Headers: map[string]string{}, TraceContext: principal.ForRequest(context.Background(), "gw-key")}
	if response := router.appendProviderCredential(state, "gateway-model", "gateway-model_gateway", ctx); response != nil {
		t.Fatalf("fail-closed chain interfered with the delegated backend: %#v", response)
	}
	if len(state.setHeaders) != 1 || string(state.setHeaders[0].GetHeader().GetRawValue()) != "Bearer gw-key" {
		t.Fatalf("set headers = %#v", state.setHeaders)
	}
}

// A Looper run that a hop or an accept check made for the request was refused
// in, for want of the request's delegated credential, answers 401 in both
// gateway modes, whether another model's answer came back or none did.
func TestALooperRunRefusedForTheDelegatedCredentialAnswers401(t *testing.T) {
	const escalate = `
      modelRefs: [{model: model-a}, {model: model-b}]
      algorithm:
        type: escalate
        escalate:
          accept: ACCEPT`
	const confidence = `{confidence: {method: avg_logprob, gte: 0.6}}`
	const contract = `global:
  services:
    authz:
      delegated_credential:
        enabled: true
`
	const checker = `  model_catalog:
    deployments:
      checker:
        provider: systemone
        endpoint: http://checker.invalid/v1/systemone
        model: checker
        auth: {credential: delegated}
`
	for _, tc := range []struct {
		name, accept, deployments string
		delegated                 []string
	}{
		{name: "the hop to the one delegated model", accept: confidence, delegated: []string{"model-b"}},
		{name: "every hop", accept: confidence, delegated: []string{"model-a", "model-b"}},
		{name: "the accept check", accept: `{systemone: {deployment: checker, gte: 0.5}}`, deployments: checker},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend := &looperBackend{}
			server := httptest.NewServer(backend)
			t.Cleanup(server.Close)
			decision := strings.ReplaceAll(strings.TrimPrefix(strings.Replace(escalate, "ACCEPT", tc.accept, 1), "\n"), "\n", "\n  ")
			yaml := strings.NewReplacer(
				"BACKEND", strings.TrimPrefix(server.URL, "http://"),
				"DECISION", "  "+decision+"\n",
				"global:\n", contract+tc.deployments,
			).Replace(looperEquivalenceConfig)
			for _, model := range tc.delegated {
				backendRef := "{name: fake-" + strings.TrimPrefix(model, "model-") + ", endpoint: " + strings.TrimPrefix(server.URL, "http://") + ", protocol: http, provider: vllm"
				yaml = strings.Replace(yaml, backendRef+"}", backendRef+", auth: {credential: delegated}}", 1)
			}
			cfg, err := config.ParseYAMLBytes([]byte(yaml))
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			router := newLooperRouter(t, cfg)
			harness := &looperHarness{router: router, backend: backend, set: router.upstreamSet()}
			for _, adapter := range []string{"extproc", "sessions"} {
				record := harness.serve(t, looperAdapters[adapter](router), looperClientRequest)
				if record.Response.Status != http.StatusUnauthorized || !strings.Contains(string(record.Response.Body), delegatedCredentialCode) {
					t.Errorf("%s: reply = %d %s, want 401 %s", adapter, record.Response.Status, record.Response.Body, delegatedCredentialCode)
				}
				for _, call := range record.Calls {
					if slices.Contains(tc.delegated, call.Model) {
						t.Errorf("%s: %s was called without the credential", adapter, call.Model)
					}
				}
			}
		})
	}
}
