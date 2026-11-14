package extproc

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// callerCredentialTestRouter serves Claude models on Anthropic with the
// client's own login and no key.
func callerCredentialTestRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.5
providers:
  models:
    - name: claude-sonnet
      provider_model_id: claude-sonnet-5-5
      api_format: anthropic
      backend_refs:
        - name: anthropic
          provider: anthropic
          endpoint: https://api.anthropic.com
          auth: {credential: caller}
recipes:
  - name: default
`))
	if err != nil {
		t.Fatal(err)
	}
	return &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}
}

const callerTestEndpoint = "claude-sonnet_anthropic"

func TestCallerDispatchSendsTheClientsLoginInTheHeaderItCameIn(t *testing.T) {
	router := callerCredentialTestRouter(t)
	profile := delegatedCredentialProfile(t, router, callerTestEndpoint)
	for header, value := range map[string]string{"authorization": "Bearer sk-ant-oat01-login", "x-api-key": "sk-ant-api03-key"} {
		ctx := &RequestContext{Headers: map[string]string{header: value, headers.UserAnthropicKey: "per-user-key"}}
		router.captureCallerCredential(ctx)
		state := &routeHeaderState{profile: profile}
		if response := router.appendProviderCredential(state, "claude-sonnet", callerTestEndpoint, ctx); response != nil {
			t.Fatalf("%s: caller dispatch returned %#v", header, response)
		}
		if len(state.setHeaders) != 1 || state.setHeaders[0].GetHeader().GetKey() != header || string(state.setHeaders[0].GetHeader().GetRawValue()) != value {
			t.Fatalf("%s: set headers = %#v, want the client's own value under its own header", header, state.setHeaders)
		}
		// The other carrier and every per-user key header leave the request.
		other := map[string]string{"authorization": "x-api-key", "x-api-key": "authorization"}[header]
		for _, want := range []string{other, headers.UserAnthropicKey} {
			if !containsFold(state.removeHeaders, want) {
				t.Fatalf("%s: remove headers = %v, want %s", header, state.removeHeaders, want)
			}
		}
	}
}

func TestCallerDispatchFailsClosedWithoutALogin(t *testing.T) {
	router := callerCredentialTestRouter(t)
	profile := delegatedCredentialProfile(t, router, callerTestEndpoint)
	ctx := &RequestContext{Headers: map[string]string{headers.UserAnthropicKey: "per-user-key"}}
	router.captureCallerCredential(ctx)
	state := &routeHeaderState{profile: profile}
	response := router.appendProviderCredential(state, "claude-sonnet", callerTestEndpoint, ctx)
	if response == nil || response.GetImmediateResponse() == nil || int(response.GetImmediateResponse().GetStatus().GetCode()) != http.StatusUnauthorized {
		t.Fatalf("a request without a login was not answered 401: %#v", response)
	}
	if ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != callerCredentialCode {
		t.Fatalf("protocol error = %+v", ctx.ImmediateProtocolError)
	}
	if len(state.setHeaders) != 0 {
		t.Fatalf("a refused request still set headers: %#v", state.setHeaders)
	}
}

func TestAHopNeverCarriesTheCallersLogin(t *testing.T) {
	router := callerCredentialTestRouter(t)
	ctx := &RequestContext{Headers: map[string]string{"authorization": "Bearer sk-ant-oat01-login"}, Hop: &routing.Hop{}}
	router.captureCallerCredential(ctx)
	if ctx.callerCredential.value != "" {
		t.Fatalf("a hop kept the caller's login: %+v", ctx.callerCredential.header)
	}
	if _, err := router.fallbackProviderAuthorizer(ctx, delegatedCredentialProfile(t, router, callerTestEndpoint), "claude-sonnet"); !errors.Is(err, errCallerCredentialMissing) {
		t.Fatalf("a hop's fallback authorizer error = %v", err)
	}
}

func TestCallerFallbackPresentsTheSameLogin(t *testing.T) {
	router := callerCredentialTestRouter(t)
	ctx := &RequestContext{Headers: map[string]string{"authorization": "Bearer sk-ant-oat01-login", "anthropic-beta": "claude-code-20250219,oauth-2025-04-20"}}
	router.captureCallerCredential(ctx)
	authorize, err := router.fallbackProviderAuthorizer(ctx, delegatedCredentialProfile(t, router, callerTestEndpoint), "claude-sonnet")
	if err != nil {
		t.Fatal(err)
	}
	request, _ := http.NewRequest(http.MethodPost, "https://api.anthropic.com/v1/messages", nil)
	request.Header.Set("x-api-key", "stale")
	if err := authorize(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if request.Header.Get("Authorization") != "Bearer sk-ant-oat01-login" || request.Header.Get("x-api-key") != "" ||
		request.Header.Get("anthropic-beta") != "claude-code-20250219,oauth-2025-04-20" {
		t.Fatalf("fallback headers = %v", request.Header)
	}
}

func TestAnEchoedLoginIsRedactedFromWhatTheRouterPassesOnOrKeeps(t *testing.T) {
	router := callerCredentialTestRouter(t)
	ctx := &RequestContext{Headers: map[string]string{"authorization": "Bearer sk-ant-oat01-login"}}
	router.captureCallerCredential(ctx)
	body := redactCallerCredential(ctx, []byte(`{"error":{"message":"bad token sk-ant-oat01-login in Bearer sk-ant-oat01-login"}}`))
	if strings.Contains(string(body), "sk-ant-oat01-login") || !strings.Contains(string(body), "[redacted]") {
		t.Fatalf("redacted body = %s", body)
	}
	if got := redactCallerCredential(&RequestContext{}, []byte("unchanged")); string(got) != "unchanged" {
		t.Fatalf("a request without a login changed its body: %s", got)
	}
}

func TestRouterOwnCallsCannotPresentTheCallersLogin(t *testing.T) {
	router := callerCredentialTestRouter(t)
	profile := delegatedCredentialProfile(t, router, callerTestEndpoint)
	if _, err := configuredProviderAuthorizer(router.Config, profile, "claude-sonnet", "test"); !errors.Is(err, principal.ErrRouterCall) {
		t.Fatalf("configured authorizer error = %v", err)
	}
}

func TestNoLoginIsKeptWhenNoBackendTakesOne(t *testing.T) {
	router := delegatedCredentialTestRouter(t, "")
	ctx := &RequestContext{Headers: map[string]string{"authorization": "Bearer raw-client-token"}}
	router.captureCallerCredential(ctx)
	if ctx.callerCredential.value != "" {
		t.Fatal("a configuration without caller backends kept the client's credential")
	}
}

func containsFold(values []string, want string) bool {
	for _, value := range values {
		if strings.EqualFold(value, want) {
			return true
		}
	}
	return false
}
