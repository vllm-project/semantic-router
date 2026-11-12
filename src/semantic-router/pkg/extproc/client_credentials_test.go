package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// A provider the Router configures receives only the credential its auth
// sets. A client's own credential headers, whatever the client meant them
// for, never travel beside it: not with a static key, not with a per-user
// key, and not when the upstream takes no key at all.
func TestProviderDispatchDropsTheClientsOwnCredentials(t *testing.T) {
	t.Setenv("CLIENT_CREDENTIALS_TEST_KEY", "router-key")
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.5
providers:
  models:
    - name: static-model
      provider_model_id: static-model
      api_format: anthropic
      backend_refs:
        - name: static
          provider: anthropic
          endpoint: https://static.example/v1
          auth: {header: Ocp-Apim-Subscription-Key, prefix: "", api_key_env: CLIENT_CREDENTIALS_TEST_KEY}
    - name: user-model
      provider_model_id: user-model
      api_format: anthropic
      backend_refs:
        - name: user
          provider: anthropic
          endpoint: https://user.example/v1
          auth: {header: Ocp-Apim-Subscription-Key, prefix: ""}
    - name: open-model
      provider_model_id: open-model
      api_format: openai
      backend_refs:
        - name: open
          provider: ollama
          endpoint: http://open.example:11434/v1
recipes:
  - name: default
`))
	if err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{Config: cfg, CredentialResolver: newTestCredentialResolver(cfg)}
	client := map[string]string{
		"authorization":             "Bearer client-router-key",
		"api-key":                   "client-api-key",
		"x-api-key":                 "client-x-api-key",
		"x-goog-api-key":            "client-goog-key",
		"api-subscription-key":      "client-subscription-key",
		"ocp-apim-subscription-key": "client-ocp-key",
	}
	for _, tc := range []struct {
		name, model, endpoint string
		userKey               string
		set                   string
	}{
		{name: "static key", model: "static-model", endpoint: "static-model_static", set: "router-key"},
		{name: "per-user key", model: "user-model", endpoint: "user-model_user", userKey: "user-key", set: "user-key"},
		{name: "no key to set", model: "user-model", endpoint: "user-model_user"},
		{name: "no credential", model: "open-model", endpoint: "open-model_open"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			requestHeaders := map[string]string{}
			for name, value := range client {
				requestHeaders[name] = value
			}
			if tc.userKey != "" {
				requestHeaders[headers.UserAnthropicKey] = tc.userKey
			}
			profile, err := router.Config.GetProviderProfileForEndpoint(tc.endpoint)
			if err != nil {
				t.Fatal(err)
			}
			state := &routeHeaderState{profile: profile}
			ctx := &RequestContext{Headers: requestHeaders, RequestID: "client-credentials-test"}
			if response := router.appendProviderCredential(state, tc.model, tc.endpoint, ctx); response != nil {
				t.Fatalf("dispatch returned %#v", response)
			}
			set := map[string]string{}
			for _, header := range state.setHeaders {
				set[strings.ToLower(header.GetHeader().GetKey())] = string(header.GetHeader().GetRawValue())
			}
			removed := map[string]bool{}
			for _, name := range state.removeHeaders {
				removed[strings.ToLower(name)] = true
			}
			for name := range client {
				if _, sets := set[name]; sets {
					continue
				}
				if !removed[name] {
					t.Fatalf("the client's %s reaches the provider: removed %v, set %v", name, state.removeHeaders, set)
				}
			}
			if tc.set == "" {
				if len(set) != 0 {
					t.Fatalf("set headers = %v, want none", set)
				}
				return
			}
			if len(set) != 1 || set["ocp-apim-subscription-key"] != tc.set || removed["ocp-apim-subscription-key"] {
				t.Fatalf("set %v, removed %v: want only ocp-apim-subscription-key: %s", set, state.removeHeaders, tc.set)
			}
		})
	}
}
