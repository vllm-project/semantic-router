package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

const clientAuthRecipeYAML = `
recipes:
  - name: privacy
    routing: {}
`

// allowClientAuth admits client_auth for one test, as the Router will once
// every request path enforces it.
func allowClientAuth(t *testing.T) {
	t.Helper()
	previous := clientAuthAvailable
	clientAuthAvailable = true
	t.Cleanup(func() { clientAuthAvailable = previous })
}

func clientAuthDocument(auth string) []byte {
	return []byte(recipeTestBaseYAML + clientAuthRecipeYAML + `
entrypoints:
  - model_names: [company/fast]
    recipe: privacy
    client_auth: ` + auth + `
`)
}

func TestEntrypointClientAuthIsNotAvailableYet(t *testing.T) {
	for _, auth := range []string{"{type: bearer_api_key, api_key_envs: [FAST_KEY]}", "null", ""} {
		_, err := ParseYAMLBytes(clientAuthDocument(auth))
		if err == nil || !strings.Contains(err.Error(), "entrypoints[0].client_auth: entrypoint client authentication is not available yet") {
			t.Fatalf("client_auth: %q gave error %v, want it rejected until it is enforced", auth, err)
		}
	}
}

func TestEntrypointClientAuthNeverExpandsAKeyIntoAnError(t *testing.T) {
	t.Setenv("FAST_KEY", "expanded-secret")
	for _, auth := range []string{
		`{type: bearer_api_key, api_key_envs: ["${FAST_KEY}"]}`,
		`{type: bearer_api_key, api_key_envs: "${FAST_KEY}"}`,
		`{type: bearer_api_key, api_key_envs: [[$FAST_KEY]]}`,
		`{type: "${FAST_KEY}", api_key_envs: [FAST_KEY]}`,
		`{type: bearer_api_key, api_key_envs: [FAST_KEY], extra: {nested: "${FAST_KEY}"}}`,
		`"${FAST_KEY}"`,
	} {
		_, err := ParseYAMLBytes(clientAuthDocument(auth))
		if err == nil || !strings.Contains(err.Error(), "entrypoints[0].client_auth must not contain $ references") {
			t.Fatalf("client_auth: %s gave error %v, want the reference rejected", auth, err)
		}
		if strings.Contains(err.Error(), "expanded-secret") {
			t.Fatalf("error %q quotes the expanded key", err)
		}
	}
}

func TestEntrypointClientAuthValidation(t *testing.T) {
	allowClientAuth(t)
	for _, tc := range []struct {
		name    string
		auth    string
		wantErr string
		secret  string
	}{
		{"missing type", `{api_key_envs: [FAST_KEY]}`, `client_auth.type must be "bearer_api_key", got ""`, ""},
		{"unsupported type", `{type: basic, api_key_envs: [FAST_KEY]}`, `client_auth.type must be "bearer_api_key", got "basic"`, ""},
		{"no variables", `{type: bearer_api_key}`, "client_auth.api_key_envs must name at least one environment variable", ""},
		{"empty variables", `{type: bearer_api_key, api_key_envs: []}`, "client_auth.api_key_envs must name at least one environment variable", ""},
		{"key written as a name", `{type: bearer_api_key, api_key_envs: [FAST_KEY, sk-live-secret]}`, "client_auth.api_key_envs[1] must be an environment variable name", "sk-live-secret"},
		{"leading digit", `{type: bearer_api_key, api_key_envs: [1KEY]}`, "client_auth.api_key_envs[0] must be an environment variable name", ""},
		{"repeated variable", `{type: bearer_api_key, api_key_envs: [FAST_KEY, FAST_KEY]}`, "client_auth.api_key_envs[1] repeats api_key_envs[0]", ""},
		{"unknown field", `{type: bearer_api_key, api_key_envs: [FAST_KEY], api_keys: [inline]}`, "api_keys", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseYAMLBytes(clientAuthDocument(tc.auth))
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %v, want %q", err, tc.wantErr)
			}
			if tc.secret != "" && strings.Contains(err.Error(), tc.secret) {
				t.Fatalf("error %q quotes a value that may be a key", err)
			}
		})
	}
}

func TestNullClientAuthIsPublicOnceAvailable(t *testing.T) {
	allowClientAuth(t)
	cfg, err := ParseYAMLBytes(clientAuthDocument("null"))
	if err != nil {
		t.Fatal(err)
	}
	if entry, ok := cfg.ResolveEntrypoint(ChatAPI, "company/fast"); !ok || entry.ClientAuth != nil {
		t.Fatalf("entrypoint = %+v (found=%v), want a public one", entry, ok)
	}
}

func TestEntrypointClientAuthKeepsDeclarationsIndependent(t *testing.T) {
	allowClientAuth(t)
	cfg, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + clientAuthRecipeYAML + `
entrypoints:
  - model_names: [company/fast, company/fast-v2]
    recipe: privacy
    client_auth:
      type: bearer_api_key
      api_key_envs: [FAST_KEY_CURRENT, FAST_KEY_NEXT]
  - model_names: [company/public]
    recipe: privacy
`))
	if err != nil {
		t.Fatal(err)
	}
	want := &EntrypointClientAuth{Type: ClientAuthBearerAPIKey, APIKeyEnvs: []string{"FAST_KEY_CURRENT", "FAST_KEY_NEXT"}}
	for _, name := range []string{"company/fast", "company/fast-v2"} {
		entry, ok := cfg.ResolveEntrypoint(ChatAPI, name)
		if !ok || !reflect.DeepEqual(entry.ClientAuth, want) {
			t.Fatalf("%s resolved to %+v (found=%v), want policy %+v", name, entry, ok, want)
		}
	}
	for _, name := range []string{"company/public", DefaultEntrypointModel} {
		entry, ok := cfg.ResolveEntrypoint(ChatAPI, name)
		if !ok || entry.ClientAuth != nil {
			t.Fatalf("%s resolved to %+v (found=%v), want a public entrypoint", name, entry, ok)
		}
	}

	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	again, err := ParseYAMLBytes(exported)
	if err != nil {
		t.Fatalf("exported config does not parse: %v\n%s", err, exported)
	}
	if !reflect.DeepEqual(cfg.Entrypoints, again.Entrypoints) {
		t.Fatalf("round trip changed entrypoints: before=%+v after=%+v", cfg.Entrypoints, again.Entrypoints)
	}

	exportedAuth := CanonicalConfigFromRouterConfig(cfg).Entrypoints[0].ClientAuth
	exportedAuth.APIKeyEnvs[0] = "CHANGED"
	resolved, _ := cfg.ResolveEntrypoint(ChatAPI, "company/fast")
	resolved.ClientAuth.APIKeyEnvs[1] = "CHANGED"
	if !reflect.DeepEqual(cfg.Entrypoints[0].ClientAuth, want) {
		t.Fatalf("export or resolution shares the normalized policy: %+v", cfg.Entrypoints[0].ClientAuth)
	}
}
