package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type savedConfigBody struct {
	Status   string                       `json:"status"`
	Warnings []routerconfig.ConfigWarning `json:"warnings"`
}

func applyWithoutRuntime(t *testing.T) {
	t.Helper()
	original := propagateConfig
	t.Cleanup(func() { propagateConfig = original })
	propagateConfig = func(string, string) error { return nil }
}

func decodeSavedConfig(t *testing.T, response *httptest.ResponseRecorder) savedConfigBody {
	t.Helper()
	if response.Code != http.StatusOK {
		t.Fatalf("save answered %d: %s", response.Code, response.Body.String())
	}
	var body savedConfigBody
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatalf("save answered %s: %v", response.Body.String(), err)
	}
	if body.Status != "success" {
		t.Fatalf("save answered %s, want a success", response.Body.String())
	}
	return body
}

// forEverySave saves config through each save path that persists model
// resources: the full update, the global patch, the raw global block and a
// DSL deploy merged into it.
func forEverySave(t *testing.T, config map[string]any, check func(*testing.T, *httptest.ResponseRecorder)) {
	t.Helper()
	document, err := yaml.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	full, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	rawGlobal := []byte("router:\n  strategy: priority\n")
	if global, ok := config["global"]; ok {
		if rawGlobal, err = yaml.Marshal(global); err != nil {
			t.Fatal(err)
		}
	}
	deploy, err := json.Marshal(DeployRequest{YAML: `recipes:
  - name: default
    decisions:
      - name: deployed-default
        priority: 5
        rules:
          operator: AND
          conditions:
            - type: domain
              name: business
        modelRefs:
          - model: test-model
`})
	if err != nil {
		t.Fatal(err)
	}
	saves := map[string]func(path, root string) *httptest.ResponseRecorder{
		"full update": func(path, root string) *httptest.ResponseRecorder {
			response := httptest.NewRecorder()
			UpdateConfigHandler(path, false, root)(response,
				httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(full)))
			return response
		},
		"global update": func(path, root string) *httptest.ResponseRecorder {
			response := httptest.NewRecorder()
			UpdateRouterDefaultsHandler(path, false, root)(response,
				httptest.NewRequest(http.MethodPost, "/api/router/config/global/update", bytes.NewBufferString(`{"router":{"strategy":"priority"}}`)))
			return response
		},
		"raw global update": func(path, root string) *httptest.ResponseRecorder {
			response := httptest.NewRecorder()
			UpdateGlobalConfigYAMLHandler(path, false, root)(response,
				httptest.NewRequest(http.MethodPost, "/api/router/config/global/raw/update", bytes.NewReader(rawGlobal)))
			return response
		},
		"dsl deploy": func(path, root string) *httptest.ResponseRecorder {
			response := httptest.NewRecorder()
			DeployHandler(path, false, root)(response,
				httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader(deploy)))
			return response
		},
	}
	for name, save := range saves {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			path := filepath.Join(root, "config.yaml")
			if err := os.WriteFile(path, document, 0o600); err != nil {
				t.Fatal(err)
			}
			check(t, save(path, root))
		})
	}
}

func withRoutingBindings(bindings map[string]any) map[string]any {
	config := canonicalConfigBody("127.0.0.1:8000")
	config["recipes"].([]map[string]interface{})[0]["model_bindings"] = bindings
	return config
}

func TestConfigSavesRefuseARemovedBindingField(t *testing.T) {
	applyWithoutRuntime(t)
	config := withRoutingBindings(map[string]any{
		"pii": map[string]any{"deployment": "primary", "contract": "decision.v1"},
	})
	forEverySave(t, config, func(t *testing.T, response *httptest.ResponseRecorder) {
		if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "contract") {
			t.Fatalf("save answered %d: %s, want the binding's contract refused", response.Code, response.Body.String())
		}
	})
}

func TestConfigSavesKeepABindingWrittenAsADeploymentName(t *testing.T) {
	applyWithoutRuntime(t)
	forEverySave(t, withRoutingBindings(map[string]any{"pii": "primary"}), func(t *testing.T, response *httptest.ResponseRecorder) {
		if warnings := decodeSavedConfig(t, response).Warnings; len(warnings) > 0 {
			t.Fatalf("a binding written as a deployment name saved with warnings: %+v", warnings)
		}
	})
}

func TestGlobalPatchRemovesKeysSetToNull(t *testing.T) {
	existing := []byte(`version: v0.5
global:
  router:
    strategy: priority
  model_catalog:
    bindings:
      domain_classifier: primary
    system:
      decision_model: {deployment: primary}
      pii_classifier: models/Vela-1.0-Encoder-307M-PII
`)
	// The Router refuses both spellings of one signal's binding, so the old
	// one must be deleted, not written as null.
	patch := []byte(`{"model_catalog":{
		"system":{"decision_model":{"deployment":"judge"},"pii_classifier":null},
		"bindings":{"domain":"judge","domain_classifier":null},
		"modules":{"prompt_guard":{"model_id":null}}}}`)
	merged, err := mergeGlobalOverridePatchYAML(existing, patch)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(merged), "null") || strings.Contains(string(merged), "modules") {
		t.Fatalf("the patch wrote its deletions into the config:\n%s", merged)
	}
	var document map[string]any
	if err := yaml.Unmarshal(merged, &document); err != nil {
		t.Fatal(err)
	}
	global := requireMapValue(t, document, "global")
	catalog := requireMapValue(t, global, "model_catalog")
	if bindings := requireMapValue(t, catalog, "bindings"); len(bindings) != 1 || bindings["domain"] != "judge" {
		t.Fatalf("bindings = %#v, want only the signal-type key", bindings)
	}
	if system := requireMapValue(t, catalog, "system"); len(system) != 1 || requireMapValue(t, system, "decision_model")["deployment"] != "judge" {
		t.Fatalf("system = %#v, want only the decision model", system)
	}
	if requireMapValue(t, global, "router")["strategy"] != "priority" {
		t.Fatal("the patch discarded a global setting it did not name")
	}
}

// The Embedding Models card saves the remote deployment's auth as a merge
// patch (configPageEmbeddingModelsSupport.test.ts lists the patch it sends for
// each saved auth). With the API key cleared, the merged deployment loads: a
// key goes with its header, while a delegated credential or a key file, which
// the card does not show, stays. Removing only api_key_env would leave a
// header with no key, which the Router refuses.
func TestEmbeddingCardAuthPatchesLoad(t *testing.T) {
	// The semantic response cache embeds each request with the deployment,
	// so a delegated credential has a request to come from.
	existing := func(auth string) []byte {
		raw, err := os.ReadFile(createValidTestConfig(t, t.TempDir()))
		if err != nil {
			t.Fatal(err)
		}
		return append(raw, []byte(`global:
  plugins:
    - type: response_cache
      configuration: {enabled: true, mode: semantic}
  services:
    authz:
      delegated_credential: {enabled: true}
  model_catalog:
    deployments:
      bge:
        provider: openai_embeddings
        endpoint: https://embedding.example.com/v1
        model: BAAI/bge-m3
        capabilities: {dimensions: 1024}
        auth: `+auth+`
    embeddings:
      default: {deployment: bge}
`)...)
	}
	for _, tc := range []struct {
		name, saved, patch string
		want               *routerconfig.DeploymentAuth
		refused            string
	}{
		{name: "a key and its header", saved: "{api_key_env: EMBEDDING_API_KEY, header: X-Api-Key}", patch: "null"},
		{
			name: "a delegated credential", saved: "{credential: delegated, header: Authorization}", patch: `{"api_key_env":null}`,
			want: &routerconfig.DeploymentAuth{Credential: routerconfig.CredentialDelegated, Header: "Authorization"},
		},
		{
			name: "a key file", saved: "{api_key_file: /run/secrets/embedding, header: X-Api-Key}", patch: `{"api_key_env":null}`,
			want: &routerconfig.DeploymentAuth{APIKeyFile: "/run/secrets/embedding", Header: "X-Api-Key"},
		},
		{
			name: "a key whose header outlives it", saved: "{api_key_env: EMBEDDING_API_KEY, header: X-Api-Key}", patch: `{"api_key_env":null}`,
			refused: "auth takes exactly one of api_key_env and api_key_file",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			merged, err := mergeGlobalOverridePatchYAML(existing(tc.saved), []byte(`{"model_catalog":{"deployments":{"bge":{"auth":`+tc.patch+`}}}}`))
			if err != nil {
				t.Fatal(err)
			}
			cfg, err := routerconfig.ParseYAMLBytes(merged)
			if tc.refused != "" {
				if err == nil || !strings.Contains(err.Error(), tc.refused) {
					t.Fatalf("load error = %v, want %q", err, tc.refused)
				}
				return
			}
			if err != nil {
				t.Fatalf("the merged config does not load: %v\n%s", err, merged)
			}
			if got := cfg.ModelDeployments["bge"].Auth; !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("auth = %+v, want %+v", got, tc.want)
			}
		})
	}
}
