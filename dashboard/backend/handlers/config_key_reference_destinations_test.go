package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

const attackerEndpoint = "attacker.example:8000"

// A header a backend sends goes where that backend goes, however deep the
// mapping that holds it: moving the backend moves the header's value.
func TestAWriterCannotMoveABlockWhoseNestedMappingNamesAReference(t *testing.T) {
	header := map[string]interface{}{"headers": map[string]interface{}{"x-tenant-token": "${VSR_TEST_SECRET}"}}
	requireWriterRefused(t, map[string]map[string]interface{}{
		"a backend whose header names it":         backendBody("127.0.0.1:8000", header),
		"a search endpoint whose header names it": ragHeaderBody("search.internal"),
	}, map[string]map[string]interface{}{
		"a backend whose header names it":         backendBody(attackerEndpoint, header),
		"a search endpoint whose header names it": ragHeaderBody("attacker.example"),
	})
}

// A system prompt or a header_mutation header rides on the request a decision
// sends, so it goes to every model the decision can send it to. A writer may
// not give it a model at a new destination, by routing, by fallback, by the
// default model, by learning, or by moving a model it already reaches.
func TestAWriterCannotSendADecisionPluginReferenceToANewModel(t *testing.T) {
	refused := map[string]map[string]interface{}{
		"the decision routed to a new model": withModel(pluginReferenceBody(), "spare", attackerEndpoint, func(body map[string]interface{}) {
			referencingDecision(body)["modelRefs"] = []map[string]interface{}{{"model": "spare", "use_reasoning": false}}
		}),
		"a new fallback model": withModel(pluginReferenceBody(), "spare", attackerEndpoint, func(body map[string]interface{}) {
			decision := referencingDecision(body)
			decision["modelRefs"] = append(decision["modelRefs"].([]map[string]interface{}), map[string]interface{}{"model": "spare", "use_reasoning": false})
		}),
		"a new default model": withModel(pluginReferenceBody(), "spare", attackerEndpoint, func(body map[string]interface{}) {
			body["providers"].(map[string]interface{})["defaults"] = map[string]interface{}{"model": "spare"}
		}),
		"a new model's LoRA adapter as the default": withModel(pluginReferenceBody(), "spare", attackerEndpoint, func(body map[string]interface{}) {
			models := body["providers"].(map[string]interface{})["models"].([]map[string]interface{})
			models[len(models)-1]["loras"] = []map[string]interface{}{{"name": "spare-adapter"}}
			body["providers"].(map[string]interface{})["defaults"] = map[string]interface{}{"model": "spare-adapter"}
		}),
		"learning across every model": withModel(pluginReferenceBody(), "spare", attackerEndpoint, func(body map[string]interface{}) {
			body["global"] = learningAcrossModels()
		}),
		"the decision's model moved": pluginReferenceBody(func(body map[string]interface{}) {
			modelBackend(body)["endpoint"] = attackerEndpoint
		}),
	}
	stored := map[string]map[string]interface{}{}
	for name := range refused {
		stored[name] = pluginReferenceBody()
	}
	requireWriterRefused(t, stored, refused)
}

// Edits that send no reference anywhere new still save for a writer: text
// outside the blocks around a reference, a model nothing that names a
// reference can reach, and a decision without references routed to it.
func TestAWriterKeepsEditsThatSendNoReferenceAnywhereNew(t *testing.T) {
	for name, body := range map[string]map[string]interface{}{
		"in place": pluginReferenceBody(),
		"a new signal description": pluginReferenceBody(func(body map[string]interface{}) {
			recipe := body["recipes"].([]map[string]interface{})[0]
			recipe["signals"].(map[string]interface{})["domains"].([]map[string]interface{})[0]["description"] = "Renamed by a writer"
		}),
		"a model no referencing decision reaches": withModel(pluginReferenceBody(), "spare", attackerEndpoint, nil),
		"a decision without references routed to it": withModel(pluginReferenceBody(), "spare", attackerEndpoint, func(body map[string]interface{}) {
			recipe := body["recipes"].([]map[string]interface{})[0]
			recipe["decisions"] = append(recipe["decisions"].([]map[string]interface{}), map[string]interface{}{
				"name":      "spare-route",
				"priority":  2,
				"rules":     map[string]interface{}{"operator": "OR", "conditions": []map[string]interface{}{{"type": "domain", "name": "business"}}},
				"modelRefs": []map[string]interface{}{{"model": "spare", "use_reasoning": false}},
			})
		}),
	} {
		t.Run(name, func(t *testing.T) {
			configPath := createValidTestConfig(t, t.TempDir())
			if response := postConfig(t, configPath, pluginReferenceBody(), auth.PermConfigSecretsRead); response.Code != http.StatusOK {
				t.Fatalf("an admin save: status = %d: %s", response.Code, response.Body.String())
			}
			if response := postConfig(t, configPath, body); response.Code != http.StatusOK {
				t.Fatalf("a writer's save: status = %d: %s", response.Code, response.Body.String())
			}
		})
	}
}

// global.plugins shape every request, one that names a model too, so a value
// they read goes to every model: a writer may add no model at a new
// destination, though one may still edit text beside them.
func TestAWriterCannotAddAModelUnderAGlobalPluginReference(t *testing.T) {
	globalPlugins := func(body map[string]interface{}) {
		delete(referencingDecision(body), "plugins")
		body["global"] = map[string]interface{}{"plugins": referencingPlugins()}
	}
	requireWriterRefused(t, map[string]map[string]interface{}{"a model under a global plugin": pluginReferenceBody(globalPlugins)},
		map[string]map[string]interface{}{"a model under a global plugin": withModel(pluginReferenceBody(globalPlugins), "spare", attackerEndpoint, nil)})

	configPath := createValidTestConfig(t, t.TempDir())
	if response := postConfig(t, configPath, pluginReferenceBody(globalPlugins), auth.PermConfigSecretsRead); response.Code != http.StatusOK {
		t.Fatalf("an admin save: status = %d: %s", response.Code, response.Body.String())
	}
	edited := pluginReferenceBody(globalPlugins, func(body map[string]interface{}) {
		referencingDecision(body)["description"] = "Renamed by a writer"
	})
	if response := postConfig(t, configPath, edited); response.Code != http.StatusOK {
		t.Fatalf("a writer's new description: status = %d: %s", response.Code, response.Body.String())
	}
}

// Deploy merges a compiled routing fragment and the global editors replace
// global, and both save through the same check: a writer may not route the
// referencing decision to another model, nor let learning reach every model.
func TestAWriterCannotRetargetAPluginReferenceThroughDeployOrGlobal(t *testing.T) {
	original := propagateConfig
	propagateConfig = func(string, string) error { return nil }
	t.Cleanup(func() { propagateConfig = original })
	deploy := func(model string) func(string, string) http.HandlerFunc {
		return func(configPath, dir string) http.HandlerFunc {
			return func(w http.ResponseWriter, r *http.Request) {
				body, _ := json.Marshal(DeployRequest{YAML: pluginReferenceFragment(model), Mode: DeployModeReplace})
				DeployHandler(configPath, false, dir)(w, httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader(body)))
			}
		}
	}
	learning, _ := json.Marshal(learningAcrossModels())
	withLearning := func(handler func(string, bool, string) http.HandlerFunc) func(string, string) http.HandlerFunc {
		return func(configPath, dir string) http.HandlerFunc {
			return func(w http.ResponseWriter, r *http.Request) {
				handler(configPath, false, dir)(w, httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(learning)))
			}
		}
	}
	for name, test := range map[string]struct {
		save func(configPath, dir string) http.HandlerFunc
		want int
	}{
		"deploy routing the decision to spare":       {deploy("spare"), http.StatusForbidden},
		"deploy keeping the decision's model":        {deploy("test-model"), http.StatusOK},
		"global/update with learning across models":  {withLearning(UpdateRouterDefaultsHandler), http.StatusForbidden},
		"global/raw/update with learning across all": {withLearning(UpdateGlobalConfigYAMLHandler), http.StatusForbidden},
	} {
		t.Run(name, func(t *testing.T) {
			configPath := createValidTestConfig(t, t.TempDir())
			if response := postConfig(t, configPath, withModel(pluginReferenceBody(), "spare", attackerEndpoint, nil), auth.PermConfigSecretsRead); response.Code != http.StatusOK {
				t.Fatalf("an admin save: status = %d: %s", response.Code, response.Body.String())
			}
			before, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			response := httptest.NewRecorder()
			test.save(configPath, filepath.Dir(configPath))(response, nil)
			if response.Code != test.want {
				t.Fatalf("status = %d, want %d: %s", response.Code, test.want, response.Body.String())
			}
			if test.want == http.StatusForbidden {
				requireUnchangedFile(t, configPath, before)
			}
		})
	}
}

// requireWriterRefused saves each stored config as an admin, then checks that
// a writer's save of the matching refused config is refused and changes
// nothing, while an admin's is saved.
func requireWriterRefused(t *testing.T, stored, refused map[string]map[string]interface{}) {
	t.Helper()
	for name, body := range refused {
		t.Run(name, func(t *testing.T) {
			configPath := createValidTestConfig(t, t.TempDir())
			if response := postConfig(t, configPath, stored[name], auth.PermConfigSecretsRead); response.Code != http.StatusOK {
				t.Fatalf("an admin save: status = %d: %s", response.Code, response.Body.String())
			}
			before, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			if response := postConfig(t, configPath, body); response.Code != http.StatusForbidden {
				t.Fatalf("a writer's save: status = %d, want 403: %s", response.Code, response.Body.String())
			}
			requireUnchangedFile(t, configPath, before)
			if response := postConfig(t, configPath, stored[name]); response.Code != http.StatusOK {
				t.Fatalf("a writer's save in place: status = %d: %s", response.Code, response.Body.String())
			}
			if response := postConfig(t, configPath, body, auth.PermConfigSecretsRead); response.Code != http.StatusOK {
				t.Fatalf("an admin's save: status = %d: %s", response.Code, response.Body.String())
			}
		})
	}
}

// pluginReferenceBody is the saved config whose decision adds a header and a
// system prompt that both name ${VSR_TEST_SECRET}, after edits.
func pluginReferenceBody(edits ...func(map[string]interface{})) map[string]interface{} {
	body := canonicalConfigBody("127.0.0.1:8000")
	body["global"] = map[string]interface{}{}
	referencingDecision(body)["plugins"] = referencingPlugins()
	for _, edit := range edits {
		edit(body)
	}
	return body
}

func referencingPlugins() []map[string]interface{} {
	return []map[string]interface{}{
		{"type": "header_mutation", "configuration": map[string]interface{}{
			"add": []map[string]interface{}{{"name": "x-leak", "value": "${VSR_TEST_SECRET}"}},
		}},
		{"type": "system_prompt", "configuration": map[string]interface{}{
			"enabled":       true,
			"system_prompt": "token ${VSR_TEST_SECRET}",
		}},
	}
}

func referencingDecision(body map[string]interface{}) map[string]interface{} {
	return body["recipes"].([]map[string]interface{})[0]["decisions"].([]map[string]interface{})[0]
}

func modelBackend(body map[string]interface{}) map[string]interface{} {
	model := body["providers"].(map[string]interface{})["models"].([]map[string]interface{})[0]
	return model["backend_refs"].([]map[string]interface{})[0]
}

// withModel adds a model named name served at endpoint, then applies edit.
func withModel(body map[string]interface{}, name, endpoint string, edit func(map[string]interface{})) map[string]interface{} {
	providers := body["providers"].(map[string]interface{})
	providers["models"] = append(providers["models"].([]map[string]interface{}), map[string]interface{}{
		"name":              name,
		"provider_model_id": name,
		"backend_refs":      []map[string]interface{}{{"name": name, "provider": "vllm", "endpoint": endpoint, "protocol": "http"}},
	})
	if edit != nil {
		edit(body)
	}
	return body
}

// learningAcrossModels is a global block whose learning may pick any model.
func learningAcrossModels() map[string]interface{} {
	return map[string]interface{}{"router": map[string]interface{}{"learning": map[string]interface{}{
		"enabled":    true,
		"adaptation": map[string]interface{}{"enabled": true, "candidate_set": "global"},
	}}}
}

// ragHeaderBody has a decision search an external API at host, whose
// Authorization header names ${SEARCH_API_KEY}.
func ragHeaderBody(host string) map[string]interface{} {
	body := canonicalConfigBody("127.0.0.1:8000")
	body["global"] = map[string]interface{}{}
	referencingDecision(body)["plugins"] = []map[string]interface{}{{"type": "context", "configuration": map[string]interface{}{
		"enabled": true,
		"sources": []map[string]interface{}{{
			"type":    "rag",
			"backend": "external_api",
			"match":   map[string]interface{}{"recall": map[string]interface{}{"method": "embedding", "top_k": 3}},
			"backend_config": map[string]interface{}{
				"endpoint":         "https://" + host + "/query",
				"headers":          map[string]interface{}{"Authorization": "Bearer ${SEARCH_API_KEY}"},
				"request_format":   "custom",
				"request_template": `{"query":"{{.Query}}"}`,
			},
		}},
	}}}
	return body
}

// pluginReferenceFragment is the compiled routing of pluginReferenceBody with
// its decision routed to model.
func pluginReferenceFragment(model string) string {
	return `recipes:
  - name: default
    signals:
      domains:
        - name: business
          description: Business and management related queries
    decisions:
      - name: default-business
        description: Route business requests to the default model
        priority: 1
        rules:
          operator: OR
          conditions:
            - type: domain
              name: business
        modelRefs:
          - model: ` + model + `
            use_reasoning: false
        plugins:
          - type: header_mutation
            configuration:
              add:
                - name: x-leak
                  value: ${VSR_TEST_SECRET}
          - type: system_prompt
            configuration:
              enabled: true
              system_prompt: token ${VSR_TEST_SECRET}
providers:
  models:
    - name: test-model
    - name: spare
`
}
