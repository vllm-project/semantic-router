package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
	sigsyaml "sigs.k8s.io/yaml"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

// A RAG source sends a reference to its own endpoint, to the store
// connection it reuses from global, or to the deployment that judges it, and
// which of those its backend or deployment field selects. A writer may change
// none of them while a reference rides along: not by a header whose name
// looks like a destination, a switched backend, a moved global connection, a
// judge or binding pointed at another deployment, or OpenAI's default host.
func TestAWriterCannotRedirectARAGSourceReference(t *testing.T) {
	t.Setenv("SEARCH_API_KEY", "test-key")
	t.Setenv("RAG_TENANT", "acme")
	t.Setenv("RAG_FILE", "file-1")
	external := func(endpoint, headers string) string {
		return externalSource(endpoint, "headers: "+headers)
	}
	keyed := externalSource("https://search.internal/query", "api_key: ${SEARCH_API_KEY}")
	judged := func(deployment string) string {
		return `- type: rag
  backend: vectorstore
  backend_config: {vector_store_id: vs_docs}
  match:
    recall: {method: embedding, top_k: 3}
    judge:
      ` + deployment + `
      question:
        type: noul
        instructions: Does this passage answer the request for tenant ${RAG_TENANT}?`
	}
	for name, test := range map[string]struct{ stored, refused string }{
		"a header named endpoint beside a moved endpoint": {
			ragConfig(external("https://search.internal/query", `{Authorization: "Bearer ${SEARCH_API_KEY}"}`), "{}", ""),
			ragConfig(external("https://attacker.example/query", `{Authorization: "Bearer ${SEARCH_API_KEY}", endpoint: "https://search.internal/query"}`), "{}", ""),
		},
		"the endpoint moved to a stored header named endpoint": {
			ragConfig(external("https://search.internal/query", `{Authorization: "Bearer ${SEARCH_API_KEY}", endpoint: "https://attacker.example/query"}`), "{}", ""),
			ragConfig(external("https://attacker.example/query", `{Authorization: "Bearer ${SEARCH_API_KEY}", endpoint: "https://attacker.example/query"}`), "{}", ""),
		},
		"external_api switched to qdrant over a moved cache": {
			ragConfig(keyed, qdrantCache("qdrant.internal"), ""),
			ragConfig(`- type: rag
  backend: qdrant
  match: {recall: {method: embedding, top_k: 3}}
  backend_config:
    endpoint: https://search.internal/query
    api_key: ${SEARCH_API_KEY}
    request_format: custom
    collection: ${SEARCH_API_KEY}
    reuse_cache_connection: true`, qdrantCache("attacker.example"), ""),
		},
		"a qdrant collection over a moved cache": {
			ragConfig(qdrantSource, qdrantCache("qdrant.internal"), ""),
			ragConfig(qdrantSource, qdrantCache("attacker.example"), ""),
		},
		"a milvus filter over a moved cache": {
			ragConfig(milvusSource, milvusCache("milvus.internal"), ""),
			ragConfig(milvusSource, milvusCache("attacker.example"), ""),
		},
		"a vectorstore file over a moved llama_stack": {
			ragConfig(vectorstoreSource, "stores:\n"+llamaStack("http://llama.internal:8321"), ""),
			ragConfig(vectorstoreSource, "stores:\n"+llamaStack("http://attacker.example:8321"), ""),
		},
		"a hybrid primary over a moved cache": {
			ragConfig(hybridSource, milvusCache("milvus.internal")+"\n"+llamaStack("http://llama.internal:8321"), ""),
			ragConfig(hybridSource, milvusCache("attacker.example")+"\n"+llamaStack("http://llama.internal:8321"), ""),
		},
		"a judge asking another deployment": {
			ragConfig(judged("deployment: trusted-judge"), judgeGlobal(""), ""),
			ragConfig(judged("deployment: other-judge"), judgeGlobal(""), ""),
		},
		"the decision model a judge asks moved": {
			ragConfig(judged("top_k: 3"), judgeGlobal("trusted-judge"), ""),
			ragConfig(judged("top_k: 3"), judgeGlobal("other-judge"), ""),
		},
		"a recipe binding to another deployment": {
			domainReferenceConfig(""),
			domainReferenceConfig("    model_bindings:\n      domain: {deployment: other-judge}\n"),
		},
		"external_api switched to openai": {
			ragConfig(keyed, "{}", ""),
			ragConfig(`- type: rag
  backend: openai
  match: {recall: {method: embedding, top_k: 3}}
  backend_config:
    endpoint: https://search.internal/query
    api_key: ${SEARCH_API_KEY}
    vector_store_id: vs_docs`, "{}", ""),
		},
	} {
		requireWriterRefused(t, map[string]map[string]interface{}{name: yamlBody(t, test.stored)},
			map[string]map[string]interface{}{name: yamlBody(t, test.refused)})
	}
}

// A writer may not edit a field set directly on a block that holds a
// reference, not even its description or priority, as such a field can be
// what selects where the block sends the value.
func TestAWriterCannotEditTheBlocksAroundAReference(t *testing.T) {
	requireWriterRefused(t, map[string]map[string]interface{}{
		"the referencing decision's description": pluginReferenceBody(),
		"the referencing decision's priority":    pluginReferenceBody(),
	}, map[string]map[string]interface{}{
		"the referencing decision's description": pluginReferenceBody(func(body map[string]interface{}) {
			referencingDecision(body)["description"] = "Renamed by a writer"
		}),
		"the referencing decision's priority": pluginReferenceBody(func(body map[string]interface{}) {
			referencingDecision(body)["priority"] = 7
		}),
	})
}

// A key saved from the Dashboard sits in a backend's auth block, so a writer
// still edits its model's limits, pricing and description, but not the
// backend's endpoint or provider, nor a model field that chooses where its
// requests go: its API format, catalog card or named deployment.
func TestAWriterEditsAKeyedModelButNotWhereItSends(t *testing.T) {
	keyed := func(edit func(model, backend map[string]interface{})) map[string]interface{} {
		body := backendBody("127.0.0.1:8000", map[string]interface{}{"auth": map[string]interface{}{"api_key_env": "VSR_KEY_TEST"}})
		model := body["providers"].(map[string]interface{})["models"].([]map[string]interface{})[0]
		edit(model, model["backend_refs"].([]map[string]interface{})[0])
		return body
	}
	for name, body := range map[string]map[string]interface{}{
		"max_output_tokens": keyed(func(model, _ map[string]interface{}) { model["max_output_tokens"] = 8000 }),
		"pricing": keyed(func(model, _ map[string]interface{}) {
			model["pricing"] = map[string]interface{}{"currency": "USD", "prompt_per_1m": 1.0, "completion_per_1m": 2.0}
		}),
		"description":       keyed(func(model, _ map[string]interface{}) { model["description"] = "Renamed by a writer" }),
		"provider_model_id": keyed(func(model, _ map[string]interface{}) { model["provider_model_id"] = "test-model-v2" }),
	} {
		configPath := createValidTestConfig(t, t.TempDir())
		if response := postConfig(t, configPath, keyed(func(_, _ map[string]interface{}) {}), auth.PermConfigSecretsRead); response.Code != http.StatusOK {
			t.Fatalf("an admin save: status = %d: %s", response.Code, response.Body.String())
		}
		if response := postConfig(t, configPath, body); response.Code != http.StatusOK {
			t.Fatalf("a writer's new %s: status = %d: %s", name, response.Code, response.Body.String())
		}
	}
	stored := map[string]map[string]interface{}{}
	refused := map[string]map[string]interface{}{
		"the backend moved":      keyed(func(_, backend map[string]interface{}) { backend["endpoint"] = attackerEndpoint }),
		"the backend's provider": keyed(func(_, backend map[string]interface{}) { backend["provider"] = "openai-compatible" }),
		"the model's API format": keyed(func(model, _ map[string]interface{}) { model["api_format"] = "messages" }),
		"the model's catalog card": keyed(func(model, _ map[string]interface{}) {
			delete(model, "reasoning")
			model["catalog"] = "vllm-sr/mom-v1-lite"
		}),
	}
	for name := range refused {
		stored[name] = keyed(func(_, _ map[string]interface{}) {})
	}
	t.Setenv("RAG_FILE", "file-1")
	t.Setenv("NATIVE_MODEL", "vllm-sr/d3-ultra")
	native := func(deployment string) map[string]interface{} {
		body := yamlBody(t, ragConfig(vectorstoreSource, judgeGlobal(""), ""))
		providers := body["providers"].(map[string]interface{})
		providers["models"] = append(providers["models"].([]interface{}), map[string]interface{}{
			"name": "native", "api_format": "systemone", "deployment": deployment, "description": "Native model ${NATIVE_MODEL}",
		})
		return body
	}
	stored["the model's deployment"], refused["the model's deployment"] = native("trusted-judge"), native("other-judge")
	requireWriterRefused(t, stored, refused)
}

// Keys kept as references in providers only reach their own backends, and a
// store password only its store, so on a config shaped like the shipped
// config.yaml, with every provider key a reference, a writer still routes,
// describes, adds plugins, adds a model no reference reaches and edits a
// service setting that holds none, but may not move a backend that holds a
// key or a store whose password is a reference.
func TestAWriterKeepsEditingTheShippedConfigWithReferencedProviderKeys(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "config", "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	base := func() map[string]interface{} { return withReferencedProviderKeys(t, data) }
	decisions := func(document map[string]interface{}) []interface{} {
		return document["recipes"].([]interface{})[0].(map[string]interface{})["decisions"].([]interface{})
	}
	edits := map[string]func(map[string]interface{}){
		"routing": func(document map[string]interface{}) {
			decisions(document)[0].(map[string]interface{})["modelRefs"] = []interface{}{map[string]interface{}{"model": "qwen3-32b"}}
		},
		"a description": func(document map[string]interface{}) {
			decisions(document)[1].(map[string]interface{})["description"] = "Renamed by a writer"
		},
		"a plugin": func(document map[string]interface{}) {
			decision := decisions(document)[2].(map[string]interface{})
			decision["plugins"] = append(sequence(decision["plugins"]), map[string]interface{}{
				"type": "system_prompt", "configuration": map[string]interface{}{"enabled": true, "system_prompt": "Answer briefly."},
			})
		},
		"a model no reference reaches": func(document map[string]interface{}) {
			providers := document["providers"].(map[string]interface{})
			providers["models"] = append(providers["models"].([]interface{}), map[string]interface{}{
				"name": "spare", "provider_model_id": "spare",
				"backend_refs": []interface{}{map[string]interface{}{"name": "spare", "provider": "vllm", "endpoint": attackerEndpoint}},
			})
		},
	}
	edits["a store setting without references"] = func(document map[string]interface{}) {
		metrics := document["global"].(map[string]interface{})["services"].(map[string]interface{})["observability"].(map[string]interface{})["metrics"]
		metrics.(map[string]interface{})["windowed_metrics"].(map[string]interface{})["max_models"] = 60
	}
	stored := marshalYAML(t, base())
	for name, edit := range edits {
		edited := base()
		edit(edited)
		if !keyReferencesKept(stored, marshalYAML(t, edited)) {
			t.Fatalf("%s: refused", name)
		}
	}
	moved := base()
	first := sequence(moved["providers"].(map[string]interface{})["models"].([]interface{})[0].(map[string]interface{})["backend_refs"])[0]
	first.(map[string]interface{})["endpoint"] = attackerEndpoint
	if keyReferencesKept(stored, marshalYAML(t, moved)) {
		t.Fatal("moving a backend that holds a key was kept")
	}
	moved = base()
	redis := moved["global"].(map[string]interface{})["services"].(map[string]interface{})["response_api"].(map[string]interface{})["redis"]
	redis.(map[string]interface{})["address"] = attackerEndpoint
	if keyReferencesKept(stored, marshalYAML(t, moved)) {
		t.Fatal("moving a store whose password is a reference was kept")
	}
}

// withReferencedProviderKeys reads a config and gives every backend that
// holds no key and delegates none a key reference.
func withReferencedProviderKeys(t *testing.T, data []byte) map[string]interface{} {
	t.Helper()
	var document map[string]interface{}
	if err := yaml.Unmarshal(data, &document); err != nil {
		t.Fatal(err)
	}
	for i, model := range document["providers"].(map[string]interface{})["models"].([]interface{}) {
		for j, backend := range sequence(model.(map[string]interface{})["backend_refs"]) {
			backend := backend.(map[string]interface{})
			if auth, ok := backend["auth"].(map[string]interface{}); !ok || (auth["api_key_env"] == nil && auth["credential"] == nil) {
				backend["api_key"] = fmt.Sprintf("${PROVIDER_KEY_%d_%d}", i, j)
			}
		}
	}
	return document
}

func sequence(value interface{}) []interface{} {
	items, _ := value.([]interface{})
	return items
}

func marshalYAML(t *testing.T, value interface{}) []byte {
	t.Helper()
	data, err := yaml.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// yamlBody is a config body written as YAML.
func yamlBody(t *testing.T, text string) map[string]interface{} {
	t.Helper()
	data, err := sigsyaml.YAMLToJSON([]byte(text))
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]interface{}
	if err = json.Unmarshal(data, &body); err != nil {
		t.Fatal(err)
	}
	return body
}

// ragConfig is a config whose decision searches source, a context plugin
// source, with global as its global block and recipe fields before the
// recipe's signals.
func ragConfig(source, global, recipe string) string {
	return fmt.Sprintf(`version: v0.5
listeners:
  - name: public
    address: 0.0.0.0
    port: 8801
providers:
  defaults:
    model: test-model
  models:
    - name: test-model
      provider_model_id: test-model
      backend_refs:
        - name: endpoint1
          provider: vllm
          endpoint: 127.0.0.1:8000
          protocol: http
recipes:
  - name: default
%s    signals:
      domains:
        - name: business
          description: Business and management related queries
    decisions:
      - name: default-business
        priority: 1
        rules:
          operator: OR
          conditions:
            - type: domain
              name: business
        modelRefs:
          - model: test-model
            use_reasoning: false
        plugins:
          - type: context
            configuration:
              enabled: true
              sources:
%s
global:
%s
`, recipe, indent(source, 16), indent(global, 2))
}

// domainReferenceConfig names a reference in a domain signal's description,
// which the deployment answering domain signals reads, with recipe fields.
func domainReferenceConfig(recipe string) string {
	return strings.Replace(ragConfig(vectorstoreSource, judgeGlobal("trusted-judge"), recipe),
		"description: Business and management related queries", "description: Business queries for tenant ${RAG_TENANT}", 1)
}

func indent(text string, spaces int) string {
	lines := strings.Split(strings.Trim(text, "\n"), "\n")
	for i, line := range lines {
		lines[i] = strings.Repeat(" ", spaces) + line
	}
	return strings.Join(lines, "\n")
}

func externalSource(endpoint, field string) string {
	return `- type: rag
  backend: external_api
  match: {recall: {method: embedding, top_k: 3}}
  backend_config:
    endpoint: ` + endpoint + `
    ` + field + `
    request_format: custom
    request_template: '{"query":"{{.Query}}"}'`
}

const qdrantSource = `- type: rag
  backend: qdrant
  match: {recall: {method: embedding, top_k: 3}}
  backend_config: {collection: "docs_${RAG_TENANT}", reuse_cache_connection: true}`

const milvusSource = `- type: rag
  backend: milvus
  match: {recall: {method: embedding, top_k: 3}}
  backend_config: {collection: docs, filter_expression: 'tenant == "${RAG_TENANT}"', reuse_cache_connection: true}`

const vectorstoreSource = `- type: rag
  backend: vectorstore
  match: {recall: {method: embedding, top_k: 3}}
  backend_config: {vector_store_id: vs_docs, file_ids: ["${RAG_FILE}"]}`

const hybridSource = `- type: rag
  backend: hybrid
  match: {recall: {method: embedding, top_k: 3}}
  backend_config:
    primary: milvus
    primary_config: {collection: "docs_${RAG_TENANT}", reuse_cache_connection: true}
    fallback: vectorstore
    fallback_config: {vector_store_id: vs_docs}`

func qdrantCache(host string) string {
	return `stores:
  response_cache:
    backend_type: qdrant
    qdrant: {host: ` + host + `, port: 6334, collection_name: semantic_cache}`
}

func milvusCache(host string) string {
	return `stores:
  response_cache:
    backend_type: milvus
    milvus: {connection: {host: ` + host + `, port: 19530}}`
}

func llamaStack(endpoint string) string {
	return `  vector_store:
    enabled: true
    backend_type: llama_stack
    llama_stack: {endpoint: "` + endpoint + `", embedding_model: all-MiniLM-L6-v2}`
}

// judgeGlobal declares a vector store, and a trusted and another systemone
// deployment, with the decision model bound to decisionModel when it is set.
func judgeGlobal(decisionModel string) string {
	global := "stores:\n" + llamaStack("http://llama.internal:8321") + `
model_catalog:
  deployments:
    trusted-judge: {provider: systemone, endpoint: "https://judge.internal/v1/systemone", model: vllm-sr/d3-ultra, schema: vllm-sr/5.0}
    other-judge: {provider: systemone, endpoint: "https://attacker.example/v1/systemone", model: vllm-sr/d3-ultra, schema: vllm-sr/5.0}`
	if decisionModel != "" {
		global += "\n  system: {decision_model: {deployment: " + decisionModel + "}}"
	}
	return global
}
