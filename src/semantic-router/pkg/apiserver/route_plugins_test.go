//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func pluginTestMux(server *ClassificationAPIServer) *http.ServeMux {
	mux := http.NewServeMux()
	for _, route := range apiPluginRoutes() {
		mux.HandleFunc(route.pattern(), route.bind(server))
	}
	return mux
}

func TestPluginCatalogCoversCanonicalRegistry(t *testing.T) {
	recorder := httptest.NewRecorder()
	pluginTestMux(&ClassificationAPIServer{}).ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, apiPluginsPath, nil))
	if recorder.Code != http.StatusOK {
		t.Fatalf("catalog: %d %s", recorder.Code, recorder.Body.String())
	}
	var result pluginCatalogResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	want := config.DecisionPluginCatalog()
	if len(result.Plugins) != len(want) {
		t.Fatalf("got %d plugins, want %d", len(result.Plugins), len(want))
	}
	for i, entry := range want {
		actual := result.Plugins[i]
		if actual.Type != entry.Type || !strings.Contains(actual.Schema, "kind=plugin&name="+entry.Type) {
			t.Fatalf("descriptor %#v does not match %#v", actual, entry)
		}
	}
	for _, name := range []string{"image_gen", "semantic-cache", "missing"} {
		recorder := httptest.NewRecorder()
		pluginTestMux(&ClassificationAPIServer{}).ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, apiPluginsPath+"/"+name, nil))
		if recorder.Code != http.StatusNotFound {
			t.Fatalf("unsupported canonical name %s: %d", name, recorder.Code)
		}
	}
}

func pluginConfigDecision(name, prompt string) config.Decision {
	return config.Decision{Name: name, Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginSystemPrompt, Configuration: config.MustStructuredPayload(config.SystemPromptPluginConfig{SystemPrompt: prompt})}}}
}

func TestPluginBindingsUsePublishedGenerationAndExactRecipe(t *testing.T) {
	cfg := &config.RouterConfig{Recipes: []config.RoutingRecipe{
		{Name: "alpha", Profile: config.RoutingProfile{Decisions: []config.Decision{pluginConfigDecision("same", "private alpha")}}},
		{Name: "beta", Profile: config.RoutingProfile{Decisions: []config.Decision{pluginConfigDecision("same", "private beta")}}},
	}}
	server := &ClassificationAPIServer{config: &config.RouterConfig{}, runtimeRegistry: routerruntime.NewRegistry(cfg)}
	recorder := httptest.NewRecorder()
	pluginTestMux(server).ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, apiPluginsPath+"/system_prompt/bindings?recipe=beta&decision=same", nil))
	if recorder.Code != http.StatusOK {
		t.Fatalf("bindings: %d %s", recorder.Code, recorder.Body.String())
	}
	var result pluginBindingsResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Bindings) != 1 || result.Bindings[0].Recipe != "beta" || !result.Bindings[0].Enabled {
		t.Fatalf("bindings: %+v", result)
	}
	if strings.Contains(recorder.Body.String(), "private") {
		t.Fatal("binding inventory exposed plugin configuration")
	}
	policy, err := previewPluginConfig[config.SystemPromptPluginConfig](server, config.DecisionPluginSystemPrompt, nil, &pluginBindingRef{Recipe: "beta", Decision: "same"})
	if err != nil || policy.SystemPrompt != "private beta" {
		t.Fatalf("wrong recipe policy: %+v %v", policy, err)
	}
	if _, err := previewPluginConfig[config.SystemPromptPluginConfig](server, config.DecisionPluginSystemPrompt, nil, &pluginBindingRef{Recipe: "missing", Decision: "same"}); err == nil {
		t.Fatal("foreign recipe lookup fell back")
	}
}

func TestPluginRequestPreviewsShareDispatchPolicy(t *testing.T) {
	cases := []struct{ name, body, contains string }{
		{"system_prompt", `{"configuration":{"system_prompt":"policy"},"request_body":{"model":"model","messages":[{"role":"system","content":"existing"},{"role":"user","content":"hello"}]}}`, `"text":"policy"`},
		{"request_params", `{"configuration":{"blocked_params":["temperature"],"default_max_tokens":256,"max_tokens_limit":128},"request_body":{"model":"model","temperature":0.7,"messages":[{"role":"user","content":"hello"}]}}`, `"capped_output_tokens":true`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			pluginTestMux(&ClassificationAPIServer{}).ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, apiPluginsPath+"/"+tc.name+"/preview", strings.NewReader(tc.body)))
			if recorder.Code != http.StatusOK {
				t.Fatalf("preview: %d %s", recorder.Code, recorder.Body.String())
			}
			if !strings.Contains(recorder.Body.String(), tc.contains) {
				t.Fatalf("missing %s: %s", tc.contains, recorder.Body.String())
			}
			var result pluginRequestPreviewResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if result.Mode != "preview" || result.Persisted || result.BackendCalls || !result.Changed {
				t.Fatalf("invalid guarantees: %+v", result)
			}
			if tc.name == "request_params" && strings.Contains(string(result.RequestBody), "temperature") {
				t.Fatal("blocked parameter survived")
			}
		})
	}
}

func TestPluginPreviewRejectsAmbiguousAndUnknownConfiguration(t *testing.T) {
	for _, body := range []string{
		`{"configuration":{"message":"hello"},"binding":{"recipe":"a","decision":"b"}}`,
		`{"configuration":{"message":"hello","unknown":true}}`,
		`{}`,
	} {
		recorder := httptest.NewRecorder()
		pluginTestMux(&ClassificationAPIServer{}).ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, apiPluginsPath+"/fast_response/preview", strings.NewReader(body)))
		if recorder.Code != http.StatusBadRequest {
			t.Fatalf("expected invalid preview: %d %s", recorder.Code, recorder.Body.String())
		}
	}
}

func TestPluginHeaderPreviewPreservesOrderAndRedactsValues(t *testing.T) {
	recorder := httptest.NewRecorder()
	body := `{"configuration":{"add":[{"name":"X-Custom","value":"private-one"}],"update":[{"name":"Authorization","value":"private-two"}],"delete":["old"]}}`
	(&ClassificationAPIServer{}).handleHeaderMutationPreview(recorder, httptest.NewRequest(http.MethodPost, apiPluginsPath+"/header_mutation/preview", strings.NewReader(body)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("preview: %d %s", recorder.Code, recorder.Body.String())
	}
	if strings.Contains(recorder.Body.String(), "private-") {
		t.Fatal("header preview leaked credential values")
	}
	var result headerMutationPreviewResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Mutations) != 3 || result.Mutations[0].Operation != "append" || result.Mutations[1].Operation != "set" || result.Mutations[2].Operation != "remove" || !result.ValuesRedacted {
		t.Fatalf("unexpected plan: %+v", result)
	}
}

func TestPluginOperationsKeepExplicitOwnershipAfterGrouping(t *testing.T) {
	for _, route := range apiRoutes() {
		for _, operation := range route.PluginOperations {
			entry, ok := canonicalPluginEntry(operation.Plugin)
			if !ok || entry.Type != operation.Plugin {
				t.Fatalf("route %s advertises unregistered plugin %q", route.Path, operation.Plugin)
			}
			switch operation.Mode {
			case "read", "preview", "probe", "mutation":
			default:
				t.Fatalf("route %s unknown plugin operation mode %q", route.Path, operation.Mode)
			}
		}
	}
	for _, operation := range pluginOperations(config.DecisionPluginSystemPrompt, apiRoutes()) {
		if operation.Path != apiPluginsPath+"/system_prompt/preview" || operation.Mode != "preview" {
			t.Fatalf("inferred unrelated plugin operation: %+v", operation)
		}
	}
	if len(pluginOperations(config.DecisionPluginSystemPrompt, apiRoutes())) != 1 {
		t.Fatal("grouping lost explicit plugin preview ownership")
	}
	if len(pluginOperations(config.DecisionPluginMemory, apiRoutes())) != 4 {
		t.Fatal("memory discovery did not reuse four owned storage operations")
	}
}
