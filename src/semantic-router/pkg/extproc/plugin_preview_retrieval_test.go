package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

func retrievalPreviewRouter(pluginType string, policy any) *OpenAIRouter {
	return &OpenAIRouter{Config: &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{Name: "selected", Plugins: []config.DecisionPlugin{{Type: pluginType, Configuration: config.MustStructuredPayload(policy)}}}}}}}
}

func TestRAGPluginPreviewAndProbePreserveResultCache(t *testing.T) {
	var calls atomic.Int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if !strings.Contains(r.URL.Path, "vs-preview") {
			t.Errorf("unexpected configured search path: %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []map[string]any{{"content": "retrieved evidence", "filename": "doc.txt"}}})
	}))
	defer backend.Close()
	policy := &config.RAGPluginConfig{
		Enabled: true, Backend: "openai", InjectionMode: "system_prompt", CacheResults: true,
		BackendConfig: config.MustStructuredPayload(config.OpenAIRAGConfig{BaseURL: backend.URL, VectorStoreID: "vs-preview"}),
	}
	router := retrievalPreviewRouter("rag", policy)
	binding := pluginruntime.Binding{Recipe: config.DefaultRecipeName, Decision: "selected"}
	body := json.RawMessage(`{"model":"sample","messages":[{"role":"user","content":"preview-cache-isolation"}]}`)
	request := pluginruntime.RAGPreviewRequest{Binding: binding, RequestBody: body, SuppliedContext: "offline evidence"}
	preview, err := router.PreviewRAG(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 0 || preview.BackendCalls || preview.Persisted || preview.Mode != pluginruntime.ModePreview {
		t.Fatalf("preview effects=%+v calls=%d", preview.Guarantees, calls.Load())
	}
	if !strings.Contains(string(preview.RequestBody), "offline evidence") {
		t.Fatalf("shared injection did not change serialized body: %s", preview.RequestBody)
	}
	// Probes must bypass the runtime result cache and leave it unchanged.
	router.setRAGCache(config.DefaultRecipeName, "preview-cache-isolation", "existing cache", policy)
	request.SuppliedContext = ""
	request.Mode = pluginruntime.ModeProbe
	for i := 0; i < 2; i++ {
		probe, probeErr := router.PreviewRAG(context.Background(), request)
		if probeErr != nil {
			t.Fatal(probeErr)
		}
		if !probe.BackendCalls || probe.Persisted || probe.Context != "retrieved evidence" {
			t.Fatalf("probe=%+v", probe)
		}
	}
	if calls.Load() != 2 {
		t.Fatalf("probe used result cache: calls=%d", calls.Load())
	}
	cached, found := router.getRAGCache(config.DefaultRecipeName, "preview-cache-isolation", policy)
	if !found || cached != "existing cache" {
		t.Fatalf("probe mutated runtime cache: found=%v context=%q", found, cached)
	}
}

func TestToolsPluginPreviewUsesPolicyAndRejectsForeignRecipe(t *testing.T) {
	disabled := false
	router := retrievalPreviewRouter("tools", config.ToolsPluginConfig{Enabled: true, Mode: config.ToolsPluginModeFiltered, SemanticSelection: &disabled, AllowTools: []string{"local_read"}})
	request := pluginruntime.ToolsPreviewRequest{Binding: pluginruntime.Binding{Recipe: config.DefaultRecipeName, Decision: "selected"}, Plugin: "tools", RequestBody: json.RawMessage(`{"model":"sample","messages":[{"role":"user","content":"read a file"}],"tool_choice":"auto","tools":[{"type":"function","function":{"name":"local_read","parameters":{"type":"object"}}},{"type":"function","function":{"name":"external_upload","parameters":{"type":"object"}}}]}`)}
	response, err := router.PreviewTools(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if response.BackendCalls || response.Persisted || !reflect.DeepEqual(response.SelectedTools, []string{"local_read"}) {
		t.Fatalf("response=%+v", response)
	}
	if strings.Contains(string(response.RequestBody), "external_upload") {
		t.Fatalf("wire output ignored policy: %s", response.RequestBody)
	}
	request.Binding.Recipe = "foreign"
	if _, err = router.PreviewTools(context.Background(), request); !errors.Is(err, pluginruntime.ErrInvalidBinding) {
		t.Fatalf("foreign recipe error=%v", err)
	}
}

type previewCountingRetriever struct{ calls int }

func (r *previewCountingRetriever) Retrieve(_ context.Context, _ tools.RetrievalInput) (tools.RetrievalResult, error) {
	r.calls++
	return tools.RetrievalResult{Tools: []tools.ToolSimilarity{sampleTool("retrieved_tool")}, StrategyID: "preview-fixture", Confidence: 0.9}, nil
}

func TestToolsPluginPreviewRequiresExplicitProbeForRetrieval(t *testing.T) {
	router := retrievalPreviewRouter("tools", config.ToolsPluginConfig{Enabled: true, Strategy: "preview-fixture"})
	router.ToolsDatabase = tools.NewToolsDatabase(tools.ToolsDatabaseOptions{Enabled: true})
	retriever := &previewCountingRetriever{}
	router.ToolsRegistry = tools.NewRegistry()
	router.ToolsRegistry.Register("preview-fixture", retriever)
	request := pluginruntime.ToolsPreviewRequest{Binding: pluginruntime.Binding{Recipe: config.DefaultRecipeName, Decision: "selected"}, Plugin: "tools", RequestBody: json.RawMessage(`{"model":"sample","messages":[{"role":"user","content":"find a tool"}],"tool_choice":"auto"}`)}
	if _, err := router.PreviewTools(context.Background(), request); !errors.Is(err, pluginruntime.ErrProbeRequired) {
		t.Fatalf("preview error=%v", err)
	}
	if retriever.calls != 0 {
		t.Fatalf("preview retrieved %d times", retriever.calls)
	}
	request.Mode = pluginruntime.ModeProbe
	result, err := router.PreviewTools(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if retriever.calls != 1 || !result.BackendCalls || result.Persisted || !reflect.DeepEqual(result.SelectedTools, []string{"retrieved_tool"}) {
		t.Fatalf("probe result=%+v calls=%d", result, retriever.calls)
	}
	if !strings.Contains(string(result.RequestBody), "retrieved_tool") {
		t.Fatalf("selected tool missing from serialized request: %s", result.RequestBody)
	}
}
