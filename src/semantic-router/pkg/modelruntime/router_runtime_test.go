package modelruntime

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestWarmupRouterRunsReadyTasksAndSkipsUnavailableTasks(t *testing.T) {
	loaded := make([]string, 0, 2)
	summary, err := WarmupRouter(context.Background(), []RouterWarmupTask{
		{
			Name:  "knowledge_bases",
			Ready: true,
			Load: func() error {
				loaded = append(loaded, "knowledge_bases")
				return nil
			},
		},
		{
			Name:       "tools_database",
			Ready:      false,
			SkipReason: "embedding runtime unavailable",
			Load: func() error {
				t.Fatal("unready warmup task must not run")
				return nil
			},
		},
	}, WarmupRouterOptions{Component: "test", MaxParallelism: 2})
	if err != nil {
		t.Fatalf("WarmupRouter() error = %v", err)
	}
	if len(loaded) != 1 || loaded[0] != "knowledge_bases" {
		t.Fatalf("loaded tasks = %v, want knowledge_bases", loaded)
	}
	if result := summary.Results["router.warmup.knowledge_bases"]; result.Status != TaskSucceeded {
		t.Fatalf("knowledge base warmup result = %+v, want success", result)
	}
}

func TestWarmupRouterTreatsTaskFailureAsBestEffort(t *testing.T) {
	summary, err := WarmupRouter(context.Background(), []RouterWarmupTask{{
		Name:  "knowledge_bases",
		Ready: true,
		Load:  func() error { return errors.New("embedding failed") },
	}}, WarmupRouterOptions{Component: "test", MaxParallelism: 1})
	if err != nil {
		t.Fatalf("WarmupRouter() best-effort error = %v", err)
	}
	if result := summary.Results["router.warmup.knowledge_bases"]; result.Status != TaskFailed {
		t.Fatalf("knowledge base warmup result = %+v, want failure", result)
	}
}

func TestEmbeddingRuntimeTasksUseOnlyRemoteProviderWhenConfigured(t *testing.T) {
	cfg := remoteEmbeddingRuntimeConfig("http://embedding-service:8000/v1")
	paths := resolveEmbeddingPaths(cfg)

	_, tasks, _ := embeddingRuntimeTasks(cfg, "test", paths)
	if len(tasks) != 1 {
		t.Fatalf("embeddingRuntimeTasks() returned %d task(s), want 1", len(tasks))
	}
	if tasks[0].Name != "router.embedding.remote_provider" {
		t.Fatalf("task name = %q, want router.embedding.remote_provider", tasks[0].Name)
	}
}

func TestPrepareRouterRuntimeProbesRemoteEmbeddingProvider(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeRuntimeEmbeddingResponse(t, w, []float64{0.1, 0.2})
	}))
	defer server.Close()
	t.Setenv("REMOTE_EMBEDDING_API_KEY", "test-secret")
	cfg := remoteEmbeddingRuntimeConfig(server.URL + "/v1")
	cfg.EmbeddingModels.Endpoint.APIKeyEnv = "REMOTE_EMBEDDING_API_KEY"

	state, err := PrepareRouterRuntime(context.Background(), cfg, PrepareRouterRuntimeOptions{
		Component:      "test-router",
		MaxParallelism: 1,
	})
	if err != nil {
		t.Fatalf("PrepareRouterRuntime() error = %v", err)
	}
	if !state.AnyReady || !state.ToolsReady {
		t.Fatalf("PrepareRouterRuntime() state = %+v, want AnyReady and ToolsReady", state)
	}
	assertReadyRemoteEmbeddingProviderStatus(t, state.EmbeddingProvider)
}

func TestPrepareRouterRuntimeReportsRemoteEmbeddingProbeFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "bad key", http.StatusUnauthorized)
	}))
	defer server.Close()

	var failedEvent *Event
	state, err := PrepareRouterRuntime(context.Background(), remoteEmbeddingRuntimeConfig(server.URL+"/v1"), PrepareRouterRuntimeOptions{
		Component:      "test-router",
		MaxParallelism: 1,
		OnEvent: func(event Event) {
			if event.Task == "router.embedding.remote_provider" && event.Status == TaskFailed {
				copyEvent := event
				failedEvent = &copyEvent
			}
		},
	})
	if err != nil {
		t.Fatalf("PrepareRouterRuntime() returned error for best-effort remote failure: %v", err)
	}
	if state.AnyReady || state.ToolsReady {
		t.Fatalf("PrepareRouterRuntime() state = %+v, want not ready", state)
	}
	assertFailedRemoteEmbeddingProviderStatus(t, state.EmbeddingProvider)
	assertRemoteEmbeddingFailureEvent(t, failedEvent)
}

func assertReadyRemoteEmbeddingProviderStatus(t *testing.T, provider *EmbeddingProviderRuntimeState) {
	t.Helper()
	if provider == nil {
		t.Fatal("expected embedding provider status")
	}
	if provider.Mode != "remote" || provider.Backend != config.EmbeddingBackendOpenAICompatible {
		t.Fatalf("embedding provider status = %+v, want remote openai-compatible", provider)
	}
	if provider.Model != "BAAI/bge-m3" {
		t.Fatalf("embedding provider model = %q", provider.Model)
	}
	if provider.APIKeyEnv != "REMOTE_EMBEDDING_API_KEY" {
		t.Fatalf("embedding provider api key env = %q", provider.APIKeyEnv)
	}
	assertRemoteEmbeddingProviderDimensionAndTimestamp(t, provider)
	assertBoolPtr(t, provider.APIKeyEnvSet, true, "embedding provider api key env set")
	assertBoolPtr(t, provider.Healthy, true, "embedding provider healthy")
	if provider.LastProbeError != "" {
		t.Fatalf("embedding provider last probe error = %q, want empty", provider.LastProbeError)
	}
}

func assertFailedRemoteEmbeddingProviderStatus(t *testing.T, provider *EmbeddingProviderRuntimeState) {
	t.Helper()
	if provider == nil {
		t.Fatal("expected embedding provider status")
	}
	assertRemoteEmbeddingProviderDimensionAndTimestamp(t, provider)
	assertBoolPtr(t, provider.Healthy, false, "embedding provider healthy")
	if provider.LastProbeError == "" {
		t.Fatal("expected embedding provider last probe error")
	}
}

func assertRemoteEmbeddingProviderDimensionAndTimestamp(t *testing.T, provider *EmbeddingProviderRuntimeState) {
	t.Helper()
	if provider.Dimension != 2 {
		t.Fatalf("embedding provider dimension = %d, want 2", provider.Dimension)
	}
	if provider.LastCheckedAt == "" {
		t.Fatal("expected embedding provider last checked timestamp")
	}
}

func assertBoolPtr(t *testing.T, got *bool, want bool, label string) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("%s = %v, want %v", label, got, want)
	}
}

func TestValidateConfiguredMultimodalDimension(t *testing.T) {
	contract := embeddingDimensionContract{Default: 384, Supported: []int{384, 256, 128, 64, 32}}

	for _, dimension := range []int{0, 384, 32} {
		if err := validateConfiguredMultimodalDimension(dimension, contract); err != nil {
			t.Fatalf("dimension %d should be accepted: %v", dimension, err)
		}
	}

	err := validateConfiguredMultimodalDimension(768, contract)
	if err == nil || !strings.Contains(err.Error(), "384") {
		t.Fatalf("expected model-specific rejection for dimension 768, got %v", err)
	}
}

// embedding_config is the single semantic-embedding block. config/config.yaml
// ships a multimodal_model_path alongside model_type: mmbert with
// target_dimension: 768, which is the text encoder's width. Reading it as the
// multimodal target rejected the shipped default and left the multimodal route
// unready while the checkpoint was already resident in the binding.
func TestConfiguredMultiModalDimensionIgnoresTextModelWidth(t *testing.T) {
	shipped := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MultiModalModelPath: "models/mom-embedding-multimodal",
				EmbeddingConfig:     config.HNSWConfig{ModelType: "mmbert", TargetDimension: 768},
			},
		},
	}
	if got := configuredMultiModalDimension(shipped); got != 0 {
		t.Fatalf("a text model_type must not contribute a multimodal dimension, got %d", got)
	}

	contract := embeddingDimensionContract{Default: 384, Supported: []int{384, 256, 128, 64, 32}}
	if err := validateConfiguredMultimodalDimension(configuredMultiModalDimension(shipped), contract); err != nil {
		t.Fatalf("shipped config must not block multimodal init: %v", err)
	}
}

// With model_type: multimodal the configured width really is the multimodal
// target, so an undeclared one must still fail at load.
func TestConfiguredMultiModalDimensionUsesMultimodalModelType(t *testing.T) {
	contract := embeddingDimensionContract{Default: 384, Supported: []int{384, 256, 128, 64, 32}}

	declared := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{ModelType: "multimodal", TargetDimension: 256},
			},
		},
	}
	if got := configuredMultiModalDimension(declared); got != 256 {
		t.Fatalf("expected the configured multimodal dimension 256, got %d", got)
	}
	if err := validateConfiguredMultimodalDimension(configuredMultiModalDimension(declared), contract); err != nil {
		t.Fatalf("declared dimension 256 should be accepted: %v", err)
	}

	// An omitted target_dimension stays unset for multimodal and resolves from
	// the loaded model instead of defaulting to 768.
	omitted := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{ModelType: "multimodal"},
			},
		},
	}
	if got := configuredMultiModalDimension(omitted); got != 0 {
		t.Fatalf("expected an omitted multimodal dimension to stay unset, got %d", got)
	}

	unsupported := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{ModelType: "multimodal", TargetDimension: 768},
			},
		},
	}
	if err := validateConfiguredMultimodalDimension(configuredMultiModalDimension(unsupported), contract); err == nil {
		t.Fatal("an undeclared multimodal dimension must be rejected at load")
	}
}

func assertRemoteEmbeddingFailureEvent(t *testing.T, failedEvent *Event) {
	t.Helper()
	if failedEvent == nil || failedEvent.Error == nil {
		t.Fatal("expected failed remote provider event with error")
	}
	if !strings.Contains(failedEvent.Error.Error(), "authentication failed") {
		t.Fatalf("remote provider event error = %v, want authentication failure", failedEvent.Error)
	}
}

func remoteEmbeddingRuntimeConfig(baseURL string) *config.RouterConfig {
	return &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MmBertModelPath: "models/mmbert-embed-32k-2d-matryoshka",
				EmbeddingConfig: config.HNSWConfig{
					Backend:         config.EmbeddingBackendOpenAICompatible,
					ModelType:       config.EmbeddingModelTypeRemote,
					TargetDimension: 2,
				},
				Endpoint: config.EmbeddingEndpointConfig{
					BaseURL: baseURL,
					Model:   "BAAI/bge-m3",
				},
			},
		},
	}
}

func writeRuntimeEmbeddingResponse(t *testing.T, w http.ResponseWriter, embedding []float64) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"data": []map[string]interface{}{
			{"index": 0, "embedding": embedding},
		},
	}); err != nil {
		t.Fatalf("encode response: %v", err)
	}
}
