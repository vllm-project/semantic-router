package modelruntime

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

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

func TestEmbeddingRuntimeTasksRemoteConfigWithCandleOverrideUsesLocalTask(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", config.EmbeddingBackendCandle)
	t.Setenv("EMBEDDING_MODEL_TYPE_OVERRIDE", "mmbert")
	cfg := remoteEmbeddingRuntimeConfig("http://embedding-service:8000/v1")
	paths := resolveEmbeddingPaths(cfg)

	_, tasks, _ := embeddingRuntimeTasks(cfg, "test", paths)
	if len(tasks) != 1 || tasks[0].Name != "router.embedding.unified_factory" {
		t.Fatalf("local override tasks = %+v, want only router.embedding.unified_factory", tasks)
	}
}

func TestEmbeddingPathsForRuntimePlanSelectsOnlyResolvedModel(t *testing.T) {
	configured := embeddingPaths{
		qwen3:      "configured/qwen3",
		gemma:      "configured/gemma",
		mmBert:     "configured/mmbert",
		multiModal: "configured/multimodal",
		bert:       "configured/bert",
	}
	tests := []struct {
		name string
		plan embedding.RuntimePlan
		want embeddingPaths
	}{
		{name: "qwen3", plan: embedding.RuntimePlan{ModelType: "qwen3", ModelPath: "raw/qwen3"}, want: embeddingPaths{qwen3: "configured/qwen3"}},
		{name: "gemma", plan: embedding.RuntimePlan{ModelType: "gemma", ModelPath: "raw/gemma"}, want: embeddingPaths{gemma: "configured/gemma"}},
		{name: "mmbert", plan: embedding.RuntimePlan{ModelType: "mmbert", ModelPath: "raw/mmbert"}, want: embeddingPaths{mmBert: "configured/mmbert"}},
		{name: "multimodal", plan: embedding.RuntimePlan{ModelType: "multimodal", ModelPath: "raw/multimodal"}, want: embeddingPaths{multiModal: "configured/multimodal"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := configured.forRuntimePlan(tt.plan); got != tt.want {
				t.Fatalf("forRuntimePlan(%+v) = %+v, want %+v", tt.plan, got, tt.want)
			}
		})
	}
}

func TestPrepareRouterRuntimeOpenVINOOverrideDoesNotClaimReadyOrProbeRemote(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", config.EmbeddingBackendOpenVINO)
	t.Setenv("EMBEDDING_MODEL_TYPE_OVERRIDE", "mmbert")
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		calls.Add(1)
	}))
	defer server.Close()
	cfg := remoteEmbeddingRuntimeConfig(server.URL + "/v1")

	state, err := PrepareRouterRuntime(context.Background(), cfg, PrepareRouterRuntimeOptions{Component: "test"})
	if err != nil {
		t.Fatalf("PrepareRouterRuntime: %v", err)
	}
	if state.AnyReady || state.ToolsReady || state.EmbeddingProvider != nil {
		t.Fatalf("OpenVINO handoff state = %+v, want explicitly not-ready without remote provider", state)
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("remote embedding HTTP calls = %d, want 0", got)
	}
}

func TestPrepareRouterRuntimeLocalConfigWithRemoteOverrideProbesEndpointOnce(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", config.EmbeddingBackendOpenAICompatible)
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		writeRuntimeEmbeddingResponse(t, w, []float64{0.1, 0.2})
	}))
	defer server.Close()
	cfg := remoteEmbeddingRuntimeConfig(server.URL + "/v1")
	cfg.EmbeddingModels.EmbeddingConfig.Backend = config.EmbeddingBackendCandle
	cfg.EmbeddingModels.EmbeddingConfig.ModelType = config.EmbeddingModelTypeQwen3
	cfg.EmbeddingModels.Qwen3ModelPath = "models/local-qwen3"

	state, err := PrepareRouterRuntime(context.Background(), cfg, PrepareRouterRuntimeOptions{Component: "test", MaxParallelism: 1})
	if err != nil {
		t.Fatalf("PrepareRouterRuntime: %v", err)
	}
	if !state.AnyReady || !state.ToolsReady || state.EmbeddingProvider == nil {
		t.Fatalf("remote override state = %+v", state)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("remote embedding HTTP calls = %d, want 1", got)
	}
}

func TestPrepareRouterRuntimeProbesRemoteEmbeddingProvider(t *testing.T) {
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeRuntimeEmbeddingResponse(t, w, []float64{0.1, 0.2})
	}))
	defer server.Close()
	originalDefaultTransport := http.DefaultTransport
	http.DefaultTransport = server.Client().Transport
	t.Cleanup(func() { http.DefaultTransport = originalDefaultTransport })
	t.Setenv(config.EmbeddingAPIKeyEnvName, "test-secret")
	cfg := remoteEmbeddingRuntimeConfig(server.URL + "/v1")
	cfg.EmbeddingModels.Endpoint.APIKeyEnv = config.EmbeddingAPIKeyEnvName

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
	if provider.APIKeyEnv != config.EmbeddingAPIKeyEnvName {
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
