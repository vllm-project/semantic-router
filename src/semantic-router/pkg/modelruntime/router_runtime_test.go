package modelruntime

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

func TestPrepareRouterRuntimeProbesRemoteEmbeddingProvider(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeRuntimeEmbeddingResponse(t, w, []float64{0.1, 0.2})
	}))
	defer server.Close()

	state, err := PrepareRouterRuntime(context.Background(), remoteEmbeddingRuntimeConfig(server.URL+"/v1"), PrepareRouterRuntimeOptions{
		Component:      "test-router",
		MaxParallelism: 1,
	})
	if err != nil {
		t.Fatalf("PrepareRouterRuntime() error = %v", err)
	}
	if !state.AnyReady || !state.ToolsReady {
		t.Fatalf("PrepareRouterRuntime() state = %+v, want AnyReady and ToolsReady", state)
	}
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
