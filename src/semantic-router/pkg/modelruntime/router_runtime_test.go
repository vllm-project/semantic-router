package modelruntime

import (
	"context"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modellifecycle"
)

func withEmbeddingModelInit(t *testing.T, init func(qwen3, gemma, mmbert string, useCPU bool) error) {
	t.Helper()

	original := initEmbeddingModels
	initEmbeddingModels = init
	t.Cleanup(func() {
		initEmbeddingModels = original
	})
}

func TestPrepareRouterRuntimeHandlesNilConfig(t *testing.T) {
	state, err := PrepareRouterRuntime(context.Background(), nil, PrepareRouterRuntimeOptions{})
	if err != nil {
		t.Fatalf("PrepareRouterRuntime() error = %v", err)
	}
	if state.AnyReady || state.ToolsReady {
		t.Fatalf("PrepareRouterRuntime() state = %#v, want empty", state)
	}
}

func TestPrepareRouterRuntimeReturnsFinalEmbeddingState(t *testing.T) {
	withEmbeddingModelInit(t, func(qwen3, gemma, mmbert string, useCPU bool) error {
		if mmbert != "models/mmbert-embed-32k-2d-matryoshka" {
			return fmt.Errorf("mmbert path = %q", mmbert)
		}
		if !useCPU {
			return fmt.Errorf("useCPU = false, want true")
		}
		return nil
	})

	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MmBertModelPath: "models/mmbert-embed-32k-2d-matryoshka",
				UseCPU:          true,
				EmbeddingConfig: config.HNSWConfig{ModelType: "mmbert"},
			},
		},
	}

	state, err := PrepareRouterRuntime(context.Background(), cfg, PrepareRouterRuntimeOptions{MaxParallelism: 1})
	if err != nil {
		t.Fatalf("PrepareRouterRuntime() error = %v", err)
	}
	if !state.AnyReady || !state.ToolsReady {
		t.Fatalf("PrepareRouterRuntime() state = %#v, want AnyReady and ToolsReady", state)
	}
}

func TestSemanticCacheBERTUsesRequiredTaskWithoutBestEffortDuplicate(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{UseCPU: true},
		},
		SemanticCache: config.SemanticCache{
			Enabled:        true,
			EmbeddingModel: "bert",
		},
	}
	paths := resolveEmbeddingPaths(modellifecycle.BuildPlan(cfg))

	_, embeddingTasks := embeddingRuntimeTasks(cfg, "test", paths)
	if containsTask(embeddingTasks, "router.embedding.bert") {
		t.Fatalf("embeddingRuntimeTasks() generated duplicate best-effort BERT task: %#v", embeddingTasks)
	}

	tasks := semanticCacheBERTTask(cfg, "test", paths)
	if len(tasks) != 1 {
		t.Fatalf("semanticCacheBERTTask() returned %d tasks, want 1", len(tasks))
	}
	if tasks[0].Name != "router.semantic_cache.bert" || tasks[0].BestEffort {
		t.Fatalf("semanticCacheBERTTask() = %#v, want required semantic cache task", tasks[0])
	}
}

func TestVectorStoreBERTTreatsEmptyEmbeddingModelAsBERT(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{UseCPU: true},
		},
		VectorStore: &config.VectorStoreConfig{
			Enabled: true,
		},
	}
	paths := resolveEmbeddingPaths(modellifecycle.BuildPlan(cfg))

	_, embeddingTasks := embeddingRuntimeTasks(cfg, "test", paths)
	if containsTask(embeddingTasks, "router.embedding.bert") {
		t.Fatalf("embeddingRuntimeTasks() generated duplicate best-effort BERT task: %#v", embeddingTasks)
	}

	tasks := vectorStoreBERTTask(cfg, "test", paths)
	if len(tasks) != 1 {
		t.Fatalf("vectorStoreBERTTask() returned %d tasks, want 1", len(tasks))
	}
	if tasks[0].Name != "router.vector_store.bert" || tasks[0].BestEffort {
		t.Fatalf("vectorStoreBERTTask() = %#v, want required vector store task", tasks[0])
	}
}

func TestMemoryBERTUsesBestEffortEmbeddingTask(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{UseCPU: true},
		},
		Memory: config.MemoryConfig{
			Enabled:        true,
			EmbeddingModel: "bert",
		},
	}
	paths := resolveEmbeddingPaths(modellifecycle.BuildPlan(cfg))

	_, embeddingTasks := embeddingRuntimeTasks(cfg, "test", paths)
	if len(embeddingTasks) != 1 {
		t.Fatalf("embeddingRuntimeTasks() returned %d tasks, want 1", len(embeddingTasks))
	}
	if embeddingTasks[0].Name != "router.embedding.bert" || !embeddingTasks[0].BestEffort {
		t.Fatalf("embeddingRuntimeTasks()[0] = %#v, want best-effort memory BERT task", embeddingTasks[0])
	}
}

func TestResolveEmbeddingPathsCarriesLifecycleUseCPUPolicy(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MmBertModelPath: "models/mmbert-embed-32k-2d-matryoshka",
				UseCPU:          false,
			},
		},
	}

	paths := resolveEmbeddingPaths(modellifecycle.BuildPlan(cfg))
	if paths.mmBert == "" {
		t.Fatal("mmBert path is empty")
	}
	if paths.useCPU {
		t.Fatal("paths.useCPU = true, want false from lifecycle plan")
	}
}

func containsTask(tasks []Task, name string) bool {
	for _, task := range tasks {
		if task.Name == name {
			return true
		}
	}
	return false
}
