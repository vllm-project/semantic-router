package modelruntime

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (p embeddingPaths) union(other embeddingPaths) embeddingPaths {
	result := p
	if other.qwen3 != "" {
		result.qwen3 = other.qwen3
	}
	if other.gemma != "" {
		result.gemma = other.gemma
	}
	if other.mmBert != "" {
		result.mmBert = other.mmBert
	}
	if other.multiModal != "" {
		result.multiModal = other.multiModal
	}
	if other.bert != "" {
		result.bert = other.bert
	}
	return result
}

// routerRuntimeEmbeddingTasks keeps the plan-selected classification/tools
// runtime separate from features that still call Candle directly. The latter
// cannot inherit a remote or OpenVINO plan until those consumers grow a
// provider-aware seam of their own.
func routerRuntimeEmbeddingTasks(
	cfg *config.RouterConfig,
	component string,
	configuredPaths embeddingPaths,
	plan embedding.RuntimePlan,
) (EmbeddingRuntimeState, []Task, *embeddingStateTracker, error) {
	planPaths := configuredPaths.forRuntimePlan(plan)
	consumerPaths, err := localConsumerEmbeddingPaths(cfg, configuredPaths)
	if err != nil {
		return EmbeddingRuntimeState{}, nil, nil, err
	}
	if plan.Backend == config.EmbeddingBackendCandle {
		classifierPaths, classifierErr := classifierEmbeddingPaths(cfg, configuredPaths, plan.Backend)
		if classifierErr != nil {
			return EmbeddingRuntimeState{}, nil, nil, classifierErr
		}
		requiredPaths := consumerPaths.union(classifierPaths)
		state, tasks, tracker := embeddingRuntimeTasksForPlanAndConsumers(
			cfg,
			component,
			planPaths.union(requiredPaths),
			plan,
			requiredPaths,
		)
		return state, tasks, tracker, nil
	}

	state, tasks, tracker := embeddingRuntimeTasksForPlan(cfg, component, planPaths, plan)
	tasks = append(tasks, localConsumerEmbeddingRuntimeTasks(cfg, component, consumerPaths)...)
	return state, tasks, tracker, nil
}

func localConsumerEmbeddingPaths(cfg *config.RouterConfig, configured embeddingPaths) (embeddingPaths, error) {
	result := embeddingPaths{}
	if cfg.Enabled {
		paths, err := pathForLocalConsumerModel(cfg, configured, "semantic cache", resolveSemanticCacheEmbeddingModel(cfg))
		if err != nil {
			return embeddingPaths{}, err
		}
		result = result.union(paths)
	}
	if vectorStoreUsesLocalEmbedding(cfg) {
		paths, err := pathForLocalConsumerModel(cfg, configured, "vector store", resolveVectorStoreEmbeddingModel(cfg))
		if err != nil {
			return embeddingPaths{}, err
		}
		result = result.union(paths)
	}
	if memoryConfigured(cfg) {
		paths, err := pathForLocalConsumerModel(cfg, configured, "memory", resolveMemoryEmbeddingModel(cfg))
		if err != nil {
			return embeddingPaths{}, err
		}
		result = result.union(paths)
	}
	return result, nil
}

func pathForLocalConsumerModel(
	cfg *config.RouterConfig,
	configured embeddingPaths,
	consumer string,
	model string,
) (embeddingPaths, error) {
	switch canonicalConsumerEmbeddingModel(model) {
	case "bert":
		return embeddingPaths{bert: resolveBertModelID(cfg.BertModelPath)}, nil
	case config.EmbeddingModelTypeQwen3:
		return requiredConsumerModelPath(consumer, model, "qwen3_model_path", configured.qwen3, func(path string) embeddingPaths {
			return embeddingPaths{qwen3: path}
		})
	case "gemma":
		return requiredConsumerModelPath(consumer, model, "gemma_model_path", configured.gemma, func(path string) embeddingPaths {
			return embeddingPaths{gemma: path}
		})
	case "mmbert":
		return requiredConsumerModelPath(consumer, model, "mmbert_model_path", configured.mmBert, func(path string) embeddingPaths {
			return embeddingPaths{mmBert: path}
		})
	case "multimodal":
		return requiredConsumerModelPath(consumer, model, "multimodal_model_path", configured.multiModal, func(path string) embeddingPaths {
			return embeddingPaths{multiModal: path}
		})
	default:
		return embeddingPaths{}, fmt.Errorf("%s selects unsupported local embedding model %q", consumer, model)
	}
}

func requiredConsumerModelPath(
	consumer string,
	model string,
	configField string,
	path string,
	build func(string) embeddingPaths,
) (embeddingPaths, error) {
	if strings.TrimSpace(path) == "" {
		return embeddingPaths{}, fmt.Errorf(
			"%s local embedding model %q requires model_catalog.embeddings.semantic.%s",
			consumer,
			model,
			configField,
		)
	}
	return build(path), nil
}

func canonicalConsumerEmbeddingModel(model string) string {
	model = strings.ToLower(strings.TrimSpace(model))
	if model == "modernbert" {
		return "mmbert"
	}
	return model
}

func resolveVectorStoreEmbeddingModel(cfg *config.RouterConfig) string {
	if cfg.VectorStore == nil {
		return ""
	}
	model := canonicalConsumerEmbeddingModel(cfg.VectorStore.EmbeddingModel)
	if model == "" {
		return "bert"
	}
	return model
}

func vectorStoreUsesLocalEmbedding(cfg *config.RouterConfig) bool {
	return cfg.VectorStore != nil &&
		cfg.VectorStore.Enabled &&
		!strings.EqualFold(strings.TrimSpace(cfg.VectorStore.BackendType), "llama_stack")
}

func localConsumerEmbeddingRuntimeTasks(
	cfg *config.RouterConfig,
	component string,
	paths embeddingPaths,
) []Task {
	tasks := make([]Task, 0, 3)
	if paths.hasUnifiedModels() {
		tasks = append(tasks, Task{
			Name: "router.local_consumers.unified_factory",
			Run: func(context.Context) error {
				if !initializeUnifiedEmbeddingModels(cfg, component, paths) {
					return fmt.Errorf("failed to initialize local consumer embedding models")
				}
				return nil
			},
		})
	}
	if paths.bert != "" {
		tasks = append(tasks, Task{
			Name: "router.local_consumers.bert",
			Run: func(context.Context) error {
				if !initializeBERTModel(component, cfg.UseCPU, paths.bert, "local_consumer_bert") {
					return fmt.Errorf("failed to initialize local consumer bert model")
				}
				return nil
			},
		})
	}
	if paths.multiModal != "" {
		tasks = append(tasks, Task{
			Name: "router.local_consumers.multimodal",
			Run: func(context.Context) error {
				if !initializeMultiModalEmbeddingModel(component, cfg.UseCPU, paths.multiModal) {
					return fmt.Errorf("failed to initialize local consumer multimodal model")
				}
				return nil
			},
		})
	}
	return tasks
}

func batchedEmbeddingNeeds(cfg *config.RouterConfig, qwen3Path string) (bool, bool, bool) {
	semanticCacheNeedsBatched := cfg.Enabled &&
		canonicalConsumerEmbeddingModel(resolveSemanticCacheEmbeddingModel(cfg)) == config.EmbeddingModelTypeQwen3 &&
		qwen3Path != ""
	mlSelectionNeedsBatched := cfg.ModelSelection.Enabled &&
		cfg.ModelSelection.ML.ModelsPath != "" &&
		qwen3Path != ""
	otherConsumersNeedBatched := qwen3Path != "" &&
		((vectorStoreUsesLocalEmbedding(cfg) && resolveVectorStoreEmbeddingModel(cfg) == config.EmbeddingModelTypeQwen3) ||
			(memoryConfigured(cfg) && canonicalConsumerEmbeddingModel(resolveMemoryEmbeddingModel(cfg)) == config.EmbeddingModelTypeQwen3))
	return semanticCacheNeedsBatched, mlSelectionNeedsBatched, otherConsumersNeedBatched
}

func logBatchedEmbeddingNeeds(
	component string,
	semanticCacheNeedsBatched bool,
	mlSelectionNeedsBatched bool,
	otherConsumersNeedBatched bool,
) {
	if !semanticCacheNeedsBatched && !mlSelectionNeedsBatched && !otherConsumersNeedBatched {
		return
	}
	logging.ComponentDebugEvent(component, "batched_embedding_mode_required", map[string]interface{}{
		"semantic_cache":  semanticCacheNeedsBatched,
		"model_selection": mlSelectionNeedsBatched,
		"local_consumers": otherConsumersNeedBatched,
	})
}

func memoryConfigured(cfg *config.RouterConfig) bool {
	if cfg.Memory.Enabled {
		return true
	}
	for _, decision := range cfg.Decisions {
		if decision.HasPlugin("memory") {
			return true
		}
	}
	return false
}

func resolveSemanticCacheEmbeddingModel(cfg *config.RouterConfig) string {
	embeddingModel := strings.ToLower(strings.TrimSpace(cfg.EmbeddingModel))
	if embeddingModel != "" {
		return embeddingModel
	}

	switch {
	case cfg.MmBertModelPath != "":
		return "mmbert"
	case cfg.MultiModalModelPath != "":
		return "multimodal"
	case cfg.Qwen3ModelPath != "":
		return "qwen3"
	case cfg.GemmaModelPath != "":
		return "gemma"
	default:
		return "bert"
	}
}

func resolveMemoryEmbeddingModel(cfg *config.RouterConfig) string {
	embeddingModel := strings.ToLower(strings.TrimSpace(cfg.Memory.EmbeddingModel))
	if embeddingModel != "" {
		return embeddingModel
	}

	switch {
	case cfg.MmBertModelPath != "":
		return "mmbert"
	case cfg.MultiModalModelPath != "":
		return "multimodal"
	case cfg.Qwen3ModelPath != "":
		return "qwen3"
	case cfg.GemmaModelPath != "":
		return "gemma"
	default:
		return "bert"
	}
}

func resolveBertModelID(modelID string) string {
	if modelID == "" {
		modelID = "sentence-transformers/all-MiniLM-L6-v2"
	}
	return config.ResolveModelPath(modelID)
}
