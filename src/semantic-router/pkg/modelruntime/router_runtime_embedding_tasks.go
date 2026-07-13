package modelruntime

import (
	"context"
	"fmt"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func initializeUnifiedEmbeddingModels(cfg *config.RouterConfig, component string, paths embeddingPaths) bool {
	if !paths.hasUnifiedModels() {
		return false
	}

	semanticCacheNeedsBatched, mlSelectionNeedsBatched, otherConsumersNeedBatched := batchedEmbeddingNeeds(cfg, paths.qwen3)
	useBatched := semanticCacheNeedsBatched || mlSelectionNeedsBatched || otherConsumersNeedBatched
	logBatchedEmbeddingNeeds(component, semanticCacheNeedsBatched, mlSelectionNeedsBatched, otherConsumersNeedBatched)
	if err := initUnifiedEmbeddingModelFactory(cfg, paths, useBatched); err != nil {
		logging.ComponentErrorEvent(component, "embedding_models_init_failed", map[string]interface{}{
			"use_batched": useBatched,
			"error":       err.Error(),
		})
		logging.ComponentWarnEvent(component, "embedding_runtime_degraded", map[string]interface{}{
			"embedding_api_placeholder": true,
			"tools_database_disabled":   true,
		})
		return false
	}

	logging.ComponentEvent(component, "embedding_models_initialized", map[string]interface{}{
		"use_batched": useBatched,
	})
	return true
}

func initUnifiedEmbeddingModelFactory(cfg *config.RouterConfig, paths embeddingPaths, useBatched bool) error {
	if !useBatched {
		return candle_binding.InitEmbeddingModels(paths.qwen3, paths.gemma, paths.mmBert, cfg.UseCPU)
	}

	if err := candle_binding.InitEmbeddingModelsBatched(paths.qwen3, 64, 10, cfg.UseCPU); err != nil {
		return err
	}
	return candle_binding.InitEmbeddingModels(paths.qwen3, paths.gemma, paths.mmBert, cfg.UseCPU)
}

func initializeBERTModel(component string, useCPU bool, bertPath string, eventPrefix string) bool {
	if bertPath == "" {
		return false
	}

	logging.ComponentEvent(component, eventPrefix+"_init_started", map[string]interface{}{
		"model_ref": bertPath,
		"use_cpu":   useCPU,
	})
	if err := candle_binding.InitModel(bertPath, useCPU); err != nil {
		logging.ComponentWarnEvent(component, eventPrefix+"_init_failed", map[string]interface{}{
			"model_ref": bertPath,
			"error":     err.Error(),
		})
		return false
	}
	logging.ComponentEvent(component, eventPrefix+"_initialized", map[string]interface{}{
		"model_ref": bertPath,
	})
	return true
}

func initializeMultiModalEmbeddingModel(component string, useCPU bool, multiModalPath string) bool {
	if multiModalPath == "" {
		return false
	}

	logging.ComponentEvent(component, "multimodal_embedding_init_started", map[string]interface{}{
		"model_ref": multiModalPath,
		"use_cpu":   useCPU,
	})
	if err := candle_binding.InitMultiModalEmbeddingModel(multiModalPath, useCPU); err != nil {
		logging.ComponentWarnEvent(component, "multimodal_embedding_init_failed", map[string]interface{}{
			"model_ref":               multiModalPath,
			"error":                   err.Error(),
			"multimodal_routes_ready": false,
		})
		return false
	}
	logging.ComponentEvent(component, "multimodal_embedding_initialized", map[string]interface{}{
		"model_ref": multiModalPath,
	})
	return true
}

func buildEmbeddingRuntimeTasks(
	cfg *config.RouterConfig,
	component string,
	paths embeddingPaths,
	consumerPaths embeddingPaths,
	requiresMultimodalTools bool,
	tracker *embeddingStateTracker,
) []Task {
	tasks := make([]Task, 0, 3)
	tasks = append(tasks, unifiedEmbeddingRuntimeTask(cfg, component, paths, consumerPaths, requiresMultimodalTools, tracker)...)
	tasks = append(tasks, bertEmbeddingRuntimeTask(cfg, component, paths, consumerPaths, tracker)...)
	tasks = append(tasks, multiModalEmbeddingRuntimeTask(cfg, component, paths, consumerPaths, requiresMultimodalTools, tracker)...)
	return tasks
}

func remoteEmbeddingRuntimeTask(
	cfg *config.RouterConfig,
	component string,
	plan embedding.RuntimePlan,
	tracker *embeddingStateTracker,
) []Task {
	return []Task{{
		Name:       "router.embedding.remote_provider",
		BestEffort: true,
		Run: func(ctx context.Context) error {
			logging.ComponentEvent(component, "remote_embedding_init_started", map[string]interface{}{
				"backend": plan.Backend,
				"model":   cfg.EmbeddingModels.Endpoint.Model,
			})
			provider, err := embedding.NewProvider(cfg.EmbeddingModels, embedding.ProviderOptions{
				BackendOverride: plan.Backend,
			})
			if err != nil {
				tracker.markEmbeddingProvider(remoteEmbeddingProviderProbeStatus(cfg, plan, nil, 0, err))
				logging.ComponentErrorEvent(component, "remote_embedding_init_failed", map[string]interface{}{
					"error": err.Error(),
				})
				return fmt.Errorf("failed to initialize remote embedding provider: %w", err)
			}
			embeddingVector, err := provider.Embed(ctx, "semantic router embedding probe")
			if err != nil {
				tracker.markEmbeddingProvider(remoteEmbeddingProviderProbeStatus(cfg, plan, provider, 0, err))
				logging.ComponentErrorEvent(component, "remote_embedding_init_failed", map[string]interface{}{
					"backend": provider.Backend(),
					"error":   err.Error(),
				})
				return fmt.Errorf("failed to probe remote embedding provider: %w", err)
			}
			tracker.markEmbeddingProvider(remoteEmbeddingProviderProbeStatus(cfg, plan, provider, len(embeddingVector), nil))
			logging.ComponentEvent(component, "remote_embedding_initialized", map[string]interface{}{
				"backend":   provider.Backend(),
				"dimension": len(embeddingVector),
			})
			tracker.markToolsReady()
			return nil
		},
	}}
}

func unifiedEmbeddingRuntimeTask(
	cfg *config.RouterConfig,
	component string,
	paths embeddingPaths,
	consumerPaths embeddingPaths,
	requiresMultimodalTools bool,
	tracker *embeddingStateTracker,
) []Task {
	if !paths.hasUnifiedModels() {
		return nil
	}

	return []Task{{
		Name:       "router.embedding.unified_factory",
		BestEffort: !consumerPaths.hasUnifiedModels(),
		Run: func(context.Context) error {
			if !initializeUnifiedEmbeddingModels(cfg, component, paths) {
				return fmt.Errorf("failed to initialize unified embedding models")
			}
			if requiresMultimodalTools {
				tracker.markAnyReady()
			} else {
				tracker.markToolsReady()
			}
			return nil
		},
	}}
}

func bertEmbeddingRuntimeTask(
	cfg *config.RouterConfig,
	component string,
	paths embeddingPaths,
	consumerPaths embeddingPaths,
	tracker *embeddingStateTracker,
) []Task {
	if paths.bert == "" {
		return nil
	}

	return []Task{{
		Name:       "router.embedding.bert",
		BestEffort: consumerPaths.bert == "",
		Run: func(context.Context) error {
			if !initializeBERTModel(component, cfg.UseCPU, paths.bert, "memory_bert") {
				return fmt.Errorf("failed to initialize bert embedding model")
			}
			tracker.markAnyReady()
			return nil
		},
	}}
}

func multiModalEmbeddingRuntimeTask(
	cfg *config.RouterConfig,
	component string,
	paths embeddingPaths,
	consumerPaths embeddingPaths,
	requiresMultimodalTools bool,
	tracker *embeddingStateTracker,
) []Task {
	if paths.multiModal == "" {
		return nil
	}

	return []Task{{
		Name:       "router.embedding.multimodal",
		BestEffort: consumerPaths.multiModal == "",
		Run: func(context.Context) error {
			if !initializeMultiModalEmbeddingModel(component, cfg.UseCPU, paths.multiModal) {
				return fmt.Errorf("failed to initialize multimodal embedding model")
			}
			if requiresMultimodalTools {
				tracker.markToolsReady()
			} else {
				tracker.markAnyReady()
			}
			return nil
		},
	}}
}
