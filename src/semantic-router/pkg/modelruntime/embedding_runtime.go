package modelruntime

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modellifecycle"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type embeddingPaths struct {
	qwen3      string
	gemma      string
	mmBert     string
	multiModal string
	bert       string
	useCPU     bool
}

type embeddingStateTracker struct {
	mu    sync.Mutex
	state EmbeddingRuntimeState
}

func embeddingRuntimeTasks(
	cfg *config.RouterConfig,
	component string,
	paths embeddingPaths,
) (*embeddingStateTracker, []Task) {
	if !paths.hasConfiguredModels() {
		return nil, nil
	}

	logEmbeddingRuntimeStart(component, paths)
	requiresMultimodalTools := toolsUseMultiModalEmbeddings(cfg)
	tracker := &embeddingStateTracker{}
	return tracker, buildEmbeddingRuntimeTasks(cfg, component, paths, requiresMultimodalTools, tracker)
}

func semanticCacheBERTTask(cfg *config.RouterConfig, component string, paths embeddingPaths) []Task {
	if !semanticCacheUsesBERT(cfg) || paths.bert == "" {
		return nil
	}

	return []Task{{
		Name: "router.semantic_cache.bert",
		Run: func(context.Context) error {
			logging.ComponentEvent(component, "semantic_cache_bert_init_started", map[string]interface{}{
				"model_ref": paths.bert,
				"use_cpu":   paths.useCPU,
			})
			if err := initModel(paths.bert, paths.useCPU); err != nil {
				logging.ComponentErrorEvent(component, "semantic_cache_bert_init_failed", map[string]interface{}{
					"model_ref": paths.bert,
					"error":     err.Error(),
				})
				return fmt.Errorf("failed to initialize semantic cache bert model: %w", err)
			}
			logging.ComponentEvent(component, "semantic_cache_bert_initialized", map[string]interface{}{
				"model_ref": paths.bert,
			})
			return nil
		},
	}}
}

func vectorStoreBERTTask(cfg *config.RouterConfig, component string, paths embeddingPaths) []Task {
	if !vectorStoreUsesBERT(cfg) || paths.bert == "" {
		return nil
	}
	if semanticCacheUsesBERT(cfg) {
		return nil
	}

	return []Task{{
		Name: "router.vector_store.bert",
		Run: func(context.Context) error {
			logging.ComponentEvent(component, "vector_store_bert_init_started", map[string]interface{}{
				"model_ref": paths.bert,
				"use_cpu":   paths.useCPU,
			})
			if err := initModel(paths.bert, paths.useCPU); err != nil {
				logging.ComponentErrorEvent(component, "vector_store_bert_init_failed", map[string]interface{}{
					"model_ref": paths.bert,
					"error":     err.Error(),
				})
				return fmt.Errorf("failed to initialize vector store bert model: %w", err)
			}
			logging.ComponentEvent(component, "vector_store_bert_initialized", map[string]interface{}{
				"model_ref": paths.bert,
			})
			return nil
		},
	}}
}

func resolveEmbeddingPaths(plan modellifecycle.Plan) embeddingPaths {
	paths := plan.EmbeddingPaths()
	return embeddingPaths{
		qwen3:      paths.Qwen3,
		gemma:      paths.Gemma,
		mmBert:     paths.MmBERT,
		multiModal: paths.MultiModal,
		bert:       paths.BERT,
		useCPU:     paths.UseCPU,
	}
}

func (p embeddingPaths) hasConfiguredModels() bool {
	return p.hasUnifiedModels() || p.multiModal != "" || p.bert != ""
}

func (p embeddingPaths) hasUnifiedModels() bool {
	return p.qwen3 != "" || p.gemma != "" || p.mmBert != ""
}

func logMissingEmbeddingModelsConfig(component string) {
	logging.ComponentEvent(component, "embedding_models_not_configured", map[string]interface{}{
		"hint": "model_catalog.embeddings.semantic",
	})
}

func toolsUseMultiModalEmbeddings(cfg *config.RouterConfig) bool {
	if cfg == nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(cfg.EmbeddingConfig.ModelType), "multimodal")
}

func semanticCacheUsesBERT(cfg *config.RouterConfig) bool {
	if cfg == nil {
		return false
	}
	return cfg.Enabled && modellifecycle.ResolveSemanticCacheEmbeddingModel(cfg) == "bert"
}

func vectorStoreUsesBERT(cfg *config.RouterConfig) bool {
	if cfg == nil {
		return false
	}
	if cfg.VectorStore == nil || !cfg.VectorStore.Enabled {
		return false
	}
	model := strings.ToLower(strings.TrimSpace(cfg.VectorStore.EmbeddingModel))
	return model == "" || model == "bert"
}

func (t *embeddingStateTracker) markAnyReady() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.state.AnyReady = true
}

func (t *embeddingStateTracker) markToolsReady() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.state.AnyReady = true
	t.state.ToolsReady = true
}

func (t *embeddingStateTracker) snapshot() EmbeddingRuntimeState {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.state
}

func embeddingRuntimeState(tracker *embeddingStateTracker) EmbeddingRuntimeState {
	if tracker == nil {
		return EmbeddingRuntimeState{}
	}
	return tracker.snapshot()
}

func logEmbeddingRuntimeStart(component string, paths embeddingPaths) {
	logging.ComponentEvent(component, "embedding_models_init_started", map[string]interface{}{
		"use_cpu":               paths.useCPU,
		"qwen3_configured":      paths.qwen3 != "",
		"gemma_configured":      paths.gemma != "",
		"mmbert_configured":     paths.mmBert != "",
		"multimodal_configured": paths.multiModal != "",
		"bert_configured":       paths.bert != "",
	})
}

func buildEmbeddingRuntimeTasks(
	cfg *config.RouterConfig,
	component string,
	paths embeddingPaths,
	requiresMultimodalTools bool,
	tracker *embeddingStateTracker,
) []Task {
	tasks := make([]Task, 0, 3)
	tasks = append(tasks, unifiedEmbeddingRuntimeTask(cfg, component, paths, requiresMultimodalTools, tracker)...)
	tasks = append(tasks, bertEmbeddingRuntimeTask(cfg, component, paths, tracker)...)
	tasks = append(tasks, multiModalEmbeddingRuntimeTask(cfg, component, paths, requiresMultimodalTools, tracker)...)
	return tasks
}

func unifiedEmbeddingRuntimeTask(
	cfg *config.RouterConfig,
	component string,
	paths embeddingPaths,
	requiresMultimodalTools bool,
	tracker *embeddingStateTracker,
) []Task {
	if !paths.hasUnifiedModels() {
		return nil
	}

	return []Task{{
		Name:       "router.embedding.unified_factory",
		BestEffort: true,
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
	tracker *embeddingStateTracker,
) []Task {
	if paths.bert == "" || semanticCacheUsesBERT(cfg) || vectorStoreUsesBERT(cfg) {
		return nil
	}

	return []Task{{
		Name:       "router.embedding.bert",
		BestEffort: true,
		Run: func(context.Context) error {
			if !initializeBERTModel(component, paths.useCPU, paths.bert, "memory_bert") {
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
	requiresMultimodalTools bool,
	tracker *embeddingStateTracker,
) []Task {
	if paths.multiModal == "" {
		return nil
	}

	return []Task{{
		Name:       "router.embedding.multimodal",
		BestEffort: true,
		Run: func(context.Context) error {
			if !initializeMultiModalEmbeddingModel(component, paths.useCPU, paths.multiModal) {
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
