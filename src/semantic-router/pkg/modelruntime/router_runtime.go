package modelruntime

import (
	"context"
	"fmt"
	"strings"
	"sync"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type EmbeddingRuntimeState struct {
	AnyReady   bool
	ToolsReady bool
}

type PrepareRouterRuntimeOptions struct {
	Component                  string
	MaxParallelism             int
	OnEvent                    func(Event)
	InitModalityClassifierFunc func(modelPath string, useCPU bool) error
}

type WarmupToolsOptions struct {
	Component      string
	MaxParallelism int
	OnEvent        func(Event)
}

type embeddingPaths struct {
	qwen3      string
	gemma      string
	mmBert     string
	multiModal string
	bert       string
}

type embeddingStateTracker struct {
	mu    sync.Mutex
	state EmbeddingRuntimeState
}

func PrepareRouterRuntime(
	ctx context.Context,
	cfg *config.RouterConfig,
	options PrepareRouterRuntimeOptions,
) (EmbeddingRuntimeState, error) {
	component := options.Component
	if component == "" {
		component = "router"
	}

	paths := resolveEmbeddingPaths(cfg)
	state, embeddingTasks := embeddingRuntimeTasks(cfg, component, paths)
	if !paths.hasConfiguredModels() {
		logMissingEmbeddingModelsConfig(component)
	}

	tasks := append([]Task{}, embeddingTasks...)
	tasks = append(tasks, semanticCacheBERTTask(cfg, component)...)
	tasks = append(tasks, vectorStoreBERTTask(cfg, component)...)
	tasks = append(tasks, modalityClassifierTask(cfg, component, options.InitModalityClassifierFunc)...)
	if len(tasks) == 0 {
		return state, nil
	}

	_, err := Execute(ctx, tasks, Options{
		MaxParallelism: options.MaxParallelism,
		OnEvent:        options.OnEvent,
	})
	if paths.hasConfiguredModels() {
		logging.ComponentEvent(component, "embedding_models_init_completed", map[string]interface{}{
			"embedding_ready": state.AnyReady,
			"tools_ready":     state.ToolsReady,
			"tools_model":     cfg.EmbeddingConfig.ModelType,
		})
	}
	return state, err
}

func WarmupToolsDatabase(
	ctx context.Context,
	toolsReady bool,
	load func() error,
	options WarmupToolsOptions,
) (Summary, error) {
	component := options.Component
	if component == "" {
		component = "router"
	}

	if !toolsReady {
		logging.ComponentEvent(component, "tools_database_load_skipped", map[string]interface{}{
			"reason": "embedding_runtime_not_ready_for_tools",
		})
		return Summary{Results: map[string]TaskResult{}}, nil
	}

	logging.ComponentEvent(component, "runtime_warmup_started", map[string]interface{}{
		"tasks": "tools_database",
	})
	summary, err := Execute(ctx, []Task{
		{
			Name:       "router.warmup.tools_database",
			BestEffort: true,
			Run: func(context.Context) error {
				logging.ComponentEvent(component, "tools_database_load_started", map[string]interface{}{})
				return load()
			},
		},
	}, Options{
		MaxParallelism: options.MaxParallelism,
		OnEvent:        options.OnEvent,
	})
	if err != nil {
		return summary, err
	}
	if result, ok := summary.Results["router.warmup.tools_database"]; ok && result.Status == TaskSucceeded {
		logging.ComponentEvent(component, "tools_database_loaded", map[string]interface{}{})
	}
	return summary, nil
}

func embeddingRuntimeTasks(
	cfg *config.RouterConfig,
	component string,
	paths embeddingPaths,
) (EmbeddingRuntimeState, []Task) {
	if !paths.hasConfiguredModels() {
		return EmbeddingRuntimeState{}, nil
	}

	logEmbeddingRuntimeStart(component, cfg, paths)
	requiresMultimodalTools := toolsUseMultiModalEmbeddings(cfg)
	tracker := &embeddingStateTracker{}
	return tracker.snapshot(), buildEmbeddingRuntimeTasks(cfg, component, paths, requiresMultimodalTools, tracker)
}

func semanticCacheBERTTask(cfg *config.RouterConfig, component string) []Task {
	if !cfg.Enabled || resolveSemanticCacheEmbeddingModel(cfg) != "bert" {
		return nil
	}

	bertModelID := resolveBertModelID(cfg.BertModelPath)
	return []Task{{
		Name: "router.semantic_cache.bert",
		Run: func(context.Context) error {
			logging.ComponentEvent(component, "semantic_cache_bert_init_started", map[string]interface{}{
				"model_ref": bertModelID,
				"use_cpu":   cfg.UseCPU,
			})
			if err := candle_binding.InitModel(bertModelID, cfg.UseCPU); err != nil {
				logging.ComponentErrorEvent(component, "semantic_cache_bert_init_failed", map[string]interface{}{
					"model_ref": bertModelID,
					"error":     err.Error(),
				})
				return fmt.Errorf("failed to initialize semantic cache bert model: %w", err)
			}
			logging.ComponentEvent(component, "semantic_cache_bert_initialized", map[string]interface{}{
				"model_ref": bertModelID,
			})
			return nil
		},
	}}
}

func vectorStoreBERTTask(cfg *config.RouterConfig, component string) []Task {
	if cfg.VectorStore == nil || !cfg.VectorStore.Enabled || cfg.VectorStore.EmbeddingModel != "bert" || cfg.Enabled {
		return nil
	}

	bertModelID := resolveBertModelID(cfg.BertModelPath)
	return []Task{{
		Name: "router.vector_store.bert",
		Run: func(context.Context) error {
			logging.ComponentEvent(component, "vector_store_bert_init_started", map[string]interface{}{
				"model_ref": bertModelID,
				"use_cpu":   cfg.UseCPU,
			})
			if err := candle_binding.InitModel(bertModelID, cfg.UseCPU); err != nil {
				logging.ComponentErrorEvent(component, "vector_store_bert_init_failed", map[string]interface{}{
					"model_ref": bertModelID,
					"error":     err.Error(),
				})
				return fmt.Errorf("failed to initialize vector store bert model: %w", err)
			}
			logging.ComponentEvent(component, "vector_store_bert_initialized", map[string]interface{}{
				"model_ref": bertModelID,
			})
			return nil
		},
	}}
}

func modalityClassifierTask(
	cfg *config.RouterConfig,
	component string,
	initFunc func(modelPath string, useCPU bool) error,
) []Task {
	md := &cfg.ModalityDetector
	if !md.Enabled {
		return nil
	}

	method := md.GetMethod()
	if method != config.ModalityDetectionClassifier && method != config.ModalityDetectionHybrid {
		return nil
	}
	if md.Classifier == nil || md.Classifier.ModelPath == "" {
		return nil
	}

	modelPath := config.ResolveModelPath(md.Classifier.ModelPath)
	bestEffort := method == config.ModalityDetectionHybrid
	return []Task{{
		Name:       "router.modality.classifier",
		BestEffort: bestEffort,
		Run: func(context.Context) error {
			logging.ComponentEvent(component, "modality_classifier_init_started", map[string]interface{}{
				"method":    method,
				"model_ref": modelPath,
				"use_cpu":   md.Classifier.UseCPU,
			})
			if initFunc == nil {
				return fmt.Errorf("modality classifier initializer is not configured")
			}
			if err := initFunc(modelPath, md.Classifier.UseCPU); err != nil {
				event := map[string]interface{}{
					"method":    method,
					"model_ref": modelPath,
					"error":     err.Error(),
				}
				if bestEffort {
					event["fallback_to_keywords"] = true
					logging.ComponentWarnEvent(component, "modality_classifier_init_failed", event)
				} else {
					logging.ComponentErrorEvent(component, "modality_classifier_init_failed", event)
				}
				return fmt.Errorf("failed to initialize modality classifier: %w", err)
			}
			logging.ComponentEvent(component, "modality_classifier_initialized", map[string]interface{}{
				"method": method,
			})
			return nil
		},
	}}
}

func resolveEmbeddingPaths(cfg *config.RouterConfig) embeddingPaths {
	return embeddingPaths{
		qwen3:      config.ResolveModelPath(cfg.Qwen3ModelPath),
		gemma:      config.ResolveModelPath(cfg.GemmaModelPath),
		mmBert:     config.ResolveModelPath(cfg.MmBertModelPath),
		multiModal: config.ResolveModelPath(cfg.MultiModalModelPath),
		bert:       config.ResolveModelPath(cfg.BertModelPath),
	}
}

func (p embeddingPaths) hasConfiguredModels() bool {
	return p.hasUnifiedModels() || p.multiModal != "" || p.bert != ""
}

func (p embeddingPaths) hasUnifiedModels() bool {
	return p.qwen3 != "" || p.gemma != "" || p.mmBert != ""
}

func initializeUnifiedEmbeddingModels(cfg *config.RouterConfig, component string, paths embeddingPaths) bool {
	if !paths.hasUnifiedModels() {
		return false
	}

	semanticCacheNeedsBatched, mlSelectionNeedsBatched := batchedEmbeddingNeeds(cfg, paths.qwen3)
	useBatched := semanticCacheNeedsBatched || mlSelectionNeedsBatched
	logBatchedEmbeddingNeeds(component, semanticCacheNeedsBatched, mlSelectionNeedsBatched)
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

func batchedEmbeddingNeeds(cfg *config.RouterConfig, qwen3Path string) (bool, bool) {
	semanticCacheNeedsBatched := cfg.Enabled &&
		strings.ToLower(strings.TrimSpace(cfg.EmbeddingModel)) == "qwen3" &&
		qwen3Path != ""
	mlSelectionNeedsBatched := cfg.ModelSelection.Enabled &&
		cfg.ModelSelection.ML.ModelsPath != "" &&
		cfg.Qwen3ModelPath != ""
	return semanticCacheNeedsBatched, mlSelectionNeedsBatched
}

func logBatchedEmbeddingNeeds(component string, semanticCacheNeedsBatched bool, mlSelectionNeedsBatched bool) {
	if !semanticCacheNeedsBatched && !mlSelectionNeedsBatched {
		return
	}
	logging.ComponentDebugEvent(component, "batched_embedding_mode_required", map[string]interface{}{
		"semantic_cache":  semanticCacheNeedsBatched,
		"model_selection": mlSelectionNeedsBatched,
	})
}

func initUnifiedEmbeddingModelFactory(cfg *config.RouterConfig, paths embeddingPaths, useBatched bool) error {
	contract := candle_binding.CurrentBackendContract()
	if err := validateEmbeddingBackendContract(contract, paths, useBatched); err != nil {
		return err
	}

	if !useBatched {
		return candle_binding.InitEmbeddingModels(paths.qwen3, paths.gemma, paths.mmBert, cfg.UseCPU)
	}

	if err := candle_binding.InitEmbeddingModelsBatched(paths.qwen3, 64, 10, cfg.UseCPU); err != nil {
		return err
	}
	return candle_binding.InitEmbeddingModels(paths.qwen3, paths.gemma, paths.mmBert, cfg.UseCPU)
}

func validateEmbeddingBackendContract(contract candle_binding.BackendContract, paths embeddingPaths, useBatched bool) error {
	if paths.qwen3 != "" {
		if err := contract.RequireEmbeddingFamily(candle_binding.EmbeddingFamilyQwen3); err != nil {
			return err
		}
		if useBatched {
			if err := contract.RequireBatchedEmbeddingFamily(candle_binding.EmbeddingFamilyQwen3); err != nil {
				return err
			}
		}
	}
	if paths.gemma != "" {
		if err := contract.RequireEmbeddingFamily(candle_binding.EmbeddingFamilyGemma); err != nil {
			return err
		}
		if useBatched {
			if err := contract.RequireBatchedEmbeddingFamily(candle_binding.EmbeddingFamilyGemma); err != nil {
				return err
			}
		}
	}
	if paths.mmBert != "" {
		if err := contract.RequireEmbeddingFamily(candle_binding.EmbeddingFamilyMMBert); err != nil {
			return err
		}
		if useBatched {
			if err := contract.RequireBatchedEmbeddingFamily(candle_binding.EmbeddingFamilyMMBert); err != nil {
				return err
			}
		}
	}
	return nil
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

func logMissingEmbeddingModelsConfig(component string) {
	logging.ComponentEvent(component, "embedding_models_not_configured", map[string]interface{}{
		"hint": "model_catalog.embeddings.semantic",
	})
}

func toolsUseMultiModalEmbeddings(cfg *config.RouterConfig) bool {
	return strings.EqualFold(strings.TrimSpace(cfg.EmbeddingConfig.ModelType), "multimodal")
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

func resolveBertModelID(modelID string) string {
	if modelID == "" {
		modelID = "sentence-transformers/all-MiniLM-L6-v2"
	}
	return config.ResolveModelPath(modelID)
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

func logEmbeddingRuntimeStart(component string, cfg *config.RouterConfig, paths embeddingPaths) {
	logging.ComponentEvent(component, "embedding_models_init_started", map[string]interface{}{
		"use_cpu":               cfg.UseCPU,
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
	if paths.bert == "" {
		return nil
	}

	return []Task{{
		Name:       "router.embedding.bert",
		BestEffort: true,
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
