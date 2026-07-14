package modelruntime

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

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
	plan, err := embedding.ResolveRuntimePlan(
		cfg.EmbeddingModels,
		embedding.BackendOverrideFromEnv(),
		embedding.ModelTypeOverrideFromEnv(),
	)
	if err != nil {
		return EmbeddingRuntimeState{}, fmt.Errorf("invalid embedding runtime plan: %w", err)
	}
	planPaths := paths.forRuntimePlan(plan)
	state, embeddingTasks, tracker, err := routerRuntimeEmbeddingTasks(cfg, component, paths, plan)
	if err != nil {
		return EmbeddingRuntimeState{}, fmt.Errorf("invalid local consumer embedding runtime: %w", err)
	}
	if !embeddingRuntimeConfigured(plan, planPaths) {
		logMissingEmbeddingModelsConfig(component)
	}

	tasks := append([]Task{}, embeddingTasks...)
	tasks = append(tasks, modalityClassifierTask(cfg, component, options.InitModalityClassifierFunc)...)
	if len(tasks) == 0 {
		return state, nil
	}

	_, err = Execute(ctx, tasks, Options{
		MaxParallelism: options.MaxParallelism,
		OnEvent:        options.OnEvent,
	})
	if tracker != nil {
		state = tracker.snapshot()
	}
	if embeddingRuntimeConfigured(plan, planPaths) {
		logging.ComponentEvent(component, "embedding_models_init_completed", map[string]interface{}{
			"embedding_ready": state.AnyReady,
			"tools_ready":     state.ToolsReady,
			"tools_model":     plan.ModelType,
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
) (EmbeddingRuntimeState, []Task, *embeddingStateTracker) {
	plan, err := embedding.ResolveRuntimePlan(
		cfg.EmbeddingModels,
		embedding.BackendOverrideFromEnv(),
		embedding.ModelTypeOverrideFromEnv(),
	)
	if err != nil {
		return EmbeddingRuntimeState{}, nil, nil
	}
	return embeddingRuntimeTasksForPlan(cfg, component, paths.forRuntimePlan(plan), plan)
}

func embeddingRuntimeTasksForPlan(
	cfg *config.RouterConfig,
	component string,
	paths embeddingPaths,
	plan embedding.RuntimePlan,
) (EmbeddingRuntimeState, []Task, *embeddingStateTracker) {
	return embeddingRuntimeTasksForPlanAndConsumers(cfg, component, paths, plan, embeddingPaths{})
}

func embeddingRuntimeTasksForPlanAndConsumers(
	cfg *config.RouterConfig,
	component string,
	paths embeddingPaths,
	plan embedding.RuntimePlan,
	consumerPaths embeddingPaths,
) (EmbeddingRuntimeState, []Task, *embeddingStateTracker) {
	if plan.Backend == config.EmbeddingBackendOpenAICompatible {
		tracker := newEmbeddingStateTracker(EmbeddingRuntimeState{
			EmbeddingProvider: remoteEmbeddingProviderRuntimeStateFromConfig(cfg, plan),
		})
		logEmbeddingRuntimeStart(component, cfg, paths, plan)
		return tracker.snapshot(), remoteEmbeddingRuntimeTask(cfg, component, plan, tracker), tracker
	}
	// OpenVINO lifecycle is owned by the classification runtime because its
	// binding is build-tagged there. The shared plan still prevents the remote
	// provider task (or Candle initializer) from running for an OpenVINO override.
	if plan.Backend == config.EmbeddingBackendOpenVINO {
		logEmbeddingRuntimeStart(component, cfg, paths, plan)
		return EmbeddingRuntimeState{}, nil, nil
	}
	if !paths.hasConfiguredModels() {
		return EmbeddingRuntimeState{}, nil, nil
	}

	logEmbeddingRuntimeStart(component, cfg, paths, plan)
	requiresMultimodalTools := toolsUseMultiModalEmbeddings(plan)
	tracker := newEmbeddingStateTracker(EmbeddingRuntimeState{})
	return tracker.snapshot(), buildEmbeddingRuntimeTasks(
		cfg,
		component,
		paths,
		consumerPaths,
		requiresMultimodalTools,
		tracker,
	), tracker
}

func (p embeddingPaths) forRuntimePlan(plan embedding.RuntimePlan) embeddingPaths {
	selected := embeddingPaths{}
	switch plan.ModelType {
	case config.EmbeddingModelTypeQwen3:
		selected.qwen3 = p.qwen3
	case "gemma":
		selected.gemma = p.gemma
	case "mmbert", "modernbert":
		selected.mmBert = p.mmBert
	case "multimodal":
		selected.multiModal = p.multiModal
	}
	return selected
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

func logMissingEmbeddingModelsConfig(component string) {
	logging.ComponentEvent(component, "embedding_models_not_configured", map[string]interface{}{
		"hint": "model_catalog.embeddings.semantic",
	})
}

func toolsUseMultiModalEmbeddings(plan embedding.RuntimePlan) bool {
	return strings.EqualFold(strings.TrimSpace(plan.ModelType), "multimodal")
}

func logEmbeddingRuntimeStart(component string, cfg *config.RouterConfig, paths embeddingPaths, plan embedding.RuntimePlan) {
	logging.ComponentEvent(component, "embedding_models_init_started", map[string]interface{}{
		"use_cpu":               cfg.UseCPU,
		"backend":               plan.Backend,
		"model_type":            plan.ModelType,
		"qwen3_configured":      paths.qwen3 != "",
		"gemma_configured":      paths.gemma != "",
		"mmbert_configured":     paths.mmBert != "",
		"multimodal_configured": paths.multiModal != "",
		"bert_configured":       paths.bert != "",
	})
}

func embeddingRuntimeConfigured(plan embedding.RuntimePlan, paths embeddingPaths) bool {
	return plan.Backend == config.EmbeddingBackendOpenAICompatible || plan.Backend == config.EmbeddingBackendOpenVINO || paths.hasConfiguredModels()
}
