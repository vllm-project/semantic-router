package modelruntime

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modellifecycle"
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

func PrepareRouterRuntime(
	ctx context.Context,
	cfg *config.RouterConfig,
	options PrepareRouterRuntimeOptions,
) (EmbeddingRuntimeState, error) {
	component := options.Component
	if component == "" {
		component = "router"
	}
	if cfg == nil {
		logMissingEmbeddingModelsConfig(component)
		return EmbeddingRuntimeState{}, nil
	}

	plan := modellifecycle.BuildPlan(cfg)
	paths := resolveEmbeddingPaths(plan)
	tracker, embeddingTasks := embeddingRuntimeTasks(cfg, component, paths)
	if !paths.hasConfiguredModels() {
		logMissingEmbeddingModelsConfig(component)
	}

	tasks := append([]Task{}, embeddingTasks...)
	tasks = append(tasks, semanticCacheBERTTask(cfg, component, paths)...)
	tasks = append(tasks, vectorStoreBERTTask(cfg, component, paths)...)
	tasks = append(tasks, modalityClassifierTask(cfg, component, options.InitModalityClassifierFunc)...)
	state := embeddingRuntimeState(tracker)
	if len(tasks) == 0 {
		return state, nil
	}

	_, err := Execute(ctx, tasks, Options{
		MaxParallelism: options.MaxParallelism,
		OnEvent:        options.OnEvent,
	})
	state = embeddingRuntimeState(tracker)
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
