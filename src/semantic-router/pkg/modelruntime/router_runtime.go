package modelruntime

import (
	"context"
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type EmbeddingRuntimeState struct {
	Embeddings        *embedding.Set
	AnyReady          bool
	ToolsReady        bool
	EmbeddingProvider *EmbeddingProviderRuntimeState
}

type EmbeddingProviderRuntimeState struct {
	Mode           string
	Backend        string
	Model          string
	Dimension      int
	APIKeyEnv      string
	APIKeyEnvSet   *bool
	Healthy        *bool
	LastProbeError string
	LastCheckedAt  string
}

type WarmupRouterOptions struct {
	Component      string
	MaxParallelism int
	OnEvent        func(Event)
}

// RouterWarmupTask describes request-path state that should be materialized
// before the router reports ready. Readiness is explicit because some tasks
// depend on an embedding runtime that may be intentionally unavailable.
type RouterWarmupTask struct {
	Name       string
	Ready      bool
	SkipReason string
	Load       func() error
}

type embeddingPaths struct {
	qwen3      string
	gemma      string
	mmBert     string
	multiModal string
	bert       string
}

func WarmupRouter(
	ctx context.Context,
	warmups []RouterWarmupTask,
	options WarmupRouterOptions,
) (Summary, error) {
	component := options.Component
	if component == "" {
		component = "router"
	}
	tasks := make([]Task, 0, len(warmups))
	taskNames := make([]string, 0, len(warmups))
	for _, warmup := range warmups {
		if !warmup.Ready {
			logging.ComponentEvent(component, warmup.Name+"_load_skipped", map[string]interface{}{
				"reason": warmup.SkipReason,
			})
			continue
		}
		if warmup.Load == nil {
			continue
		}
		taskName := "router.warmup." + warmup.Name
		taskNames = append(taskNames, warmup.Name)
		tasks = append(tasks, Task{
			Name:       taskName,
			BestEffort: true,
			Run: func(ctx context.Context) error {
				if err := ctx.Err(); err != nil {
					return err
				}
				logging.ComponentEvent(component, warmup.Name+"_load_started", map[string]interface{}{})
				return warmup.Load()
			},
		})
	}
	if len(tasks) == 0 {
		return Summary{Results: map[string]TaskResult{}}, nil
	}

	logging.ComponentEvent(component, "runtime_warmup_started", map[string]interface{}{
		"tasks": strings.Join(taskNames, ","),
	})
	summary, err := Execute(ctx, tasks, Options{
		MaxParallelism: options.MaxParallelism,
		OnEvent:        options.OnEvent,
	})
	if err != nil {
		return summary, err
	}
	for _, warmup := range warmups {
		result, ok := summary.Results["router.warmup."+warmup.Name]
		if ok && result.Status == TaskSucceeded {
			logging.ComponentEvent(component, warmup.Name+"_loaded", map[string]interface{}{})
		}
	}
	return summary, nil
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

func semanticCacheNeedsBERT(cfg *config.RouterConfig) bool {
	if cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() {
		return false
	}
	return cfg.Enabled && resolveSemanticCacheEmbeddingModel(cfg) == "bert"
}

func vectorStoreNeedsBERT(cfg *config.RouterConfig) bool {
	if cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() {
		return false
	}
	return cfg.VectorStore != nil && cfg.VectorStore.Enabled && cfg.VectorStore.EmbeddingModel == "bert" && !cfg.Enabled
}

func memoryNeedsBERT(cfg *config.RouterConfig) bool {
	if cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() {
		return false
	}
	if !memoryConfigured(cfg) {
		return false
	}
	return resolveMemoryEmbeddingModel(cfg) == "bert"
}

func memoryConfigured(cfg *config.RouterConfig) bool { return config.MemoryConfigured(cfg) }
func resolveSemanticCacheEmbeddingModel(cfg *config.RouterConfig) string {
	return config.SemanticCacheEmbeddingModel(cfg)
}

func resolveMemoryEmbeddingModel(cfg *config.RouterConfig) string {
	return config.MemoryEmbeddingModel(cfg)
}

func resolveBertModelID(modelID string) string {
	if modelID == "" {
		modelID = "sentence-transformers/all-MiniLM-L6-v2"
	}
	return config.ResolveModelPath(modelID)
}

func remoteEmbeddingProviderRuntimeStateFromConfig(cfg *config.RouterConfig) *EmbeddingProviderRuntimeState {
	if cfg == nil || !cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() {
		return nil
	}

	apiKeyEnv := strings.TrimSpace(cfg.EmbeddingModels.Endpoint.APIKeyEnv)
	var apiKeyEnvSet *bool
	if apiKeyEnv != "" {
		value := os.Getenv(apiKeyEnv) != ""
		apiKeyEnvSet = &value
	}

	dimension := cfg.EmbeddingModels.Endpoint.Dimensions
	if dimension == 0 {
		dimension = cfg.EmbeddingModels.EmbeddingConfig.TargetDimension
	}

	return &EmbeddingProviderRuntimeState{
		Mode:         "remote",
		Backend:      cfg.EmbeddingModels.EmbeddingBackend(),
		Model:        strings.TrimSpace(cfg.EmbeddingModels.Endpoint.Model),
		Dimension:    dimension,
		APIKeyEnv:    apiKeyEnv,
		APIKeyEnvSet: apiKeyEnvSet,
	}
}

func remoteEmbeddingProviderProbeStatus(
	cfg *config.RouterConfig,
	provider embedding.Provider,
	dimension int,
	probeErr error,
) *EmbeddingProviderRuntimeState {
	status := remoteEmbeddingProviderRuntimeStateFromConfig(cfg)
	if status == nil {
		status = &EmbeddingProviderRuntimeState{Mode: "remote"}
	}
	if provider != nil {
		status.Backend = provider.Backend()
	}
	if dimension > 0 {
		status.Dimension = dimension
	}
	healthy := probeErr == nil
	status.Healthy = &healthy
	status.LastCheckedAt = time.Now().UTC().Format(time.RFC3339)
	if probeErr != nil {
		status.LastProbeError = probeErr.Error()
	}
	return status
}
