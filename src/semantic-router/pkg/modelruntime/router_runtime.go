package modelruntime

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
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

type PrepareRouterRuntimeOptions struct {
	NativeRuntime              *native.Runtime
	Component                  string
	MaxParallelism             int
	OnEvent                    func(Event)
	InitModalityClassifierFunc func(modelPath string, useCPU bool) error
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

type embeddingStateTracker struct {
	mu    sync.Mutex
	state EmbeddingRuntimeState
}

func newEmbeddingStateTracker(state EmbeddingRuntimeState) *embeddingStateTracker {
	return &embeddingStateTracker{state: cloneEmbeddingRuntimeState(state)}
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

	// Compatibility health probe: callers outside generation construction can
	// inspect a remote endpoint without creating native model ownership.
	if cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() {
		_, tasks, tracker := embeddingRuntimeTasks(cfg, component, resolveEmbeddingPaths(cfg))
		_, err := Execute(ctx, tasks, Options{MaxParallelism: options.MaxParallelism, OnEvent: options.OnEvent})
		return tracker.snapshot(), err
	}

	owned, err := PrepareOwnedEmbeddings(ctx, cfg, options.NativeRuntime)
	state := EmbeddingRuntimeState{Embeddings: owned, AnyReady: owned.Ready(), ToolsReady: owned.Has("")}
	if err != nil {
		return state, err
	}
	if provider, providerErr := owned.Default(); providerErr == nil {
		state.AnyReady = true
		state.ToolsReady = true
		if cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() {
			state.EmbeddingProvider = remoteEmbeddingProviderProbeStatus(cfg, provider, provider.Dimension(), nil)
		}
	}
	return state, nil
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

func embeddingRuntimeTasks(
	cfg *config.RouterConfig,
	component string,
	paths embeddingPaths,
) (EmbeddingRuntimeState, []Task, *embeddingStateTracker) {
	if cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() {
		tracker := newEmbeddingStateTracker(EmbeddingRuntimeState{
			EmbeddingProvider: remoteEmbeddingProviderRuntimeStateFromConfig(cfg),
		})
		logEmbeddingRuntimeStart(component, cfg, paths)
		return tracker.snapshot(), remoteEmbeddingRuntimeTask(cfg, component, tracker), tracker
	}
	// OpenVINO model initialization belongs to the classifier runtime. Do not
	// send its model paths through the Candle-only shared runtime.
	if cfg.EmbeddingModels.EmbeddingBackend() == config.EmbeddingBackendOpenVINO {
		return EmbeddingRuntimeState{}, nil, nil
	}
	if !paths.hasConfiguredModels() {
		return EmbeddingRuntimeState{}, nil, nil
	}

	logEmbeddingRuntimeStart(component, cfg, paths)
	requiresMultimodalTools := toolsUseMultiModalEmbeddings(cfg)
	tracker := newEmbeddingStateTracker(EmbeddingRuntimeState{})
	return tracker.snapshot(), buildEmbeddingRuntimeTasks(cfg, component, paths, requiresMultimodalTools, tracker), tracker
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

func toolsUseMultiModalEmbeddings(cfg *config.RouterConfig) bool {
	return strings.EqualFold(strings.TrimSpace(cfg.EmbeddingConfig.ModelType), "multimodal")
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

func (t *embeddingStateTracker) markEmbeddingProvider(status *EmbeddingProviderRuntimeState) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.state.EmbeddingProvider = cloneEmbeddingProviderRuntimeState(status)
}

func (t *embeddingStateTracker) snapshot() EmbeddingRuntimeState {
	t.mu.Lock()
	defer t.mu.Unlock()
	return cloneEmbeddingRuntimeState(t.state)
}

func cloneEmbeddingRuntimeState(state EmbeddingRuntimeState) EmbeddingRuntimeState {
	state.EmbeddingProvider = cloneEmbeddingProviderRuntimeState(state.EmbeddingProvider)
	return state
}

func cloneEmbeddingProviderRuntimeState(status *EmbeddingProviderRuntimeState) *EmbeddingProviderRuntimeState {
	if status == nil {
		return nil
	}
	clone := *status
	if status.APIKeyEnvSet != nil {
		value := *status.APIKeyEnvSet
		clone.APIKeyEnvSet = &value
	}
	if status.Healthy != nil {
		value := *status.Healthy
		clone.Healthy = &value
	}
	return &clone
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

func logEmbeddingRuntimeStart(component string, cfg *config.RouterConfig, paths embeddingPaths) {
	logging.ComponentEvent(component, "embedding_models_init_started", map[string]interface{}{
		"use_cpu":               cfg.UseCPU,
		"backend":               cfg.EmbeddingModels.EmbeddingBackend(),
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

func remoteEmbeddingRuntimeTask(
	cfg *config.RouterConfig,
	component string,
	tracker *embeddingStateTracker,
) []Task {
	return []Task{{
		Name:       "router.embedding.remote_provider",
		BestEffort: true,
		Run: func(ctx context.Context) error {
			logging.ComponentEvent(component, "remote_embedding_init_started", map[string]interface{}{
				"backend": cfg.EmbeddingModels.EmbeddingBackend(),
				"model":   cfg.EmbeddingModels.Endpoint.Model,
			})
			provider, err := embedding.NewProvider(cfg.EmbeddingModels, embedding.ProviderOptions{})
			if err != nil {
				tracker.markEmbeddingProvider(remoteEmbeddingProviderProbeStatus(cfg, nil, 0, err))
				logging.ComponentErrorEvent(component, "remote_embedding_init_failed", map[string]interface{}{
					"error": err.Error(),
				})
				return fmt.Errorf("failed to initialize remote embedding provider: %w", err)
			}
			if closer, ok := provider.(interface{ Close() error }); ok {
				defer closer.Close()
			}
			embeddingVector, err := provider.Embed(ctx, "semantic router embedding probe")
			if err != nil {
				tracker.markEmbeddingProvider(remoteEmbeddingProviderProbeStatus(cfg, provider, 0, err))
				logging.ComponentErrorEvent(component, "remote_embedding_init_failed", map[string]interface{}{
					"backend": provider.Backend(),
					"error":   err.Error(),
				})
				return fmt.Errorf("failed to probe remote embedding provider: %w", err)
			}
			tracker.markEmbeddingProvider(remoteEmbeddingProviderProbeStatus(cfg, provider, len(embeddingVector), nil))
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
