package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apiserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

type runtimeOptions struct {
	configPath   string
	certPath     string
	kubeconfig   string
	namespace    string
	port         int
	apiPort      int
	metricsPort  int
	enableAPI    bool
	secure       bool
	downloadOnly bool
}

type embeddingPaths struct {
	qwen3      string
	gemma      string
	mmBert     string
	multiModal string
	bert       string
}

func parseRuntimeOptions() runtimeOptions {
	var (
		configPath   = flag.String("config", "config/config.yaml", "Path to the configuration file")
		port         = flag.Int("port", 50051, "Port to listen on for gRPC ExtProc")
		apiPort      = flag.Int("api-port", 8080, "Port to listen on for the router apiserver")
		metricsPort  = flag.Int("metrics-port", 9190, "Port for Prometheus metrics")
		enableAPI    = flag.Bool("enable-api", true, "Enable the router apiserver")
		secure       = flag.Bool("secure", false, "Enable secure gRPC server with TLS")
		certPath     = flag.String("cert-path", "", "Path to TLS certificate directory (containing tls.crt and tls.key)")
		kubeconfig   = flag.String("kubeconfig", "", "Path to kubeconfig file (optional, uses in-cluster config if not specified)")
		namespace    = flag.String("namespace", "default", "Kubernetes namespace to watch for CRDs")
		downloadOnly = flag.Bool("download-only", false, "Download required models and exit (useful for CI/testing)")
	)
	flag.Parse()

	return runtimeOptions{
		configPath:   *configPath,
		certPath:     *certPath,
		kubeconfig:   *kubeconfig,
		namespace:    *namespace,
		port:         *port,
		apiPort:      *apiPort,
		metricsPort:  *metricsPort,
		enableAPI:    *enableAPI,
		secure:       *secure,
		downloadOnly: *downloadOnly,
	}
}

func initializeRuntimeLogger() {
	if _, err := logging.InitLoggerFromEnv(); err != nil {
		fmt.Fprintf(os.Stderr, "failed to initialize logger: %v\n", err)
	}
}

func loadRuntimeConfigOrFatal(configPath string) *config.RouterConfig {
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		logging.ComponentFatalEvent("router", "runtime_config_missing", map[string]interface{}{
			"config_path": configPath,
		})
	}

	cfg, err := config.Parse(configPath)
	if err != nil {
		logging.ComponentFatalEvent("router", "runtime_config_load_failed", map[string]interface{}{
			"config_path": configPath,
			"error":       err.Error(),
		})
	}
	logging.ComponentDebugEvent("router", "runtime_config_loaded", map[string]interface{}{
		"config_path":    configPath,
		"config_source":  cfg.ConfigSource,
		"decision_count": len(cfg.Decisions),
	})
	return cfg
}

func newStartupWriter(configPath string) *startupstatus.Writer {
	writer := startupstatus.NewWriter(configPath)
	writeStartupState(writer, startupstatus.State{
		Phase:   "starting",
		Ready:   false,
		Message: "Router process booting...",
	}, "Failed to write initial startup status")
	return writer
}

func writeStartupState(writer *startupstatus.Writer, state startupstatus.State, warning string) {
	if err := writer.Write(state); err != nil {
		logging.ComponentWarnEvent("router", "startup_state_write_failed", map[string]interface{}{
			"warning": warning,
			"phase":   state.Phase,
			"ready":   state.Ready,
			"error":   err.Error(),
		})
	}
}

func failStartup(writer *startupstatus.Writer, format string, args ...interface{}) {
	message := fmt.Sprintf(format, args...)
	_ = writer.Write(startupstatus.State{
		Phase:   "error",
		Ready:   false,
		Message: message,
	})
	logging.ComponentFatalEvent("router", "startup_failed", map[string]interface{}{
		"message": message,
	})
}

func ensureModelsDownloadedOrFatal(cfg *config.RouterConfig, writer *startupstatus.Writer) {
	if err := ensureModelsDownloaded(cfg, writer); err != nil {
		failStartup(writer, "Failed to ensure models are downloaded: %v", err)
	}
}

func exitIfDownloadOnly(downloadOnly bool) {
	if !downloadOnly {
		return
	}

	logging.ComponentEvent("router", "download_only_complete", map[string]interface{}{
		"mode": "download_only",
	})
	os.Exit(0)
}

func initializeTracing(cfg *config.RouterConfig) func() {
	if !cfg.Observability.Tracing.Enabled {
		return func() {}
	}

	tracingCfg := tracing.TracingConfig{
		Enabled:               cfg.Observability.Tracing.Enabled,
		Provider:              cfg.Observability.Tracing.Provider,
		ExporterType:          cfg.Observability.Tracing.Exporter.Type,
		ExporterEndpoint:      cfg.Observability.Tracing.Exporter.Endpoint,
		ExporterInsecure:      cfg.Observability.Tracing.Exporter.Insecure,
		SamplingType:          cfg.Observability.Tracing.Sampling.Type,
		SamplingRate:          cfg.Observability.Tracing.Sampling.Rate,
		ServiceName:           cfg.Observability.Tracing.Resource.ServiceName,
		ServiceVersion:        cfg.Observability.Tracing.Resource.ServiceVersion,
		DeploymentEnvironment: cfg.Observability.Tracing.Resource.DeploymentEnvironment,
	}
	if err := tracing.InitTracing(context.Background(), tracingCfg); err != nil {
		logging.ComponentWarnEvent("router", "tracing_init_failed", map[string]interface{}{
			"provider":          tracingCfg.Provider,
			"exporter_type":     tracingCfg.ExporterType,
			"exporter_endpoint": tracingCfg.ExporterEndpoint,
			"error":             err.Error(),
		})
	}
	return shutdownTracing
}

func shutdownTracing() {
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := tracing.ShutdownTracing(shutdownCtx); err != nil {
		logging.ComponentErrorEvent("router", "tracing_shutdown_failed", map[string]interface{}{
			"error": err.Error(),
		})
	}
}

func initializeWindowedMetricsIfEnabled(cfg *config.RouterConfig) {
	if !cfg.Observability.Metrics.WindowedMetrics.Enabled {
		return
	}

	if err := metrics.InitializeWindowedMetrics(cfg.Observability.Metrics.WindowedMetrics); err != nil {
		logging.ComponentWarnEvent("router", "windowed_metrics_init_failed", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}
	logging.ComponentEvent("router", "windowed_metrics_initialized", map[string]interface{}{
		"mode": "load_balancing",
	})
}

func registerSignalHandler(shutdownHooks *[]func()) {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		sig := <-sigChan
		logging.ComponentEvent("router", "shutdown_signal_received", map[string]interface{}{
			"signal": sig.String(),
		})
		for _, hook := range *shutdownHooks {
			hook()
		}
		shutdownTracing()
		os.Exit(0)
	}()
}

func startMetricsServerIfEnabled(cfg *config.RouterConfig, metricsPort int) {
	metricsEnabled := true
	if cfg.Observability.Metrics.Enabled != nil {
		metricsEnabled = *cfg.Observability.Metrics.Enabled
	}
	if metricsPort <= 0 {
		metricsEnabled = false
	}
	if !metricsEnabled {
		logging.ComponentEvent("router", "metrics_server_disabled", map[string]interface{}{
			"metrics_port": metricsPort,
		})
		return
	}

	go func() {
		http.Handle("/metrics", promhttp.Handler())
		metricsAddr := fmt.Sprintf(":%d", metricsPort)
		logging.ComponentEvent("router", "metrics_server_starting", map[string]interface{}{
			"address": metricsAddr,
		})
		if err := http.ListenAndServe(metricsAddr, nil); err != nil {
			logging.ComponentErrorEvent("router", "metrics_server_failed", map[string]interface{}{
				"address": metricsAddr,
				"error":   err.Error(),
			})
		}
	}()
}

func initializeRuntimeDependencies(
	cfg *config.RouterConfig,
	writer *startupstatus.Writer,
	shutdownHooks *[]func(),
) bool {
	writeStartupState(writer, startupstatus.State{
		Phase:   "initializing_models",
		Ready:   false,
		Message: "Initializing embedding models and router dependencies...",
	}, "Failed to write initialization startup status")

	embeddingModelsInitialized := initializeEmbeddingModels(cfg)
	initializeSemanticCacheBERTIfNeeded(cfg)
	initializeVectorStoreIfEnabled(cfg, shutdownHooks)
	initializeModalityClassifierIfEnabled(cfg)
	return embeddingModelsInitialized
}

func initializeEmbeddingModels(cfg *config.RouterConfig) bool {
	paths := resolveEmbeddingPaths(cfg)
	if !paths.hasConfiguredModels() {
		logMissingEmbeddingModelsConfig()
		return false
	}

	logging.ComponentEvent("router", "embedding_models_init_started", map[string]interface{}{
		"use_cpu":               cfg.UseCPU,
		"qwen3_configured":      paths.qwen3 != "",
		"gemma_configured":      paths.gemma != "",
		"mmbert_configured":     paths.mmBert != "",
		"multimodal_configured": paths.multiModal != "",
		"bert_configured":       paths.bert != "",
	})

	initialized := initializeUnifiedEmbeddingModels(cfg, paths)
	if initializeBERTModelIfConfigured(cfg, paths.bert) {
		initialized = true
	}
	if initializeMultiModalEmbeddingModelIfConfigured(cfg, paths.multiModal) {
		initialized = true
	}
	return initialized
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

func initializeUnifiedEmbeddingModels(cfg *config.RouterConfig, paths embeddingPaths) bool {
	if !paths.hasUnifiedModels() {
		return false
	}

	semanticCacheNeedsBatched, mlSelectionNeedsBatched := batchedEmbeddingNeeds(cfg, paths.qwen3)
	useBatched := semanticCacheNeedsBatched || mlSelectionNeedsBatched
	logBatchedEmbeddingNeeds(semanticCacheNeedsBatched, mlSelectionNeedsBatched)
	if err := initUnifiedEmbeddingModelFactory(cfg, paths, useBatched); err != nil {
		logging.ComponentErrorEvent("router", "embedding_models_init_failed", map[string]interface{}{
			"use_batched": useBatched,
			"error":       err.Error(),
		})
		logging.ComponentWarnEvent("router", "embedding_runtime_degraded", map[string]interface{}{
			"embedding_api_placeholder": true,
			"tools_database_disabled":   true,
		})
		return false
	}

	logging.ComponentEvent("router", "embedding_models_initialized", map[string]interface{}{
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

func logBatchedEmbeddingNeeds(semanticCacheNeedsBatched bool, mlSelectionNeedsBatched bool) {
	if !semanticCacheNeedsBatched && !mlSelectionNeedsBatched {
		return
	}
	logging.ComponentDebugEvent("router", "batched_embedding_mode_required", map[string]interface{}{
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

func initializeBERTModelIfConfigured(cfg *config.RouterConfig, bertPath string) bool {
	if bertPath == "" {
		return false
	}

	logging.ComponentEvent("router", "memory_bert_init_started", map[string]interface{}{
		"model_ref": bertPath,
		"use_cpu":   cfg.UseCPU,
	})
	if err := candle_binding.InitModel(bertPath, cfg.UseCPU); err != nil {
		logging.ComponentWarnEvent("router", "memory_bert_init_failed", map[string]interface{}{
			"model_ref":              bertPath,
			"error":                  err.Error(),
			"memory_retrieval_ready": false,
		})
		return false
	}
	logging.ComponentEvent("router", "memory_bert_initialized", map[string]interface{}{
		"model_ref": bertPath,
	})
	return true
}

func initializeMultiModalEmbeddingModelIfConfigured(cfg *config.RouterConfig, multiModalPath string) bool {
	if multiModalPath == "" {
		return false
	}

	logging.ComponentEvent("router", "multimodal_embedding_init_started", map[string]interface{}{
		"model_ref": multiModalPath,
		"use_cpu":   cfg.UseCPU,
	})
	if err := candle_binding.InitMultiModalEmbeddingModel(multiModalPath, cfg.UseCPU); err != nil {
		logging.ComponentWarnEvent("router", "multimodal_embedding_init_failed", map[string]interface{}{
			"model_ref":               multiModalPath,
			"error":                   err.Error(),
			"multimodal_routes_ready": false,
		})
		return false
	}
	logging.ComponentEvent("router", "multimodal_embedding_initialized", map[string]interface{}{
		"model_ref": multiModalPath,
	})
	return true
}

func logMissingEmbeddingModelsConfig() {
	logging.ComponentEvent("router", "embedding_models_not_configured", map[string]interface{}{
		"hint": "model_catalog.embeddings.semantic",
	})
}

func initializeSemanticCacheBERTIfNeeded(cfg *config.RouterConfig) {
	if !cfg.Enabled || resolveSemanticCacheEmbeddingModel(cfg) != "bert" {
		return
	}

	bertModelID := resolveBertModelID(cfg.BertModelPath)
	logging.ComponentEvent("router", "semantic_cache_bert_init_started", map[string]interface{}{
		"model_ref": bertModelID,
		"use_cpu":   cfg.UseCPU,
	})
	if err := candle_binding.InitModel(bertModelID, cfg.UseCPU); err != nil {
		logging.ComponentFatalEvent("router", "semantic_cache_bert_init_failed", map[string]interface{}{
			"model_ref": bertModelID,
			"error":     err.Error(),
		})
	}
	logging.ComponentEvent("router", "semantic_cache_bert_initialized", map[string]interface{}{
		"model_ref": bertModelID,
	})
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

func initializeVectorStoreIfEnabled(cfg *config.RouterConfig, shutdownHooks *[]func()) {
	if cfg.VectorStore == nil || !cfg.VectorStore.Enabled {
		return
	}

	logging.ComponentEvent("router", "vector_store_init_started", map[string]interface{}{
		"backend": cfg.VectorStore.BackendType,
	})
	if err := cfg.VectorStore.Validate(); err != nil {
		logging.ComponentFatalEvent("router", "vector_store_config_invalid", map[string]interface{}{
			"backend": cfg.VectorStore.BackendType,
			"error":   err.Error(),
		})
	}
	cfg.VectorStore.ApplyDefaults()
	ensureVectorStoreBERTIfNeeded(cfg)

	vsFileStore := newVectorStoreFileStoreOrFatal(cfg)
	vsBackend := newVectorStoreBackendOrFatal(cfg)
	vsPipeline := configureVectorStoreRuntime(cfg, vsBackend, vsFileStore)
	registerVectorStoreShutdownHook(shutdownHooks, vsPipeline, vsBackend)

	logging.ComponentEvent("router", "vector_store_initialized", map[string]interface{}{
		"backend": cfg.VectorStore.BackendType,
		"model":   cfg.VectorStore.EmbeddingModel,
		"dim":     cfg.VectorStore.EmbeddingDimension,
		"workers": cfg.VectorStore.IngestionWorkers,
	})
}

func ensureVectorStoreBERTIfNeeded(cfg *config.RouterConfig) {
	if cfg.VectorStore.EmbeddingModel != "bert" || cfg.Enabled {
		return
	}

	bertModelID := resolveBertModelID(cfg.BertModelPath)
	logging.ComponentEvent("router", "vector_store_bert_init_started", map[string]interface{}{
		"model_ref": bertModelID,
		"use_cpu":   cfg.UseCPU,
	})
	if err := candle_binding.InitModel(bertModelID, cfg.UseCPU); err != nil {
		logging.ComponentFatalEvent("router", "vector_store_bert_init_failed", map[string]interface{}{
			"model_ref": bertModelID,
			"error":     err.Error(),
		})
	}
	logging.ComponentEvent("router", "vector_store_bert_initialized", map[string]interface{}{
		"model_ref": bertModelID,
	})
}

func newVectorStoreFileStoreOrFatal(cfg *config.RouterConfig) *vectorstore.FileStore {
	fileStore, err := vectorstore.NewFileStore(cfg.VectorStore.FileStorageDir)
	if err != nil {
		logging.ComponentFatalEvent("router", "vector_store_filestore_create_failed", map[string]interface{}{
			"storage_dir": cfg.VectorStore.FileStorageDir,
			"error":       err.Error(),
		})
	}
	apiserver.SetFileStore(fileStore)
	return fileStore
}

func newVectorStoreBackendOrFatal(cfg *config.RouterConfig) vectorstore.VectorStoreBackend {
	backend, err := vectorstore.NewBackend(cfg.VectorStore.BackendType, buildVectorStoreBackendConfigs(cfg))
	if err != nil {
		logging.ComponentFatalEvent("router", "vector_store_backend_create_failed", map[string]interface{}{
			"backend": cfg.VectorStore.BackendType,
			"error":   err.Error(),
		})
	}
	return backend
}

func buildVectorStoreBackendConfigs(cfg *config.RouterConfig) vectorstore.BackendConfigs {
	switch cfg.VectorStore.BackendType {
	case "memory":
		maxEntries := 100000
		if cfg.VectorStore.Memory != nil && cfg.VectorStore.Memory.MaxEntriesPerStore > 0 {
			maxEntries = cfg.VectorStore.Memory.MaxEntriesPerStore
		}
		return vectorstore.BackendConfigs{
			Memory: vectorstore.MemoryBackendConfig{MaxEntriesPerStore: maxEntries},
		}
	case "milvus":
		return vectorstore.BackendConfigs{
			Milvus: vectorstore.MilvusBackendConfig{
				Address: fmt.Sprintf("%s:%d", cfg.VectorStore.Milvus.Connection.Host, cfg.VectorStore.Milvus.Connection.Port),
			},
		}
	case "llama_stack":
		lsCfg := cfg.VectorStore.LlamaStack
		return vectorstore.BackendConfigs{
			LlamaStack: vectorstore.LlamaStackBackendConfig{
				Endpoint:              lsCfg.Endpoint,
				AuthToken:             lsCfg.AuthToken,
				EmbeddingModel:        lsCfg.EmbeddingModel,
				EmbeddingDimension:    cfg.VectorStore.EmbeddingDimension,
				RequestTimeoutSeconds: lsCfg.RequestTimeoutSeconds,
				SearchType:            lsCfg.SearchType,
			},
		}
	case "valkey":
		vCfg := cfg.VectorStore.Valkey
		return vectorstore.BackendConfigs{
			Valkey: vectorstore.ValkeyBackendConfig{
				Host:             vCfg.Host,
				Port:             vCfg.Port,
				Password:         vCfg.Password,
				Database:         vCfg.Database,
				CollectionPrefix: vCfg.CollectionPrefix,
				MetricType:       vCfg.MetricType,
				IndexM:           vCfg.IndexM,
				IndexEf:          vCfg.IndexEfConstruction,
				ConnectTimeout:   vCfg.ConnectTimeout,
			},
		}
	default:
		return vectorstore.BackendConfigs{}
	}
}

func configureVectorStoreRuntime(
	cfg *config.RouterConfig,
	vsBackend vectorstore.VectorStoreBackend,
	vsFileStore *vectorstore.FileStore,
) *vectorstore.IngestionPipeline {
	vsMgr := vectorstore.NewManager(vsBackend, cfg.VectorStore.EmbeddingDimension, cfg.VectorStore.BackendType)
	apiserver.SetVectorStoreManager(vsMgr)

	vsEmbedder := vectorstore.NewCandleEmbedder(cfg.VectorStore.EmbeddingModel, cfg.VectorStore.EmbeddingDimension)
	apiserver.SetEmbedder(vsEmbedder)

	vsPipeline := vectorstore.NewIngestionPipeline(vsBackend, vsFileStore, vsMgr, vsEmbedder, vectorstore.PipelineConfig{
		Workers:   cfg.VectorStore.IngestionWorkers,
		QueueSize: 100,
	})
	vsPipeline.Start()
	apiserver.SetIngestionPipeline(vsPipeline)
	return vsPipeline
}

func registerVectorStoreShutdownHook(
	shutdownHooks *[]func(),
	vsPipeline *vectorstore.IngestionPipeline,
	vsBackend vectorstore.VectorStoreBackend,
) {
	*shutdownHooks = append(*shutdownHooks, func() {
		logging.ComponentEvent("router", "vector_store_shutdown_started", map[string]interface{}{})
		vsPipeline.Stop()
		if err := vsBackend.Close(); err != nil {
			logging.ComponentErrorEvent("router", "vector_store_shutdown_failed", map[string]interface{}{
				"error": err.Error(),
			})
		}
	})
}

func initializeModalityClassifierIfEnabled(cfg *config.RouterConfig) {
	md := &cfg.ModalityDetector
	if !md.Enabled {
		return
	}

	method := md.GetMethod()
	if method != config.ModalityDetectionClassifier && method != config.ModalityDetectionHybrid {
		return
	}
	if md.Classifier == nil || md.Classifier.ModelPath == "" {
		return
	}

	modelPath := config.ResolveModelPath(md.Classifier.ModelPath)
	logging.ComponentEvent("router", "modality_classifier_init_started", map[string]interface{}{
		"method":    method,
		"model_ref": modelPath,
		"use_cpu":   md.Classifier.UseCPU,
	})
	if err := extproc.InitModalityClassifier(modelPath, md.Classifier.UseCPU); err != nil {
		if method == config.ModalityDetectionClassifier {
			logging.ComponentFatalEvent("router", "modality_classifier_init_failed", map[string]interface{}{
				"method":    method,
				"model_ref": modelPath,
				"error":     err.Error(),
			})
		}
		logging.ComponentWarnEvent("router", "modality_classifier_init_failed", map[string]interface{}{
			"method":               method,
			"model_ref":            modelPath,
			"error":                err.Error(),
			"fallback_to_keywords": true,
		})
		return
	}
	logging.ComponentEvent("router", "modality_classifier_initialized", map[string]interface{}{
		"method": method,
	})
}

func newExtProcServerOrFatal(opts runtimeOptions, writer *startupstatus.Writer) *extproc.Server {
	server, err := extproc.NewServer(opts.configPath, opts.port, opts.secure, opts.certPath)
	if err != nil {
		failStartup(writer, "Failed to create ExtProc server: %v", err)
	}

	return server
}

func loadToolsDatabaseIfReady(server *extproc.Server, embeddingModelsInitialized bool) {
	router := server.GetRouter()
	if router == nil {
		return
	}
	if !embeddingModelsInitialized {
		logging.ComponentEvent("router", "tools_database_load_skipped", map[string]interface{}{
			"reason": "embedding_models_not_initialized",
		})
		return
	}

	logging.ComponentEvent("router", "tools_database_load_started", map[string]interface{}{})
	if err := router.LoadToolsDatabase(); err != nil {
		logging.ComponentWarnEvent("router", "tools_database_load_failed", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}
	logging.ComponentEvent("router", "tools_database_loaded", map[string]interface{}{})
}

func startAPIServerIfEnabled(opts runtimeOptions) {
	if !opts.enableAPI {
		return
	}

	go func() {
		logging.ComponentEvent("router", "api_server_starting", map[string]interface{}{
			"api_port": opts.apiPort,
		})
		if err := apiserver.Init(opts.configPath, opts.apiPort); err != nil {
			logging.ComponentErrorEvent("router", "api_server_failed", map[string]interface{}{
				"api_port": opts.apiPort,
				"error":    err.Error(),
			})
		}
	}()
}

func markRouterReady(writer *startupstatus.Writer) {
	writeStartupState(writer, startupstatus.State{
		Phase:   "ready",
		Ready:   true,
		Message: "Router models are ready. Starting router services...",
	}, "Failed to write ready startup status")
}

func startKubernetesControllerIfNeeded(cfg *config.RouterConfig, kubeconfig, namespace string) {
	if cfg.ConfigSource == config.ConfigSourceKubernetes {
		go startKubernetesController(cfg, kubeconfig, namespace)
	}
}

func startExtProcServerOrFatal(server *extproc.Server, writer *startupstatus.Writer) {
	if err := server.Start(); err != nil {
		failStartup(writer, "ExtProc server error: %v", err)
	}
}
