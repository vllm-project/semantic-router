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
		logging.Fatalf("Config file not found: %s", configPath)
	}

	cfg, err := config.Parse(configPath)
	if err != nil {
		logging.Fatalf("Failed to load config: %v", err)
	}
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
		logging.Warnf("%s: %v", warning, err)
	}
}

func failStartup(writer *startupstatus.Writer, format string, args ...interface{}) {
	message := fmt.Sprintf(format, args...)
	_ = writer.Write(startupstatus.State{
		Phase:   "error",
		Ready:   false,
		Message: message,
	})
	logging.Fatalf("%s", message)
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

	logging.Infof("Download-only mode: models downloaded successfully, exiting")
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
		logging.Warnf("Failed to initialize tracing: %v", err)
	}
	return shutdownTracing
}

func shutdownTracing() {
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := tracing.ShutdownTracing(shutdownCtx); err != nil {
		logging.Errorf("Failed to shutdown tracing: %v", err)
	}
}

func initializeWindowedMetricsIfEnabled(cfg *config.RouterConfig) {
	if !cfg.Observability.Metrics.WindowedMetrics.Enabled {
		return
	}

	logging.Infof("Initializing windowed metrics for load balancing...")
	if err := metrics.InitializeWindowedMetrics(cfg.Observability.Metrics.WindowedMetrics); err != nil {
		logging.Warnf("Failed to initialize windowed metrics: %v", err)
		return
	}
	logging.Infof("Windowed metrics initialized successfully")
}

func registerSignalHandler(shutdownHooks *[]func()) {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		logging.Infof("Received shutdown signal, cleaning up...")
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
		logging.Infof("Metrics server disabled")
		return
	}

	go func() {
		http.Handle("/metrics", promhttp.Handler())
		metricsAddr := fmt.Sprintf(":%d", metricsPort)
		logging.Infof("Starting metrics server on %s", metricsAddr)
		if err := http.ListenAndServe(metricsAddr, nil); err != nil {
			logging.Errorf("Metrics server error: %v", err)
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

	logging.Infof("Initializing embedding models: qwen3=%q, gemma=%q, mmbert=%q, multimodal=%q, bert=%q, useCPU=%t",
		paths.qwen3, paths.gemma, paths.mmBert, paths.multiModal, paths.bert, cfg.UseCPU)

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
	logBatchedEmbeddingNeeds(semanticCacheNeedsBatched, mlSelectionNeedsBatched)
	if err := initUnifiedEmbeddingModelFactory(cfg, paths, semanticCacheNeedsBatched || mlSelectionNeedsBatched); err != nil {
		logging.Errorf("Failed to initialize unified embedding models: %v", err)
		logging.Warnf("Embedding API endpoints will return placeholder embeddings")
		logging.Warnf("Tools database will NOT be loaded (requires embedding models)")
		return false
	}

	logging.Infof("Unified embedding models initialized successfully")
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
	if semanticCacheNeedsBatched {
		logging.Infof("Semantic cache uses qwen3, initializing with batched embedding model...")
	}
	if mlSelectionNeedsBatched {
		logging.Infof("ML model selection enabled, initializing with batched embedding model...")
	}
}

func initUnifiedEmbeddingModelFactory(cfg *config.RouterConfig, paths embeddingPaths, useBatched bool) error {
	if !useBatched {
		return candle_binding.InitEmbeddingModels(paths.qwen3, paths.gemma, paths.mmBert, cfg.UseCPU)
	}

	if err := candle_binding.InitEmbeddingModelsBatched(paths.qwen3, 64, 10, cfg.UseCPU); err != nil {
		return err
	}
	logging.Infof("Batched embedding model initialized successfully")
	return candle_binding.InitEmbeddingModels(paths.qwen3, paths.gemma, paths.mmBert, cfg.UseCPU)
}

func initializeBERTModelIfConfigured(cfg *config.RouterConfig, bertPath string) bool {
	if bertPath == "" {
		return false
	}

	logging.Infof("Initializing BERT model for memory: %s", bertPath)
	if err := candle_binding.InitModel(bertPath, cfg.UseCPU); err != nil {
		logging.Warnf("Failed to initialize BERT model: %v", err)
		logging.Warnf("Memory retrieval with 'bert' embedding type will not work")
		return false
	}
	logging.Infof("BERT model initialized successfully (384-dim for memory)")
	return true
}

func initializeMultiModalEmbeddingModelIfConfigured(cfg *config.RouterConfig, multiModalPath string) bool {
	if multiModalPath == "" {
		return false
	}

	logging.Infof("Initializing MultiModal embedding model: %s", multiModalPath)
	if err := candle_binding.InitMultiModalEmbeddingModel(multiModalPath, cfg.UseCPU); err != nil {
		logging.Warnf("Failed to initialize MultiModal embedding model: %v", err)
		logging.Warnf("Embedding paths using 'multimodal' will not work")
		return false
	}
	logging.Infof("MultiModal embedding model initialized successfully (384-dim default)")
	return true
}

func logMissingEmbeddingModelsConfig() {
	logging.Infof("No embedding models configured, skipping initialization")
	logging.Infof("To enable embedding models, add to config.yaml:")
	logging.Infof("  embedding_models:")
	logging.Infof("    qwen3_model_path: 'models/mom-embedding-pro'")
	logging.Infof("    gemma_model_path: 'models/mom-embedding-flash'")
	logging.Infof("    mmbert_model_path: 'models/mom-embedding-ultra'")
	logging.Infof("    multimodal_model_path: 'models/mom-embedding-multimodal'")
	logging.Infof("    bert_model_path: 'models/all-MiniLM-L12-v2'  # For memory (384-dim)")
	logging.Infof("    use_cpu: true")
}

func initializeSemanticCacheBERTIfNeeded(cfg *config.RouterConfig) {
	if !cfg.Enabled || resolveSemanticCacheEmbeddingModel(cfg) != "bert" {
		return
	}

	bertModelID := resolveBertModelID(cfg.BertModelPath)
	logging.Infof("Semantic cache uses BERT embeddings, initializing BERT model: %s", bertModelID)
	if err := candle_binding.InitModel(bertModelID, cfg.UseCPU); err != nil {
		logging.Fatalf("Failed to initialize BERT model for semantic cache: %v", err)
	}
	logging.Infof("BERT model initialized successfully for semantic cache")
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

	logging.Infof("Initializing vector store feature...")
	if err := cfg.VectorStore.Validate(); err != nil {
		logging.Fatalf("Invalid vector store configuration: %v", err)
	}
	cfg.VectorStore.ApplyDefaults()
	ensureVectorStoreBERTIfNeeded(cfg)

	vsFileStore := newVectorStoreFileStoreOrFatal(cfg)
	vsBackend := newVectorStoreBackendOrFatal(cfg)
	vsPipeline := configureVectorStoreRuntime(cfg, vsBackend, vsFileStore)
	registerVectorStoreShutdownHook(shutdownHooks, vsPipeline, vsBackend)

	logging.Infof("Vector store initialized: backend=%s, model=%s, dim=%d, workers=%d",
		cfg.VectorStore.BackendType, cfg.VectorStore.EmbeddingModel,
		cfg.VectorStore.EmbeddingDimension, cfg.VectorStore.IngestionWorkers)
}

func ensureVectorStoreBERTIfNeeded(cfg *config.RouterConfig) {
	if cfg.VectorStore.EmbeddingModel != "bert" || cfg.Enabled {
		return
	}

	bertModelID := resolveBertModelID(cfg.BertModelPath)
	logging.Infof("Vector store uses BERT embeddings, initializing BERT model: %s", bertModelID)
	if err := candle_binding.InitModel(bertModelID, cfg.UseCPU); err != nil {
		logging.Fatalf("Failed to initialize BERT model for vector store: %v", err)
	}
}

func newVectorStoreFileStoreOrFatal(cfg *config.RouterConfig) *vectorstore.FileStore {
	fileStore, err := vectorstore.NewFileStore(cfg.VectorStore.FileStorageDir)
	if err != nil {
		logging.Fatalf("Failed to create vector store file store: %v", err)
	}
	apiserver.SetFileStore(fileStore)
	return fileStore
}

func newVectorStoreBackendOrFatal(cfg *config.RouterConfig) vectorstore.VectorStoreBackend {
	backend, err := vectorstore.NewBackend(cfg.VectorStore.BackendType, buildVectorStoreBackendConfigs(cfg))
	if err != nil {
		logging.Fatalf("Failed to create vector store backend: %v", err)
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
		logging.Infof("Shutting down vector store pipeline...")
		vsPipeline.Stop()
		if err := vsBackend.Close(); err != nil {
			logging.Errorf("Failed to close vector store backend: %v", err)
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
	logging.Infof("Initializing modality classifier (method=%s) from model: %s", method, modelPath)
	if err := extproc.InitModalityClassifier(modelPath, md.Classifier.UseCPU); err != nil {
		if method == config.ModalityDetectionClassifier {
			logging.Fatalf("Failed to initialize modality classifier (required for method=%q): %v", method, err)
		}
		logging.Warnf("Failed to initialize modality classifier (hybrid will fall back to keywords): %v", err)
		return
	}
	logging.Infof("Modality classifier initialized successfully")
}

func newExtProcServerOrFatal(opts runtimeOptions, writer *startupstatus.Writer) *extproc.Server {
	server, err := extproc.NewServer(opts.configPath, opts.port, opts.secure, opts.certPath)
	if err != nil {
		failStartup(writer, "Failed to create ExtProc server: %v", err)
	}

	logging.Infof("Starting vLLM Semantic Router ExtProc with config: %s", opts.configPath)
	return server
}

func loadToolsDatabaseIfReady(server *extproc.Server, embeddingModelsInitialized bool) {
	router := server.GetRouter()
	if router == nil {
		return
	}
	if !embeddingModelsInitialized {
		logging.Infof("Skipping tools database loading (embedding models not initialized)")
		return
	}

	logging.Infof("Loading tools database (embedding models are ready)...")
	if err := router.LoadToolsDatabase(); err != nil {
		logging.Warnf("Failed to load tools database: %v", err)
	}
}

func startAPIServerIfEnabled(opts runtimeOptions) {
	if !opts.enableAPI {
		return
	}

	go func() {
		logging.Infof("Starting API server on port %d", opts.apiPort)
		if err := apiserver.Init(opts.configPath, opts.apiPort); err != nil {
			logging.Errorf("Start API server error: %v", err)
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
		logging.Infof("ConfigSource is kubernetes, starting Kubernetes controller")
		go startKubernetesController(cfg, kubeconfig, namespace)
		return
	}

	logging.Infof("ConfigSource is file (or not specified), using file-based configuration")
}

func startExtProcServerOrFatal(server *extproc.Server, writer *startupstatus.Writer) {
	if err := server.Start(); err != nil {
		failStartup(writer, "ExtProc server error: %v", err)
	}
}
