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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/logo"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modeldownload"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

func main() {
	// Display vLLM logo
	logo.PrintVLLMLogo()

	// Parse command-line flags
	var (
		configPath            = flag.String("config", "config/config.yaml", "Path to the configuration file")
		port                  = flag.Int("port", 50051, "Port to listen on for gRPC ExtProc")
		apiPort               = flag.Int("api-port", 8080, "Port to listen on for Classification API")
		metricsPort           = flag.Int("metrics-port", 9190, "Port for Prometheus metrics")
		enableAPI             = flag.Bool("enable-api", true, "Enable Classification API server")
		enableSystemPromptAPI = flag.Bool("enable-system-prompt-api", false, "Enable system prompt configuration endpoints (SECURITY: only enable in trusted environments)")
		secure                = flag.Bool("secure", false, "Enable secure gRPC server with TLS")
		certPath              = flag.String("cert-path", "", "Path to TLS certificate directory (containing tls.crt and tls.key)")
		kubeconfig            = flag.String("kubeconfig", "", "Path to kubeconfig file (optional, uses in-cluster config if not specified)")
		namespace             = flag.String("namespace", "default", "Kubernetes namespace to watch for CRDs")
		downloadOnly          = flag.Bool("download-only", false, "Download required models and exit (useful for CI/testing)")
	)
	flag.Parse()

	// Initialize logging (zap) from environment.
	if _, err := logging.InitLoggerFromEnv(); err != nil {
		// Fallback to stderr since logger initialization failed
		fmt.Fprintf(os.Stderr, "failed to initialize logger: %v\n", err)
	}

	// Check if config file exists
	if _, err := os.Stat(*configPath); os.IsNotExist(err) {
		logging.Fatalf("Config file not found: %s", *configPath)
	}

	// Load configuration to initialize tracing
	cfg, err := config.Parse(*configPath)
	if err != nil {
		logging.Fatalf("Failed to load config: %v", err)
	}

	// Set the initial configuration in the global config
	// This is important for Kubernetes mode where the controller will update it
	config.Replace(cfg)

	// Ensure required models are downloaded
	if modelErr := ensureModelsDownloaded(cfg); modelErr != nil {
		logging.Fatalf("Failed to ensure models are downloaded: %v", modelErr)
	}

	// If download-only mode, exit after downloading models
	if *downloadOnly {
		logging.Infof("Download-only mode: models downloaded successfully, exiting")
		os.Exit(0)
	}

	// Initialize distributed tracing if enabled
	ctx := context.Background()
	if cfg.Observability.Tracing.Enabled {
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
		if tracingErr := tracing.InitTracing(ctx, tracingCfg); tracingErr != nil {
			logging.Warnf("Failed to initialize tracing: %v", tracingErr)
		}

		// Set up graceful shutdown for tracing
		defer func() {
			shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if shutdownErr := tracing.ShutdownTracing(shutdownCtx); shutdownErr != nil {
				logging.Errorf("Failed to shutdown tracing: %v", shutdownErr)
			}
		}()
	}

	// Initialize windowed metrics if enabled
	if cfg.Observability.Metrics.WindowedMetrics.Enabled {
		logging.Infof("Initializing windowed metrics for load balancing...")
		if initErr := metrics.InitializeWindowedMetrics(cfg.Observability.Metrics.WindowedMetrics); initErr != nil {
			logging.Warnf("Failed to initialize windowed metrics: %v", initErr)
		} else {
			logging.Infof("Windowed metrics initialized successfully")
		}
	}

	// Shutdown hooks for components initialized later (e.g. vector store pipeline).
	// The signal handler calls these before exiting since os.Exit skips defers.
	var shutdownHooks []func()

	// Set up signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		logging.Infof("Received shutdown signal, cleaning up...")

		// Run shutdown hooks (vector store pipeline, backend, etc.)
		for _, hook := range shutdownHooks {
			hook()
		}

		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if shutdownErr := tracing.ShutdownTracing(shutdownCtx); shutdownErr != nil {
			logging.Errorf("Failed to shutdown tracing: %v", shutdownErr)
		}
		os.Exit(0)
	}()

	// Start metrics server if enabled
	metricsEnabled := true
	if cfg.Observability.Metrics.Enabled != nil {
		metricsEnabled = *cfg.Observability.Metrics.Enabled
	}
	if *metricsPort <= 0 {
		metricsEnabled = false
	}
	if metricsEnabled {
		go func() {
			http.Handle("/metrics", promhttp.Handler())
			metricsAddr := fmt.Sprintf(":%d", *metricsPort)
			logging.Infof("Starting metrics server on %s", metricsAddr)
			if metricsErr := http.ListenAndServe(metricsAddr, nil); metricsErr != nil {
				logging.Errorf("Metrics server error: %v", metricsErr)
			}
		}()
	} else {
		logging.Infof("Metrics server disabled")
	}

	// Initialize embedding models BEFORE creating server, this ensures Qwen3/Gemma/mmBERT/BERT models are ready when semantic cache is initialized
	// Use the already loaded config instead of calling config.Load() again
	var embeddingModelsInitialized bool

	// Resolve model paths using registry (supports aliases like "qwen3", "gemma", "mmbert", "bert")
	qwen3Path := config.ResolveModelPath(cfg.Qwen3ModelPath)
	gemmaPath := config.ResolveModelPath(cfg.GemmaModelPath)
	mmBertPath := config.ResolveModelPath(cfg.EmbeddingModels.MmBertModelPath)
	bertPath := config.ResolveModelPath(cfg.EmbeddingModels.BertModelPath)

	// Check if any unified models (qwen3/gemma/mmbert) are configured
	hasUnifiedModels := qwen3Path != "" || gemmaPath != "" || mmBertPath != ""
	hasBertModel := bertPath != ""

	if hasUnifiedModels || hasBertModel {
		logging.Infof("Initializing embedding models: qwen3=%q, gemma=%q, mmbert=%q, bert=%q, useCPU=%t",
			qwen3Path, gemmaPath, mmBertPath, bertPath, cfg.EmbeddingModels.UseCPU)

		// Initialize unified models (qwen3/gemma/mmbert) if any are configured
		if hasUnifiedModels {
			var initErr error

			// Check if semantic cache uses qwen3 and needs batched initialization
			// The cache uses GetEmbeddingBatched() which requires InitEmbeddingModelsBatched()
			semanticCacheNeedsBatched := cfg.SemanticCache.Enabled &&
				strings.ToLower(strings.TrimSpace(cfg.SemanticCache.EmbeddingModel)) == "qwen3" &&
				qwen3Path != ""

			// ML model selection (KNN, KMeans, SVM) also needs batched embeddings for query embedding
			mlSelectionNeedsBatched := cfg.ModelSelection.Enabled &&
				cfg.ModelSelection.ML.ModelsPath != "" &&
				cfg.Qwen3ModelPath != ""

			useBatchedInit := semanticCacheNeedsBatched || mlSelectionNeedsBatched

			if useBatchedInit {
				if semanticCacheNeedsBatched {
					logging.Infof("Semantic cache uses qwen3, initializing with batched embedding model...")
				}
				if mlSelectionNeedsBatched {
					logging.Infof("ML model selection enabled, initializing with batched embedding model...")
				}
				maxBatchSize := 64      // Batch up to 64 requests together
				maxWaitMs := uint64(10) // Wait max 10ms for batch to fill
				initErr = candle_binding.InitEmbeddingModelsBatched(
					qwen3Path,
					maxBatchSize,
					maxWaitMs,
					cfg.EmbeddingModels.UseCPU,
				)
				if initErr == nil {
					logging.Infof("Batched embedding model initialized successfully")
					// Also initialize standard ModelFactory
					initErr = candle_binding.InitEmbeddingModels(
						qwen3Path,
						gemmaPath,
						mmBertPath,
						cfg.EmbeddingModels.UseCPU,
					)
				}
			} else {
				initErr = candle_binding.InitEmbeddingModels(
					qwen3Path,
					gemmaPath,
					mmBertPath,
					cfg.EmbeddingModels.UseCPU,
				)
			}

			if initErr != nil {
				logging.Errorf("Failed to initialize unified embedding models: %v", initErr)
				logging.Warnf("Embedding API endpoints will return placeholder embeddings")
				logging.Warnf("Tools database will NOT be loaded (requires embedding models)")
			} else {
				logging.Infof("Unified embedding models initialized successfully")
				embeddingModelsInitialized = true
			}
		}

		// Initialize BERT model separately (for memory with 384-dim embeddings)
		// This uses a different initialization path (InitModel vs InitEmbeddingModels)
		if hasBertModel {
			logging.Infof("Initializing BERT model for memory: %s", bertPath)
			if bertErr := candle_binding.InitModel(bertPath, cfg.EmbeddingModels.UseCPU); bertErr != nil {
				logging.Warnf("Failed to initialize BERT model: %v", bertErr)
				logging.Warnf("Memory retrieval with 'bert' embedding type will not work")
			} else {
				logging.Infof("BERT model initialized successfully (384-dim for memory)")
				embeddingModelsInitialized = true
			}
		}
	} else {
		logging.Infof("No embedding models configured, skipping initialization")
		logging.Infof("To enable embedding models, add to config.yaml:")
		logging.Infof("  embedding_models:")
		logging.Infof("    qwen3_model_path: 'models/mom-embedding-pro'")
		logging.Infof("    gemma_model_path: 'models/mom-embedding-flash'")
		logging.Infof("    mmbert_model_path: 'models/mom-embedding-ultra'")
		logging.Infof("    bert_model_path: 'models/all-MiniLM-L12-v2'  # For memory (384-dim)")
		logging.Infof("    use_cpu: true")
		embeddingModelsInitialized = false
	}

	// Initialize BERT model if semantic cache is configured to use it
	// This is required because Redis/Milvus caches call candle_binding.GetEmbedding() for "bert" embeddings
	// which requires the BERT model to be initialized via InitModel()
	if cfg.SemanticCache.Enabled {
		embeddingModel := strings.ToLower(strings.TrimSpace(cfg.SemanticCache.EmbeddingModel))
		// Auto-detect embedding model if not explicitly set (matches router.go logic)
		if embeddingModel == "" {
			if cfg.EmbeddingModels.MmBertModelPath != "" {
				embeddingModel = "mmbert"
			} else if cfg.Qwen3ModelPath != "" {
				embeddingModel = "qwen3"
			} else if cfg.GemmaModelPath != "" {
				embeddingModel = "gemma"
			} else {
				embeddingModel = "bert"
			}
		}

		if embeddingModel == "bert" {
			bertModelID := cfg.BertModel.ModelID
			if bertModelID == "" {
				bertModelID = "sentence-transformers/all-MiniLM-L6-v2" // Default BERT model
			}
			bertModelID = config.ResolveModelPath(bertModelID)

			logging.Infof("Semantic cache uses BERT embeddings, initializing BERT model: %s", bertModelID)
			if initErr := candle_binding.InitModel(bertModelID, cfg.BertModel.UseCPU); initErr != nil {
				logging.Fatalf("Failed to initialize BERT model for semantic cache: %v", initErr)
			}
			logging.Infof("BERT model initialized successfully for semantic cache")
		}
	}

	// Initialize vector store if configured
	if cfg.VectorStore != nil && cfg.VectorStore.Enabled {
		logging.Infof("Initializing vector store feature...")

		if validateErr := cfg.VectorStore.Validate(); validateErr != nil {
			logging.Fatalf("Invalid vector store configuration: %v", validateErr)
		}
		cfg.VectorStore.ApplyDefaults()

		// Ensure BERT is initialized if vector store uses it and semantic cache didn't already
		if cfg.VectorStore.EmbeddingModel == "bert" && !cfg.SemanticCache.Enabled {
			bertModelID := cfg.BertModel.ModelID
			if bertModelID == "" {
				bertModelID = "sentence-transformers/all-MiniLM-L6-v2"
			}
			bertModelID = config.ResolveModelPath(bertModelID)
			logging.Infof("Vector store uses BERT embeddings, initializing BERT model: %s", bertModelID)
			if initErr := candle_binding.InitModel(bertModelID, cfg.BertModel.UseCPU); initErr != nil {
				logging.Fatalf("Failed to initialize BERT model for vector store: %v", initErr)
			}
		}

		// Create file store
		vsFileStore, vsErr := vectorstore.NewFileStore(cfg.VectorStore.FileStorageDir)
		if vsErr != nil {
			logging.Fatalf("Failed to create vector store file store: %v", vsErr)
		}
		apiserver.SetFileStore(vsFileStore)

		// Create backend
		var memoryCfg vectorstore.MemoryBackendConfig
		var milvusCfg vectorstore.MilvusBackendConfig
		switch cfg.VectorStore.BackendType {
		case "memory":
			maxEntries := 100000
			if cfg.VectorStore.Memory != nil && cfg.VectorStore.Memory.MaxEntriesPerStore > 0 {
				maxEntries = cfg.VectorStore.Memory.MaxEntriesPerStore
			}
			memoryCfg = vectorstore.MemoryBackendConfig{MaxEntriesPerStore: maxEntries}
		case "milvus":
			milvusCfg = vectorstore.MilvusBackendConfig{
				Address: fmt.Sprintf("%s:%d", cfg.VectorStore.Milvus.Connection.Host, cfg.VectorStore.Milvus.Connection.Port),
			}
		}
		vsBackend, vsErr := vectorstore.NewBackend(cfg.VectorStore.BackendType, memoryCfg, milvusCfg)
		if vsErr != nil {
			logging.Fatalf("Failed to create vector store backend: %v", vsErr)
		}

		// Create manager
		vsMgr := vectorstore.NewManager(vsBackend, cfg.VectorStore.EmbeddingDimension, cfg.VectorStore.BackendType)
		apiserver.SetVectorStoreManager(vsMgr)

		// Create embedder
		vsEmbedder := vectorstore.NewCandleEmbedder(cfg.VectorStore.EmbeddingModel, cfg.VectorStore.EmbeddingDimension)
		apiserver.SetEmbedder(vsEmbedder)

		// Create and start ingestion pipeline
		vsPipeline := vectorstore.NewIngestionPipeline(vsBackend, vsFileStore, vsMgr, vsEmbedder, vectorstore.PipelineConfig{
			Workers:   cfg.VectorStore.IngestionWorkers,
			QueueSize: 100,
		})
		vsPipeline.Start()
		apiserver.SetIngestionPipeline(vsPipeline)

		// Register shutdown hooks so signal handler can clean up
		// (defer won't run when os.Exit is called from signal handler)
		shutdownHooks = append(shutdownHooks, func() {
			logging.Infof("Shutting down vector store pipeline...")
			vsPipeline.Stop()
			vsBackend.Close()
		})

		logging.Infof("Vector store initialized: backend=%s, model=%s, dim=%d, workers=%d",
			cfg.VectorStore.BackendType, cfg.VectorStore.EmbeddingModel,
			cfg.VectorStore.EmbeddingDimension, cfg.VectorStore.IngestionWorkers)
	}

	// Create and start the ExtProc server
	server, err := extproc.NewServer(*configPath, *port, *secure, *certPath)
	if err != nil {
		logging.Fatalf("Failed to create ExtProc server: %v", err)
	}

	logging.Infof("Starting vLLM Semantic Router ExtProc with config: %s", *configPath)

	// Load tools database after server initialization
	// Note: Tools database requires embedding models to be initialized first
	// If embedding models failed to initialize, tools will not be loaded
	router := server.GetRouter()
	if router != nil {
		// Only load tools if embedding models were successfully initialized
		if embeddingModelsInitialized {
			logging.Infof("Loading tools database (embedding models are ready)...")
			if err := router.LoadToolsDatabase(); err != nil {
				logging.Warnf("Failed to load tools database: %v", err)
			}
		} else {
			logging.Infof("Skipping tools database loading (embedding models not initialized)")
		}
	}

	// Start API server if enabled
	if *enableAPI {
		go func() {
			logging.Infof("Starting API server on port %d", *apiPort)
			if err := apiserver.Init(*configPath, *apiPort, *enableSystemPromptAPI); err != nil {
				logging.Errorf("Start API server error: %v", err)
			}
		}()
	}

	// Start Kubernetes controller if ConfigSource is kubernetes
	if cfg.ConfigSource == config.ConfigSourceKubernetes {
		logging.Infof("ConfigSource is kubernetes, starting Kubernetes controller")
		go startKubernetesController(cfg, *kubeconfig, *namespace)
	} else {
		logging.Infof("ConfigSource is file (or not specified), using file-based configuration")
	}

	if err := server.Start(); err != nil {
		logging.Fatalf("ExtProc server error: %v", err)
	}
}

// ensureModelsDownloaded checks and downloads required models
func ensureModelsDownloaded(cfg *config.RouterConfig) error {
	logging.Infof("Installing required models...")

	// Build model specs from config
	specs, err := modeldownload.BuildModelSpecs(cfg)
	if err != nil {
		return fmt.Errorf("failed to build model specs: %w", err)
	}

	// Skip download if no local models are configured (API-only mode)
	if len(specs) == 0 {
		logging.Infof("No local models configured, skipping model download (API-only mode)")
		return nil
	}

	// Calculate unique models based on RepoID
	uniqueModels := make(map[string]bool)
	for _, repoID := range cfg.MoMRegistry {
		uniqueModels[repoID] = true
	}

	// Print model registry configuration
	logging.Infof("MoM Families: %d unique models (total %d registry aliases)", len(uniqueModels), len(cfg.MoMRegistry))
	logging.Debugf("Registry Details:")
	for localPath, repoID := range cfg.MoMRegistry {
		logging.Debugf("  %s -> %s", localPath, repoID)
	}

	// Check if huggingface-cli is available
	if err := modeldownload.CheckHuggingFaceCLI(); err != nil {
		return fmt.Errorf("huggingface-cli check failed: %w", err)
	}

	// Get download configuration from environment
	downloadConfig := modeldownload.GetDownloadConfig()

	// Log environment configuration (mask sensitive token)
	maskedToken := "***"
	if downloadConfig.HFToken == "" {
		maskedToken = "<not set>"
	}
	logging.Infof("HF_ENDPOINT: %s; HF_TOKEN: %s; HF_HOME: %s", downloadConfig.HFEndpoint, maskedToken, downloadConfig.HFHome)
	// Ensure all models are downloaded
	if err := modeldownload.EnsureModels(specs, downloadConfig); err != nil {
		return fmt.Errorf("failed to download models: %w", err)
	}

	logging.Infof("All required models are ready")
	return nil
}

// startKubernetesController starts the Kubernetes controller for watching CRDs
func startKubernetesController(staticConfig *config.RouterConfig, kubeconfig, namespace string) {
	// Import k8s package here to avoid import errors when k8s dependencies are not available
	// This is a lazy import pattern
	logging.Infof("Initializing Kubernetes controller for namespace: %s", namespace)

	logging.Infof("Starting Kubernetes controller for namespace: %s", namespace)

	controller, err := k8s.NewController(k8s.ControllerConfig{
		Namespace:    namespace,
		Kubeconfig:   kubeconfig,
		StaticConfig: staticConfig,
		OnConfigUpdate: func(newConfig *config.RouterConfig) error {
			config.Replace(newConfig)
			logging.Infof("Configuration updated from Kubernetes CRDs")
			return nil
		},
	})
	if err != nil {
		logging.Fatalf("Failed to create Kubernetes controller: %v", err)
	}

	ctx := context.Background()
	if err := controller.Start(ctx); err != nil {
		logging.Fatalf("Kubernetes controller error: %v", err)
	}
}
