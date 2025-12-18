package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
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

	// Set up signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		logging.Infof("Received shutdown signal, cleaning up...")
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

	// Create and start the ExtProc server
	server, err := extproc.NewServer(*configPath, *port, *secure, *certPath)
	if err != nil {
		logging.Fatalf("Failed to create ExtProc server: %v", err)
	}

	logging.Infof("Starting vLLM Semantic Router ExtProc with config: %s", *configPath)

	// Initialize embedding models if configured (Long-context support)
	// Use the already loaded config instead of calling config.Load() again
	if cfg.Qwen3ModelPath != "" || cfg.GemmaModelPath != "" {
		if err := candle_binding.InitEmbeddingModels(
			cfg.Qwen3ModelPath,
			cfg.GemmaModelPath,
			cfg.EmbeddingModels.UseCPU,
		); err != nil {
			logging.Errorf("Failed to initialize embedding models: %v", err)
			logging.Warnf("Embedding API endpoints will return placeholder embeddings")
		} else {
			logging.Infof("Embedding models initialized successfully")

			// Load tools database after embedding models are initialized
			// This ensures ModelFactory is ready when generating tool embeddings
			router := server.GetRouter()
			if router != nil {
				if err := router.LoadToolsDatabase(); err != nil {
					logging.Warnf("Failed to load tools database: %v", err)
				}
			}
		}
	} else {
		logging.Infof("No embedding models configured, skipping initialization")
		logging.Infof("To enable embedding models, add to config.yaml:")
		logging.Infof("  embedding_models:")
		logging.Infof("    qwen3_model_path: 'models/Qwen3-Embedding-0.6B'")
		logging.Infof("    gemma_model_path: 'models/embeddinggemma-300m'")
		logging.Infof("    use_cpu: true")
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

	// Print model registry configuration
	logging.Infof("MoM Families (%d models)", len(cfg.MoMRegistry))
	for localPath, repoID := range cfg.MoMRegistry {
		logging.Debugf("  %s -> %s", localPath, repoID)
	}

	// Check if huggingface-cli is available
	if err := modeldownload.CheckHuggingFaceCLI(); err != nil {
		return fmt.Errorf("huggingface-cli check failed: %w", err)
	}

	// Build model specs from config
	specs, err := modeldownload.BuildModelSpecs(cfg)
	if err != nil {
		return fmt.Errorf("failed to build model specs: %w", err)
	}

	// Get download configuration from environment
	downloadConfig := modeldownload.GetDownloadConfig()

	// Log environment configuration
	logging.Infof("HF_ENDPOINT: %s; HF_TOKEN: %s; HF_HOME: %s", downloadConfig.HFEndpoint, downloadConfig.HFToken, downloadConfig.HFHome)
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
