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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

func main() {
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
	)
	flag.Parse()

	// Initialize logging (zap) from environment.
	if _, err := observability.InitLoggerFromEnv(); err != nil {
		// Fallback to stderr since logger initialization failed
		fmt.Fprintf(os.Stderr, "failed to initialize logger: %v\n", err)
	}

	// Check if config file exists
	if _, err := os.Stat(*configPath); os.IsNotExist(err) {
		observability.Fatalf("Config file not found: %s", *configPath)
	}

	// Load configuration to initialize tracing
	cfg, err := config.ParseConfigFile(*configPath)
	if err != nil {
		observability.Fatalf("Failed to load config: %v", err)
	}

	// Initialize distributed tracing if enabled
	ctx := context.Background()
	if cfg.Observability.Tracing.Enabled {
		tracingCfg := observability.TracingConfig{
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
		if tracingErr := observability.InitTracing(ctx, tracingCfg); tracingErr != nil {
			observability.Warnf("Failed to initialize tracing: %v", tracingErr)
		}

		// Set up graceful shutdown for tracing
		defer func() {
			shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if shutdownErr := observability.ShutdownTracing(shutdownCtx); shutdownErr != nil {
				observability.Errorf("Failed to shutdown tracing: %v", shutdownErr)
			}
		}()
	}

	// Set up signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		observability.Infof("Received shutdown signal, cleaning up...")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if shutdownErr := observability.ShutdownTracing(shutdownCtx); shutdownErr != nil {
			observability.Errorf("Failed to shutdown tracing: %v", shutdownErr)
		}
		os.Exit(0)
	}()

	// Start metrics server
	go func() {
		http.Handle("/metrics", promhttp.Handler())
		metricsAddr := fmt.Sprintf(":%d", *metricsPort)
		observability.Infof("Starting metrics server on %s", metricsAddr)
		if metricsErr := http.ListenAndServe(metricsAddr, nil); metricsErr != nil {
			observability.Errorf("Metrics server error: %v", metricsErr)
		}
	}()

	// Create and start the ExtProc server
	server, err := extproc.NewServer(*configPath, *port, *secure, *certPath)
	if err != nil {
		observability.Fatalf("Failed to create ExtProc server: %v", err)
	}

	observability.Infof("Starting vLLM Semantic Router ExtProc with config: %s", *configPath)

	// Initialize embedding models if configured (Long-context support)
	cfg, err := config.LoadConfig(*configPath)
	if err != nil {
		observability.Warnf("Failed to load config for embedding models: %v", err)
	} else if cfg.EmbeddingModels.Qwen3ModelPath != "" || cfg.EmbeddingModels.GemmaModelPath != "" {
		observability.Infof("Initializing embedding models...")
		observability.Infof("  Qwen3 model: %s", cfg.EmbeddingModels.Qwen3ModelPath)
		observability.Infof("  Gemma model: %s", cfg.EmbeddingModels.GemmaModelPath)
		observability.Infof("  Use CPU: %v", cfg.EmbeddingModels.UseCPU)

		if err := candle_binding.InitEmbeddingModels(
			cfg.EmbeddingModels.Qwen3ModelPath,
			cfg.EmbeddingModels.GemmaModelPath,
			cfg.EmbeddingModels.UseCPU,
		); err != nil {
			observability.Errorf("Failed to initialize embedding models: %v", err)
			observability.Warnf("Embedding API endpoints will return placeholder embeddings")
		} else {
			observability.Infof("Embedding models initialized successfully")
		}
	} else {
		observability.Infof("No embedding models configured, skipping initialization")
		observability.Infof("To enable embedding models, add to config.yaml:")
		observability.Infof("  embedding_models:")
		observability.Infof("    qwen3_model_path: 'models/Qwen3-Embedding-0.6B'")
		observability.Infof("    gemma_model_path: 'models/embeddinggemma-300m'")
		observability.Infof("    use_cpu: true")
	}

	// Start API server if enabled
	if *enableAPI {
		go func() {
			observability.Infof("Starting Classification API server on port %d", *apiPort)
			if err := api.StartClassificationAPI(*configPath, *apiPort, *enableSystemPromptAPI); err != nil {
				observability.Errorf("Classification API server error: %v", err)
			}
		}()
	}

	if err := server.Start(); err != nil {
		observability.Fatalf("ExtProc server error: %v", err)
	}
}
