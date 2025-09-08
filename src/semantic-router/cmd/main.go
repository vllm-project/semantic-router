package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/api"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/semantic-router/pkg/observability"
)

func main() {
	// Parse command-line flags
	var (
		configPath  = flag.String("config", "config/config.yaml", "Path to the configuration file")
		port        = flag.Int("port", 50051, "Port to listen on for gRPC ExtProc")
		apiPort     = flag.Int("api-port", 8080, "Port to listen on for Classification API")
		metricsPort = flag.Int("metrics-port", 9190, "Port for Prometheus metrics")
		enableAPI   = flag.Bool("enable-api", true, "Enable Classification API server")
	)
	flag.Parse()

	// Initialize logging (zap) from environment.
	if _, err := observability.InitLoggerFromEnv(); err != nil {
		log.Printf("failed to initialize logger, falling back to stdlib: %v", err)
	}

	// Check if config file exists
	if _, err := os.Stat(*configPath); os.IsNotExist(err) {
		log.Fatalf("Config file not found: %s", *configPath)
	}

	// Start metrics server
	go func() {
		http.Handle("/metrics", promhttp.Handler())
		metricsAddr := fmt.Sprintf(":%d", *metricsPort)
		log.Printf("Starting metrics server on %s", metricsAddr)
		if err := http.ListenAndServe(metricsAddr, nil); err != nil {
			log.Printf("Metrics server error: %v", err)
		}
	}()

	// Create and start the ExtProc server
	server, err := extproc.NewServer(*configPath, *port)
	if err != nil {
		log.Fatalf("Failed to create ExtProc server: %v", err)
	}

	log.Printf("Starting vLLM Semantic Router ExtProc with config: %s", *configPath)

	// Start API server if enabled
	if *enableAPI {
		go func() {
			log.Printf("Starting Classification API server on port %d", *apiPort)
			if err := api.StartClassificationAPI(*configPath, *apiPort); err != nil {
				log.Printf("Classification API server error: %v", err)
			}
		}()
	}

	if err := server.Start(); err != nil {
		log.Fatalf("ExtProc server error: %v", err)
	}
}
