package main

import (
	"log"
	"net/http"

	backendapp "github.com/vllm-project/semantic-router/dashboard/backend/app"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/router"
)

func main() {
	// Load configuration
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	log.Printf("Config file path: %s", cfg.AbsConfigPath)

	application, err := backendapp.New(cfg)
	if err != nil {
		log.Fatalf("Failed to initialize backend application: %v", err)
	}

	// Setup routes
	mux := router.Setup(application)

	// Log configuration
	addr := ":" + cfg.Port
	log.Printf("Semantic Router Dashboard listening on %s", addr)
	log.Printf("Static dir: %s", cfg.StaticDir)
	if cfg.GrafanaURL != "" {
		log.Printf("Grafana: %s → /embedded/grafana/", cfg.GrafanaURL)
	}
	if cfg.PrometheusURL != "" {
		log.Printf("Prometheus: %s → /embedded/prometheus/", cfg.PrometheusURL)
	}
	if cfg.JaegerURL != "" {
		log.Printf("Jaeger: %s → /embedded/jaeger/", cfg.JaegerURL)
	}
	if cfg.EnvoyURL != "" {
		log.Printf("Envoy: %s → /api/router/v1/chat/completions", cfg.EnvoyURL)
	}
	log.Printf("Router API: %s → /api/router/*", cfg.RouterAPIURL)
	log.Printf("Router Metrics: %s → /metrics/router", cfg.RouterMetrics)
	log.Printf("Console Store: %s (%s)", cfg.ConsoleStoreBackend, cfg.ConsoleDBPath)
	if cfg.ReadonlyMode {
		log.Printf("Read-only mode: ENABLED (config editing disabled)")
	}

	// Start server
	serverErr := http.ListenAndServe(addr, mux)
	if closeErr := application.Close(); closeErr != nil {
		log.Printf("Warning: failed to close backend application: %v", closeErr)
	}
	if serverErr != nil {
		log.Fatalf("server error: %v", serverErr)
	}
}
