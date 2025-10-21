package main

import (
	"log"
	"net/http"

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

	// Setup routes
	mux := router.Setup(cfg)

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
	if cfg.OpenWebUIURL != "" {
		log.Printf("OpenWebUI: %s → /embedded/openwebui/", cfg.OpenWebUIURL)
	}
	if cfg.ChatUIURL != "" {
		log.Printf("HuggingChat: %s → /embedded/chatui/", cfg.ChatUIURL)
	}
	log.Printf("Router API: %s → /api/router/*", cfg.RouterAPIURL)
	log.Printf("Router Metrics: %s → /metrics/router", cfg.RouterMetrics)

	// Start server
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
