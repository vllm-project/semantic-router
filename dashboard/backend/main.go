package main

import (
	"log"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/router"
)

const (
	dashboardReadHeaderTimeout = 10 * time.Second
	dashboardReadTimeout       = 2 * time.Minute
	dashboardIdleTimeout       = 2 * time.Minute
	dashboardMaxHeaderBytes    = 64 * 1024
)

func main() {
	// Load configuration
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	log.Printf("Config file path: %s", cfg.AbsConfigPath)

	// Setup routes
	srv, err := router.Setup(cfg)
	if err != nil {
		log.Fatalf("Failed to setup dashboard: %v", err)
	}

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
	if cfg.FleetSimURL != "" {
		log.Printf("Fleet Sim: %s → /api/fleet-sim/*", cfg.FleetSimURL)
	}
	log.Printf("Router API: %s → /api/router/*", cfg.RouterAPIURL)
	log.Printf("Router Metrics: %s → /metrics/router", cfg.RouterMetrics)
	if cfg.ReadonlyMode {
		log.Printf("Read-only mode: ENABLED (config editing disabled)")
	}

	// Start server
	httpServer := newDashboardHTTPServer(addr, srv.Handler)
	serveErr := httpServer.ListenAndServe()
	if closeErr := srv.Close(); closeErr != nil {
		log.Printf("Warning: dashboard store shutdown: %v", closeErr)
	}
	if serveErr != nil {
		log.Fatalf("server error: %v", serveErr)
	}
}

func newDashboardHTTPServer(addr string, handler http.Handler) *http.Server {
	return &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: dashboardReadHeaderTimeout,
		ReadTimeout:       dashboardReadTimeout,
		IdleTimeout:       dashboardIdleTimeout,
		MaxHeaderBytes:    dashboardMaxHeaderBytes,
		// WriteTimeout remains zero because the dashboard serves intentional
		// WebSocket/SSE streams. Those transports enforce their own deadlines.
	}
}
