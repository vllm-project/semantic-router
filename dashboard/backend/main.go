package main

import (
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/router"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

func main() {
	// Load configuration
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	log.Printf("Config file path: %s", cfg.AbsConfigPath)

	if cfg.SetupMode {
		log.Printf("DEPRECATED: --setup-mode / DASHBOARD_SETUP_MODE no longer decides setup mode; " +
			"it is read only so a disagreement with the config file's setup.mode block can be detected and reported.")
	}

	// One setup-mode source for the whole process, built once and passed down.
	setupResolver := setupmode.New(cfg.AbsConfigPath, cfg.SetupMode)
	resolution := setupResolver.Resolve()
	if resolution.Active {
		log.Printf("Setup mode: ACTIVE (source: %s), first-run bootstrap is open", resolution.Source)
	} else if resolution.LegacyFlag {
		log.Printf("Setup mode: inactive")
	}
	// A conflict warning is logged by the resolver itself, on this first call.
	// Repeating it here would duplicate the line.

	// Setup routes
	srv := router.Setup(cfg, setupResolver)

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
	serveErr := http.ListenAndServe(addr, srv.Handler)
	if closeErr := srv.Close(); closeErr != nil {
		log.Printf("Warning: dashboard store shutdown: %v", closeErr)
	}
	if serveErr != nil {
		log.Fatalf("server error: %v", serveErr)
	}
}
