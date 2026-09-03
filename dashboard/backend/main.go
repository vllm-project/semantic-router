package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

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
	log.Printf("Router API: %s → /api/router/*", cfg.RouterAPIURL)
	log.Printf("Router Metrics: %s → /metrics/router", cfg.RouterMetrics)
	if cfg.ReadonlyMode {
		log.Printf("Read-only mode: ENABLED (config editing disabled)")
	}

	server := &http.Server{Addr: addr, Handler: srv.Handler, ReadHeaderTimeout: 15 * time.Second}
	shutdownContext, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	shutdown := func(ctx context.Context) error {
		if err := server.Shutdown(ctx); err != nil {
			return errors.Join(err, server.Close())
		}
		return nil
	}
	lifecycleErr := runServerLifecycle(shutdownContext, server.ListenAndServe, shutdown, srv.Close, 15*time.Second)
	stop()
	if lifecycleErr != nil {
		log.Fatalf("server error: %v", lifecycleErr)
	}
}

func runServerLifecycle(
	ctx context.Context,
	serve func() error,
	shutdown func(context.Context) error,
	closeResources func() error,
	timeout time.Duration,
) error {
	serveErrors := make(chan error, 1)
	go func() { serveErrors <- serve() }()
	select {
	case serveErr := <-serveErrors:
		if errors.Is(serveErr, http.ErrServerClosed) {
			serveErr = nil
		}
		return errors.Join(serveErr, closeResources())
	case <-ctx.Done():
		shutdownContext, cancel := context.WithTimeout(context.Background(), timeout)
		shutdownErr := shutdown(shutdownContext)
		cancel()
		serveErr := <-serveErrors
		if errors.Is(serveErr, http.ErrServerClosed) {
			serveErr = nil
		}
		return errors.Join(shutdownErr, serveErr, closeResources())
	}
}
