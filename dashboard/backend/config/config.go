package config

import (
	"flag"
	"os"
	"path/filepath"
)

// Config holds all application configuration
type Config struct {
	Port          string
	StaticDir     string
	ConfigFile    string
	AbsConfigPath string
	ConfigDir     string

	// Upstream targets
	GrafanaURL    string
	PrometheusURL string
	RouterAPIURL  string
	RouterMetrics string
	JaegerURL     string
	EnvoyURL      string // Envoy proxy for chat completions
}

// env returns the env var or default
func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// LoadConfig loads configuration from flags and environment variables
func LoadConfig() (*Config, error) {
	cfg := &Config{}

	// Flags/env for configuration
	port := flag.String("port", env("DASHBOARD_PORT", "8700"), "dashboard port")
	staticDir := flag.String("static", env("DASHBOARD_STATIC_DIR", "../frontend"), "static assets directory")
	configFile := flag.String("config", env("ROUTER_CONFIG_PATH", "../../config/config.yaml"), "path to config.yaml")

	// Upstream targets
	grafanaURL := flag.String("grafana", env("TARGET_GRAFANA_URL", ""), "Grafana base URL")
	promURL := flag.String("prometheus", env("TARGET_PROMETHEUS_URL", ""), "Prometheus base URL")
	routerAPI := flag.String("router_api", env("TARGET_ROUTER_API_URL", "http://localhost:8080"), "Router API base URL")
	routerMetrics := flag.String("router_metrics", env("TARGET_ROUTER_METRICS_URL", "http://localhost:9190/metrics"), "Router metrics URL")
	jaegerURL := flag.String("jaeger", env("TARGET_JAEGER_URL", ""), "Jaeger base URL")
	envoyURL := flag.String("envoy", env("TARGET_ENVOY_URL", ""), "Envoy proxy URL for chat completions")

	flag.Parse()

	cfg.Port = *port
	cfg.StaticDir = *staticDir
	cfg.ConfigFile = *configFile
	cfg.GrafanaURL = *grafanaURL
	cfg.PrometheusURL = *promURL
	cfg.RouterAPIURL = *routerAPI
	cfg.RouterMetrics = *routerMetrics
	cfg.JaegerURL = *jaegerURL
	cfg.EnvoyURL = *envoyURL

	// Resolve config file path to absolute path
	absConfigPath, err := filepath.Abs(cfg.ConfigFile)
	if err != nil {
		return nil, err
	}
	cfg.AbsConfigPath = absConfigPath
	cfg.ConfigDir = filepath.Dir(absConfigPath)

	return cfg, nil
}
