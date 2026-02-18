package config

import (
	"flag"
	"os"
	"path/filepath"
	"runtime"
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

	// Read-only mode for public beta deployments
	ReadonlyMode bool

	// Platform branding (e.g., "amd" for AMD GPU deployments)
	Platform string

	// Evaluation configuration
	EvaluationEnabled    bool
	EvaluationDBPath     string
	EvaluationResultsDir string
	PythonPath           string

	// MCP configuration
	MCPEnabled bool

	// ML Pipeline configuration
	MLPipelineEnabled bool
	MLPipelineDataDir string
	MLTrainingDir     string // path to src/training/ml_model_selection
	MLServiceURL      string // URL of the Python ML service sidecar (empty = subprocess mode)
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

	// Read-only mode for public beta deployments
	readonlyMode := flag.Bool("readonly", env("DASHBOARD_READONLY", "false") == "true", "enable read-only mode (disable config editing)")

	// Platform branding
	platform := flag.String("platform", env("DASHBOARD_PLATFORM", ""), "platform branding (e.g., 'amd' for AMD GPU deployments)")

	// Evaluation configuration
	evaluationEnabled := flag.Bool("evaluation", env("EVALUATION_ENABLED", "true") == "true", "enable evaluation feature")
	evaluationDBPath := flag.String("evaluation-db", env("EVALUATION_DB_PATH", "./data/evaluations.db"), "evaluation database path")
	evaluationResultsDir := flag.String("evaluation-results", env("EVALUATION_RESULTS_DIR", "./data/results"), "evaluation results directory")
	defaultPython := "python3"
	if runtime.GOOS == "windows" {
		defaultPython = "python"
	}
	pythonPath := flag.String("python", env("PYTHON_PATH", defaultPython), "path to Python interpreter")

	// MCP configuration
	mcpEnabled := flag.Bool("mcp", env("MCP_ENABLED", "true") == "true", "enable MCP (Model Context Protocol) feature")

	// ML Onboarding configuration
	mlPipelineEnabled := flag.Bool("ml-pipeline", env("ML_PIPELINE_ENABLED", "true") == "true", "enable ML pipeline (benchmark, train, config)")
	mlPipelineDataDir := flag.String("ml-pipeline-data", env("ML_PIPELINE_DATA_DIR", "./data/ml-pipeline"), "ML pipeline data directory")
	mlTrainingDir := flag.String("ml-training-dir", env("ML_TRAINING_DIR", ""), "path to src/training/ml_model_selection")
	mlServiceURL := flag.String("ml-service-url", env("ML_SERVICE_URL", ""), "URL of Python ML service sidecar (empty = subprocess mode)")

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
	cfg.ReadonlyMode = *readonlyMode
	cfg.Platform = *platform
	cfg.EvaluationEnabled = *evaluationEnabled
	cfg.EvaluationDBPath = *evaluationDBPath
	cfg.EvaluationResultsDir = *evaluationResultsDir
	cfg.PythonPath = *pythonPath
	cfg.MCPEnabled = *mcpEnabled
	cfg.MLPipelineEnabled = *mlPipelineEnabled
	cfg.MLPipelineDataDir = *mlPipelineDataDir
	cfg.MLTrainingDir = *mlTrainingDir
	cfg.MLServiceURL = *mlServiceURL

	// Resolve config file path to absolute path
	absConfigPath, err := filepath.Abs(cfg.ConfigFile)
	if err != nil {
		return nil, err
	}
	cfg.AbsConfigPath = absConfigPath
	cfg.ConfigDir = filepath.Dir(absConfigPath)

	return cfg, nil
}
