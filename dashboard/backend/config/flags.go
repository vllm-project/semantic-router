package config

import (
	"flag"
	"path/filepath"
	"runtime"
)

type flagValues struct {
	port          *string
	staticDir     *string
	configFile    *string
	grafanaURL    *string
	prometheusURL *string
	routerAPIURL  *string
	routerMetrics *string
	jaegerURL     *string
	envoyURL      *string
	readonlyMode  *bool
	setupMode     *bool
	platform      *string

	evaluationEnabled    *bool
	evaluationDBPath     *string
	evaluationResultsDir *string
	pythonPath           *string

	consoleStoreBackend *string
	consoleDBPath       *string
	consoleStoreDSN     *string

	mcpEnabled *bool

	mlPipelineEnabled *bool
	mlPipelineDataDir *string
	mlTrainingDir     *string
	mlServiceURL      *string

	openClawEnabled *bool
	openClawURL     *string
	openClawDataDir *string
	openClawToken   *string
}

// LoadConfig loads configuration from flags and environment variables.
func LoadConfig() (*Config, error) {
	values := registerFlags()
	flag.Parse()

	cfg := values.buildConfig()
	if err := resolveConfigPaths(cfg); err != nil {
		return nil, err
	}
	return cfg, nil
}

func registerFlags() *flagValues {
	values := &flagValues{}
	registerDashboardFlags(values)
	registerUpstreamFlags(values)
	registerEvaluationFlags(values)
	registerConsoleFlags(values)
	registerFeatureFlags(values)
	return values
}

func registerDashboardFlags(values *flagValues) {
	values.port = flag.String("port", env("DASHBOARD_PORT", "8700"), "dashboard port")
	values.staticDir = flag.String("static", env("DASHBOARD_STATIC_DIR", "../frontend"), "static assets directory")
	values.configFile = flag.String("config", env("ROUTER_CONFIG_PATH", "../../config/config.yaml"), "path to config.yaml")
	values.readonlyMode = flag.Bool("readonly", env("DASHBOARD_READONLY", "false") == "true", "enable read-only mode (disable config editing)")
	values.setupMode = flag.Bool("setup-mode", env("DASHBOARD_SETUP_MODE", "false") == "true", "enable dashboard setup mode")
	values.platform = flag.String("platform", env("DASHBOARD_PLATFORM", ""), "platform branding (e.g., 'amd' for AMD GPU deployments)")
}

func registerUpstreamFlags(values *flagValues) {
	values.grafanaURL = flag.String("grafana", env("TARGET_GRAFANA_URL", ""), "Grafana base URL")
	values.prometheusURL = flag.String("prometheus", env("TARGET_PROMETHEUS_URL", ""), "Prometheus base URL")
	values.routerAPIURL = flag.String("router_api", env("TARGET_ROUTER_API_URL", "http://localhost:8080"), "Router API base URL")
	values.routerMetrics = flag.String("router_metrics", env("TARGET_ROUTER_METRICS_URL", "http://localhost:9190/metrics"), "Router metrics URL")
	values.jaegerURL = flag.String("jaeger", env("TARGET_JAEGER_URL", ""), "Jaeger base URL")
	values.envoyURL = flag.String("envoy", env("TARGET_ENVOY_URL", ""), "Envoy proxy URL for chat completions")
}

func registerEvaluationFlags(values *flagValues) {
	values.evaluationEnabled = flag.Bool("evaluation", env("EVALUATION_ENABLED", "true") == "true", "enable evaluation feature")
	values.evaluationDBPath = flag.String("evaluation-db", env("EVALUATION_DB_PATH", "./data/evaluations.db"), "evaluation database path")
	values.evaluationResultsDir = flag.String("evaluation-results", env("EVALUATION_RESULTS_DIR", "./data/results"), "evaluation results directory")
	values.pythonPath = flag.String("python", env("PYTHON_PATH", defaultPythonBinary()), "path to Python interpreter")
}

func registerConsoleFlags(values *flagValues) {
	values.consoleStoreBackend = flag.String("console-store-backend", env("CONSOLE_STORE_BACKEND", "sqlite"), "dashboard console store backend")
	values.consoleDBPath = flag.String("console-db", env("CONSOLE_DB_PATH", "./data/console.db"), "dashboard console database path")
	values.consoleStoreDSN = flag.String("console-store-dsn", env("CONSOLE_STORE_DSN", ""), "dashboard console store DSN")
}

func registerFeatureFlags(values *flagValues) {
	values.mcpEnabled = flag.Bool("mcp", env("MCP_ENABLED", "true") == "true", "enable MCP (Model Context Protocol) feature")
	values.mlPipelineEnabled = flag.Bool("ml-pipeline", env("ML_PIPELINE_ENABLED", "true") == "true", "enable ML pipeline (benchmark, train, config)")
	values.mlPipelineDataDir = flag.String("ml-pipeline-data", env("ML_PIPELINE_DATA_DIR", "./data/ml-pipeline"), "ML pipeline data directory")
	values.mlTrainingDir = flag.String("ml-training-dir", env("ML_TRAINING_DIR", ""), "path to src/training/model_selection/ml_model_selection")
	values.mlServiceURL = flag.String("ml-service-url", env("ML_SERVICE_URL", ""), "URL of Python ML service sidecar (empty = subprocess mode)")
	values.openClawEnabled = flag.Bool("openclaw", env("OPENCLAW_ENABLED", "true") == "true", "enable OpenClaw agent provisioning")
	values.openClawURL = flag.String("openclaw-url", env("OPENCLAW_URL", "http://localhost:18788"), "OpenClaw gateway URL")
	values.openClawDataDir = flag.String("openclaw-data", env("OPENCLAW_DATA_DIR", "./data/openclaw"), "OpenClaw workspace directory")
	values.openClawToken = flag.String("openclaw-token", env("OPENCLAW_TOKEN", ""), "OpenClaw gateway auth token")
}

func defaultPythonBinary() string {
	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}

func (values *flagValues) buildConfig() *Config {
	return &Config{
		Port:                 *values.port,
		StaticDir:            *values.staticDir,
		ConfigFile:           *values.configFile,
		GrafanaURL:           *values.grafanaURL,
		PrometheusURL:        *values.prometheusURL,
		RouterAPIURL:         *values.routerAPIURL,
		RouterMetrics:        *values.routerMetrics,
		JaegerURL:            *values.jaegerURL,
		EnvoyURL:             *values.envoyURL,
		ReadonlyMode:         *values.readonlyMode,
		SetupMode:            *values.setupMode,
		Platform:             *values.platform,
		EvaluationEnabled:    *values.evaluationEnabled,
		EvaluationDBPath:     *values.evaluationDBPath,
		EvaluationResultsDir: *values.evaluationResultsDir,
		PythonPath:           *values.pythonPath,
		ConsoleStoreBackend:  *values.consoleStoreBackend,
		ConsoleDBPath:        *values.consoleDBPath,
		ConsoleStoreDSN:      *values.consoleStoreDSN,
		MCPEnabled:           *values.mcpEnabled,
		MLPipelineEnabled:    *values.mlPipelineEnabled,
		MLPipelineDataDir:    *values.mlPipelineDataDir,
		MLTrainingDir:        *values.mlTrainingDir,
		MLServiceURL:         *values.mlServiceURL,
		OpenClawEnabled:      *values.openClawEnabled,
		OpenClawURL:          *values.openClawURL,
		OpenClawDataDir:      *values.openClawDataDir,
		OpenClawToken:        *values.openClawToken,
	}
}

func resolveConfigPaths(cfg *Config) error {
	absConfigPath, err := filepath.Abs(cfg.ConfigFile)
	if err != nil {
		return err
	}
	cfg.AbsConfigPath = absConfigPath
	cfg.ConfigDir = filepath.Dir(absConfigPath)
	return nil
}
