package config

import (
	"flag"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
)

// Config holds all application configuration
type Config struct {
	Port                   string
	AuthDBPath             string
	JWTSecret              string
	JWTExpiryHours         int
	BootstrapAdminEmail    string
	BootstrapAdminPassword string
	BootstrapAdminName     string
	StaticDir              string
	ConfigFile             string
	AbsConfigPath          string
	ConfigDir              string

	// Upstream targets
	GrafanaURL    string
	PrometheusURL string
	RouterAPIURL  string
	RouterMetrics string
	JaegerURL     string
	EnvoyURL      string // Envoy proxy for chat completions
	FleetSimURL   string // Fleet simulator base URL

	// Read-only mode for public beta deployments
	ReadonlyMode bool
	SetupMode    bool

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
	MLTrainingDir     string // path to src/training/model_selection/ml_model_selection
	MLServiceURL      string // URL of the Python ML service sidecar (empty = subprocess mode)

	// OpenClaw configuration
	OpenClawEnabled bool
	OpenClawURL     string // URL of OpenClaw gateway (default: http://localhost:18788)
	OpenClawDataDir string // workspace generation directory
	OpenClawToken   string // auth token for OpenClaw gateway

	// Durable workflow state (ML pipeline jobs, OpenClaw entities)
	WorkflowDBPath string
}

// env returns the env var or default
func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

type authFlags struct {
	dbPath            *string
	jwtSecret         *string
	jwtTTL            *string
	bootstrapEmail    *string
	bootstrapPassword *string
	bootstrapName     *string
}

func bindAuthFlags() authFlags {
	return authFlags{
		dbPath:            flag.String("auth-db", env("DASHBOARD_AUTH_DB_PATH", "./data/auth.db"), "auth database path"),
		jwtSecret:         flag.String("auth-jwt-secret", env("DASHBOARD_JWT_SECRET", ""), "JWT signing secret"),
		jwtTTL:            flag.String("auth-jwt-expiry-hours", env("DASHBOARD_JWT_EXPIRY_HOURS", "12"), "JWT expiry in hours"),
		bootstrapEmail:    flag.String("bootstrap-admin-email", env("DASHBOARD_ADMIN_EMAIL", ""), "bootstrap admin email"),
		bootstrapPassword: flag.String("bootstrap-admin-password", env("DASHBOARD_ADMIN_PASSWORD", ""), "bootstrap admin password"),
		bootstrapName:     flag.String("bootstrap-admin-name", env("DASHBOARD_ADMIN_NAME", ""), "bootstrap admin name"),
	}
}

type openClawFlags struct {
	enabled *bool
	url     *string
	dataDir *string
	token   *string
}

func bindOpenClawFlags() openClawFlags {
	return openClawFlags{
		enabled: flag.Bool("openclaw", env("OPENCLAW_ENABLED", "true") == "true", "enable OpenClaw agent provisioning"),
		url:     flag.String("openclaw-url", env("OPENCLAW_URL", "http://localhost:18788"), "OpenClaw gateway URL"),
		dataDir: flag.String("openclaw-data", env("OPENCLAW_DATA_DIR", "./data/openclaw"), "OpenClaw workspace directory"),
		token:   flag.String("openclaw-token", env("OPENCLAW_TOKEN", ""), "OpenClaw gateway auth token"),
	}
}

func defaultPythonBinary() string {
	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}

type parsedFlags struct {
	port                 *string
	staticDir            *string
	configFile           *string
	grafanaURL           *string
	promURL              *string
	routerAPI            *string
	routerMetrics        *string
	jaegerURL            *string
	envoyURL             *string
	fleetSimURL          *string
	readonlyMode         *bool
	setupMode            *bool
	platform             *string
	evaluationEnabled    *bool
	evaluationDBPath     *string
	evaluationResultsDir *string
	pythonPath           *string
	mcpEnabled           *bool
	mlPipelineEnabled    *bool
	mlPipelineDataDir    *string
	mlTrainingDir        *string
	mlServiceURL         *string
	workflowDBPath       *string
	auth                 authFlags
	openClaw             openClawFlags
}

func applyCoreConfig(cfg *Config, flags parsedFlags) {
	cfg.Port = *flags.port
	cfg.StaticDir = *flags.staticDir
	cfg.ConfigFile = *flags.configFile
	cfg.GrafanaURL = *flags.grafanaURL
	cfg.PrometheusURL = *flags.promURL
	cfg.RouterAPIURL = *flags.routerAPI
	cfg.RouterMetrics = *flags.routerMetrics
	cfg.JaegerURL = *flags.jaegerURL
	cfg.EnvoyURL = *flags.envoyURL
	cfg.FleetSimURL = *flags.fleetSimURL
	cfg.ReadonlyMode = *flags.readonlyMode
	cfg.SetupMode = *flags.setupMode
	cfg.Platform = *flags.platform
}

func applyFeatureConfig(cfg *Config, flags parsedFlags) {
	cfg.EvaluationEnabled = *flags.evaluationEnabled
	cfg.EvaluationDBPath = *flags.evaluationDBPath
	cfg.EvaluationResultsDir = *flags.evaluationResultsDir
	cfg.PythonPath = *flags.pythonPath
	cfg.MCPEnabled = *flags.mcpEnabled
	cfg.MLPipelineEnabled = *flags.mlPipelineEnabled
	cfg.MLPipelineDataDir = *flags.mlPipelineDataDir
	cfg.MLTrainingDir = *flags.mlTrainingDir
	cfg.MLServiceURL = *flags.mlServiceURL
	cfg.WorkflowDBPath = *flags.workflowDBPath
}

func applyAuthConfig(cfg *Config, flags authFlags) error {
	cfg.AuthDBPath = *flags.dbPath
	cfg.JWTSecret = *flags.jwtSecret
	cfg.BootstrapAdminEmail = *flags.bootstrapEmail
	cfg.BootstrapAdminPassword = *flags.bootstrapPassword
	cfg.BootstrapAdminName = *flags.bootstrapName

	ttl, err := strconv.Atoi(*flags.jwtTTL)
	if err != nil {
		return err
	}
	cfg.JWTExpiryHours = ttl
	return nil
}

func applyOpenClawConfig(cfg *Config, flags openClawFlags) {
	cfg.OpenClawEnabled = *flags.enabled
	cfg.OpenClawURL = *flags.url
	cfg.OpenClawDataDir = *flags.dataDir
	cfg.OpenClawToken = *flags.token
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
	fleetSimURL := flag.String("fleet-sim", env("TARGET_FLEET_SIM_URL", ""), "Fleet simulator base URL")

	// Read-only mode for public beta deployments
	readonlyMode := flag.Bool("readonly", env("DASHBOARD_READONLY", "false") == "true", "enable read-only mode (disable config editing)")
	setupMode := flag.Bool("setup-mode", env("DASHBOARD_SETUP_MODE", "false") == "true", "enable dashboard setup mode")

	// Platform branding
	platform := flag.String("platform", env("DASHBOARD_PLATFORM", ""), "platform branding (e.g., 'amd' for AMD GPU deployments)")

	// Evaluation configuration
	evaluationEnabled := flag.Bool("evaluation", env("EVALUATION_ENABLED", "true") == "true", "enable evaluation feature")
	evaluationDBPath := flag.String("evaluation-db", env("EVALUATION_DB_PATH", "./data/evaluations.db"), "evaluation database path")
	evaluationResultsDir := flag.String("evaluation-results", env("EVALUATION_RESULTS_DIR", "./data/results"), "evaluation results directory")
	pythonPath := flag.String("python", env("PYTHON_PATH", defaultPythonBinary()), "path to Python interpreter")

	// MCP configuration
	mcpEnabled := flag.Bool("mcp", env("MCP_ENABLED", "true") == "true", "enable MCP (Model Context Protocol) feature")

	// ML Onboarding configuration
	mlPipelineEnabled := flag.Bool("ml-pipeline", env("ML_PIPELINE_ENABLED", "true") == "true", "enable ML pipeline (benchmark, train, config)")
	mlPipelineDataDir := flag.String("ml-pipeline-data", env("ML_PIPELINE_DATA_DIR", "./data/ml-pipeline"), "ML pipeline data directory")
	mlTrainingDir := flag.String("ml-training-dir", env("ML_TRAINING_DIR", ""), "path to src/training/model_selection/ml_model_selection")
	mlServiceURL := flag.String("ml-service-url", env("ML_SERVICE_URL", ""), "URL of Python ML service sidecar (empty = subprocess mode)")
	workflowDBPath := flag.String("workflow-db", env("DASHBOARD_WORKFLOW_DB_PATH", "./data/workflow.sqlite"), "SQLite path for durable dashboard workflow state")

	// Authentication configuration
	auth := bindAuthFlags()

	// OpenClaw configuration
	openClaw := bindOpenClawFlags()

	flags := parsedFlags{
		port:                 port,
		staticDir:            staticDir,
		configFile:           configFile,
		grafanaURL:           grafanaURL,
		promURL:              promURL,
		routerAPI:            routerAPI,
		routerMetrics:        routerMetrics,
		jaegerURL:            jaegerURL,
		envoyURL:             envoyURL,
		fleetSimURL:          fleetSimURL,
		readonlyMode:         readonlyMode,
		setupMode:            setupMode,
		platform:             platform,
		evaluationEnabled:    evaluationEnabled,
		evaluationDBPath:     evaluationDBPath,
		evaluationResultsDir: evaluationResultsDir,
		pythonPath:           pythonPath,
		mcpEnabled:           mcpEnabled,
		mlPipelineEnabled:    mlPipelineEnabled,
		mlPipelineDataDir:    mlPipelineDataDir,
		mlTrainingDir:        mlTrainingDir,
		mlServiceURL:         mlServiceURL,
		workflowDBPath:       workflowDBPath,
		auth:                 auth,
		openClaw:             openClaw,
	}

	flag.Parse()

	applyCoreConfig(cfg, flags)
	applyFeatureConfig(cfg, flags)
	if err := applyAuthConfig(cfg, flags.auth); err != nil {
		return nil, err
	}
	applyOpenClawConfig(cfg, flags.openClaw)
	if err := resolveConfigPaths(cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}
