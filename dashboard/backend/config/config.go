package config

import (
	"flag"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
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
	// ConfigBaseDir is the shared resource root, independent of mutable state.
	ConfigBaseDir string

	// Upstream targets
	GrafanaURL    string
	PrometheusURL string
	RouterAPIURL  string
	RouterMetrics string
	JaegerURL     string
	EnvoyURL      string // Envoy proxy for chat completions

	// ReadonlyMode is the explicit, process-wide hard deny. The writable flags
	// describe the two independent persisted surfaces discovered by the
	// container entrypoint: runtime config and the managed Recipe package store.
	ReadonlyMode          bool
	RuntimeConfigWritable bool
	RecipeStoreWritable   bool

	// SetupMode is the legacy --setup-mode / DASHBOARD_SETUP_MODE input. It no
	// longer decides anything; setup mode resolves from the router config's
	// setup.mode block (see dashboard/backend/setupmode). Kept so a stale value
	// can be reported. The vllm-sr CLI still sets it; remove this field once it
	// does not.
	SetupMode bool

	// AllowOpenBootstrap enables first-admin creation via the public, unauthenticated
	// web-form bootstrap endpoint. Off by default; production should provision the
	// admin via DASHBOARD_ADMIN_* instead of exposing an open registration path.
	// SetupMode is a separate trusted bootstrap path for dashboard-first local install.
	AllowOpenBootstrap bool

	// Browser origins permitted to make state-changing requests, "scheme://host[:port]".
	// Empty means our own origin only, which rejects the Vite dev proxy.
	AllowedOrigins []string

	// Platform branding (e.g., "amd" for AMD GPU deployments)
	Platform string

	// sr-bench is a separate durable service shared by CLI and Dashboard.
	SRBenchURL               string
	SRBenchTokenEnv          string
	SRBenchAvailable         bool
	SRBenchUnavailableReason string
	PythonPath               string

	// MCP configuration
	MCPEnabled bool

	// ML Pipeline configuration
	MLPipelineEnabled           bool
	MLPipelineDataDir           string
	MLPipelineAvailable         bool
	MLPipelineUnavailableReason string
	MLTrainingDir               string // path to src/training/model_selection/ml_model_selection
	MLServiceURL                string // URL of the Python ML service sidecar (empty = subprocess mode)

	// Durable workflow state (ML pipeline jobs, MCP servers)
	WorkflowDBPath string
	// Durable hourly availability history for the public status page.
	StatusDBPath string

	// Durable deployed-config projection read model
	ConfigProjectionDBPath string

	// IgnoredOpenClawSettings names the removed OpenClaw flags and variables
	// this process was started with, so startup can warn about them.
	IgnoredOpenClawSettings []string
}

// removedOpenClawSettings are the flags of the removed OpenClaw integration,
// with the variables they defaulted from. The flags still parse, for one
// release, so a manifest that passes them keeps starting; their values are
// ignored.
var removedOpenClawSettings = []struct {
	flag   string
	env    string
	isBool bool
}{
	{flag: "openclaw", env: "OPENCLAW_ENABLED", isBool: true},
	{flag: "openclaw-url", env: "OPENCLAW_URL"},
	{flag: "openclaw-data", env: "OPENCLAW_DATA_DIR"},
	{flag: "openclaw-token", env: "OPENCLAW_TOKEN"},
}

func bindRemovedOpenClawFlags() {
	const usage = "DEPRECATED and ignored: OpenClaw was removed"
	for _, setting := range removedOpenClawSettings {
		if setting.isBool {
			flag.Bool(setting.flag, false, usage)
		} else {
			flag.String(setting.flag, "", usage)
		}
	}
}

func ignoredOpenClawSettings() []string {
	passed := map[string]bool{}
	flag.Visit(func(f *flag.Flag) { passed[f.Name] = true })
	var ignored []string
	for _, setting := range removedOpenClawSettings {
		if passed[setting.flag] {
			ignored = append(ignored, "-"+setting.flag)
		}
		if os.Getenv(setting.env) != "" {
			ignored = append(ignored, setting.env)
		}
	}
	return ignored
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

func defaultPythonBinary() string {
	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}

type parsedFlags struct {
	port                   *string
	staticDir              *string
	configFile             *string
	grafanaURL             *string
	promURL                *string
	routerAPI              *string
	routerMetrics          *string
	jaegerURL              *string
	envoyURL               *string
	readonlyMode           *bool
	runtimeConfigWritable  *bool
	recipeStoreWritable    *bool
	setupMode              *bool
	allowOpenBootstrap     *bool
	allowedOrigins         *string
	platform               *string
	srBenchURL             *string
	srBenchTokenEnv        *string
	pythonPath             *string
	mcpEnabled             *bool
	mlPipelineEnabled      *bool
	mlPipelineDataDir      *string
	mlTrainingDir          *string
	mlServiceURL           *string
	workflowDBPath         *string
	statusDBPath           *string
	configProjectionDBPath *string
	auth                   authFlags
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
	cfg.ReadonlyMode = *flags.readonlyMode
	cfg.RuntimeConfigWritable = *flags.runtimeConfigWritable
	cfg.RecipeStoreWritable = *flags.recipeStoreWritable
	cfg.SetupMode = *flags.setupMode
	cfg.AllowOpenBootstrap = *flags.allowOpenBootstrap
	cfg.AllowedOrigins = parseAllowedOrigins(*flags.allowedOrigins)
	cfg.Platform = *flags.platform
}

func parseAllowedOrigins(raw string) []string {
	var origins []string
	for _, entry := range strings.Split(raw, ",") {
		// An Origin header never has a trailing slash, so an entry with one would
		// silently match nothing.
		entry = strings.TrimSuffix(strings.ToLower(strings.TrimSpace(entry)), "/")
		if entry != "" {
			origins = append(origins, entry)
		}
	}
	return origins
}

func applyFeatureConfig(cfg *Config, flags parsedFlags) error {
	cfg.SRBenchURL = *flags.srBenchURL
	cfg.SRBenchTokenEnv = *flags.srBenchTokenEnv
	cfg.PythonPath = *flags.pythonPath
	if err := ValidateSRBenchConfig(cfg.SRBenchURL, cfg.SRBenchTokenEnv); err != nil {
		return err
	}
	cfg.MCPEnabled = *flags.mcpEnabled
	cfg.MLPipelineEnabled = *flags.mlPipelineEnabled
	cfg.MLPipelineDataDir = *flags.mlPipelineDataDir
	cfg.MLTrainingDir = *flags.mlTrainingDir
	cfg.MLServiceURL = *flags.mlServiceURL
	cfg.WorkflowDBPath = *flags.workflowDBPath
	cfg.StatusDBPath = *flags.statusDBPath
	cfg.ConfigProjectionDBPath = *flags.configProjectionDBPath
	return nil
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

func resolveConfigPaths(cfg *Config) error {
	absConfigPath, err := filepath.Abs(cfg.ConfigFile)
	if err != nil {
		return err
	}
	cfg.AbsConfigPath = absConfigPath
	configDir := strings.TrimSpace(os.Getenv("DASHBOARD_CONFIG_DIR"))
	if configDir == "" {
		configDir = filepath.Dir(absConfigPath)
	}
	absConfigDir, err := filepath.Abs(configDir)
	if err != nil {
		return err
	}
	cfg.ConfigDir = absConfigDir
	cfg.ConfigBaseDir, err = resolveConfigBaseDir()
	return err
}

func bindCoreFlags() parsedFlags {
	return parsedFlags{
		port:       flag.String("port", env("DASHBOARD_PORT", "8700"), "dashboard port"),
		staticDir:  flag.String("static", env("DASHBOARD_STATIC_DIR", "../frontend"), "static assets directory"),
		configFile: flag.String("config", env("ROUTER_CONFIG_PATH", "../../config/config.yaml"), "path to config.yaml"),
		grafanaURL: flag.String("grafana", env("TARGET_GRAFANA_URL", ""), "Grafana base URL"),
		promURL:    flag.String("prometheus", env("TARGET_PROMETHEUS_URL", ""), "Prometheus base URL"),
		routerAPI: flag.String(
			"router_api", env("TARGET_ROUTER_API_URL", "http://localhost:8080"), "Router API base URL",
		),
		routerMetrics: flag.String(
			"router_metrics", env("TARGET_ROUTER_METRICS_URL", "http://localhost:9190/metrics"), "Router metrics URL",
		),
		jaegerURL: flag.String("jaeger", env("TARGET_JAEGER_URL", ""), "Jaeger base URL"),
		envoyURL:  flag.String("envoy", env("TARGET_ENVOY_URL", ""), "Envoy proxy URL for chat completions"),
		readonlyMode: flag.Bool(
			"readonly", env("DASHBOARD_READONLY", "false") == "true", "enable read-only mode (disable config editing)",
		),
		runtimeConfigWritable: flag.Bool(
			"runtime-config-writable", env("DASHBOARD_RUNTIME_CONFIG_WRITABLE", "true") == "true",
			"allow runtime config mutation when the mounted config state is writable",
		),
		recipeStoreWritable: flag.Bool(
			"recipe-store-writable", env("DASHBOARD_RECIPE_STORE_WRITABLE", "true") == "true",
			"allow Recipe package import when the package store is writable",
		),
		setupMode: flag.Bool(
			"setup-mode", env("DASHBOARD_SETUP_MODE", "false") == "true",
			"DEPRECATED: setup mode is resolved from the setup.mode block in the router config. "+
				"This flag is ignored except to warn when it disagrees with the config.",
		),
		allowOpenBootstrap: flag.Bool(
			"allow-open-bootstrap", env("DASHBOARD_ALLOW_OPEN_BOOTSTRAP", "false") == "true",
			"allow first-admin creation via the public web-form bootstrap endpoint (off by default; production should provision the admin via DASHBOARD_ADMIN_*)",
		),
		allowedOrigins: flag.String(
			"allowed-origins", env("DASHBOARD_ALLOWED_ORIGINS", ""),
			"comma-separated origins permitted to make state-changing requests, e.g. http://localhost:3001 for the Vite dev proxy (empty = own origin only)",
		),
		platform: flag.String("platform", env("DASHBOARD_PLATFORM", ""), "platform branding (e.g., 'amd' for AMD GPU deployments)"),
	}
}

func bindFeatureFlags(flags parsedFlags) parsedFlags {
	flags.srBenchURL = flag.String("sr-bench-url", env("SR_BENCH_URL", "http://127.0.0.1:8090"), "sr-bench service origin")
	flags.srBenchTokenEnv = flag.String("sr-bench-token-env", env("SR_BENCH_TOKEN_ENV", "SR_BENCH_TOKEN"), "environment variable holding the sr-bench service token")
	flags.pythonPath = flag.String("python", env("PYTHON_PATH", defaultPythonBinary()), "path to Python interpreter")
	flags.mcpEnabled = flag.Bool("mcp", env("MCP_ENABLED", "true") == "true", "enable MCP (Model Context Protocol) feature")
	flags.mlPipelineEnabled = flag.Bool("ml-pipeline", env("ML_PIPELINE_ENABLED", "false") == "true", "enable ML pipeline (benchmark, train, config)")
	flags.mlPipelineDataDir = flag.String("ml-pipeline-data", env("ML_PIPELINE_DATA_DIR", "./data/ml-pipeline"), "ML pipeline data directory")
	flags.mlTrainingDir = flag.String("ml-training-dir", env("ML_TRAINING_DIR", ""), "path to src/training/model_selection/ml_model_selection")
	flags.mlServiceURL = flag.String("ml-service-url", env("ML_SERVICE_URL", ""), "URL of Python ML service sidecar (empty = subprocess mode)")
	flags.workflowDBPath = flag.String("workflow-db", env("DASHBOARD_WORKFLOW_DB_PATH", "./data/workflow.sqlite"), "SQLite path for durable dashboard workflow state")
	flags.statusDBPath = flag.String("status-db", env("DASHBOARD_STATUS_DB_PATH", ""), "SQLite path for durable hourly service history")
	flags.configProjectionDBPath = flag.String("config-projection-db", env("DASHBOARD_CONFIG_PROJECTION_DB_PATH", "./data/config-projection.sqlite"), "SQLite path for deployed config projection state")
	flags.auth = bindAuthFlags()
	return flags
}

// LoadConfig loads configuration from flags and environment variables.
func LoadConfig() (*Config, error) {
	cfg := &Config{}
	flags := bindFeatureFlags(bindCoreFlags())
	bindRemovedOpenClawFlags()

	flag.Parse()
	cfg.IgnoredOpenClawSettings = ignoredOpenClawSettings()

	applyCoreConfig(cfg, flags)
	if err := applyFeatureConfig(cfg, flags); err != nil {
		return nil, err
	}
	if err := applyAuthConfig(cfg, flags.auth); err != nil {
		return nil, err
	}
	if err := resolveConfigPaths(cfg); err != nil {
		return nil, err
	}
	if strings.TrimSpace(cfg.StatusDBPath) == "" {
		cfg.StatusDBPath = filepath.Join(filepath.Dir(cfg.AuthDBPath), "status.sqlite")
	}

	return cfg, nil
}
