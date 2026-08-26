package config

import (
	"flag"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// Config holds all application configuration
type Config struct {
	Port                              string
	AuthDBPath                        string
	JWTSecret                         string
	JWTExpiryHours                    int
	BootstrapAdminEmail               string
	BootstrapAdminPassword            string
	BootstrapAdminName                string
	DashboardPublicURL                string
	DashboardIssuer                   string
	DashboardIssuerID                 string
	DashboardSigningKeyFile           string
	DashboardKeyID                    string
	DashboardIssuerTLSListenAddr      string
	DashboardIssuerTLSCertificateFile string
	DashboardIssuerTLSPrivateKeyFile  string
	RouterBootstrapTokenFile          string
	SMTPHost                          string
	SMTPPort                          int
	SMTPUsername                      string
	SMTPPassword                      string
	SMTPFrom                          string
	StaticDir                         string
	ConfigFile                        string
	AbsConfigPath                     string
	ConfigDir                         string

	// Upstream targets
	GrafanaURL      string
	PrometheusURL   string
	RouterAPIURL    string
	RouterPublicURL string // Browser-reachable Router inference origin; Dashboard never proxies it
	RouterMetrics   string
	JaegerURL       string
	EnvoyURL        string // Router inference endpoint used by trusted Dashboard workflows
	FleetSimURL     string // Fleet simulator base URL

	// ReadonlyMode is the explicit, process-wide hard deny for Dashboard-owned
	// workflows. Router Management permissions remain authoritative for Router
	// resources.
	ReadonlyMode bool

	// AllowOpenBootstrap enables first-admin creation via the public, unauthenticated
	// web-form bootstrap endpoint. Off by default; production should provision the
	// admin via DASHBOARD_ADMIN_* instead of exposing an open registration path.
	AllowOpenBootstrap bool

	// Platform branding (e.g., "amd" for AMD GPU deployments)
	Platform string

	// Evaluation configuration
	EvaluationEnabled    bool
	EvaluationDBPath     string
	EvaluationResultsDir string
	PythonPath           string

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

	// Durable Dashboard-local public service history.
	StatusDBPath string
}

// env returns the env var or default
func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func defaultRouterAPIURL() string {
	return defaultRouterAPIURLForEnvironment(runningInContainer())
}

func defaultRouterAPIURLForEnvironment(inContainer bool) string {
	if !inContainer {
		return "http://localhost:8080"
	}

	containerName := strings.TrimSpace(os.Getenv("VLLM_SR_ROUTER_CONTAINER_NAME"))
	if containerName == "" {
		containerName = "vllm-sr-router-container"
	}
	return "http://" + containerName + ":8080"
}

func runningInContainer() bool {
	_, err := os.Stat("/.dockerenv")
	return err == nil
}

type authFlags struct {
	dbPath                   *string
	jwtSecret                *string
	jwtTTL                   *string
	bootstrapEmail           *string
	bootstrapPassword        *string
	bootstrapName            *string
	publicURL                *string
	issuer                   *string
	issuerID                 *string
	signingKeyFile           *string
	keyID                    *string
	issuerTLSListenAddr      *string
	issuerTLSCertificateFile *string
	issuerTLSPrivateKeyFile  *string
	routerBootstrapTokenFile *string
	smtpHost                 *string
	smtpPort                 *string
	smtpUsername             *string
	smtpPassword             *string
	smtpFrom                 *string
}

func bindAuthFlags() authFlags {
	return authFlags{
		dbPath:                   flag.String("auth-db", env("DASHBOARD_AUTH_DB_PATH", "./data/auth.db"), "auth database path"),
		jwtSecret:                flag.String("auth-jwt-secret", env("DASHBOARD_JWT_SECRET", ""), "JWT signing secret"),
		jwtTTL:                   flag.String("auth-jwt-expiry-hours", env("DASHBOARD_JWT_EXPIRY_HOURS", "12"), "JWT expiry in hours"),
		bootstrapEmail:           flag.String("bootstrap-admin-email", env("DASHBOARD_ADMIN_EMAIL", ""), "bootstrap admin email"),
		bootstrapPassword:        flag.String("bootstrap-admin-password", env("DASHBOARD_ADMIN_PASSWORD", ""), "bootstrap admin password"),
		bootstrapName:            flag.String("bootstrap-admin-name", env("DASHBOARD_ADMIN_NAME", ""), "bootstrap admin name"),
		publicURL:                flag.String("dashboard-public-url", env("DASHBOARD_PUBLIC_URL", ""), "public dashboard URL used in member invitations"),
		issuer:                   flag.String("dashboard-issuer", env("DASHBOARD_ISSUER", ""), "canonical HTTPS issuer origin used for Router identity exchange"),
		issuerID:                 flag.String("dashboard-issuer-id", env("DASHBOARD_ISSUER_ID", ""), "Router trusted identity issuer UUID"),
		signingKeyFile:           flag.String("dashboard-signing-key-file", env("DASHBOARD_SIGNING_KEY_FILE", ""), "PEM PKCS#8 Ed25519 assertion signing key"),
		keyID:                    flag.String("dashboard-key-id", env("DASHBOARD_KEY_ID", ""), "public signing key identifier"),
		issuerTLSListenAddr:      flag.String("dashboard-issuer-tls-listen", env("DASHBOARD_ISSUER_TLS_LISTEN_ADDR", ""), "private HTTPS listener for Router issuer discovery"),
		issuerTLSCertificateFile: flag.String("dashboard-issuer-tls-cert", env("DASHBOARD_ISSUER_TLS_CERT_FILE", ""), "issuer HTTPS certificate file"),
		issuerTLSPrivateKeyFile:  flag.String("dashboard-issuer-tls-key", env("DASHBOARD_ISSUER_TLS_KEY_FILE", ""), "issuer HTTPS private key file"),
		routerBootstrapTokenFile: flag.String("router-bootstrap-token-file", env("DASHBOARD_ROUTER_BOOTSTRAP_TOKEN_FILE", ""), "one-time Router bootstrap token file"),
		smtpHost:                 flag.String("smtp-host", env("DASHBOARD_SMTP_HOST", ""), "SMTP host for dashboard member invitations"),
		smtpPort:                 flag.String("smtp-port", env("DASHBOARD_SMTP_PORT", "587"), "SMTP port for dashboard member invitations"),
		smtpUsername:             flag.String("smtp-username", env("DASHBOARD_SMTP_USERNAME", ""), "SMTP username"),
		smtpPassword:             flag.String("smtp-password", env("DASHBOARD_SMTP_PASSWORD", ""), "SMTP password"),
		smtpFrom:                 flag.String("smtp-from", env("DASHBOARD_SMTP_FROM", ""), "SMTP From address for dashboard member invitations"),
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
	routerPublicURL      *string
	routerMetrics        *string
	jaegerURL            *string
	envoyURL             *string
	fleetSimURL          *string
	readonlyMode         *bool
	allowOpenBootstrap   *bool
	platform             *string
	evaluationEnabled    *bool
	evaluationDBPath     *string
	evaluationResultsDir *string
	pythonPath           *string
	mlPipelineEnabled    *bool
	mlPipelineDataDir    *string
	mlTrainingDir        *string
	mlServiceURL         *string
	workflowDBPath       *string
	statusDBPath         *string
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
	cfg.RouterPublicURL = strings.TrimRight(strings.TrimSpace(*flags.routerPublicURL), "/")
	cfg.RouterMetrics = *flags.routerMetrics
	cfg.JaegerURL = *flags.jaegerURL
	cfg.EnvoyURL = *flags.envoyURL
	cfg.FleetSimURL = *flags.fleetSimURL
	cfg.ReadonlyMode = *flags.readonlyMode
	cfg.AllowOpenBootstrap = *flags.allowOpenBootstrap
	cfg.Platform = *flags.platform
}

func canonicalRouterPublicURL(raw string) (string, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return "", nil
	}
	parsed, err := url.Parse(trimmed)
	if err != nil || parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", fmt.Errorf("DASHBOARD_ROUTER_PUBLIC_URL must be an HTTP(S) origin")
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" || parsed.Path != "" && parsed.Path != "/" {
		return "", fmt.Errorf("DASHBOARD_ROUTER_PUBLIC_URL must be an HTTP(S) origin")
	}
	return parsed.Scheme + "://" + strings.ToLower(parsed.Host), nil
}

func applyFeatureConfig(cfg *Config, flags parsedFlags) {
	cfg.EvaluationEnabled = *flags.evaluationEnabled
	cfg.EvaluationDBPath = *flags.evaluationDBPath
	cfg.EvaluationResultsDir = *flags.evaluationResultsDir
	cfg.PythonPath = *flags.pythonPath
	cfg.MLPipelineEnabled = *flags.mlPipelineEnabled
	cfg.MLPipelineDataDir = *flags.mlPipelineDataDir
	cfg.MLTrainingDir = *flags.mlTrainingDir
	cfg.MLServiceURL = *flags.mlServiceURL
	cfg.WorkflowDBPath = *flags.workflowDBPath
	cfg.StatusDBPath = *flags.statusDBPath
}

func applyAuthConfig(cfg *Config, flags authFlags) error {
	cfg.AuthDBPath = *flags.dbPath
	cfg.JWTSecret = *flags.jwtSecret
	cfg.BootstrapAdminEmail = *flags.bootstrapEmail
	cfg.BootstrapAdminPassword = *flags.bootstrapPassword
	cfg.BootstrapAdminName = *flags.bootstrapName
	cfg.DashboardPublicURL = strings.TrimRight(strings.TrimSpace(*flags.publicURL), "/")
	cfg.DashboardIssuer = strings.TrimSpace(*flags.issuer)
	cfg.DashboardIssuerID = strings.TrimSpace(*flags.issuerID)
	cfg.DashboardSigningKeyFile = strings.TrimSpace(*flags.signingKeyFile)
	cfg.DashboardKeyID = strings.TrimSpace(*flags.keyID)
	cfg.DashboardIssuerTLSListenAddr = strings.TrimSpace(*flags.issuerTLSListenAddr)
	cfg.DashboardIssuerTLSCertificateFile = strings.TrimSpace(*flags.issuerTLSCertificateFile)
	cfg.DashboardIssuerTLSPrivateKeyFile = strings.TrimSpace(*flags.issuerTLSPrivateKeyFile)
	cfg.RouterBootstrapTokenFile = strings.TrimSpace(*flags.routerBootstrapTokenFile)
	issuerTLSConfigured := cfg.DashboardIssuerTLSListenAddr != "" || cfg.DashboardIssuerTLSCertificateFile != "" || cfg.DashboardIssuerTLSPrivateKeyFile != ""
	if issuerTLSConfigured && (cfg.DashboardIssuerTLSListenAddr == "" || cfg.DashboardIssuerTLSCertificateFile == "" || cfg.DashboardIssuerTLSPrivateKeyFile == "") {
		return fmt.Errorf("dashboard issuer TLS listener, certificate, and private key must be configured together")
	}
	if err := validateRouterBootstrapConfig(cfg, issuerTLSConfigured); err != nil {
		return err
	}
	cfg.SMTPHost = strings.TrimSpace(*flags.smtpHost)
	cfg.SMTPUsername = strings.TrimSpace(*flags.smtpUsername)
	cfg.SMTPPassword = *flags.smtpPassword
	cfg.SMTPFrom = strings.TrimSpace(*flags.smtpFrom)

	ttl, err := strconv.Atoi(*flags.jwtTTL)
	if err != nil {
		return err
	}
	cfg.JWTExpiryHours = ttl
	smtpPort, err := strconv.Atoi(*flags.smtpPort)
	if err != nil {
		return err
	}
	cfg.SMTPPort = smtpPort
	return nil
}

func validateRouterBootstrapConfig(cfg *Config, issuerTLSConfigured bool) error {
	if cfg.RouterBootstrapTokenFile == "" {
		return nil
	}
	if !filepath.IsAbs(cfg.RouterBootstrapTokenFile) || filepath.Clean(cfg.RouterBootstrapTokenFile) != cfg.RouterBootstrapTokenFile {
		return fmt.Errorf("router bootstrap token file must be an absolute canonical path")
	}
	if cfg.DashboardIssuer == "" || cfg.DashboardIssuerID == "" || cfg.DashboardSigningKeyFile == "" ||
		cfg.DashboardKeyID == "" || !issuerTLSConfigured {
		return fmt.Errorf("router bootstrap requires the complete private Dashboard issuer configuration")
	}
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
	configDir := strings.TrimSpace(os.Getenv("DASHBOARD_CONFIG_DIR"))
	if configDir == "" {
		configDir = filepath.Dir(absConfigPath)
	}
	absConfigDir, err := filepath.Abs(configDir)
	if err != nil {
		return err
	}
	cfg.ConfigDir = absConfigDir
	return nil
}

func bindParsedFlags() parsedFlags {
	// Flags/env for configuration
	port := flag.String("port", env("DASHBOARD_PORT", "8700"), "dashboard port")
	staticDir := flag.String("static", env("DASHBOARD_STATIC_DIR", "../frontend"), "static assets directory")
	configFile := flag.String("config", env("ROUTER_CONFIG_PATH", "../../config/config.yaml"), "path to config.yaml")

	// Upstream targets
	grafanaURL := flag.String("grafana", env("TARGET_GRAFANA_URL", ""), "Grafana base URL")
	promURL := flag.String("prometheus", env("TARGET_PROMETHEUS_URL", ""), "Prometheus base URL")
	routerAPI := flag.String("router_api", env("TARGET_ROUTER_API_URL", defaultRouterAPIURL()), "Router API base URL")
	routerPublicURL := flag.String("router-public-url", env("DASHBOARD_ROUTER_PUBLIC_URL", ""), "browser-reachable Router public API origin")
	routerMetrics := flag.String("router_metrics", env("TARGET_ROUTER_METRICS_URL", "http://localhost:9190/metrics"), "Router metrics URL")
	jaegerURL := flag.String("jaeger", env("TARGET_JAEGER_URL", ""), "Jaeger base URL")
	envoyURL := flag.String("envoy", env("TARGET_ENVOY_URL", ""), "Router inference URL for trusted Dashboard workflows")
	fleetSimURL := flag.String("fleet-sim", env("TARGET_FLEET_SIM_URL", ""), "Fleet simulator base URL")

	// Read-only mode for Dashboard-owned workflows.
	readonlyMode := flag.Bool("readonly", env("DASHBOARD_READONLY", "false") == "true", "enable read-only mode (disable config editing)")
	allowOpenBootstrap := flag.Bool("allow-open-bootstrap", env("DASHBOARD_ALLOW_OPEN_BOOTSTRAP", "false") == "true", "allow first-admin creation via the public web-form bootstrap endpoint (off by default; production should provision the admin via DASHBOARD_ADMIN_*)")

	// Platform branding
	platform := flag.String("platform", env("DASHBOARD_PLATFORM", ""), "platform branding (e.g., 'amd' for AMD GPU deployments)")

	// Evaluation configuration
	evaluationEnabled := flag.Bool("evaluation", env("EVALUATION_ENABLED", "true") == "true", "enable evaluation feature")
	evaluationDBPath := flag.String("evaluation-db", env("EVALUATION_DB_PATH", "./data/evaluations.db"), "evaluation database path")
	evaluationResultsDir := flag.String("evaluation-results", env("EVALUATION_RESULTS_DIR", "./data/results"), "evaluation results directory")
	pythonPath := flag.String("python", env("PYTHON_PATH", defaultPythonBinary()), "path to Python interpreter")

	// ML Onboarding configuration
	mlPipelineEnabled := flag.Bool("ml-pipeline", env("ML_PIPELINE_ENABLED", "true") == "true", "enable ML pipeline (benchmark, train, config)")
	mlPipelineDataDir := flag.String("ml-pipeline-data", env("ML_PIPELINE_DATA_DIR", "./data/ml-pipeline"), "ML pipeline data directory")
	mlTrainingDir := flag.String("ml-training-dir", env("ML_TRAINING_DIR", ""), "path to src/training/model_selection/ml_model_selection")
	mlServiceURL := flag.String("ml-service-url", env("ML_SERVICE_URL", ""), "URL of Python ML service sidecar (empty = subprocess mode)")
	workflowDBPath := flag.String("workflow-db", env("DASHBOARD_WORKFLOW_DB_PATH", "./data/workflow.sqlite"), "SQLite path for durable dashboard workflow state")
	statusDBPath := flag.String("status-db", env("DASHBOARD_STATUS_DB_PATH", ""), "SQLite path for durable dashboard service history (defaults beside the auth database)")

	// Authentication configuration
	auth := bindAuthFlags()

	// OpenClaw configuration
	openClaw := bindOpenClawFlags()

	return parsedFlags{
		port:                 port,
		staticDir:            staticDir,
		configFile:           configFile,
		grafanaURL:           grafanaURL,
		promURL:              promURL,
		routerAPI:            routerAPI,
		routerPublicURL:      routerPublicURL,
		routerMetrics:        routerMetrics,
		jaegerURL:            jaegerURL,
		envoyURL:             envoyURL,
		fleetSimURL:          fleetSimURL,
		readonlyMode:         readonlyMode,
		allowOpenBootstrap:   allowOpenBootstrap,
		platform:             platform,
		evaluationEnabled:    evaluationEnabled,
		evaluationDBPath:     evaluationDBPath,
		evaluationResultsDir: evaluationResultsDir,
		pythonPath:           pythonPath,
		mlPipelineEnabled:    mlPipelineEnabled,
		mlPipelineDataDir:    mlPipelineDataDir,
		mlTrainingDir:        mlTrainingDir,
		mlServiceURL:         mlServiceURL,
		workflowDBPath:       workflowDBPath,
		statusDBPath:         statusDBPath,
		auth:                 auth,
		openClaw:             openClaw,
	}
}

// LoadConfig loads configuration from flags and environment variables
func LoadConfig() (*Config, error) {
	cfg := &Config{}
	flags := bindParsedFlags()
	flag.Parse()

	applyCoreConfig(cfg, flags)
	publicURL, err := canonicalRouterPublicURL(cfg.RouterPublicURL)
	if err != nil {
		return nil, err
	}
	cfg.RouterPublicURL = publicURL
	applyFeatureConfig(cfg, flags)
	if err := applyAuthConfig(cfg, flags.auth); err != nil {
		return nil, err
	}
	resolveDashboardStatePaths(cfg)
	applyOpenClawConfig(cfg, flags.openClaw)
	if err := resolveConfigPaths(cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}

func resolveDashboardStatePaths(cfg *Config) {
	if strings.TrimSpace(cfg.StatusDBPath) == "" {
		cfg.StatusDBPath = filepath.Join(filepath.Dir(cfg.AuthDBPath), "status.sqlite")
	}
}
