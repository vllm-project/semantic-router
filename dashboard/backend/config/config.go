package config

import (
	"os"
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
	SetupMode    bool

	// Platform branding (e.g., "amd" for AMD GPU deployments)
	Platform string

	// Evaluation configuration
	EvaluationEnabled    bool
	EvaluationDBPath     string
	EvaluationResultsDir string
	PythonPath           string

	// Console persistence configuration
	ConsoleStoreBackend string
	ConsoleDBPath       string
	ConsoleStoreDSN     string

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
}

// env returns the env var or default
func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
