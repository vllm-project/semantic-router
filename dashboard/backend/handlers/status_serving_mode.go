package handlers

import (
	"encoding/json"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

const (
	servingModeRouter   = "router"
	servingModeEngine   = "engine"
	servingModeUnknown  = "unknown"
	servingProbeTimeout = 2 * time.Second
	servingHealthLimit  = 64 << 10
	servingOpenAPILimit = 128 << 10
)

var engineAPIVersion = regexp.MustCompile(`^\d+\.\d+\.\d+$`)

type servingHealthProbe struct {
	mode    string
	healthy bool
	message string
	state   string
}

// probeServingHealth reads identity from the same response that establishes
// process health. Only the configured upstream identifies the serving mode;
// a Router's managed model engines do not turn it into an engine-only service.
func probeServingHealth(upstream string, providers ...routerauth.CredentialProvider) servingHealthProbe {
	probe := servingHealthProbe{mode: servingModeUnknown}
	if strings.TrimSpace(upstream) == "" {
		return probe
	}
	upstream = strings.TrimRight(upstream, "/")
	response, err := routerManagementGET(upstream+"/health", servingProbeTimeout, providers...)
	if err != nil {
		return probe
	}
	defer func() { _ = response.Body.Close() }()
	probe.healthy = response.StatusCode >= 200 && response.StatusCode < 300
	if probe.healthy {
		probe.message = "HTTP health check OK"
	}
	if !probe.healthy && response.StatusCode != http.StatusServiceUnavailable {
		return probe
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, servingHealthLimit+1))
	if err != nil || len(body) > servingHealthLimit {
		return probe
	}
	var health struct {
		Service    string          `json:"service"`
		Status     string          `json:"status"`
		APIVersion string          `json:"api_version"`
		Reason     json.RawMessage `json:"reason"`
		Model      json.RawMessage `json:"model"`
	}
	if json.Unmarshal(body, &health) != nil {
		return probe
	}
	if probe.healthy && health.Service == "classification-api" && health.Status == "healthy" {
		probe.mode = servingModeRouter
		return probe
	}
	if !engineAPIVersion.MatchString(health.APIVersion) || len(health.Reason) == 0 || len(health.Model) == 0 {
		return probe
	}
	switch health.Status {
	case "starting", "loading", "warming", "ready", "failed", "degraded":
	default:
		return probe
	}
	if !isModelEngineAPI(upstream, health.APIVersion, providers...) {
		return probe
	}
	probe.mode, probe.state = servingModeEngine, health.Status
	return probe
}

// The health shape alone is not a unique identity. Confirm the runtime's
// versioned API contract before describing an arbitrary healthy API as Engine.
func isModelEngineAPI(upstream, version string, providers ...routerauth.CredentialProvider) bool {
	response, err := routerManagementGET(upstream+"/openapi.yaml", servingProbeTimeout, providers...)
	if err != nil {
		return false
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return false
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, servingOpenAPILimit+1))
	if err != nil || len(body) > servingOpenAPILimit {
		return false
	}
	var contract struct {
		OpenAPI string `yaml:"openapi"`
		Info    struct {
			Title   string `yaml:"title"`
			Version string `yaml:"version"`
		} `yaml:"info"`
		Paths map[string]map[string]any `yaml:"paths"`
	}
	if yaml.Unmarshal(body, &contract) != nil || !strings.HasPrefix(contract.OpenAPI, "3.") ||
		contract.Info.Title != "vLLM Semantic Router Model Runtime" || contract.Info.Version != version {
		return false
	}
	for path, method := range map[string]string{"/health": "get", "/v1/models": "get", "/v1/decisions": "post"} {
		if _, exists := contract.Paths[path][method]; !exists {
			return false
		}
	}
	return true
}

func collectModelEngineStatus(upstream, deploymentType, component string, probe servingHealthProbe) SystemStatus {
	status := baseSystemStatus()
	status.ServingMode, status.DeploymentType = servingModeEngine, deploymentType
	status.Endpoints = []string{strings.TrimRight(upstream, "/")}
	ready := probe.healthy && probe.state == "ready"
	status.Overall = "healthy"
	setDegradedWhenUnhealthy(&status, ready)
	state, message := "running", "Ready"
	if !ready {
		state, message = probe.state, "Model Engine reports "+probe.state
		if probe.state == "loading" || probe.state == "warming" {
			state = "starting"
		}
	}
	status.Services = []ServiceStatus{
		buildServiceStatus("Model Engine", state, ready, message, component),
		buildServiceStatus("Dashboard", "running", true, "Running", component),
	}
	return status
}
