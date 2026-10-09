package handlers

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

const (
	servingModeRouter   = "router"
	servingModeEngine   = "engine"
	servingModeUnknown  = "unknown"
	servingProbeTimeout = 2 * time.Second
	servingHealthLimit  = 64 << 10
)

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
		ServingMode string `json:"serving_mode"`
		Service     string `json:"service"`
		Status      string `json:"status"`
	}
	if json.Unmarshal(body, &health) != nil {
		return probe
	}
	if probe.healthy && health.Service == "classification-api" && health.Status == "healthy" {
		switch health.ServingMode {
		case servingModeRouter, servingModeEngine:
			probe.mode, probe.state = health.ServingMode, "ready"
		}
	}
	return probe
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
