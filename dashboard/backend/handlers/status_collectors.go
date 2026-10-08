package handlers

import (
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

// StackState locates what tells a managed service that does not answer apart:
// the runtime config, whose setup block says the stack waits for first-run
// setup, and the files `vllm-sr serve` and the Dashboard keep beside it (the
// CLI's heartbeat and the pending activation). The Dashboard reads them instead
// of asking a container runtime.
type StackState struct {
	ConfigPath string
	Setup      *setupmode.Resolver
}

// stoppedService is the state a managed service reports while it does not
// answer its HTTP probe. Only the starting state may mention starting: the
// status history counts a message that does as a start, not an outage.
type stoppedService struct {
	status  string
	message string
}

func (s StackState) stoppedService() stoppedService {
	switch {
	case s.Setup != nil && s.Setup.Active():
		return stoppedService{"standby", "Waiting for setup: activate a config in the Dashboard"}
	case s.ConfigPath == "" || !servedByCLI():
		return stoppedService{"not running", "Not running"}
	case serveRunning(s.ConfigPath):
		return stoppedService{"starting", "Starting: `vllm-sr serve` is starting it"}
	case pendingActivationRecorded(s.ConfigPath):
		return stoppedService{"not running", "Not running: run `vllm-sr serve` to apply the saved configuration"}
	default:
		return stoppedService{"not running", "Not running: run `vllm-sr serve`"}
	}
}

// servedByCLI reports whether `vllm-sr serve` created this stack. It names the
// runtime config on every container it starts; a Kubernetes deployment does
// not, keeps none of the CLI's files, and takes no `vllm-sr serve` hint.
func servedByCLI() bool {
	return strings.TrimSpace(os.Getenv("VLLM_SR_RUNTIME_CONFIG_PATH")) != ""
}

func collectInContainerStatus(runtimePath string, stack StackState, routerAPIURL, envoyURL string, credentialProvider ...routerauth.CredentialProvider) SystemStatus {
	return collectManagedStackStatus(runtimePath, stack, routerAPIURL, envoyURL, credentialProvider...)
}

func collectHostStatus(runtimePath, routerAPIURL, envoyURL string, credentialProvider ...routerauth.CredentialProvider) SystemStatus {
	if status, ok := collectDirectStatus(runtimePath, routerAPIURL, envoyURL, credentialProvider...); ok {
		return status
	}
	return collectDashboardOnlyHostStatus(routerAPIURL, envoyURL)
}

// collectManagedStackStatus reports the stack `vllm-sr serve` runs, from its
// Dashboard container. A service that answers its HTTP probe runs; the state
// of one that does not comes from the stack's files.
func collectManagedStackStatus(runtimePath string, stack StackState, routerAPIURL, envoyURL string, credentialProvider ...routerauth.CredentialProvider) SystemStatus {
	routerAPIURL = strings.TrimRight(routerAPIURL, "/")
	status := baseSystemStatus()
	status.DeploymentType = "docker"
	status.Overall = "healthy"
	status.Endpoints = []string{"http://localhost:8899"}

	stopped := stack.stoppedService()
	router := resolveManagedRouterStatus(routerAPIURL, stopped)
	envoy := resolveManagedEnvoyStatus(envoyURL, stopped)

	status.RouterRuntime = resolveRouterRuntimeStatus(runtimePath, routerAPIURL, router.Healthy, credentialProvider...)
	routerReady := resolveRouterReadiness(routerAPIURL, router.Healthy, status.RouterRuntime, credentialProvider...)
	router.Message = applyRuntimeMessage(router.Message, status.RouterRuntime)
	status.Models = fetchModelsWhenReady(routerAPIURL, routerReady, credentialProvider...)
	status.Services = append(status.Services,
		buildServiceStatus("Routing access", boolToStatus(routerReady && envoy.Healthy), routerReady && envoy.Healthy, routingAccessMessage(routerReady, envoy.Healthy), "gateway"),
		router,
	)
	// A standalone Router serves the listeners that the routing-access probe
	// reaches; there is no Envoy container to report.
	if managedStackRunsEnvoy() {
		status.Services = append(status.Services, envoy)
	}
	status.Services = append(status.Services,
		buildServiceStatus("Dashboard", "running", true, "Running", "container"),
	)
	setDegradedWhenUnhealthy(&status, router.Healthy, routerReady, envoy.Healthy)

	return status
}

func collectDirectStatus(runtimePath, routerAPIURL, envoyURL string, credentialProvider ...routerauth.CredentialProvider) (SystemStatus, bool) {
	routerAPIURL = strings.TrimRight(routerAPIURL, "/")
	if routerAPIURL == "" {
		return SystemStatus{}, false
	}

	routerHealthy, routerMsg := checkHTTPHealth(routerAPIURL + "/health")
	if !routerHealthy {
		return SystemStatus{}, false
	}

	status := baseSystemStatus()
	status.DeploymentType = "local (direct)"
	status.Overall = "healthy"
	status.Endpoints = []string{routerAPIURL}
	status.RouterRuntime = resolveRouterRuntimeStatus(runtimePath, routerAPIURL, routerHealthy, credentialProvider...)
	routerReady := resolveRouterReadiness(routerAPIURL, routerHealthy, status.RouterRuntime, credentialProvider...)
	routerMsg = applyRuntimeMessage(routerMsg, status.RouterRuntime)
	status.Models = fetchModelsWhenReady(routerAPIURL, routerReady, credentialProvider...)
	status.Services = append(status.Services, buildServiceStatus("Router", "running", true, routerMsg, "process"))

	envoyHealthy := appendDirectEnvoyStatus(&status, envoyURL)
	status.Services = append([]ServiceStatus{buildServiceStatus("Routing access", boolToStatus(routerReady && envoyHealthy), routerReady && envoyHealthy, routingAccessMessage(routerReady, envoyHealthy), "gateway")}, status.Services...)
	status.Services = append(status.Services, buildServiceStatus("Dashboard", "running", true, "Running", "process"))
	setDegradedWhenUnhealthy(&status, routerReady, envoyHealthy)

	return status, true
}

func collectDashboardOnlyHostStatus(routerAPIURL, envoyURL string) SystemStatus {
	status := baseSystemStatus()
	routerMsg := "Router API URL is not configured"
	if routerAPIURL != "" {
		status.Endpoints = []string{routerAPIURL}
		routerMsg = "Router health check failed"
	}

	status.Services = append(status.Services,
		buildServiceStatus("Routing access", "unavailable", false, "Router or gateway is unavailable", "gateway"),
		buildServiceStatus("Router", "not running", false, routerMsg, "process"),
	)
	appendDirectEnvoyStatus(&status, envoyURL)
	status.Services = append(status.Services,
		buildServiceStatus("Dashboard", "running", true, "Running", "process"),
	)

	return status
}

func routingAccessMessage(routerHealthy, envoyHealthy bool) string {
	if routerHealthy && envoyHealthy {
		return "Ready"
	}
	return "Router or gateway is unavailable"
}

func appendDirectEnvoyStatus(status *SystemStatus, envoyURL string) bool {
	readyURL := "http://localhost:8801/ready"
	if envoyURL != "" {
		readyURL = strings.TrimRight(envoyURL, "/") + "/v1/models"
	}
	envoyRunning, envoyHealthy, envoyMsg := checkEnvoyHealth(readyURL)
	if !envoyRunning {
		return false
	}

	status.Services = append(status.Services, buildServiceStatus("Envoy", boolToStatus(envoyHealthy), envoyHealthy, envoyMsg, "proxy"))
	if !envoyHealthy {
		status.Overall = "degraded"
	}
	return envoyHealthy
}

func buildServiceStatus(name, serviceStatus string, healthy bool, message, component string) ServiceStatus {
	return ServiceStatus{
		Name:      name,
		Status:    serviceStatus,
		Healthy:   healthy,
		Message:   message,
		Component: component,
	}
}

func setDegradedWhenUnhealthy(status *SystemStatus, checks ...bool) {
	for _, healthy := range checks {
		if !healthy {
			status.Overall = "degraded"
			return
		}
	}
}

func resolveManagedRouterStatus(routerAPIURL string, stopped stoppedService) ServiceStatus {
	if routerAPIURL != "" {
		if healthy, msg := checkHTTPHealth(routerAPIURL + "/health"); healthy {
			return buildServiceStatus("Router", "running", true, msg, "container")
		}
	}
	return buildServiceStatus("Router", stopped.status, false, stopped.message, "container")
}

func resolveManagedEnvoyStatus(envoyURL string, stopped stoppedService) ServiceStatus {
	readyURLs := []string{}
	if envoyURL != "" {
		readyURLs = append(readyURLs, strings.TrimRight(envoyURL, "/")+"/v1/models")
	}
	if readyURL := managedEnvoyReadyURL(); readyURL != "" {
		readyURLs = append(readyURLs, readyURL)
	}
	for _, readyURL := range readyURLs {
		if running, healthy, msg := checkEnvoyHealth(readyURL); running {
			return buildServiceStatus("Envoy", boolToStatus(healthy), healthy, msg, "container")
		}
	}
	return buildServiceStatus("Envoy", stopped.status, false, stopped.message, "container")
}

func applyRuntimeMessage(message string, runtime *RouterRuntimeStatus) string {
	if runtime != nil && runtime.Message != "" {
		return runtime.Message
	}
	return message
}

// Keep process liveness separate from whether the Router can serve requests.
func resolveRouterReadiness(routerAPIURL string, routerHealthy bool, runtime *RouterRuntimeStatus, credentialProvider ...routerauth.CredentialProvider) bool {
	if !routerHealthy {
		return false
	}
	if runtime != nil {
		return runtime.Ready
	}
	return checkRouterManagementHealth(routerAPIURL+"/ready", credentialProvider...)
}

func fetchModelsWhenReady(routerAPIURL string, routerReady bool, credentialProvider ...routerauth.CredentialProvider) *RouterModelsInfo {
	if !routerReady {
		return nil
	}

	return fetchRouterModelsInfo(routerAPIURL, credentialProvider...)
}
