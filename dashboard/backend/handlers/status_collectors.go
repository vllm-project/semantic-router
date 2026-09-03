package handlers

import (
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

func collectInContainerStatus(runtimePath, routerAPIURL, envoyURL string, credentialProvider ...routerauth.CredentialProvider) SystemStatus {
	return collectManagedDockerStatus(runtimePath, routerAPIURL, envoyURL, credentialProvider...)
}

func collectHostStatus(runtimePath, routerAPIURL, envoyURL string, credentialProvider ...routerauth.CredentialProvider) SystemStatus {
	if status, ok := collectSplitManagedHostStatus(runtimePath, routerAPIURL, envoyURL, credentialProvider...); ok {
		return status
	}

	if status, ok := collectDirectStatus(runtimePath, routerAPIURL, envoyURL, credentialProvider...); ok {
		return status
	}
	return collectDashboardOnlyHostStatus(routerAPIURL, envoyURL)
}

func collectSplitManagedHostStatus(runtimePath, routerAPIURL, envoyURL string, credentialProvider ...routerauth.CredentialProvider) (SystemStatus, bool) {
	if !managedRuntimeUsesSplitContainers() {
		return SystemStatus{}, false
	}

	switch managedStatus := managedRuntimeContainerStatus(); managedStatus {
	case "running", "exited":
		return collectManagedDockerStatus(runtimePath, routerAPIURL, envoyURL, credentialProvider...), true
	case "not found":
		return SystemStatus{}, false
	default:
		return unknownContainerStatus(managedStatus), true
	}
}

func collectManagedDockerStatus(runtimePath, routerAPIURL, envoyURL string, credentialProvider ...routerauth.CredentialProvider) SystemStatus {
	status := baseSystemStatus()
	status.DeploymentType = "docker"
	status.Overall = "healthy"
	status.Endpoints = []string{"http://localhost:8899"}

	routerLogContent := getContainerLogsTailForContainer(managedContainerNameForService("router"), 500)
	routerHealthy, routerMsg := resolveManagedRouterStatus(routerAPIURL, routerLogContent)
	envoyHealthy, envoyMsg := resolveManagedEnvoyStatus(envoyURL)
	dashboardHealthy, dashboardMsg := resolveManagedDashboardStatus()

	status.RouterRuntime = resolveRouterRuntimeStatus(runtimePath, routerAPIURL, routerHealthy, credentialProvider...)
	routerMsg = applyRuntimeMessage(routerMsg, status.RouterRuntime)
	status.Models = fetchModelsWhenReady(routerAPIURL, routerHealthy, credentialProvider...)
	status.Services = append(status.Services,
		buildServiceStatus("Routing access", boolToStatus(routerHealthy && envoyHealthy), routerHealthy && envoyHealthy, routingAccessMessage(routerHealthy, envoyHealthy), "gateway"),
		buildServiceStatus("Router", boolToStatus(routerHealthy), routerHealthy, routerMsg, "container"),
		buildServiceStatus("Envoy", boolToStatus(envoyHealthy), envoyHealthy, envoyMsg, "container"),
		buildServiceStatus("Dashboard", boolToStatus(dashboardHealthy), dashboardHealthy, dashboardMsg, "container"),
	)
	setManagedDockerOverall(&status, routerHealthy, envoyHealthy, dashboardHealthy)

	return status
}

func unknownContainerStatus(containerStatus string) SystemStatus {
	status := baseSystemStatus()
	status.DeploymentType = "docker"
	status.Overall = containerStatus
	status.Services = append(status.Services, ServiceStatus{
		Name:      "Runtime",
		Status:    containerStatus,
		Healthy:   false,
		Component: "container",
	})
	return status
}

func collectDirectStatus(runtimePath, routerAPIURL, envoyURL string, credentialProvider ...routerauth.CredentialProvider) (SystemStatus, bool) {
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
	routerMsg = applyRuntimeMessage(routerMsg, status.RouterRuntime)
	status.Models = fetchModelsWhenReady(routerAPIURL, true, credentialProvider...)
	status.Services = append(status.Services, buildServiceStatus("Router", "running", true, routerMsg, "process"))

	envoyHealthy := appendDirectEnvoyStatus(&status, envoyURL)
	status.Services = append([]ServiceStatus{buildServiceStatus("Routing access", boolToStatus(envoyHealthy), envoyHealthy, routingAccessMessage(true, envoyHealthy), "gateway")}, status.Services...)
	status.Services = append(status.Services, buildServiceStatus("Dashboard", "running", true, "Running", "process"))

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

func setManagedDockerOverall(status *SystemStatus, checks ...bool) {
	for _, healthy := range checks {
		if healthy {
			setDegradedWhenUnhealthy(status, checks...)
			return
		}
	}
	status.Overall = "stopped"
}

func resolveManagedRouterStatus(routerAPIURL string, logContent string) (bool, string) {
	containerStatus := getDockerContainerStatus(managedContainerNameForService("router"))
	if routerAPIURL != "" {
		if healthy, msg := checkHTTPHealth(routerAPIURL + "/health"); healthy {
			return healthy, msg
		}
		if containerStatus == "running" {
			return false, "Starting"
		}
	}
	return resolveManagedServiceStatus("router", containerStatus, logContent)
}

func resolveManagedEnvoyStatus(envoyURL string) (bool, string) {
	if envoyURL != "" {
		if running, healthy, msg := checkEnvoyHealth(strings.TrimRight(envoyURL, "/") + "/v1/models"); running {
			return healthy, msg
		}
	}
	if readyURL := managedEnvoyReadyURL(); readyURL != "" {
		if running, healthy, msg := checkEnvoyHealth(readyURL); running {
			return healthy, msg
		}
	}
	return resolveManagedServiceStatus("envoy", getDockerContainerStatus(managedContainerNameForService("envoy")), "")
}

func resolveManagedDashboardStatus() (bool, string) {
	if isRunningInContainer() {
		return true, "Running"
	}
	return resolveManagedServiceStatus("dashboard", getDockerContainerStatus(managedContainerNameForService("dashboard")), "")
}

func resolveManagedServiceStatus(service string, containerStatus string, logContent string) (bool, string) {
	switch containerStatus {
	case "running":
		if logContent != "" && serviceLogLooksHealthy(service, logContent) {
			return true, "Running"
		}
		return true, "Running"
	case "created":
		return false, "Standby (setup mode)"
	case "exited":
		return false, "Exited"
	case "not found":
		return false, "Not found"
	default:
		return false, containerStatus
	}
}

func applyRuntimeMessage(message string, runtime *RouterRuntimeStatus) string {
	if runtime != nil && runtime.Message != "" {
		return runtime.Message
	}
	return message
}

func fetchModelsWhenReady(routerAPIURL string, routerHealthy bool, credentialProvider ...routerauth.CredentialProvider) *RouterModelsInfo {
	if !routerHealthy {
		return nil
	}

	return fetchRouterModelsInfo(routerAPIURL, credentialProvider...)
}
