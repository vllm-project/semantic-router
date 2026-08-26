package handlers

func collectInContainerStatus(routerAPIURL string) SystemStatus {
	return collectManagedDockerStatus(routerAPIURL)
}

func collectHostStatus(routerAPIURL string) SystemStatus {
	if status, ok := collectSplitManagedHostStatus(routerAPIURL); ok {
		return status
	}

	if status, ok := collectDirectStatus(routerAPIURL); ok {
		return status
	}
	return collectDashboardOnlyHostStatus(routerAPIURL)
}

func collectSplitManagedHostStatus(routerAPIURL string) (SystemStatus, bool) {
	if !managedRuntimeUsesSplitContainers() {
		return SystemStatus{}, false
	}

	switch managedStatus := managedRuntimeContainerStatus(); managedStatus {
	case "running", "exited":
		return collectManagedDockerStatus(routerAPIURL), true
	case "not found":
		return SystemStatus{}, false
	default:
		return unknownContainerStatus(managedStatus), true
	}
}

func collectManagedDockerStatus(routerAPIURL string) SystemStatus {
	status := baseSystemStatus()
	status.Overall = "healthy"

	routerHealthy, routerMsg := resolveManagedRouterStatus(routerAPIURL)
	envoyHealthy, envoyMsg := resolveManagedEnvoyStatus()
	dashboardHealthy, dashboardMsg := resolveManagedDashboardStatus()

	status.Services = append(status.Services,
		buildServiceStatus("Router", boolToStatus(routerHealthy), routerHealthy, routerMsg, "container"),
		buildServiceStatus("Envoy", boolToStatus(envoyHealthy), envoyHealthy, envoyMsg, "container"),
		buildServiceStatus("Dashboard", boolToStatus(dashboardHealthy), dashboardHealthy, dashboardMsg, "container"),
	)
	setManagedDockerOverall(&status, routerHealthy, envoyHealthy, dashboardHealthy)

	return status
}

func unknownContainerStatus(containerStatus string) SystemStatus {
	status := baseSystemStatus()
	status.Overall = "degraded"
	status.Services = append(status.Services, ServiceStatus{
		Name:    "Runtime",
		Status:  publicServiceStatus(containerStatus, false),
		Healthy: false,
	})
	return status
}

func collectDirectStatus(routerAPIURL string) (SystemStatus, bool) {
	if routerAPIURL == "" {
		return SystemStatus{}, false
	}

	routerHealthy, routerMsg := checkHTTPHealth(routerAPIURL + "/health")
	if !routerHealthy {
		return SystemStatus{}, false
	}

	status := baseSystemStatus()
	status.Overall = "healthy"
	status.Services = append(status.Services, buildServiceStatus("Router", "running", true, routerMsg, "process"))

	appendDirectEnvoyStatus(&status)
	status.Services = append(status.Services, buildServiceStatus("Dashboard", "running", true, "Running", "process"))

	return status, true
}

func collectDashboardOnlyHostStatus(routerAPIURL string) SystemStatus {
	status := baseSystemStatus()
	routerMsg := "Router API URL is not configured"
	if routerAPIURL != "" {
		routerMsg = "Router health check failed"
	}

	status.Services = append(status.Services,
		buildServiceStatus("Router", "not running", false, routerMsg, "process"),
	)
	appendDirectEnvoyStatus(&status)
	status.Services = append(status.Services,
		buildServiceStatus("Dashboard", "running", true, "Running", "process"),
	)

	return status
}

func appendDirectEnvoyStatus(status *SystemStatus) {
	envoyRunning, envoyHealthy, envoyMsg := checkEnvoyHealth(managedEnvoyReadyURL())
	if !envoyRunning {
		status.Services = append(status.Services, buildServiceStatus("Envoy", "not running", false, "", "proxy"))
		if status.Overall == "healthy" {
			status.Overall = "degraded"
		}
		return
	}

	status.Services = append(status.Services, buildServiceStatus("Envoy", boolToStatus(envoyHealthy), envoyHealthy, envoyMsg, "proxy"))
	if !envoyHealthy {
		status.Overall = "degraded"
	}
}

func buildServiceStatus(name, serviceStatus string, healthy bool, _ string, _ string) ServiceStatus {
	return ServiceStatus{
		Name:    name,
		Status:  publicServiceStatus(serviceStatus, healthy),
		Healthy: healthy,
	}
}

func publicServiceStatus(status string, healthy bool) string {
	if healthy {
		return "operational"
	}
	switch status {
	case "created", "starting", "unknown":
		return "starting"
	default:
		return "unavailable"
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

func resolveManagedRouterStatus(routerAPIURL string) (bool, string) {
	containerStatus := getDockerContainerStatus(managedContainerNameForService("router"))
	if routerAPIURL != "" {
		if healthy, msg := checkHTTPHealth(routerAPIURL + "/health"); healthy {
			return healthy, msg
		}
		if containerStatus == "running" {
			return false, "Starting"
		}
	}
	return resolveManagedServiceStatus(containerStatus)
}

func resolveManagedEnvoyStatus() (bool, string) {
	if readyURL := managedEnvoyReadyURL(); readyURL != "" {
		if running, healthy, msg := checkEnvoyHealth(readyURL); running {
			return healthy, msg
		}
	}
	return resolveManagedServiceStatus(getDockerContainerStatus(managedContainerNameForService("envoy")))
}

func resolveManagedDashboardStatus() (bool, string) {
	if isRunningInContainer() {
		return true, "Running"
	}
	return resolveManagedServiceStatus(getDockerContainerStatus(managedContainerNameForService("dashboard")))
}

func resolveManagedServiceStatus(containerStatus string) (bool, string) {
	switch containerStatus {
	case "running":
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
