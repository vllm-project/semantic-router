package handlers

func collectInContainerStatus(runtimePath, routerAPIURL string) SystemStatus {
	if managedRuntimeUsesSplitContainers() {
		return collectManagedDockerStatus(runtimePath, routerAPIURL)
	}

	status := baseSystemStatus()
	status.DeploymentType = "docker"
	status.Overall = "healthy"
	status.Endpoints = []string{"http://localhost:8899"}

	routerHealthy, routerMsg := checkServiceFromContainerLogs("router")
	envoyHealthy, envoyMsg := checkServiceFromContainerLogs("envoy")
	dashboardHealthy := true
	dashboardMsg := "Running"

	status.RouterRuntime = resolveRouterRuntimeStatus(runtimePath, routerAPIURL, routerHealthy)
	routerMsg = applyRuntimeMessage(routerMsg, status.RouterRuntime)
	status.Models = fetchModelsWhenReady(routerAPIURL, routerHealthy)
	status.Services = append(status.Services,
		buildServiceStatus("Router", boolToStatus(routerHealthy), routerHealthy, routerMsg, "container"),
		buildServiceStatus("Envoy", boolToStatus(envoyHealthy), envoyHealthy, envoyMsg, "container"),
		buildServiceStatus("Dashboard", boolToStatus(dashboardHealthy), dashboardHealthy, dashboardMsg, "container"),
	)
	setDegradedWhenUnhealthy(&status, routerHealthy, envoyHealthy, dashboardHealthy)

	return status
}

func collectHostStatus(runtimePath, routerAPIURL string) SystemStatus {
	if status, ok := collectSplitManagedHostStatus(runtimePath, routerAPIURL); ok {
		return status
	}

	switch containerStatus := getDockerContainerStatus(vllmSrContainerName); containerStatus {
	case "running":
		return collectRunningDockerStatus(runtimePath, routerAPIURL)
	case "exited":
		return exitedContainerStatus()
	case "not found":
		if status, ok := collectDirectStatus(runtimePath, routerAPIURL); ok {
			return status
		}
		return baseSystemStatus()
	default:
		return unknownContainerStatus(containerStatus)
	}
}

func collectSplitManagedHostStatus(runtimePath, routerAPIURL string) (SystemStatus, bool) {
	if !managedRuntimeUsesSplitContainers() {
		return SystemStatus{}, false
	}

	switch managedStatus := managedRuntimeContainerStatus(); managedStatus {
	case "running", "exited":
		return collectManagedDockerStatus(runtimePath, routerAPIURL), true
	case "not found":
		return SystemStatus{}, false
	default:
		return unknownContainerStatus(managedStatus), true
	}
}

func collectManagedDockerStatus(runtimePath, routerAPIURL string) SystemStatus {
	status := baseSystemStatus()
	status.DeploymentType = "docker"
	status.Overall = "healthy"
	status.Endpoints = []string{"http://localhost:8899"}

	routerLogContent := getContainerLogsTailForContainer(managedContainerNameForService("router"), 500)
	routerHealthy, routerMsg := resolveManagedRouterStatus(routerAPIURL, routerLogContent)
	envoyHealthy, envoyMsg := resolveManagedEnvoyStatus()
	dashboardHealthy, dashboardMsg := resolveManagedDashboardStatus()

	status.RouterRuntime = resolveRouterRuntimeStatus(runtimePath, routerAPIURL, routerHealthy)
	routerMsg = applyRuntimeMessage(routerMsg, status.RouterRuntime)
	status.Models = fetchModelsWhenReady(routerAPIURL, routerHealthy)
	status.Services = append(status.Services,
		buildServiceStatus("Router", boolToStatus(routerHealthy), routerHealthy, routerMsg, "container"),
		buildServiceStatus("Envoy", boolToStatus(envoyHealthy), envoyHealthy, envoyMsg, "container"),
		buildServiceStatus("Dashboard", boolToStatus(dashboardHealthy), dashboardHealthy, dashboardMsg, "container"),
	)
	setManagedDockerOverall(&status, routerHealthy, envoyHealthy, dashboardHealthy)

	return status
}

func collectRunningDockerStatus(runtimePath, routerAPIURL string) SystemStatus {
	status := baseSystemStatus()
	status.DeploymentType = "docker"
	status.Overall = "healthy"
	status.Endpoints = []string{"http://localhost:8899"}

	logContent := getContainerLogsTail(500)
	routerHealthy, routerMsg := checkServiceInLogContent("router", logContent)
	envoyHealthy, envoyMsg := checkServiceInLogContent("envoy", logContent)
	dashboardHealthy, dashboardMsg := checkServiceInLogContent("dashboard", logContent)

	status.RouterRuntime = resolveRouterRuntimeStatus(runtimePath, routerAPIURL, routerHealthy)
	routerMsg = applyRuntimeMessage(routerMsg, status.RouterRuntime)
	status.Models = fetchModelsWhenReady(routerAPIURL, routerHealthy)
	status.Services = append(status.Services,
		buildServiceStatus("Router", boolToStatus(routerHealthy), routerHealthy, routerMsg, "container"),
		buildServiceStatus("Envoy", boolToStatus(envoyHealthy), envoyHealthy, envoyMsg, "container"),
		buildServiceStatus("Dashboard", boolToStatus(dashboardHealthy), dashboardHealthy, dashboardMsg, "container"),
	)
	setDegradedWhenUnhealthy(&status, routerHealthy, envoyHealthy, dashboardHealthy)

	return status
}

func exitedContainerStatus() SystemStatus {
	status := baseSystemStatus()
	status.DeploymentType = "docker"
	status.Overall = "stopped"
	status.Services = append(status.Services, ServiceStatus{
		Name:    "vllm-sr-container",
		Status:  "exited",
		Healthy: false,
		Message: "Container exited. Check logs with: vllm-sr logs router",
	})
	return status
}

func unknownContainerStatus(containerStatus string) SystemStatus {
	status := baseSystemStatus()
	status.DeploymentType = "docker"
	status.Overall = containerStatus
	status.Services = append(status.Services, ServiceStatus{
		Name:    "vllm-sr-container",
		Status:  containerStatus,
		Healthy: false,
	})
	return status
}

func collectDirectStatus(runtimePath, routerAPIURL string) (SystemStatus, bool) {
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
	status.RouterRuntime = resolveRouterRuntimeStatus(runtimePath, routerAPIURL, routerHealthy)
	routerMsg = applyRuntimeMessage(routerMsg, status.RouterRuntime)
	status.Models = fetchModelsWhenReady(routerAPIURL, true)
	status.Services = append(status.Services, buildServiceStatus("Router", "running", true, routerMsg, "process"))

	appendDirectEnvoyStatus(&status)
	status.Services = append(status.Services, buildServiceStatus("Dashboard", "running", true, "Running", "process"))

	return status, true
}

func appendDirectEnvoyStatus(status *SystemStatus) {
	envoyRunning, envoyHealthy, envoyMsg := checkEnvoyHealth("http://localhost:8801/ready")
	if !envoyRunning {
		return
	}

	status.Services = append(status.Services, buildServiceStatus("Envoy", boolToStatus(envoyHealthy), envoyHealthy, envoyMsg, "proxy"))
	if !envoyHealthy {
		status.Overall = "degraded"
	}
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

func resolveManagedEnvoyStatus() (bool, string) {
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

func fetchModelsWhenReady(routerAPIURL string, routerHealthy bool) *RouterModelsInfo {
	if !routerHealthy {
		return nil
	}

	return fetchRouterModelsInfo(routerAPIURL)
}
