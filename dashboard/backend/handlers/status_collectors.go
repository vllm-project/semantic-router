package handlers

func collectInContainerStatus(runtimePath, routerAPIURL string) SystemStatus {
	status := baseSystemStatus()
	status.DeploymentType = "docker"
	status.Overall = "healthy"
	status.Endpoints = []string{"http://localhost:8899"}

	routerHealthy, routerMsg := checkServiceFromContainerLogs("router")
	envoyHealthy, envoyMsg := checkServiceFromContainerLogs("envoy")
	dashboardHealthy := true
	dashboardMsg := "Running"

	status.RouterRuntime = resolveRouterRuntimeStatus(runtimePath, routerAPIURL, routerHealthy, readRouterLogContentInContainer())
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

func collectRunningDockerStatus(runtimePath, routerAPIURL string) SystemStatus {
	status := baseSystemStatus()
	status.DeploymentType = "docker"
	status.Overall = "healthy"
	status.Endpoints = []string{"http://localhost:8899"}

	logContent := getContainerLogsTail(500)
	routerHealthy, routerMsg := checkServiceInLogContent("router", logContent)
	envoyHealthy, envoyMsg := checkServiceInLogContent("envoy", logContent)
	dashboardHealthy, dashboardMsg := checkServiceInLogContent("dashboard", logContent)

	status.RouterRuntime = resolveRouterRuntimeStatus(runtimePath, routerAPIURL, routerHealthy, logContent)
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
	status.RouterRuntime = resolveRouterRuntimeStatus(runtimePath, routerAPIURL, routerHealthy, "")
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
