package handlers

import (
	"os"
	"strings"
)

const (
	routerContainerNameEnv    = "VLLM_SR_ROUTER_CONTAINER_NAME"
	envoyContainerNameEnv     = "VLLM_SR_ENVOY_CONTAINER_NAME"
	dashboardContainerNameEnv = "VLLM_SR_DASHBOARD_CONTAINER_NAME"
)

func managedContainerNameForService(service string) string {
	switch service {
	case "router":
		return envOrDefaultTrimmed(routerContainerNameEnv, vllmSrContainerName)
	case "envoy":
		return envOrDefaultTrimmed(envoyContainerNameEnv, vllmSrContainerName)
	case "dashboard":
		return managedDashboardContainerName()
	default:
		return vllmSrContainerName
	}
}

func managedDashboardContainerName() string {
	return envOrDefaultTrimmed(dashboardContainerNameEnv, vllmSrContainerName)
}

func managedRuntimeSyncContainerName() string {
	return managedDashboardContainerName()
}

func managedServiceCanBeControlledLocally(service string) bool {
	return isRunningInContainer() &&
		managedContainerNameForService(service) == managedDashboardContainerName()
}

func managedRuntimeUsesSplitContainers() bool {
	dashboardContainer := managedDashboardContainerName()
	return managedContainerNameForService("router") != dashboardContainer ||
		managedContainerNameForService("envoy") != dashboardContainer
}

func managedContainerNamesForComponent(component string) []string {
	switch component {
	case "router", "envoy", "dashboard":
		return uniqueNonEmptyStrings([]string{managedContainerNameForService(component)})
	case "all":
		return uniqueNonEmptyStrings([]string{
			managedContainerNameForService("router"),
			managedContainerNameForService("envoy"),
			managedContainerNameForService("dashboard"),
		})
	default:
		return uniqueNonEmptyStrings([]string{managedContainerNameForService(component)})
	}
}

func managedServiceForContainerName(containerName string) string {
	switch containerName {
	case managedContainerNameForService("router"):
		return "router"
	case managedContainerNameForService("envoy"):
		return "envoy"
	case managedContainerNameForService("dashboard"):
		return "dashboard"
	default:
		return ""
	}
}

func managedRuntimeContainerStatus() string {
	fallbackStatus := ""
	for _, containerName := range managedContainerNamesForComponent("all") {
		switch status := getDockerContainerStatus(containerName); status {
		case "running":
			return status
		case "not found":
			continue
		case "exited":
			if fallbackStatus == "" {
				fallbackStatus = status
			}
		default:
			if fallbackStatus == "" {
				fallbackStatus = status
			}
		}
	}

	if fallbackStatus != "" {
		return fallbackStatus
	}
	return "not found"
}

func managedEnvoyReadyURL() string {
	if candidate := strings.TrimSpace(os.Getenv("TARGET_ENVOY_ADMIN_URL")); candidate != "" {
		return strings.TrimRight(candidate, "/") + "/ready"
	}

	if candidate := strings.TrimSpace(os.Getenv("TARGET_ENVOY_URL")); candidate != "" {
		return strings.TrimRight(candidate, "/") + "/ready"
	}

	if managedRuntimeUsesSplitContainers() && isRunningInContainer() {
		return ""
	}

	return "http://localhost:8801/ready"
}

func uniqueNonEmptyStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	return result
}

func envOrDefaultTrimmed(key string, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}
