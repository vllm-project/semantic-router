package handlers

import (
	"os"
	"regexp"
	"strings"
)

const (
	routerContainerNameEnv    = "VLLM_SR_ROUTER_CONTAINER_NAME"
	envoyContainerNameEnv     = "VLLM_SR_ENVOY_CONTAINER_NAME"
	dashboardContainerNameEnv = "VLLM_SR_DASHBOARD_CONTAINER_NAME"

	defaultRouterContainerName    = "vllm-sr-router-container"
	defaultEnvoyContainerName     = "vllm-sr-envoy-container"
	defaultDashboardContainerName = "vllm-sr-dashboard-container"
)

func managedContainerNameForService(service string) string {
	switch service {
	case "router":
		return envOrDefaultTrimmed(routerContainerNameEnv, defaultRouterContainerName)
	case "envoy":
		return envOrDefaultTrimmed(envoyContainerNameEnv, defaultEnvoyContainerName)
	case "dashboard":
		return managedDashboardContainerName()
	default:
		return defaultRouterContainerName
	}
}

func managedContainerNameForStorage(backend string) string {
	stack := normalizedManagedStackName(os.Getenv("VLLM_SR_STACK_NAME"))
	if stack == "" || stack == "vllm-sr" {
		return "vllm-sr-" + backend
	}
	return stack + "-vllm-sr-" + backend
}

func normalizedManagedStackName(raw string) string {
	value := strings.TrimSpace(raw)
	if value == "" {
		return "vllm-sr"
	}
	value = regexp.MustCompile(`[^A-Za-z0-9_.-]+`).ReplaceAllString(value, "-")
	return strings.Trim(value, "._-")
}

func managedDashboardContainerName() string {
	return envOrDefaultTrimmed(dashboardContainerNameEnv, defaultDashboardContainerName)
}

// managedStackRunsEnvoy reports whether an Envoy container serves the
// listeners in front of the Router. The CLI sets VLLM_SR_GATEWAY on every
// container of the stack; in a standalone stack the Router serves them and no
// Envoy container exists. Stacks from earlier releases always ran Envoy.
func managedStackRunsEnvoy() bool {
	return strings.TrimSpace(os.Getenv("VLLM_SR_GATEWAY")) != "standalone"
}

func managedRuntimeUsesSplitContainers() bool {
	dashboardContainer := managedDashboardContainerName()
	return managedContainerNameForService("router") != dashboardContainer ||
		managedContainerNameForService("envoy") != dashboardContainer
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

func envOrDefaultTrimmed(key string, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}
