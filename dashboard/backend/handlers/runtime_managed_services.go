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

// managedStackNamePattern matches the CLI's STACK_NAME_PATTERN.
var managedStackNamePattern = regexp.MustCompile(`[^A-Za-z0-9_.-]+`)

// ManagedContainerName returns the container name the CLI gives a stack
// service: the name it exports into every stack container, else the default
// name scoped to the stack the way the CLI scopes it.
func ManagedContainerName(service string) string {
	switch service {
	case "router":
		return envOrDefaultTrimmed(routerContainerNameEnv, stackScopedContainerName(defaultRouterContainerName))
	case "envoy":
		return envOrDefaultTrimmed(envoyContainerNameEnv, stackScopedContainerName(defaultEnvoyContainerName))
	case "dashboard":
		return managedDashboardContainerName()
	default:
		return defaultRouterContainerName
	}
}

func managedContainerNameForStorage(backend string) string {
	return stackScopedContainerName("vllm-sr-" + backend)
}

// stackScopedContainerName prefixes a default container name with the
// normalized stack name, except on the default stack.
func stackScopedContainerName(name string) string {
	stack := normalizedManagedStackName(os.Getenv("VLLM_SR_STACK_NAME"))
	if stack == "" || stack == "vllm-sr" {
		return name
	}
	return stack + "-" + name
}

func normalizedManagedStackName(raw string) string {
	value := strings.TrimSpace(raw)
	if value == "" {
		return "vllm-sr"
	}
	value = managedStackNamePattern.ReplaceAllString(value, "-")
	return strings.Trim(value, "._-")
}

func managedDashboardContainerName() string {
	return envOrDefaultTrimmed(dashboardContainerNameEnv, stackScopedContainerName(defaultDashboardContainerName))
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
	return ManagedContainerName("router") != dashboardContainer ||
		ManagedContainerName("envoy") != dashboardContainer
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
