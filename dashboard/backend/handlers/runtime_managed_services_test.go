package handlers

import "testing"

func TestManagedContainerNameForServiceDefaultsToSplitRuntimeContainers(t *testing.T) {
	t.Setenv(routerContainerNameEnv, "")
	t.Setenv(envoyContainerNameEnv, "")
	t.Setenv(dashboardContainerNameEnv, "")

	if got := managedContainerNameForService("router"); got != defaultRouterContainerName {
		t.Fatalf("router container default = %q, want %q", got, defaultRouterContainerName)
	}
	if got := managedContainerNameForService("envoy"); got != defaultEnvoyContainerName {
		t.Fatalf("envoy container default = %q, want %q", got, defaultEnvoyContainerName)
	}
	if got := managedContainerNameForService("dashboard"); got != defaultDashboardContainerName {
		t.Fatalf("dashboard container default = %q, want %q", got, defaultDashboardContainerName)
	}
}

func TestManagedContainerNameForServiceUsesExplicitEnvOverrides(t *testing.T) {
	t.Setenv(routerContainerNameEnv, "lane-a-vllm-sr-router-container")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv(dashboardContainerNameEnv, "lane-a-vllm-sr-dashboard-container")

	if got := managedContainerNameForService("router"); got != "lane-a-vllm-sr-router-container" {
		t.Fatalf("router container = %q", got)
	}
	if got := managedContainerNameForService("envoy"); got != "lane-a-vllm-sr-envoy-container" {
		t.Fatalf("envoy container = %q", got)
	}
	if got := managedContainerNameForService("dashboard"); got != "lane-a-vllm-sr-dashboard-container" {
		t.Fatalf("dashboard container = %q", got)
	}
}

func TestManagedRuntimeUsesSplitContainersDefaultsToTrue(t *testing.T) {
	t.Setenv(routerContainerNameEnv, "")
	t.Setenv(envoyContainerNameEnv, "")
	t.Setenv(dashboardContainerNameEnv, "")

	if !managedRuntimeUsesSplitContainers() {
		t.Fatal("expected split runtime by default")
	}
}

func TestManagedRuntimeUsesSplitContainersDetectsRoleOverrides(t *testing.T) {
	t.Setenv(routerContainerNameEnv, "lane-a-vllm-sr-router-container")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv(dashboardContainerNameEnv, "lane-a-vllm-sr-dashboard-container")

	if !managedRuntimeUsesSplitContainers() {
		t.Fatal("expected split runtime when role container names differ")
	}
}

func TestManagedEnvoyReadyURLPrefersExplicitAdminURL(t *testing.T) {
	t.Setenv("TARGET_ENVOY_ADMIN_URL", "http://lane-a-vllm-sr-envoy-container:9901")
	t.Setenv("TARGET_ENVOY_URL", "http://lane-a-vllm-sr-envoy-container:8899")

	if got := managedEnvoyReadyURL(); got != "http://lane-a-vllm-sr-envoy-container:9901/ready" {
		t.Fatalf("envoy ready url = %q", got)
	}
}

func TestManagedEnvoyReadyURLFallsBackToListenerURL(t *testing.T) {
	t.Setenv("TARGET_ENVOY_ADMIN_URL", "")
	t.Setenv("TARGET_ENVOY_URL", "http://lane-a-vllm-sr-envoy-container:8899")

	if got := managedEnvoyReadyURL(); got != "http://lane-a-vllm-sr-envoy-container:8899/ready" {
		t.Fatalf("envoy ready url = %q", got)
	}
}
