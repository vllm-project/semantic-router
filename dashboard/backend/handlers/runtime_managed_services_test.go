package handlers

import "testing"

func TestManagedContainerNameForServiceDefaultsToLegacyRuntimeContainer(t *testing.T) {
	t.Setenv(routerContainerNameEnv, "")
	t.Setenv(envoyContainerNameEnv, "")
	t.Setenv(dashboardContainerNameEnv, "")

	if got := managedContainerNameForService("router"); got != vllmSrContainerName {
		t.Fatalf("router container default = %q, want %q", got, vllmSrContainerName)
	}
	if got := managedContainerNameForService("envoy"); got != vllmSrContainerName {
		t.Fatalf("envoy container default = %q, want %q", got, vllmSrContainerName)
	}
	if got := managedContainerNameForService("dashboard"); got != vllmSrContainerName {
		t.Fatalf("dashboard container default = %q, want %q", got, vllmSrContainerName)
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
	if got := managedRuntimeSyncContainerName(); got != "lane-a-vllm-sr-dashboard-container" {
		t.Fatalf("runtime sync container = %q", got)
	}
}

func TestManagedRuntimeUsesSplitContainersDefaultsToFalse(t *testing.T) {
	t.Setenv(routerContainerNameEnv, "")
	t.Setenv(envoyContainerNameEnv, "")
	t.Setenv(dashboardContainerNameEnv, "")

	if managedRuntimeUsesSplitContainers() {
		t.Fatal("expected legacy single-container runtime by default")
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

func TestManagedContainerNamesForComponentAllDeduplicatesLegacyContainer(t *testing.T) {
	t.Setenv(routerContainerNameEnv, "")
	t.Setenv(envoyContainerNameEnv, "")
	t.Setenv(dashboardContainerNameEnv, "")

	got := managedContainerNamesForComponent("all")
	if len(got) != 1 || got[0] != vllmSrContainerName {
		t.Fatalf("all managed containers = %#v", got)
	}
}

func TestManagedContainerNamesForComponentAllKeepsSplitRoleOrder(t *testing.T) {
	t.Setenv(routerContainerNameEnv, "lane-a-vllm-sr-router-container")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv(dashboardContainerNameEnv, "lane-a-vllm-sr-dashboard-container")

	got := managedContainerNamesForComponent("all")
	want := []string{
		"lane-a-vllm-sr-router-container",
		"lane-a-vllm-sr-envoy-container",
		"lane-a-vllm-sr-dashboard-container",
	}

	if len(got) != len(want) {
		t.Fatalf("managed container count = %d, want %d (%#v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("managed containers[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestManagedServiceForContainerNameResolvesRole(t *testing.T) {
	t.Setenv(routerContainerNameEnv, "lane-a-vllm-sr-router-container")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv(dashboardContainerNameEnv, "lane-a-vllm-sr-dashboard-container")

	if got := managedServiceForContainerName("lane-a-vllm-sr-router-container"); got != "router" {
		t.Fatalf("router role = %q", got)
	}
	if got := managedServiceForContainerName("lane-a-vllm-sr-envoy-container"); got != "envoy" {
		t.Fatalf("envoy role = %q", got)
	}
	if got := managedServiceForContainerName("lane-a-vllm-sr-dashboard-container"); got != "dashboard" {
		t.Fatalf("dashboard role = %q", got)
	}
	if got := managedServiceForContainerName("unknown"); got != "" {
		t.Fatalf("unknown role = %q", got)
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
