package handlers

import "testing"

func TestManagedContainerNameDefaultsToSplitRuntimeContainers(t *testing.T) {
	t.Setenv("VLLM_SR_STACK_NAME", "")
	t.Setenv(routerContainerNameEnv, "")
	t.Setenv(envoyContainerNameEnv, "")
	t.Setenv(dashboardContainerNameEnv, "")

	if got := ManagedContainerName("router"); got != defaultRouterContainerName {
		t.Fatalf("router container default = %q, want %q", got, defaultRouterContainerName)
	}
	if got := ManagedContainerName("envoy"); got != defaultEnvoyContainerName {
		t.Fatalf("envoy container default = %q, want %q", got, defaultEnvoyContainerName)
	}
	if got := ManagedContainerName("dashboard"); got != defaultDashboardContainerName {
		t.Fatalf("dashboard container default = %q, want %q", got, defaultDashboardContainerName)
	}
}

func TestManagedContainerNameUsesExplicitEnvOverrides(t *testing.T) {
	t.Setenv(routerContainerNameEnv, "lane-a-vllm-sr-router-container")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv(dashboardContainerNameEnv, "lane-a-vllm-sr-dashboard-container")

	if got := ManagedContainerName("router"); got != "lane-a-vllm-sr-router-container" {
		t.Fatalf("router container = %q", got)
	}
	if got := ManagedContainerName("envoy"); got != "lane-a-vllm-sr-envoy-container" {
		t.Fatalf("envoy container = %q", got)
	}
	if got := ManagedContainerName("dashboard"); got != "lane-a-vllm-sr-dashboard-container" {
		t.Fatalf("dashboard container = %q", got)
	}
}

// Without an exported container name, the default is scoped to the stack the
// way the CLI names it: the normalized stack name prefixes every container
// except on the default stack.
func TestManagedContainerNameScopesDefaultsToTheStack(t *testing.T) {
	cases := []struct {
		stack, router, envoy, dashboard string
	}{
		{"", defaultRouterContainerName, defaultEnvoyContainerName, defaultDashboardContainerName},
		{"vllm-sr", defaultRouterContainerName, defaultEnvoyContainerName, defaultDashboardContainerName},
		{"lane-a", "lane-a-vllm-sr-router-container", "lane-a-vllm-sr-envoy-container", "lane-a-vllm-sr-dashboard-container"},
		{" lab a ", "lab-a-vllm-sr-router-container", "lab-a-vllm-sr-envoy-container", "lab-a-vllm-sr-dashboard-container"},
	}
	for _, tc := range cases {
		t.Run(tc.stack, func(t *testing.T) {
			t.Setenv("VLLM_SR_STACK_NAME", tc.stack)
			t.Setenv(routerContainerNameEnv, "")
			t.Setenv(envoyContainerNameEnv, "")
			t.Setenv(dashboardContainerNameEnv, "")

			for service, want := range map[string]string{"router": tc.router, "envoy": tc.envoy, "dashboard": tc.dashboard} {
				if got := ManagedContainerName(service); got != want {
					t.Errorf("%s container = %q, want %q", service, got, want)
				}
			}
		})
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
