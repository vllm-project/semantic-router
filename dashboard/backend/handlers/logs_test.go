package handlers

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRuntimeContainerStatusForLogsPrefersSplitManagedRuntimeOverLegacyContainerResidue(t *testing.T) {
	dockerPath := writeFakeStatusDockerCLI(t)
	t.Setenv("PATH", filepath.Dir(dockerPath)+":"+os.Getenv("PATH"))

	t.Setenv(routerContainerNameEnv, "lane-a-vllm-sr-router-container")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv(dashboardContainerNameEnv, "lane-a-vllm-sr-dashboard-container")

	t.Setenv("TEST_LEGACY_CONTAINER", vllmSrContainerName)
	t.Setenv("TEST_LEGACY_STATUS", "exited")
	t.Setenv("TEST_ROUTER_CONTAINER", "lane-a-vllm-sr-router-container")
	t.Setenv("TEST_ROUTER_STATUS", "running")
	t.Setenv("TEST_ENVOY_CONTAINER", "lane-a-vllm-sr-envoy-container")
	t.Setenv("TEST_ENVOY_STATUS", "running")
	t.Setenv("TEST_DASHBOARD_CONTAINER", "lane-a-vllm-sr-dashboard-container")
	t.Setenv("TEST_DASHBOARD_STATUS", "running")

	if got := runtimeContainerStatusForLogs(); got != "running" {
		t.Fatalf("runtimeContainerStatusForLogs() = %q, want running", got)
	}
}
