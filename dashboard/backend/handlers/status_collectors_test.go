package handlers

import (
	"os"
	"path/filepath"
	"testing"
)

func writeFakeStatusDockerCLI(t *testing.T) string {
	t.Helper()

	tempDir := t.TempDir()
	dockerPath := filepath.Join(tempDir, "docker")
	script := `#!/bin/sh
if [ "$1" = "inspect" ] && [ "$2" = "-f" ]; then
  container="$4"
  case "$container" in
    "$TEST_LEGACY_CONTAINER") status="$TEST_LEGACY_STATUS" ;;
    "$TEST_ROUTER_CONTAINER") status="$TEST_ROUTER_STATUS" ;;
    "$TEST_ENVOY_CONTAINER") status="$TEST_ENVOY_STATUS" ;;
    "$TEST_DASHBOARD_CONTAINER") status="$TEST_DASHBOARD_STATUS" ;;
    *) status="" ;;
  esac
  if [ -n "$status" ]; then
    printf "%s\n" "$status"
    exit 0
  fi
  exit 1
fi

if [ "$1" = "logs" ] && [ "$2" = "--tail" ]; then
  container="$4"
  case "$container" in
    "$TEST_LEGACY_CONTAINER") printf "%s" "$TEST_LEGACY_LOG" ;;
    "$TEST_ROUTER_CONTAINER") printf "%s" "$TEST_ROUTER_LOG" ;;
    "$TEST_ENVOY_CONTAINER") printf "%s" "$TEST_ENVOY_LOG" ;;
    "$TEST_DASHBOARD_CONTAINER") printf "%s" "$TEST_DASHBOARD_LOG" ;;
  esac
  exit 0
fi

exit 1
`
	if err := os.WriteFile(dockerPath, []byte(script), 0o755); err != nil {
		t.Fatalf("failed to write fake docker CLI: %v", err)
	}
	return dockerPath
}

func TestCollectHostStatusPrefersSplitManagedRuntimeOverLegacyContainerResidue(t *testing.T) {
	dockerPath := writeFakeStatusDockerCLI(t)
	t.Setenv("PATH", filepath.Dir(dockerPath)+":"+os.Getenv("PATH"))

	t.Setenv(routerContainerNameEnv, "lane-a-vllm-sr-router-container")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv(dashboardContainerNameEnv, "lane-a-vllm-sr-dashboard-container")
	t.Setenv("TARGET_ENVOY_URL", "http://127.0.0.1:1")

	t.Setenv("TEST_LEGACY_CONTAINER", vllmSrContainerName)
	t.Setenv("TEST_LEGACY_STATUS", "exited")
	t.Setenv("TEST_ROUTER_CONTAINER", "lane-a-vllm-sr-router-container")
	t.Setenv("TEST_ROUTER_STATUS", "running")
	t.Setenv("TEST_ENVOY_CONTAINER", "lane-a-vllm-sr-envoy-container")
	t.Setenv("TEST_ENVOY_STATUS", "running")
	t.Setenv("TEST_DASHBOARD_CONTAINER", "lane-a-vllm-sr-dashboard-container")
	t.Setenv("TEST_DASHBOARD_STATUS", "running")

	status := collectHostStatus("", "")
	if status.Overall != "healthy" {
		t.Fatalf("overall status = %q, want healthy", status.Overall)
	}
	if len(status.Services) != 3 {
		t.Fatalf("service count = %d, want 3 (%#v)", len(status.Services), status.Services)
	}
	if status.Services[0].Name != "Router" {
		t.Fatalf("first service = %q, want Router", status.Services[0].Name)
	}
	for _, service := range status.Services {
		if service.Name == vllmSrContainerName {
			t.Fatalf("legacy container residue leaked into split status: %#v", status.Services)
		}
		if service.Status != "running" {
			t.Fatalf("service %q status = %q, want running", service.Name, service.Status)
		}
	}
}

func TestCollectHostStatusReportsStandbyForCreatedSplitServices(t *testing.T) {
	dockerPath := writeFakeStatusDockerCLI(t)
	t.Setenv("PATH", filepath.Dir(dockerPath)+":"+os.Getenv("PATH"))

	t.Setenv(routerContainerNameEnv, "lane-a-vllm-sr-router-container")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv(dashboardContainerNameEnv, "lane-a-vllm-sr-dashboard-container")
	t.Setenv("TARGET_ENVOY_URL", "http://127.0.0.1:1")

	t.Setenv("TEST_ROUTER_CONTAINER", "lane-a-vllm-sr-router-container")
	t.Setenv("TEST_ROUTER_STATUS", "created")
	t.Setenv("TEST_ENVOY_CONTAINER", "lane-a-vllm-sr-envoy-container")
	t.Setenv("TEST_ENVOY_STATUS", "created")
	t.Setenv("TEST_DASHBOARD_CONTAINER", "lane-a-vllm-sr-dashboard-container")
	t.Setenv("TEST_DASHBOARD_STATUS", "running")

	status := collectHostStatus("", "")
	if status.Overall != "degraded" {
		t.Fatalf("overall status = %q, want degraded", status.Overall)
	}
	if len(status.Services) != 3 {
		t.Fatalf("service count = %d, want 3 (%#v)", len(status.Services), status.Services)
	}
	if got := status.Services[0].Message; got != "Standby (setup mode)" {
		t.Fatalf("router message = %q, want standby", got)
	}
	if got := status.Services[1].Message; got != "Standby (setup mode)" {
		t.Fatalf("envoy message = %q, want standby", got)
	}
}
