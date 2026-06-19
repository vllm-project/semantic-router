package handlers

import (
	"net"
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

func TestCollectHostStatusUsesSplitManagedRuntime(t *testing.T) {
	dockerPath := writeFakeStatusDockerCLI(t)
	t.Setenv("PATH", filepath.Dir(dockerPath)+":"+os.Getenv("PATH"))

	t.Setenv(routerContainerNameEnv, "lane-a-vllm-sr-router-container")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv(dashboardContainerNameEnv, "lane-a-vllm-sr-dashboard-container")
	t.Setenv("TARGET_ENVOY_URL", "http://127.0.0.1:1")

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

func TestCollectHostStatusReportsDashboardWhenRouterIsUnavailable(t *testing.T) {
	t.Setenv(routerContainerNameEnv, "vllm-sr-runtime-container")
	t.Setenv(envoyContainerNameEnv, "vllm-sr-runtime-container")
	t.Setenv(dashboardContainerNameEnv, "vllm-sr-runtime-container")

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen on unused port: %v", err)
	}
	routerAPIURL := "http://" + listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatalf("close listener: %v", err)
	}

	status := collectHostStatus("", routerAPIURL)
	if status.Overall != "not_running" {
		t.Fatalf("overall status = %q, want not_running", status.Overall)
	}
	if len(status.Services) != 2 {
		t.Fatalf("service count = %d, want 2 (%#v)", len(status.Services), status.Services)
	}

	router := status.Services[0]
	if router.Name != "Router" || router.Healthy || router.Status != "not running" {
		t.Fatalf("router service = %#v, want Router not running and unhealthy", router)
	}

	dashboard := status.Services[1]
	if dashboard.Name != "Dashboard" || !dashboard.Healthy || dashboard.Status != "running" {
		t.Fatalf("dashboard service = %#v, want Dashboard running and healthy", dashboard)
	}
}
