package handlers

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type fakeLifecycleDocker struct {
	path             string
	logPath          string
	routerStatusPath string
	envoyStatusPath  string
}

func writeFakeLifecycleDockerCLI(t *testing.T) fakeLifecycleDocker {
	t.Helper()

	tempDir := t.TempDir()
	dockerPath := filepath.Join(tempDir, "docker")
	logPath := filepath.Join(tempDir, "docker.log")
	routerStatusPath := filepath.Join(tempDir, "router.status")
	envoyStatusPath := filepath.Join(tempDir, "envoy.status")

	script := `#!/bin/sh
status_file_for() {
  container="$1"
  case "$container" in
    "$TEST_ROUTER_CONTAINER") printf "%s" "$TEST_ROUTER_STATUS_FILE" ;;
    "$TEST_ENVOY_CONTAINER") printf "%s" "$TEST_ENVOY_STATUS_FILE" ;;
    "$TEST_DASHBOARD_CONTAINER") printf "%s" "$TEST_DASHBOARD_STATUS_FILE" ;;
    *) printf "" ;;
  esac
}

if [ -n "$TEST_DOCKER_LOG_FILE" ]; then
  printf "%s\n" "$*" >> "$TEST_DOCKER_LOG_FILE"
fi

case "$1" in
  inspect)
    status_file=$(status_file_for "$4")
    if [ -z "$status_file" ] || [ ! -f "$status_file" ]; then
      exit 1
    fi
    cat "$status_file"
    exit 0
    ;;
  start|restart)
    status_file=$(status_file_for "$2")
    if [ -z "$status_file" ]; then
      exit 1
    fi
    printf "running\n" > "$status_file"
    printf "%s\n" "$2"
    exit 0
    ;;
esac

exit 1
`

	if err := os.WriteFile(dockerPath, []byte(script), 0o755); err != nil {
		t.Fatalf("failed to write fake docker CLI: %v", err)
	}

	return fakeLifecycleDocker{
		path:             dockerPath,
		logPath:          logPath,
		routerStatusPath: routerStatusPath,
		envoyStatusPath:  envoyStatusPath,
	}
}

func TestRestartManagedServiceUsesDockerStartForCreatedSplitContainers(t *testing.T) {
	fakeDocker := writeFakeLifecycleDockerCLI(t)
	t.Setenv("PATH", filepath.Dir(fakeDocker.path)+":"+os.Getenv("PATH"))
	t.Setenv(routerContainerNameEnv, "lane-a-vllm-sr-router-container")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv(dashboardContainerNameEnv, "lane-a-vllm-sr-dashboard-container")
	t.Setenv("TEST_DOCKER_LOG_FILE", fakeDocker.logPath)
	t.Setenv("TEST_ROUTER_CONTAINER", "lane-a-vllm-sr-router-container")
	t.Setenv("TEST_ROUTER_STATUS_FILE", fakeDocker.routerStatusPath)

	if err := os.WriteFile(fakeDocker.routerStatusPath, []byte("created\n"), 0o644); err != nil {
		t.Fatalf("failed to seed router status: %v", err)
	}

	if err := restartManagedService("router", 2*time.Second); err != nil {
		t.Fatalf("restartManagedService(router) failed: %v", err)
	}

	logData, err := os.ReadFile(fakeDocker.logPath)
	if err != nil {
		t.Fatalf("failed to read docker log: %v", err)
	}
	logText := string(logData)
	if !strings.Contains(logText, "start lane-a-vllm-sr-router-container") {
		t.Fatalf("expected docker start for router, got %q", logText)
	}
	if strings.Contains(logText, "supervisorctl") {
		t.Fatalf("split runtime should not use supervisorctl, got %q", logText)
	}
}

func TestRestartManagedServiceUsesDockerRestartForRunningSplitContainers(t *testing.T) {
	fakeDocker := writeFakeLifecycleDockerCLI(t)
	t.Setenv("PATH", filepath.Dir(fakeDocker.path)+":"+os.Getenv("PATH"))
	t.Setenv(routerContainerNameEnv, "lane-a-vllm-sr-router-container")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv(dashboardContainerNameEnv, "lane-a-vllm-sr-dashboard-container")
	t.Setenv("TEST_DOCKER_LOG_FILE", fakeDocker.logPath)
	t.Setenv("TEST_ENVOY_CONTAINER", "lane-a-vllm-sr-envoy-container")
	t.Setenv("TEST_ENVOY_STATUS_FILE", fakeDocker.envoyStatusPath)

	if err := os.WriteFile(fakeDocker.envoyStatusPath, []byte("running\n"), 0o644); err != nil {
		t.Fatalf("failed to seed envoy status: %v", err)
	}

	if err := restartManagedService("envoy", 2*time.Second); err != nil {
		t.Fatalf("restartManagedService(envoy) failed: %v", err)
	}

	logData, err := os.ReadFile(fakeDocker.logPath)
	if err != nil {
		t.Fatalf("failed to read docker log: %v", err)
	}
	logText := string(logData)
	if !strings.Contains(logText, "restart lane-a-vllm-sr-envoy-container") {
		t.Fatalf("expected docker restart for envoy, got %q", logText)
	}
	if strings.Contains(logText, "supervisorctl") {
		t.Fatalf("split runtime should not use supervisorctl, got %q", logText)
	}
}
