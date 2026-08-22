package handlers

import (
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

// getDockerContainerStatus checks the status of a Docker container.
// Returns: "running", "exited", "not found", or other Docker status.
func getDockerContainerStatus(containerName string) string {
	cmd := exec.Command("docker", "inspect", "-f", "{{.State.Status}}", containerName)
	output, err := cmd.Output()
	if err != nil {
		return "not found"
	}
	return strings.TrimSpace(string(output))
}

var isRunningInContainer = detectRunningInContainer

// detectRunningInContainer checks if the current process is running inside a Docker container.
func detectRunningInContainer() bool {
	if _, err := os.Stat("/.dockerenv"); err == nil {
		return true
	}

	data, err := os.ReadFile("/proc/1/cgroup")
	if err == nil {
		content := string(data)
		if strings.Contains(content, "docker") || strings.Contains(content, "containerd") {
			return true
		}
	}

	return false
}

func boolToStatus(healthy bool) string {
	if healthy {
		return "running"
	}
	return "unknown"
}

func checkHTTPHealth(url string) (bool, string) {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false, ""
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return true, "HTTP health check OK"
	}
	return false, ""
}

func checkRouterManagementHealth(url string, credentialProvider ...routerauth.CredentialProvider) bool {
	resp, err := routerManagementGET(url, 2*time.Second, credentialProvider...)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}

// checkEnvoyHealth checks if Envoy is running and healthy.
// Returns: (isRunning, isHealthy, message)
func checkEnvoyHealth(url string) (bool, bool, string) {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false, false, ""
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	isRunning := true
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return isRunning, true, "Ready"
	}

	return isRunning, false, "Running (upstream not ready)"
}
