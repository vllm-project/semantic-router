package handlers

import (
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"
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

// isRunningInContainer checks if the current process is running inside a Docker container.
func isRunningInContainer() bool {
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

// checkServiceFromContainerLogs checks service status from supervisorctl within the same container.
func checkServiceFromContainerLogs(service string) (bool, string) {
	cmd := exec.Command("supervisorctl", "status", service)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return false, "Status unknown"
	}

	outputStr := string(output)
	switch {
	case strings.Contains(outputStr, "RUNNING"):
		return true, "Running"
	case strings.Contains(outputStr, "STOPPED"):
		return false, "Stopped"
	case strings.Contains(outputStr, "FATAL"), strings.Contains(outputStr, "EXITED"):
		return false, "Failed"
	case strings.Contains(outputStr, "STARTING"):
		return false, "Starting"
	default:
		return false, "Status unknown"
	}
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
