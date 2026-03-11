package handlers

import (
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"
)

// ServiceStatus represents the status of a single service
type ServiceStatus struct {
	Name      string `json:"name"`
	Status    string `json:"status"`
	Healthy   bool   `json:"healthy"`
	Message   string `json:"message,omitempty"`
	Component string `json:"component,omitempty"`
}

// RouterRuntimeStatus captures router startup progress beyond process-level health.
type RouterRuntimeStatus struct {
	Phase            string   `json:"phase"`
	Ready            bool     `json:"ready"`
	Message          string   `json:"message,omitempty"`
	DownloadingModel string   `json:"downloading_model,omitempty"`
	PendingModels    []string `json:"pending_models,omitempty"`
	ReadyModels      int      `json:"ready_models,omitempty"`
	TotalModels      int      `json:"total_models,omitempty"`
}

// SystemStatus represents the overall system status
type SystemStatus struct {
	Overall        string               `json:"overall"`
	DeploymentType string               `json:"deployment_type"`
	Services       []ServiceStatus      `json:"services"`
	RouterRuntime  *RouterRuntimeStatus `json:"router_runtime,omitempty"`
	Models         *RouterModelsInfo    `json:"models,omitempty"`
	Endpoints      []string             `json:"endpoints,omitempty"`
	Version        string               `json:"version,omitempty"`
}

// vllmSrContainerName is the container name used by the Python vllm-sr CLI
const vllmSrContainerName = "vllm-sr-container"

// StatusHandler returns the status of vLLM-SR services
// Aligns with the vllm-sr Python CLI by using the same Docker-based detection
func StatusHandler(routerAPIURL, configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		status := detectSystemStatus(routerAPIURL, configDir)

		if err := json.NewEncoder(w).Encode(status); err != nil {
			http.Error(w, `{"error":"Failed to encode response"}`, http.StatusInternalServerError)
			return
		}
	}
}

// getDockerContainerStatus checks the status of a Docker container
// Returns: "running", "exited", "not found", or other Docker status
func getDockerContainerStatus(containerName string) string {
	cmd := exec.Command("docker", "inspect", "-f", "{{.State.Status}}", containerName)
	output, err := cmd.Output()
	if err != nil {
		return "not found"
	}
	return strings.TrimSpace(string(output))
}

// isRunningInContainer checks if the current process is running inside a Docker container
func isRunningInContainer() bool {
	// Check for /.dockerenv file (common indicator)
	if _, err := os.Stat("/.dockerenv"); err == nil {
		return true
	}

	// Check /proc/1/cgroup for docker/containerd
	data, err := os.ReadFile("/proc/1/cgroup")
	if err == nil {
		content := string(data)
		if strings.Contains(content, "docker") || strings.Contains(content, "containerd") {
			return true
		}
	}

	return false
}

// checkServiceFromContainerLogs checks service status from supervisorctl within the same container
func checkServiceFromContainerLogs(service string) (bool, string) {
	// Use supervisorctl to check service status
	cmd := exec.Command("supervisorctl", "status", service)
	output, err := cmd.CombinedOutput()
	if err != nil {
		// If supervisorctl fails, service might not be configured
		return false, "Status unknown"
	}

	outputStr := string(output)

	// Parse supervisorctl output
	// Format: "service_name  RUNNING   pid 123, uptime 0:01:23"
	// or:     "service_name  STOPPED   Not started"
	// or:     "service_name  FATAL     Exited too quickly"

	if strings.Contains(outputStr, "RUNNING") {
		return true, "Running"
	} else if strings.Contains(outputStr, "STOPPED") {
		return false, "Stopped"
	} else if strings.Contains(outputStr, "FATAL") || strings.Contains(outputStr, "EXITED") {
		return false, "Failed"
	} else if strings.Contains(outputStr, "STARTING") {
		return false, "Starting"
	}

	return false, "Status unknown"
}

// boolToStatus converts a boolean to a status string
func boolToStatus(healthy bool) string {
	if healthy {
		return "running"
	}
	return "unknown"
}

// checkHTTPHealth performs an HTTP health check
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

// checkEnvoyHealth checks if Envoy is running and healthy
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

	// Envoy is running if we got ANY response
	isRunning := true

	// Healthy only if 200
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return isRunning, true, "Ready"
	}

	// Running but not healthy (e.g., 503 "no healthy upstream")
	return isRunning, false, "Running (upstream not ready)"
}
