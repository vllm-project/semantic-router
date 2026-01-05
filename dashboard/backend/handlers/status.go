package handlers

import (
	"encoding/json"
	"net/http"
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

// SystemStatus represents the overall system status
type SystemStatus struct {
	Overall        string          `json:"overall"`
	DeploymentType string          `json:"deployment_type"`
	Services       []ServiceStatus `json:"services"`
	Endpoints      []string        `json:"endpoints,omitempty"`
	Version        string          `json:"version,omitempty"`
}

// vllmSrContainerName is the container name used by the Python vllm-sr CLI
const vllmSrContainerName = "vllm-sr-container"

// StatusHandler returns the status of vLLM-SR services
// Aligns with the vllm-sr Python CLI by using the same Docker-based detection
func StatusHandler(routerAPIURL string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "application/json")

		status := SystemStatus{
			Overall:        "not_running",
			DeploymentType: "none",
			Services:       []ServiceStatus{},
			Version:        "v0.1.0",
		}

		// Check for vllm-sr Docker container (same as vllm-sr Python CLI)
		containerStatus := getDockerContainerStatus(vllmSrContainerName)

		switch containerStatus {
		case "running":
			status.DeploymentType = "docker"
			status.Overall = "healthy"
			status.Endpoints = []string{"http://localhost:8888"}

			// Check individual services by examining container logs (same as Python CLI)
			routerHealthy, routerMsg := checkServiceFromLogs("router")
			envoyHealthy, envoyMsg := checkServiceFromLogs("envoy")

			status.Services = append(status.Services, ServiceStatus{
				Name:      "Router",
				Status:    boolToStatus(routerHealthy),
				Healthy:   routerHealthy,
				Message:   routerMsg,
				Component: "container",
			})

			status.Services = append(status.Services, ServiceStatus{
				Name:      "Envoy",
				Status:    boolToStatus(envoyHealthy),
				Healthy:   envoyHealthy,
				Message:   envoyMsg,
				Component: "container",
			})

			// Update overall status based on services
			if !routerHealthy || !envoyHealthy {
				status.Overall = "degraded"
			}

		case "exited":
			status.DeploymentType = "docker"
			status.Overall = "stopped"
			status.Services = append(status.Services, ServiceStatus{
				Name:    "vllm-sr-container",
				Status:  "exited",
				Healthy: false,
				Message: "Container exited. Check logs with: vllm-sr logs router",
			})

		case "not found":
			// Fallback: Check if router is accessible via HTTP (direct run)
			if routerAPIURL != "" {
				routerHealthy, routerMsg := checkHTTPHealth(routerAPIURL + "/health")
				if routerHealthy {
					status.DeploymentType = "local (direct)"
					status.Overall = "healthy"
					status.Services = append(status.Services, ServiceStatus{
						Name:      "Router",
						Status:    "running",
						Healthy:   true,
						Message:   routerMsg,
						Component: "process",
					})
					status.Endpoints = []string{routerAPIURL}

					// Also check Envoy if running locally
					envoyRunning, envoyHealthy, envoyMsg := checkEnvoyHealth("http://localhost:8801/ready")
					if envoyRunning {
						status.Services = append(status.Services, ServiceStatus{
							Name:      "Envoy",
							Status:    boolToStatus(envoyHealthy),
							Healthy:   envoyHealthy,
							Message:   envoyMsg,
							Component: "proxy",
						})
						if !envoyHealthy {
							status.Overall = "degraded"
						}
					}
				}
			}

		default:
			status.DeploymentType = "docker"
			status.Overall = containerStatus
			status.Services = append(status.Services, ServiceStatus{
				Name:    "vllm-sr-container",
				Status:  containerStatus,
				Healthy: false,
			})
		}

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

// checkServiceFromLogs checks if a service is running by examining container logs
// This mirrors the vllm-sr Python CLI approach
func checkServiceFromLogs(service string) (bool, string) {
	// Get container logs directly without shell
	// #nosec G204 -- vllmSrContainerName is a compile-time constant, not user input
	cmd := exec.Command("docker", "logs", "--tail", "100", vllmSrContainerName)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return false, "Status unknown (check logs)"
	}

	logContent := strings.ToLower(string(output))

	// Check for service-specific patterns
	if service == "router" {
		if strings.Contains(logContent, "starting") && strings.Contains(logContent, "router") ||
			strings.Contains(logContent, "router entered running") {
			return true, "Running"
		}
	} else {
		if strings.Contains(logContent, "envoy entered running") {
			return true, "Running"
		}
	}

	return false, "Status unknown (check logs)"
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
	defer resp.Body.Close()

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
	defer resp.Body.Close()

	// Envoy is running if we got ANY response
	isRunning := true

	// Healthy only if 200
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return isRunning, true, "Ready"
	}

	// Running but not healthy (e.g., 503 "no healthy upstream")
	return isRunning, false, "Running (upstream not ready)"
}
