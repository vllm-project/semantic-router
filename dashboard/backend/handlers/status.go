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

		// Check if we're running inside a container
		runningInContainer := isRunningInContainer()

		// If running in container, report container services directly
		if runningInContainer {
			status.DeploymentType = "docker"
			status.Overall = "healthy"
			status.Endpoints = []string{"http://localhost:8888"}

			// Check services from logs within the same container
			routerHealthy, routerMsg := checkServiceFromContainerLogs("router")
			envoyHealthy, envoyMsg := checkServiceFromContainerLogs("envoy")
			dashboardHealthy := true
			dashboardMsg := "Running"

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

			status.Services = append(status.Services, ServiceStatus{
				Name:      "Dashboard",
				Status:    boolToStatus(dashboardHealthy),
				Healthy:   dashboardHealthy,
				Message:   dashboardMsg,
				Component: "container",
			})

			// Update overall status based on services
			if !routerHealthy || !envoyHealthy || !dashboardHealthy {
				status.Overall = "degraded"
			}

			if err := json.NewEncoder(w).Encode(status); err != nil {
				http.Error(w, `{"error":"Failed to encode response"}`, http.StatusInternalServerError)
			}
			return
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
			dashboardHealthy, dashboardMsg := checkServiceFromLogs("dashboard")

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

			status.Services = append(status.Services, ServiceStatus{
				Name:      "Dashboard",
				Status:    boolToStatus(dashboardHealthy),
				Healthy:   dashboardHealthy,
				Message:   dashboardMsg,
				Component: "container",
			})

			// Update overall status based on services
			if !routerHealthy || !envoyHealthy || !dashboardHealthy {
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

					// Dashboard is always running in local mode (since we're serving this page)
					status.Services = append(status.Services, ServiceStatus{
						Name:      "Dashboard",
						Status:    "running",
						Healthy:   true,
						Message:   "Running",
						Component: "process",
					})
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

// checkServiceFromLogs checks if a service is running by examining container logs
// This mirrors the vllm-sr Python CLI approach
func checkServiceFromLogs(service string) (bool, string) {
	// Get container logs directly without shell
	// #nosec G204 -- vllmSrContainerName is a compile-time constant, not user input
	cmd := exec.Command("docker", "logs", "--tail", "200", vllmSrContainerName)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return false, "Status unknown (check logs)"
	}

	logContent := string(output)
	logContentLower := strings.ToLower(logContent)

	// Check for service-specific patterns from supervisord
	switch service {
	case "router":
		// Check for supervisord spawn message or router startup messages
		if strings.Contains(logContent, "spawned: 'router'") ||
			strings.Contains(logContentLower, "starting router") ||
			strings.Contains(logContentLower, "router entered running") ||
			strings.Contains(logContent, "Starting insecure LLM Router ExtProc server") ||
			strings.Contains(logContent, `"caller"`) {
			return true, "Running"
		}
	case "envoy":
		// Check for supervisord spawn message or envoy startup messages
		if strings.Contains(logContent, "spawned: 'envoy'") ||
			strings.Contains(logContentLower, "envoy entered running") ||
			strings.Contains(logContent, "[info] initializing epoch") ||
			(strings.Contains(logContent, "[20") && strings.Contains(logContent, "[info]")) ||
			(strings.Contains(logContent, "[20") && strings.Contains(logContent, "[debug]")) {
			return true, "Running"
		}
	case "dashboard":
		// Check for supervisord spawn message or dashboard startup messages
		if strings.Contains(logContent, "spawned: 'dashboard'") ||
			strings.Contains(logContentLower, "dashboard entered running") ||
			strings.Contains(logContent, "Dashboard listening on") ||
			strings.Contains(logContent, "Semantic Router Dashboard listening") {
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
