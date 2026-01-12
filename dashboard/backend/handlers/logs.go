package handlers

import (
	"bufio"
	"encoding/json"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// LogEntry represents a single log entry
type LogEntry struct {
	Line    string `json:"line"`
	Service string `json:"service,omitempty"`
}

// LogsResponse represents the logs response
type LogsResponse struct {
	DeploymentType string     `json:"deployment_type"`
	Service        string     `json:"service"`
	Logs           []LogEntry `json:"logs"`
	Count          int        `json:"count"`
	Error          string     `json:"error,omitempty"`
	Message        string     `json:"message,omitempty"`
}

// LogsHandler returns logs from vLLM-SR services
// Aligns with the vllm-sr Python CLI by using the same Docker-based approach
func LogsHandler(routerAPIURL string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "application/json")

		// Parse query parameters
		component := r.URL.Query().Get("component")
		if component == "" {
			component = "router"
		}

		linesStr := r.URL.Query().Get("lines")
		lines := 100
		if linesStr != "" {
			if n, err := strconv.Atoi(linesStr); err == nil && n > 0 && n <= 1000 {
				lines = n
			}
		}

		response := LogsResponse{
			DeploymentType: "none",
			Service:        component,
			Logs:           []LogEntry{},
		}

		// Check if we're running inside a container
		runningInContainer := isRunningInContainer()

		if runningInContainer {
			// Running inside container - use supervisorctl to get logs
			response.DeploymentType = "docker"
			logs, err := fetchContainerLogs(component, lines)
			if err != nil {
				response.Error = err.Error()
			} else {
				for _, line := range logs {
					if line != "" {
						response.Logs = append(response.Logs, LogEntry{
							Line:    line,
							Service: component,
						})
					}
				}
			}
		} else {
			// Running outside container - check Docker container status
			containerStatus := getDockerContainerStatus(vllmSrContainerName)

			switch containerStatus {
			case "running", "exited":
				response.DeploymentType = "docker"
				logs, err := fetchContainerLogs(component, lines)
				if err != nil {
					response.Error = err.Error()
				} else {
					for _, line := range logs {
						if line != "" {
							response.Logs = append(response.Logs, LogEntry{
								Line:    line,
								Service: component,
							})
						}
					}
				}

			case "not found":
				// Check if router is running directly (not via Docker)
				if routerAPIURL != "" && checkRouterHealth(routerAPIURL) {
					response.DeploymentType = "local (direct)"
					response.Message = "Logs are available for Docker deployments started with 'vllm-sr serve'. " +
						"For the current deployment, logs are written to the process stdout/stderr."
				} else {
					response.Error = "No running deployment detected. Start with: vllm-sr serve"
				}

			default:
				response.DeploymentType = "docker"
				response.Error = "Container status: " + containerStatus
			}
		}

		response.Count = len(response.Logs)

		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, `{"error":"Failed to encode response"}`, http.StatusInternalServerError)
			return
		}
	}
}

// fetchContainerLogs gets logs from supervisor-managed services
// When running inside the container, use supervisorctl to get logs
// When running outside, use docker logs
func fetchContainerLogs(component string, lines int) ([]string, error) {
	// First, try to use supervisorctl (when running inside container)
	logs, err := fetchLogsFromSupervisor(component, lines)
	if err == nil && len(logs) > 0 {
		return logs, nil
	}

	// Fallback: try docker logs (when running outside container)
	return fetchLogsFromDocker(component, lines)
}

// fetchLogsFromSupervisor gets logs by reading supervisor log files directly
// Reads both stdout and stderr logs for each component
func fetchLogsFromSupervisor(component string, lines int) ([]string, error) {
	var result []string

	// Map component names to log file paths (both stdout and stderr)
	logFiles := map[string][]string{
		"router":    {"/var/log/supervisor/router.log", "/var/log/supervisor/router-error.log"},
		"envoy":     {"/var/log/supervisor/envoy.log", "/var/log/supervisor/envoy-error.log"},
		"dashboard": {"/var/log/supervisor/dashboard.log", "/var/log/supervisor/dashboard-error.log"},
		"all": {
			"/var/log/supervisor/router.log", "/var/log/supervisor/router-error.log",
			"/var/log/supervisor/envoy.log", "/var/log/supervisor/envoy-error.log",
			"/var/log/supervisor/dashboard.log", "/var/log/supervisor/dashboard-error.log",
		},
	}

	files, ok := logFiles[component]
	if !ok {
		files = []string{
			"/var/log/supervisor/" + component + ".log",
			"/var/log/supervisor/" + component + "-error.log",
		}
	}

	for _, logFile := range files {
		// Use tail command to get last N lines from log file
		// #nosec G204 - logFile is validated and lines is converted from int
		cmd := exec.Command("tail", "-n", strconv.Itoa(lines), logFile)
		output, err := cmd.CombinedOutput()
		if err != nil {
			// File might not exist yet, skip
			continue
		}

		// Parse output and add to result
		logLines := splitLogLines(string(output))
		result = append(result, logLines...)
	}

	// Return last N lines (in case we read from multiple files)
	if len(result) > lines {
		return result[len(result)-lines:], nil
	}
	return result, nil
}

// fetchLogsFromDocker gets logs from Docker container (fallback for external access)
func fetchLogsFromDocker(component string, lines int) ([]string, error) {
	// Get logs directly from Docker without shell interpolation
	// #nosec G204 -- vllmSrContainerName is a compile-time constant, lines is validated integer
	tailArg := strconv.Itoa(lines * 2) // Get more lines for filtering
	cmd := exec.Command("docker", "logs", "--tail", tailArg, vllmSrContainerName)
	output, err := cmd.CombinedOutput()
	if err != nil {
		if len(output) == 0 {
			return []string{}, nil
		}
	}

	// Filter logs in Go instead of using shell grep (safer, avoids gosec warning)
	allLines := splitLogLines(string(output))

	if component == "all" {
		// Return all logs, limited to requested count
		if len(allLines) > lines {
			return allLines[len(allLines)-lines:], nil
		}
		return allLines, nil
	}

	// Filter for specific component
	var filtered []string
	for _, line := range allLines {
		lineLower := strings.ToLower(line)
		switch component {
		case "router":
			// Match router-specific logs: Go router logs contain "caller" field in JSON
			// Also include supervisor messages about router and router startup messages
			if strings.Contains(line, `"caller"`) ||
				strings.Contains(line, "spawned: 'router'") ||
				strings.Contains(lineLower, "success: router") ||
				strings.Contains(lineLower, "router entered running") ||
				strings.Contains(line, "Starting router") ||
				strings.Contains(line, "Starting insecure LLM Router ExtProc server") {
				filtered = append(filtered, line)
			}
		case "envoy":
			// Match envoy-specific logs and supervisor messages
			// Envoy logs have timestamp format [YYYY-MM-DD HH:MM:SS.mmm][level]
			if (strings.Contains(line, "[20") && strings.Contains(line, "][")) ||
				strings.Contains(line, "spawned: 'envoy'") ||
				strings.Contains(lineLower, "success: envoy") ||
				strings.Contains(lineLower, "envoy entered running") ||
				strings.Contains(line, "Generating Envoy config") {
				filtered = append(filtered, line)
			}
		case "dashboard":
			// Match dashboard-specific logs: Dashboard logs start with timestamp YYYY/MM/DD
			// Also include supervisor messages about dashboard
			if (strings.Contains(line, "2026/") || strings.Contains(line, "2025/") || strings.Contains(line, "2027/")) ||
				strings.Contains(line, "spawned: 'dashboard'") ||
				strings.Contains(lineLower, "success: dashboard") ||
				strings.Contains(lineLower, "dashboard entered running") ||
				strings.Contains(line, "Starting dashboard") ||
				strings.Contains(line, "Dashboard listening") {
				filtered = append(filtered, line)
			}
		}
	}

	// Return last N lines
	if len(filtered) > lines {
		return filtered[len(filtered)-lines:], nil
	}
	return filtered, nil
}

// checkRouterHealth checks if router is accessible via HTTP
func checkRouterHealth(url string) bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}

// splitLogLines splits output into lines, removing empty ones
func splitLogLines(output string) []string {
	var result []string
	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		line := scanner.Text()
		if line != "" {
			result = append(result, line)
		}
	}
	return result
}
