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

		// Check container status (same as vllm-sr Python CLI)
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

		response.Count = len(response.Logs)

		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, `{"error":"Failed to encode response"}`, http.StatusInternalServerError)
			return
		}
	}
}

// fetchContainerLogs gets logs from the vllm-sr container with filtering
// Uses the same approach as the vllm-sr Python CLI
func fetchContainerLogs(component string, lines int) ([]string, error) {
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
			// Also include supervisor messages about router
			if strings.Contains(line, `"caller"`) ||
				strings.Contains(lineLower, "spawned: 'router'") ||
				strings.Contains(lineLower, "success: router") ||
				strings.Contains(lineLower, "router entered running") {
				filtered = append(filtered, line)
			}
		case "envoy":
			// Match envoy-specific logs and supervisor messages
			if (strings.Contains(line, "[20") && strings.Contains(line, "][")) ||
				strings.Contains(lineLower, "spawned: 'envoy'") ||
				strings.Contains(lineLower, "success: envoy") ||
				strings.Contains(lineLower, "envoy entered running") {
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
