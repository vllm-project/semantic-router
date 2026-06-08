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

		if isRunningInContainer() {
			populateRunningContainerLogs(&response, component, lines)
		} else {
			populateExternalLogs(&response, component, lines, routerAPIURL)
		}

		response.Count = len(response.Logs)

		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, `{"error":"Failed to encode response"}`, http.StatusInternalServerError)
			return
		}
	}
}

func populateRunningContainerLogs(
	response *LogsResponse, component string, lines int,
) {
	response.DeploymentType = "docker"
	populateFetchedLogs(response, component, lines)
}

func populateExternalLogs(
	response *LogsResponse, component string, lines int, routerAPIURL string,
) {
	switch containerStatus := runtimeContainerStatusForLogs(); containerStatus {
	case "running", "exited":
		response.DeploymentType = "docker"
		populateFetchedLogs(response, component, lines)
	case "not found":
		populateDirectRuntimeMessage(response, routerAPIURL)
	default:
		response.DeploymentType = "docker"
		response.Error = "Container status: " + containerStatus
	}
}

func populateFetchedLogs(response *LogsResponse, component string, lines int) {
	logs, err := fetchContainerLogs(component, lines)
	if err != nil {
		response.Error = err.Error()
		return
	}
	response.Logs = append(response.Logs, buildLogEntries(component, logs)...)
}

func buildLogEntries(component string, logs []string) []LogEntry {
	entries := make([]LogEntry, 0, len(logs))
	for _, line := range logs {
		if line == "" {
			continue
		}
		entries = append(entries, LogEntry{
			Line:    line,
			Service: component,
		})
	}
	return entries
}

func populateDirectRuntimeMessage(response *LogsResponse, routerAPIURL string) {
	if routerAPIURL != "" && checkRouterHealth(routerAPIURL) {
		response.DeploymentType = "local (direct)"
		response.Message = "Logs are available for Docker deployments started with 'vllm-sr serve'. " +
			"For the current deployment, logs are written to the process stdout/stderr."
		return
	}
	response.Error = "No running deployment detected. Start with: vllm-sr serve"
}

func runtimeContainerStatusForLogs() string {
	return managedRuntimeContainerStatus()
}

func fetchContainerLogs(component string, lines int) ([]string, error) {
	return fetchLogsFromManagedDocker(component, lines)
}

func fetchLogsFromManagedDocker(component string, lines int) ([]string, error) {
	var result []string

	for _, containerName := range managedContainerNamesForComponent(component) {
		// #nosec G204 -- containerName is repository-managed and lines is converted from int.
		cmd := exec.Command("docker", "logs", "--tail", strconv.Itoa(lines), containerName)
		output, err := cmd.CombinedOutput()
		if err != nil && len(output) == 0 {
			continue
		}

		logLines := splitLogLines(string(output))
		service := managedServiceForContainerName(containerName)
		if component == "all" && service != "" {
			for _, line := range logLines {
				result = append(result, "["+service+"] "+line)
			}
			continue
		}
		result = append(result, logLines...)
	}

	if len(result) > lines {
		return result[len(result)-lines:], nil
	}
	return result, nil
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
