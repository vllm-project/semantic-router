package handlers

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
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

		status := SystemStatus{
			Overall:        "not_running",
			DeploymentType: "none",
			Services:       []ServiceStatus{},
			Version:        "v0.1.0",
		}
		runtimePath := filepath.Join(configDir, ".vllm-sr", "router-runtime.json")

		// Check if we're running inside a container
		runningInContainer := isRunningInContainer()

		// If running in container, report container services directly
		if runningInContainer {
			status.DeploymentType = "docker"
			status.Overall = "healthy"
			status.Endpoints = []string{"http://localhost:8899"}

			// Check services from logs within the same container
			routerHealthy, routerMsg := checkServiceFromContainerLogs("router")
			envoyHealthy, envoyMsg := checkServiceFromContainerLogs("envoy")
			dashboardHealthy := true
			dashboardMsg := "Running"
			status.RouterRuntime = resolveRouterRuntimeStatus(runtimePath, routerAPIURL, routerHealthy, readRouterLogContentInContainer())
			if status.RouterRuntime != nil && status.RouterRuntime.Message != "" {
				routerMsg = status.RouterRuntime.Message
			}

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
			status.Endpoints = []string{"http://localhost:8899"}

			// Check individual services by examining container logs (same as Python CLI)
			logContent := getContainerLogsTail(500)
			routerHealthy, routerMsg := checkServiceInLogContent("router", logContent)
			envoyHealthy, envoyMsg := checkServiceInLogContent("envoy", logContent)
			dashboardHealthy, dashboardMsg := checkServiceInLogContent("dashboard", logContent)
			status.RouterRuntime = resolveRouterRuntimeStatus(runtimePath, routerAPIURL, routerHealthy, logContent)
			if status.RouterRuntime != nil && status.RouterRuntime.Message != "" {
				routerMsg = status.RouterRuntime.Message
			}

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
					status.RouterRuntime = resolveRouterRuntimeStatus(runtimePath, routerAPIURL, routerHealthy, "")
					if status.RouterRuntime != nil && status.RouterRuntime.Message != "" {
						routerMsg = status.RouterRuntime.Message
					}
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

func checkServiceInLogContent(service, logContent string) (bool, string) {
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

type modelDownloadSummary struct {
	DownloadingModel string
	PendingModels    []string
	ReadyModels      int
	TotalModels      int
}

func resolveRouterRuntimeStatus(runtimePath, routerAPIURL string, routerHealthy bool, fallbackLogContent string) *RouterRuntimeStatus {
	if state, err := loadRouterRuntimeState(runtimePath); err == nil && state != nil {
		runtime := &RouterRuntimeStatus{
			Phase:            state.Phase,
			Ready:            state.Ready,
			Message:          state.Message,
			DownloadingModel: state.DownloadingModel,
			PendingModels:    state.PendingModels,
			ReadyModels:      state.ReadyModels,
			TotalModels:      state.TotalModels,
		}

		if runtime.Ready && routerAPIURL != "" {
			readyHealthy, _ := checkHTTPHealth(routerAPIURL + "/ready")
			if !readyHealthy {
				runtime.Ready = false
				runtime.Phase = "starting"
				runtime.Message = "Router services are starting..."
			}
		}

		return runtime
	}

	if fallbackLogContent == "" {
		return nil
	}

	return detectRouterRuntimeStatus(fallbackLogContent, routerHealthy)
}

func loadRouterRuntimeState(runtimePath string) (*startupstatus.State, error) {
	state, err := startupstatus.Load(runtimePath)
	if err == nil || runtimePath == "" {
		return state, err
	}

	parentDir := filepath.Dir(filepath.Dir(runtimePath))
	if parentDir == "." || parentDir == "/" || parentDir == "" {
		return nil, err
	}

	fallbackPath := filepath.Join(parentDir, "router-runtime.json")
	return startupstatus.Load(fallbackPath)
}

func detectRouterRuntimeStatus(logContent string, routerHealthy bool) *RouterRuntimeStatus {
	trimmed := strings.TrimSpace(logContent)
	if trimmed == "" {
		if routerHealthy {
			return &RouterRuntimeStatus{
				Phase:   "ready",
				Ready:   true,
				Message: "Ready",
			}
		}
		return nil
	}

	if strings.Contains(strings.ToLower(trimmed), "setup mode enabled: router disabled") {
		return &RouterRuntimeStatus{
			Phase:   "setup_mode",
			Ready:   false,
			Message: "Setup mode: router disabled until activation",
		}
	}

	installIdx := strings.LastIndex(trimmed, "Installing required models...")
	allReadyIdx := strings.LastIndex(trimmed, "All required models are ready")
	apiOnlyIdx := strings.LastIndex(trimmed, "No local models configured, skipping model download (API-only mode)")
	if apiOnlyIdx > allReadyIdx {
		allReadyIdx = apiOnlyIdx
	}
	embeddingInitIdx := strings.LastIndex(trimmed, "Initializing embedding models:")
	embeddingReadyIdx := max(
		strings.LastIndex(trimmed, "Unified embedding models initialized successfully"),
		strings.LastIndex(trimmed, "No embedding models configured, skipping initialization"),
	)
	serverReadyIdx := max(
		strings.LastIndex(trimmed, "Starting insecure LLM Router ExtProc server"),
		strings.LastIndex(trimmed, "Starting secure LLM Router ExtProc server"),
	)
	models := summarizeModelDownloadState(trimmed, installIdx)

	if installIdx >= 0 && allReadyIdx < installIdx {
		if models.TotalModels == 0 {
			return &RouterRuntimeStatus{
				Phase:   "checking_models",
				Ready:   false,
				Message: "Checking required router models...",
			}
		}

		message := fmt.Sprintf("Downloading required router models (%d/%d ready)", models.ReadyModels, models.TotalModels)
		if models.DownloadingModel == "" {
			message = fmt.Sprintf("Preparing required router models (%d/%d ready)", models.ReadyModels, models.TotalModels)
		}

		return &RouterRuntimeStatus{
			Phase:            "downloading_models",
			Ready:            false,
			Message:          message,
			DownloadingModel: models.DownloadingModel,
			PendingModels:    models.PendingModels,
			ReadyModels:      models.ReadyModels,
			TotalModels:      models.TotalModels,
		}
	}

	if allReadyIdx >= 0 {
		if embeddingInitIdx > allReadyIdx && embeddingReadyIdx < embeddingInitIdx {
			return &RouterRuntimeStatus{
				Phase:       "initializing_models",
				Ready:       false,
				Message:     "Required models downloaded. Initializing embedding models...",
				ReadyModels: models.TotalModels,
				TotalModels: models.TotalModels,
			}
		}

		if serverReadyIdx < allReadyIdx {
			return &RouterRuntimeStatus{
				Phase:       "initializing_models",
				Ready:       false,
				Message:     "Required models downloaded. Starting router services...",
				ReadyModels: models.TotalModels,
				TotalModels: models.TotalModels,
			}
		}
	}

	if !routerHealthy {
		return &RouterRuntimeStatus{
			Phase:   "starting",
			Ready:   false,
			Message: "Router services are starting...",
		}
	}

	return &RouterRuntimeStatus{
		Phase:            "ready",
		Ready:            true,
		Message:          "Ready",
		DownloadingModel: models.DownloadingModel,
		PendingModels:    models.PendingModels,
		ReadyModels:      max(models.ReadyModels, models.TotalModels),
		TotalModels:      models.TotalModels,
	}
}

func summarizeModelDownloadState(logContent string, installIdx int) modelDownloadSummary {
	relevant := logContent
	if installIdx >= 0 && installIdx < len(logContent) {
		relevant = logContent[installIdx:]
	}

	states := map[string]string{}
	downloadingModel := ""
	scanner := bufio.NewScanner(strings.NewReader(relevant))
	for scanner.Scan() {
		line := scanner.Text()

		if model := extractModelBeforeMarker(line, "(need download)"); model != "" {
			states[model] = "pending"
			continue
		}
		if model := extractModelBeforeMarker(line, "(ready)"); model != "" {
			states[model] = "ready"
			if downloadingModel == model {
				downloadingModel = ""
			}
			continue
		}
		if model := extractModelAfterPrefix(line, "Downloading model: "); model != "" {
			states[model] = "downloading"
			downloadingModel = model
			continue
		}
		if model := extractModelAfterPrefix(line, "Successfully downloaded model: "); model != "" {
			states[model] = "ready"
			if downloadingModel == model {
				downloadingModel = ""
			}
			continue
		}
		if strings.Contains(line, "All required models are ready") {
			downloadingModel = ""
		}
	}

	pendingModels := make([]string, 0)
	readyModels := 0
	for model, state := range states {
		switch state {
		case "ready":
			readyModels++
		case "pending", "downloading":
			pendingModels = append(pendingModels, model)
			if state == "downloading" && downloadingModel == "" {
				downloadingModel = model
			}
		}
	}
	sort.Strings(pendingModels)

	return modelDownloadSummary{
		DownloadingModel: downloadingModel,
		PendingModels:    pendingModels,
		ReadyModels:      readyModels,
		TotalModels:      len(states),
	}
}

func extractModelBeforeMarker(line, marker string) string {
	idx := strings.Index(line, marker)
	if idx == -1 {
		return ""
	}

	prefix := strings.TrimSpace(line[:idx])
	if modelIdx := strings.LastIndex(prefix, "models/"); modelIdx >= 0 {
		return cleanModelPath(prefix[modelIdx:])
	}

	fields := strings.Fields(prefix)
	if len(fields) == 0 {
		return ""
	}
	return cleanModelPath(fields[len(fields)-1])
}

func extractModelAfterPrefix(line, prefix string) string {
	idx := strings.Index(line, prefix)
	if idx == -1 {
		return ""
	}
	return cleanModelPath(line[idx+len(prefix):])
}

func cleanModelPath(value string) string {
	cleaned := strings.TrimSpace(value)
	cleaned = strings.TrimPrefix(cleaned, "✗ ")
	cleaned = strings.TrimPrefix(cleaned, "x ")
	cleaned = strings.Trim(cleaned, "\"'`,")

	if modelIdx := strings.Index(cleaned, "models/"); modelIdx >= 0 {
		cleaned = cleaned[modelIdx:]
	}

	if end := strings.IndexAny(cleaned, " )\"\t,"); end >= 0 {
		cleaned = cleaned[:end]
	}

	return strings.TrimSpace(cleaned)
}

func readRouterLogContentInContainer() string {
	var parts []string
	for _, path := range []string{"/var/log/supervisor/router.log", "/var/log/supervisor/router-error.log"} {
		data, err := os.ReadFile(path)
		if err == nil && len(data) > 0 {
			parts = append(parts, string(data))
		}
	}

	return tailText(strings.Join(parts, "\n"), 400)
}

func getContainerLogsTail(lines int) string {
	// #nosec G204 -- container name is a compile-time constant, line count is formatted from an integer
	cmd := exec.Command("docker", "logs", "--tail", strconv.Itoa(lines), vllmSrContainerName)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return ""
	}
	return string(output)
}

func tailText(content string, maxLines int) string {
	if maxLines <= 0 || content == "" {
		return content
	}

	lines := strings.Split(content, "\n")
	if len(lines) <= maxLines {
		return content
	}
	return strings.Join(lines[len(lines)-maxLines:], "\n")
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
