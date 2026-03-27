package handlers

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

type modelDownloadSummary struct {
	DownloadingModel string
	PendingModels    []string
	ReadyModels      int
	TotalModels      int
}

type routerRuntimeMarkers struct {
	installIdx        int
	allReadyIdx       int
	embeddingInitIdx  int
	embeddingReadyIdx int
	serverReadyIdx    int
}

func resolveRouterRuntimeStatus(runtimePath, routerAPIURL string, routerHealthy bool, fallbackLogContent string) *RouterRuntimeStatus {
	if state, err := loadRouterRuntimeState(runtimePath); err == nil && state != nil {
		runtime := runtimeStatusFromState(state)
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

func runtimeStatusFromState(state *startupstatus.State) *RouterRuntimeStatus {
	return &RouterRuntimeStatus{
		Phase:            state.Phase,
		Ready:            state.Ready,
		Message:          state.Message,
		DownloadingModel: state.DownloadingModel,
		PendingModels:    state.PendingModels,
		ReadyModels:      state.ReadyModels,
		TotalModels:      state.TotalModels,
	}
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
	if runtime := detectEmptyOrSetupRuntime(trimmed, routerHealthy); runtime != nil {
		return runtime
	}

	markers := parseRuntimeMarkers(trimmed)
	models := summarizeModelDownloadState(trimmed, markers.installIdx)
	if runtime := runtimeDuringModelDownload(markers, models); runtime != nil {
		return runtime
	}
	if runtime := runtimeWhileInitializing(markers, models); runtime != nil {
		return runtime
	}

	return finalRouterRuntimeStatus(routerHealthy, models)
}

func detectEmptyOrSetupRuntime(trimmed string, routerHealthy bool) *RouterRuntimeStatus {
	if trimmed == "" {
		if routerHealthy {
			return &RouterRuntimeStatus{Phase: "ready", Ready: true, Message: "Ready"}
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

	return nil
}

func parseRuntimeMarkers(trimmed string) routerRuntimeMarkers {
	allReadyIdx := strings.LastIndex(trimmed, "All required models are ready")
	apiOnlyIdx := strings.LastIndex(trimmed, "No local models configured, skipping model download (API-only mode)")
	if apiOnlyIdx > allReadyIdx {
		allReadyIdx = apiOnlyIdx
	}

	return routerRuntimeMarkers{
		installIdx:        strings.LastIndex(trimmed, "Installing required models..."),
		allReadyIdx:       allReadyIdx,
		embeddingInitIdx:  strings.LastIndex(trimmed, "Initializing embedding models:"),
		embeddingReadyIdx: max(strings.LastIndex(trimmed, "Unified embedding models initialized successfully"), strings.LastIndex(trimmed, "No embedding models configured, skipping initialization")),
		serverReadyIdx:    max(strings.LastIndex(trimmed, "Starting insecure LLM Router ExtProc server"), strings.LastIndex(trimmed, "Starting secure LLM Router ExtProc server")),
	}
}

func runtimeDuringModelDownload(markers routerRuntimeMarkers, models modelDownloadSummary) *RouterRuntimeStatus {
	if markers.installIdx < 0 || markers.allReadyIdx >= markers.installIdx {
		return nil
	}

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

func runtimeWhileInitializing(markers routerRuntimeMarkers, models modelDownloadSummary) *RouterRuntimeStatus {
	if markers.allReadyIdx < 0 {
		return nil
	}

	if markers.embeddingInitIdx > markers.allReadyIdx && markers.embeddingReadyIdx < markers.embeddingInitIdx {
		return &RouterRuntimeStatus{
			Phase:       "initializing_models",
			Ready:       false,
			Message:     "Required models downloaded. Initializing embedding models...",
			ReadyModels: models.TotalModels,
			TotalModels: models.TotalModels,
		}
	}

	if markers.serverReadyIdx < markers.allReadyIdx {
		return &RouterRuntimeStatus{
			Phase:       "initializing_models",
			Ready:       false,
			Message:     "Required models downloaded. Starting router services...",
			ReadyModels: models.TotalModels,
			TotalModels: models.TotalModels,
		}
	}

	return nil
}

func finalRouterRuntimeStatus(routerHealthy bool, models modelDownloadSummary) *RouterRuntimeStatus {
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
	relevant := relevantModelLogContent(logContent, installIdx)
	states := map[string]string{}
	downloadingModel := ""

	scanner := bufio.NewScanner(strings.NewReader(relevant))
	for scanner.Scan() {
		downloadingModel = updateModelDownloadState(scanner.Text(), states, downloadingModel)
	}

	return finalizeModelDownloadSummary(states, downloadingModel)
}

func relevantModelLogContent(logContent string, installIdx int) string {
	if installIdx >= 0 && installIdx < len(logContent) {
		return logContent[installIdx:]
	}

	return logContent
}

func updateModelDownloadState(line string, states map[string]string, downloadingModel string) string {
	switch {
	case extractModelBeforeMarker(line, "(need download)") != "":
		states[extractModelBeforeMarker(line, "(need download)")] = "pending"
	case extractModelBeforeMarker(line, "(ready)") != "":
		model := extractModelBeforeMarker(line, "(ready)")
		states[model] = "ready"
		if downloadingModel == model {
			return ""
		}
	case extractModelAfterPrefix(line, "Downloading model: ") != "":
		model := extractModelAfterPrefix(line, "Downloading model: ")
		states[model] = "downloading"
		return model
	case extractModelAfterPrefix(line, "Successfully downloaded model: ") != "":
		model := extractModelAfterPrefix(line, "Successfully downloaded model: ")
		states[model] = "ready"
		if downloadingModel == model {
			return ""
		}
	case strings.Contains(line, "All required models are ready"):
		return ""
	}

	return downloadingModel
}

func finalizeModelDownloadSummary(states map[string]string, downloadingModel string) modelDownloadSummary {
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
	if managedRuntimeUsesSplitContainers() && !managedServiceCanBeControlledLocally("router") {
		return getContainerLogsTailForContainer(managedContainerNameForService("router"), 400)
	}

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
	return getContainerLogsTailForContainer(vllmSrContainerName, lines)
}

func getContainerLogsTailForContainer(containerName string, lines int) string {
	// #nosec G204 -- containerName is repository-managed and lines is converted from int.
	tailArg := strconv.Itoa(lines)
	cmd := exec.Command("docker", "logs", "--tail", tailArg, containerName)
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
