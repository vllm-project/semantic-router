package handlers

import (
	"encoding/json"
	"io"
	"net/http"
	"os/exec"
	"path/filepath"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func resolveRouterRuntimeStatus(runtimePath, routerAPIURL string, routerHealthy bool) *RouterRuntimeStatus {
	if routerAPIURL != "" {
		if state := fetchStartupStatusFromAPI(routerAPIURL); state != nil {
			return runtimeStatusFromState(state)
		}
	}

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

	return nil
}

func fetchStartupStatusFromAPI(routerAPIURL string) *startupstatus.State {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(routerAPIURL + "/startup-status")
	if err != nil {
		return nil
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusServiceUnavailable {
		return nil
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil
	}

	var state startupstatus.State
	if err := json.Unmarshal(body, &state); err != nil {
		return nil
	}

	return &state
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
