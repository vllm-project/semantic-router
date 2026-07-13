package handlers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
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
		Phase:             state.Phase,
		Ready:             state.Ready,
		Message:           state.Message,
		DownloadingModel:  state.DownloadingModel,
		PendingModels:     state.PendingModels,
		ReadyModels:       state.ReadyModels,
		TotalModels:       state.TotalModels,
		EmbeddingProvider: cloneStartupEmbeddingProviderStatus(state.EmbeddingProvider),
	}
}

func cloneStartupEmbeddingProviderStatus(status *startupstatus.EmbeddingProviderStatus) *startupstatus.EmbeddingProviderStatus {
	if status == nil {
		return nil
	}
	clone := *status
	if status.APIKeyEnvSet != nil {
		value := *status.APIKeyEnvSet
		clone.APIKeyEnvSet = &value
	}
	if status.Healthy != nil {
		value := *status.Healthy
		clone.Healthy = &value
	}
	return &clone
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

func getContainerLogsTailForContainerWithContext(parent context.Context, containerName string, lines int) string {
	// #nosec G204 -- containerName is repository-managed and lines is converted from int.
	tailArg := strconv.Itoa(lines)
	ctx, cancel := context.WithTimeout(parent, 3*time.Second)
	defer cancel()
	output, err := runBoundedCommand(
		ctx,
		"docker",
		4*1024*1024,
		"logs",
		"--tail",
		tailArg,
		containerName,
	)
	if err != nil {
		return ""
	}
	return string(output)
}
