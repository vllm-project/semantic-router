//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"runtime"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

// getSystemInfo returns system information.
func (s *ClassificationAPIServer) getSystemInfo() SystemInfo {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return SystemInfo{
		GoVersion:    runtime.Version(),
		Architecture: runtime.GOARCH,
		OS:           runtime.GOOS,
		MemoryUsage:  fmt.Sprintf("%.2f MB", float64(m.Alloc)/1024/1024),
		GPUAvailable: false, // TODO: Implement GPU detection.
	}
}

func (s *ClassificationAPIServer) loadModelsRuntimeState() *startupstatus.State {
	if s == nil || s.configPath == "" {
		return nil
	}

	state, err := startupstatus.Load(startupstatus.StatusPathFromConfigPath(s.configPath))
	if err != nil {
		return nil
	}

	return state
}

func buildModelsInfoSummary(runtimeState *startupstatus.State, models []ModelInfo) ModelsInfoSummary {
	loadedModels := 0
	for _, model := range models {
		if model.Loaded {
			loadedModels++
		}
	}

	totalModels := len(models)
	summary := ModelsInfoSummary{
		Ready:        totalModels == 0 || loadedModels == totalModels,
		LoadedModels: loadedModels,
		TotalModels:  totalModels,
	}

	if runtimeState == nil {
		if summary.Ready {
			summary.Phase = "ready"
			summary.Message = "All known router models are ready."
		} else if totalModels > 0 {
			summary.Phase = "starting"
			summary.Message = "Router models are still initializing."
		}
		return summary
	}

	summary.Ready = runtimeState.Ready
	summary.Phase = runtimeState.Phase
	summary.Message = runtimeState.Message
	summary.DownloadingModel = runtimeState.DownloadingModel
	summary.PendingModels = runtimeState.PendingModels
	summary.UpdatedAt = runtimeState.UpdatedAt
	if runtimeState.TotalModels > summary.TotalModels {
		summary.TotalModels = runtimeState.TotalModels
	}
	if loadedModels == 0 && runtimeState.ReadyModels > 0 {
		summary.LoadedModels = runtimeState.ReadyModels
	}

	return summary
}

func enrichModelInfo(model ModelInfo, runtimeState *startupstatus.State) ModelInfo {
	resolvedPath := canonicalModelPath(model.ModelPath)
	if resolvedPath != "" && resolvedPath != model.ModelPath {
		model.ResolvedModelPath = resolvedPath
	}

	if registry := lookupModelRegistryInfo(model.ModelPath); registry != nil {
		model.Registry = registry
	}

	model.State = resolveModelState(model, runtimeState)
	return model
}

func resolveModelState(model ModelInfo, runtimeState *startupstatus.State) string {
	if model.Loaded {
		return "ready"
	}

	if runtimeState == nil {
		return "not_loaded"
	}

	modelPath := canonicalModelPath(model.ModelPath)
	downloadingPath := canonicalModelPath(runtimeState.DownloadingModel)
	if modelPath != "" && modelPath == downloadingPath {
		return "downloading"
	}

	for _, pending := range runtimeState.PendingModels {
		if modelPath != "" && modelPath == canonicalModelPath(pending) {
			return "pending"
		}
	}

	switch runtimeState.Phase {
	case "downloading_models":
		return "pending"
	case "checking_models", "initializing_models", "starting":
		return "initializing"
	default:
		return "not_loaded"
	}
}

func lookupModelRegistryInfo(modelPath string) *ModelRegistryInfo {
	resolvedPath := canonicalModelPath(modelPath)
	if resolvedPath == "" {
		return nil
	}

	return routerconfig.GetModelRegistryInfoByPath(resolvedPath)
}

func canonicalModelPath(modelPath string) string {
	if modelPath == "" {
		return ""
	}

	return routerconfig.ResolveModelPath(modelPath)
}
