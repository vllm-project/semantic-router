package handlers

import (
	"context"
	"path/filepath"
)

func detectSystemStatusWithContext(ctx context.Context, routerAPIURL, configDir string) SystemStatus {
	runtimePath := filepath.Join(configDir, ".vllm-sr", "router-runtime.json")
	if isRunningInContainer() {
		return collectInContainerStatusWithContext(ctx, runtimePath, routerAPIURL)
	}

	return collectHostStatusWithContext(ctx, runtimePath, routerAPIURL)
}

func baseSystemStatus() SystemStatus {
	return SystemStatus{
		Overall:        "not_running",
		DeploymentType: "none",
		Services:       []ServiceStatus{},
		Version:        statusVersion(),
	}
}
