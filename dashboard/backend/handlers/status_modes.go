package handlers

import (
	"context"
	"path/filepath"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

func detectSystemStatus(ctx context.Context, routerAPIURL, envoyURL, configDir string, stack StackState, credentialProvider ...routerauth.CredentialProvider) SystemStatus {
	runtimePath := filepath.Join(configDir, ".vllm-sr", "router-runtime.json")
	if isRunningInContainer() {
		return collectInContainerStatus(ctx, runtimePath, stack, routerAPIURL, envoyURL, credentialProvider...)
	}

	return collectHostStatus(ctx, runtimePath, routerAPIURL, envoyURL, credentialProvider...)
}

func baseSystemStatus() SystemStatus {
	return SystemStatus{
		Overall:        "not_running",
		DeploymentType: "none",
		ServingMode:    servingModeUnknown,
		Services:       []ServiceStatus{},
		Version:        statusVersion(),
	}
}
