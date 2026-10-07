package handlers

import (
	"path/filepath"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

func detectSystemStatus(routerAPIURL, envoyURL, configDir string, stack StackState, credentialProvider ...routerauth.CredentialProvider) SystemStatus {
	runtimePath := filepath.Join(configDir, ".vllm-sr", "router-runtime.json")
	if isRunningInContainer() {
		return collectInContainerStatus(runtimePath, stack, routerAPIURL, envoyURL, credentialProvider...)
	}

	return collectHostStatus(runtimePath, routerAPIURL, envoyURL, credentialProvider...)
}

func baseSystemStatus() SystemStatus {
	return SystemStatus{
		Overall:        "not_running",
		DeploymentType: "none",
		Services:       []ServiceStatus{},
		Version:        statusVersion(),
	}
}
