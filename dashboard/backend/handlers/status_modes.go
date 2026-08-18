package handlers

import (
	"path/filepath"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

func detectSystemStatus(routerAPIURL, configDir string, credentialProvider ...routerauth.CredentialProvider) SystemStatus {
	runtimePath := filepath.Join(configDir, ".vllm-sr", "router-runtime.json")
	if isRunningInContainer() {
		return collectInContainerStatus(runtimePath, routerAPIURL, credentialProvider...)
	}

	return collectHostStatus(runtimePath, routerAPIURL, credentialProvider...)
}

func baseSystemStatus() SystemStatus {
	return SystemStatus{
		Overall:        "not_running",
		DeploymentType: "none",
		Services:       []ServiceStatus{},
		Version:        statusVersion(),
	}
}
