package handlers

import "path/filepath"

func detectSystemStatus(routerAPIURL, configDir string) SystemStatus {
	runtimePath := filepath.Join(configDir, ".vllm-sr", "router-runtime.json")
	if isRunningInContainer() {
		return collectInContainerStatus(runtimePath, routerAPIURL)
	}

	return collectHostStatus(runtimePath, routerAPIURL)
}

func baseSystemStatus() SystemStatus {
	return SystemStatus{
		Overall:        "not_running",
		DeploymentType: "none",
		Services:       []ServiceStatus{},
		Version:        "v0.1.0",
	}
}
