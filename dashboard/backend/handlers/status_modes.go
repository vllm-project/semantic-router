package handlers

func detectSystemStatus(routerAPIURL string) SystemStatus {
	if isRunningInContainer() {
		return collectInContainerStatus(routerAPIURL)
	}

	return collectHostStatus(routerAPIURL)
}

func baseSystemStatus() SystemStatus {
	return SystemStatus{
		Overall:  "not_running",
		Services: []ServiceStatus{},
	}
}
