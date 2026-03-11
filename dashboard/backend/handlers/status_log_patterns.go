package handlers

import "strings"

func checkServiceInLogContent(service, logContent string) (bool, string) {
	if serviceLogLooksHealthy(service, logContent) {
		return true, "Running"
	}

	return false, "Status unknown (check logs)"
}

func serviceLogLooksHealthy(service, logContent string) bool {
	logContentLower := strings.ToLower(logContent)

	switch service {
	case "router":
		return containsAny(logContent, []string{
			"spawned: 'router'",
			"Starting insecure LLM Router ExtProc server",
			`"caller"`,
		}) || containsAny(logContentLower, []string{
			"starting router",
			"router entered running",
		})
	case "envoy":
		return containsAny(logContent, []string{
			"spawned: 'envoy'",
			"[info] initializing epoch",
		}) || containsAny(logContentLower, []string{
			"envoy entered running",
		}) || containsAll(logContent, "[20", "[info]") || containsAll(logContent, "[20", "[debug]")
	case "dashboard":
		return containsAny(logContent, []string{
			"spawned: 'dashboard'",
			"Dashboard listening on",
			"Semantic Router Dashboard listening",
		}) || containsAny(logContentLower, []string{
			"dashboard entered running",
		})
	default:
		return false
	}
}

func containsAny(content string, patterns []string) bool {
	for _, pattern := range patterns {
		if strings.Contains(content, pattern) {
			return true
		}
	}

	return false
}

func containsAll(content string, patterns ...string) bool {
	for _, pattern := range patterns {
		if !strings.Contains(content, pattern) {
			return false
		}
	}

	return true
}
