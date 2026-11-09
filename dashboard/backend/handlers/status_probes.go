package handlers

import (
	"context"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

// isRunningInContainer checks if the current process is running inside a Docker container.
func isRunningInContainer() bool {
	if _, err := os.Stat("/.dockerenv"); err == nil {
		return true
	}

	data, err := os.ReadFile("/proc/1/cgroup")
	if err == nil {
		content := string(data)
		if strings.Contains(content, "docker") || strings.Contains(content, "containerd") {
			return true
		}
	}

	return false
}

func boolToStatus(healthy bool) string {
	if healthy {
		return "running"
	}
	return "unknown"
}

func checkRouterManagementHealth(ctx context.Context, url string, credentialProvider ...routerauth.CredentialProvider) bool {
	resp, err := routerManagementGET(ctx, url, 2*time.Second, credentialProvider...)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}

// checkEnvoyHealth checks if Envoy is running and healthy.
// Returns: (isRunning, isHealthy, message)
func checkEnvoyHealth(ctx context.Context, url string) (bool, bool, string) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false, false, ""
	}
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Do(request)
	if err != nil {
		return false, false, ""
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	isRunning := true
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return isRunning, true, "Ready"
	}

	return isRunning, false, "Running (upstream not ready)"
}
