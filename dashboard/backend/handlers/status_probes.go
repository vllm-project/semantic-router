package handlers

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

const dockerStatusProbeTimeout = 2 * time.Second

const dockerStatusProbeOutputLimit = 64 * 1024

type dockerStatusProbe func(context.Context, string) ([]byte, error)

// managedContainerStatusProbe is the process boundary used by status and
// mutation orchestration. Keeping the runner behind one narrow seam makes
// orchestration tests deterministic without weakening the production timeout.
var managedContainerStatusProbe dockerStatusProbe = runDockerStatusProbe

type dockerStatusProbeError struct {
	err    error
	stderr []byte
}

func (e *dockerStatusProbeError) Error() string { return e.err.Error() }
func (e *dockerStatusProbeError) Unwrap() error { return e.err }

// getDockerContainerStatus checks the status of a Docker container without
// allowing an unavailable container runtime to block an HTTP request forever.
// Returns "unknown" when the probe times out, "not found" when Docker rejects
// the inspect, or the status reported by Docker.
func getDockerContainerStatus(containerName string) string {
	return getDockerContainerStatusWithProbe(
		containerName,
		dockerStatusProbeTimeout,
		managedContainerStatusProbe,
	)
}

func getDockerContainerStatusWithProbe(
	containerName string,
	timeout time.Duration,
	probe dockerStatusProbe,
) string {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	output, err := probe(ctx, containerName)
	if ctx.Err() != nil {
		return "unknown"
	}
	if err != nil {
		if dockerInspectMeansNoManagedRuntime(output, err) {
			return "not found"
		}
		return "unknown"
	}

	status := strings.TrimSpace(string(output))
	if status == "" {
		return "unknown"
	}
	return status
}

func runDockerStatusProbe(ctx context.Context, containerName string) ([]byte, error) {
	output, err := runBoundedCommandSplit(
		ctx,
		"docker",
		dockerStatusProbeOutputLimit,
		"inspect",
		"-f",
		"{{.State.Status}}",
		containerName,
	)
	if err != nil {
		return output.stdout, &dockerStatusProbeError{err: err, stderr: output.stderr}
	}
	return output.stdout, nil
}

func dockerInspectMeansNoManagedRuntime(output []byte, err error) bool {
	message := string(output)
	var probeErr *dockerStatusProbeError
	if errors.As(err, &probeErr) {
		message = string(probeErr.stderr)
	}
	message = strings.ToLower(message)
	return strings.Contains(message, "no such object") ||
		strings.Contains(message, "no such container")
}

func managedContainerRunningOrAbsent(status string, component string) (bool, error) {
	switch status {
	case "running":
		return true, nil
	case "not found":
		return false, nil
	case "unknown":
		return false, fmt.Errorf("managed %s status probe is unavailable", component)
	default:
		return false, fmt.Errorf("managed %s is not running (status %s)", component, status)
	}
}

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

func checkHTTPHealth(url string) (bool, string) {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false, ""
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return true, "HTTP health check OK"
	}
	return false, ""
}

// checkEnvoyHealth checks if Envoy is running and healthy.
// Returns: (isRunning, isHealthy, message)
func checkEnvoyHealth(url string) (bool, bool, string) {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url)
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
