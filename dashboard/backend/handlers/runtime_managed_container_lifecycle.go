package handlers

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

func managedServiceUsesContainerLifecycle(service string) bool {
	if !managedRuntimeUsesSplitContainers() {
		return false
	}

	switch service {
	case "router", "envoy":
		return managedContainerNameForService(service) != managedDashboardContainerName()
	default:
		return false
	}
}

func restartOrStartManagedSplitContainerService(service string, timeout time.Duration) error {
	containerName := managedContainerNameForService(service)
	status := getDockerContainerStatus(containerName)
	if status == "not found" {
		return fmt.Errorf("%s container %s not found", service, containerName)
	}

	action, err := splitContainerLifecycleAction(status)
	if err != nil {
		return fmt.Errorf("%s container %s cannot be started from status %s: %w", service, containerName, status, err)
	}

	output, err := execManagedContainerLifecycleAction(containerName, action, 20*time.Second)
	if err != nil {
		return fmt.Errorf("%s %s failed: %s", action, service, strings.TrimSpace(output))
	}

	return waitForManagedSplitContainerService(containerName, timeout)
}

func splitContainerLifecycleAction(status string) (string, error) {
	switch status {
	case "running", "paused", "restarting":
		return "restart", nil
	case "created", "exited":
		return "start", nil
	default:
		return "", fmt.Errorf("unsupported container status")
	}
}

func execManagedContainerLifecycleAction(containerName string, action string, timeout time.Duration) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "docker", action, containerName)
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func waitForManagedSplitContainerService(containerName string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	lastStatus := ""

	for time.Now().Before(deadline) {
		status := getDockerContainerStatus(containerName)
		lastStatus = status
		switch status {
		case "running":
			return nil
		case "exited", "dead", "removing":
			return fmt.Errorf("container failed to start: %s", status)
		case "not found":
			return fmt.Errorf("container disappeared")
		}
		time.Sleep(500 * time.Millisecond)
	}

	return fmt.Errorf("timed out waiting for %s to become running (last status: %s)", containerName, lastStatus)
}
