package configlifecycle

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const managedContainerConfigDestination = "/app/config.yaml"

type containerMount struct {
	Source      string `json:"Source"`
	Destination string `json:"Destination"`
}

type containerInspect struct {
	Mounts []containerMount `json:"Mounts"`
}

func (s *Service) shouldPropagateToManagedContainer() bool {
	outputDir := filepath.Join(s.ConfigDir, ".vllm-sr")
	if info, err := os.Stat(outputDir); err != nil || !info.IsDir() {
		return false
	}

	mounts, err := inspectManagedContainerMounts(vllmSrContainerName)
	if err != nil {
		return false
	}
	return mountsIncludeConfigPath(mounts, s.ConfigPath)
}

func inspectManagedContainerMounts(containerName string) ([]containerMount, error) {
	cmd := exec.Command("docker", "inspect", containerName)
	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	var inspected []containerInspect
	if err := json.Unmarshal(output, &inspected); err != nil {
		return nil, fmt.Errorf("failed to decode docker inspect output: %w", err)
	}
	if len(inspected) == 0 {
		return nil, fmt.Errorf("no docker inspect results for %s", containerName)
	}
	return inspected[0].Mounts, nil
}

func mountsIncludeConfigPath(mounts []containerMount, configPath string) bool {
	expected := normalizeMountPath(configPath)
	for _, mount := range mounts {
		if mount.Destination != managedContainerConfigDestination {
			continue
		}
		if normalizeMountPath(mount.Source) == expected {
			return true
		}
	}
	return false
}

func normalizeMountPath(path string) string {
	if path == "" {
		return ""
	}
	if resolved, err := filepath.EvalSymlinks(path); err == nil {
		path = resolved
	}
	return filepath.Clean(strings.TrimSpace(path))
}
