package handlers

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	routerprojection "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerprojection"
)

const activeConfigProjectionFile = "active-config-projection.json"

func activeConfigProjectionPath(configDir string) string {
	return filepath.Join(configDir, ".vllm-sr", activeConfigProjectionFile)
}

func archivedDSLPath(configDir string) string {
	return filepath.Join(configDir, ".vllm-sr", "config.dsl")
}

func persistActiveConfigProjection(configPath string, configDir string) error {
	projection, err := routerprojection.BuildActiveConfigProjectionFromFile(configPath, archivedDSLPath(configDir))
	if err != nil {
		return fmt.Errorf("failed to build active config projection: %w", err)
	}

	data, err := json.MarshalIndent(projection, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to encode active config projection: %w", err)
	}

	projectionDir := filepath.Dir(activeConfigProjectionPath(configDir))
	if err := os.MkdirAll(projectionDir, 0o755); err != nil {
		return fmt.Errorf("failed to create config projection directory: %w", err)
	}

	targetPath := activeConfigProjectionPath(configDir)
	tmpPath := targetPath + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0o644); err != nil {
		return fmt.Errorf("failed to write active config projection: %w", err)
	}
	if err := os.Rename(tmpPath, targetPath); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("failed to replace active config projection: %w", err)
	}

	return nil
}

func readActiveConfigProjection(configDir string) (*routerprojection.ActiveConfigProjection, error) {
	data, err := os.ReadFile(activeConfigProjectionPath(configDir))
	if err != nil {
		return nil, err
	}

	var projection routerprojection.ActiveConfigProjection
	if err := json.Unmarshal(data, &projection); err != nil {
		return nil, fmt.Errorf("failed to decode active config projection: %w", err)
	}
	return &projection, nil
}
