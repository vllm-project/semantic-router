package mcpconfig

import (
	"os"
	"path/filepath"
	"strings"
)

const (
	sourceConfigPathEnv  = "VLLM_SR_SOURCE_CONFIG_PATH"
	runtimeConfigPathEnv = "VLLM_SR_RUNTIME_CONFIG_PATH"
)

type persistencePaths struct {
	activePath  string
	sourcePath  string
	runtimePath string
}

func resolvePersistencePaths(activeConfigPath string) persistencePaths {
	activePath := filepath.Clean(activeConfigPath)
	sourcePath := strings.TrimSpace(os.Getenv(sourceConfigPathEnv))
	if sourcePath == "" {
		sourcePath = deriveSourceConfigPath(activePath)
	}
	if sourcePath == "" {
		sourcePath = activePath
	}
	sourcePath = filepath.Clean(sourcePath)

	runtimePath := strings.TrimSpace(os.Getenv(runtimeConfigPathEnv))
	if runtimePath == "" {
		runtimePath = activePath
	}
	runtimePath = filepath.Clean(runtimePath)

	return persistencePaths{
		activePath:  activePath,
		sourcePath:  sourcePath,
		runtimePath: runtimePath,
	}
}

func deriveSourceConfigPath(activeConfigPath string) string {
	activePath := filepath.Clean(activeConfigPath)
	if filepath.Base(activePath) == "config.yaml" {
		return activePath
	}

	parent := filepath.Base(filepath.Dir(activePath))
	base := filepath.Base(activePath)
	if parent == ".vllm-sr" && strings.HasPrefix(base, "runtime-config") && strings.HasSuffix(base, ".yaml") {
		return filepath.Join(filepath.Dir(filepath.Dir(activePath)), "config.yaml")
	}

	return ""
}

func (p persistencePaths) usesRuntimeOverride() bool {
	return p.sourcePath != "" && p.runtimePath != "" && p.sourcePath != p.runtimePath
}

func defaultAuditLogPath(paths persistencePaths) string {
	return filepath.Join(filepath.Dir(paths.sourcePath), ".vllm-sr", "mcp-config-audit.jsonl")
}
