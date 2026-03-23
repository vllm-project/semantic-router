//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	yamlv2 "gopkg.in/yaml.v2"
	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// handleConfigDeploy handles POST /config/deploy.
// The Router writes its own config file and triggers hot-reload via fsnotify.
func (s *ClassificationAPIServer) handleConfigDeploy(w http.ResponseWriter, r *http.Request) {
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return
	}
	if !deployMu.TryLock() {
		s.writeErrorResponse(w, http.StatusConflict, "DEPLOY_IN_PROGRESS", "Another deploy operation is in progress. Please try again.")
		return
	}
	defer deployMu.Unlock()

	req, parsedCfg, ok := s.parseConfigDeployRequest(w, r)
	if !ok {
		return
	}

	paths := resolveConfigPersistencePaths(s.configPath)
	yamlBytes, existingData, ok := s.prepareConfigDeployPayload(w, parsedCfg, paths.sourcePath)
	if !ok {
		return
	}

	version, backupDir := s.recordConfigDeployArtifacts(paths.sourcePath, existingData, req.DSL)
	if !s.writeConfigDeployFiles(w, paths, yamlBytes) {
		return
	}

	logging.Infof(
		"Config deployed via API: version=%s, size=%d bytes, sourceConfigPath=%s, runtimeConfigPath=%s",
		version,
		len(yamlBytes),
		paths.sourcePath,
		paths.runtimePath,
	)
	configCleanupBackups(backupDir)
	s.writeJSONResponse(w, http.StatusOK, ConfigDeployResponse{
		Status:  "success",
		Version: version,
		Message: "Config deployed successfully. Router will reload automatically via fsnotify.",
	})
}

func (s *ClassificationAPIServer) parseConfigDeployRequest(w http.ResponseWriter, r *http.Request) (ConfigDeployRequest, *config.RouterConfig, bool) {
	var req ConfigDeployRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return ConfigDeployRequest{}, nil, false
	}
	if strings.TrimSpace(req.YAML) == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "YAML content is required")
		return ConfigDeployRequest{}, nil, false
	}

	yamlBytes := []byte(req.YAML)
	var yamlMap interface{}
	if err := yaml.Unmarshal(yamlBytes, &yamlMap); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "YAML_PARSE_ERROR", fmt.Sprintf("Invalid YAML syntax: %v", err))
		return ConfigDeployRequest{}, nil, false
	}

	tempFile := filepath.Join(os.TempDir(), fmt.Sprintf("deploy_validate_%d.yaml", time.Now().UnixNano()))
	if err := os.WriteFile(tempFile, yamlBytes, 0o644); err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "TEMP_FILE_ERROR", fmt.Sprintf("Failed to create temp file: %v", err))
		return ConfigDeployRequest{}, nil, false
	}
	defer func() { _ = os.Remove(tempFile) }()

	parsedCfg, err := config.Parse(tempFile)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "CONFIG_VALIDATION_ERROR", fmt.Sprintf("Config validation failed: %v", err))
		return ConfigDeployRequest{}, nil, false
	}

	logging.Infof("[Deploy] After config.Parse: decisions=%d", len(parsedCfg.Decisions))
	for i, d := range parsedCfg.Decisions {
		logging.Infof("[Deploy]   parsed decision[%d]: name=%q, modelRefs=%d, priority=%d", i, d.Name, len(d.ModelRefs), d.Priority)
	}

	return req, parsedCfg, true
}

func (s *ClassificationAPIServer) prepareConfigDeployPayload(w http.ResponseWriter, parsedCfg *config.RouterConfig, sourceConfigPath string) ([]byte, []byte, bool) {
	canonicalCfg := config.CanonicalConfigFromRouterConfig(parsedCfg)
	yamlBytes, err := yaml.Marshal(canonicalCfg)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NORMALIZE_ERROR", fmt.Sprintf("Failed to normalize config: %v", err))
		return nil, nil, false
	}

	logging.Infof("[Deploy] After canonical normalize: yaml size=%d bytes", len(yamlBytes))
	existingData, mergedYAML := s.mergeConfigDeployPayload(sourceConfigPath, yamlBytes)
	return mergedYAML, existingData, true
}

func (s *ClassificationAPIServer) mergeConfigDeployPayload(sourceConfigPath string, newYAML []byte) ([]byte, []byte) {
	existingData, err := os.ReadFile(sourceConfigPath)
	if err != nil || len(existingData) == 0 {
		logging.Infof("[Deploy] No existing config at path=%s (err=%v), using new config as-is", sourceConfigPath, err)
		return existingData, newYAML
	}

	logging.Infof("[Deploy] Existing config found: path=%s, size=%d bytes", sourceConfigPath, len(existingData))
	existingMap := make(map[string]interface{})
	if unmarshalErr := yamlv2.Unmarshal(existingData, &existingMap); unmarshalErr != nil {
		return existingData, newYAML
	}

	newMap := make(map[string]interface{})
	if unmarshalErr := yamlv2.Unmarshal(newYAML, &newMap); unmarshalErr != nil {
		return existingData, newYAML
	}

	merged := configDeepMerge(existingMap, newMap)
	mergedYAML, err := yamlv2.Marshal(merged)
	if err != nil {
		return existingData, newYAML
	}
	return existingData, mergedYAML
}

func (s *ClassificationAPIServer) recordConfigDeployArtifacts(sourceConfigPath string, existingData []byte, dsl string) (string, string) {
	configDir := filepath.Dir(sourceConfigPath)
	backupDir := filepath.Join(configDir, ".vllm-sr", "config-backups")
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		logging.Warnf("Failed to create backup directory: %v", err)
	}

	version := time.Now().Format("20060102-150405")
	if len(existingData) > 0 {
		backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version))
		if err := os.WriteFile(backupFile, existingData, 0o644); err != nil {
			logging.Warnf("Failed to create backup: %v", err)
		} else {
			logging.Infof("Config backup created: %s", backupFile)
		}
	}

	if strings.TrimSpace(dsl) != "" {
		dslDir := filepath.Join(configDir, ".vllm-sr")
		dslFile := filepath.Join(dslDir, "config.dsl")
		if err := os.WriteFile(dslFile, []byte(dsl), 0o644); err != nil {
			logging.Warnf("Failed to archive DSL source: %v", err)
		}
	}

	return version, backupDir
}

func (s *ClassificationAPIServer) writeConfigDeployFiles(w http.ResponseWriter, paths configPersistencePaths, yamlBytes []byte) bool {
	if err := writeConfigAtomically(paths.sourcePath, yamlBytes); err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "WRITE_ERROR", fmt.Sprintf("Failed to write source config: %v", err))
		return false
	}

	if !paths.usesRuntimeOverride() {
		return true
	}

	if _, err := runtimeConfigSyncRunner(paths.sourcePath); err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "RUNTIME_SYNC_ERROR", err.Error())
		return false
	}
	return true
}

func writeConfigAtomically(configPath string, yamlBytes []byte) error {
	tmpConfigFile := configPath + ".tmp"
	if err := os.WriteFile(tmpConfigFile, yamlBytes, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmpConfigFile, configPath); err != nil {
		if writeErr := os.WriteFile(configPath, yamlBytes, 0o644); writeErr != nil {
			return writeErr
		}
	}
	return nil
}
