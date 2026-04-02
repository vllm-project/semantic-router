//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type routerConfigMutationMode string

const (
	routerConfigMutationMerge   routerConfigMutationMode = "merge"
	routerConfigMutationReplace routerConfigMutationMode = "replace"
)

// handleConfigPatch handles PATCH /config/router with merge semantics.
func (s *ClassificationAPIServer) handleConfigPatch(w http.ResponseWriter, r *http.Request) {
	s.handleConfigMutation(w, r, routerConfigMutationMerge)
}

// handleConfigPut handles PUT /config/router with replace semantics.
func (s *ClassificationAPIServer) handleConfigPut(w http.ResponseWriter, r *http.Request) {
	s.handleConfigMutation(w, r, routerConfigMutationReplace)
}

func (s *ClassificationAPIServer) handleConfigMutation(
	w http.ResponseWriter,
	r *http.Request,
	mode routerConfigMutationMode,
) {
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return
	}
	if !deployMu.TryLock() {
		s.writeErrorResponse(w, http.StatusConflict, "DEPLOY_IN_PROGRESS", "Another config update operation is in progress. Please try again.")
		return
	}
	defer deployMu.Unlock()

	req, patchDoc, ok := s.parseRouterConfigUpdateRequest(w, r)
	if !ok {
		return
	}

	paths := resolveConfigPersistencePaths(s.configPath)
	yamlBytes, existingData, ok := s.prepareRouterConfigMutationPayload(w, patchDoc, paths.sourcePath, mode)
	if !ok {
		return
	}

	version, backupDir := s.recordRouterConfigArtifacts(paths.sourcePath, existingData, req.DSL)
	if !s.writeRouterConfigFiles(w, paths, yamlBytes) {
		return
	}

	logging.Infof(
		"Router config %s via API: version=%s, size=%d bytes, sourceConfigPath=%s, runtimeConfigPath=%s",
		mode,
		version,
		len(yamlBytes),
		paths.sourcePath,
		paths.runtimePath,
	)
	configCleanupBackups(backupDir)
	s.writeJSONResponse(w, http.StatusOK, RouterConfigUpdateResponse{
		Status:  "success",
		Version: version,
		Message: routerConfigMutationMessage(mode),
	})
}

func routerConfigMutationMessage(mode routerConfigMutationMode) string {
	if mode == routerConfigMutationMerge {
		return "Router config merged successfully. Router will reload automatically via fsnotify."
	}
	return "Router config replaced successfully. Router will reload automatically via fsnotify."
}

func (s *ClassificationAPIServer) parseRouterConfigUpdateRequest(
	w http.ResponseWriter,
	r *http.Request,
) (RouterConfigUpdateRequest, map[string]any, bool) {
	var req RouterConfigUpdateRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return RouterConfigUpdateRequest{}, nil, false
	}
	if strings.TrimSpace(req.YAML) == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "YAML content is required")
		return RouterConfigUpdateRequest{}, nil, false
	}

	doc, err := decodeYAMLDocument([]byte(req.YAML))
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "YAML_PARSE_ERROR", fmt.Sprintf("Invalid YAML syntax: %v", err))
		return RouterConfigUpdateRequest{}, nil, false
	}

	return req, doc, true
}

func (s *ClassificationAPIServer) prepareRouterConfigMutationPayload(
	w http.ResponseWriter,
	patchDoc map[string]any,
	sourceConfigPath string,
	mode routerConfigMutationMode,
) ([]byte, []byte, bool) {
	existingDoc, existingData, err := readConfigDocument(sourceConfigPath)
	if err != nil && !os.IsNotExist(err) {
		s.writeErrorResponse(w, http.StatusInternalServerError, "READ_ERROR", fmt.Sprintf("Failed to read existing config: %v", err))
		return nil, nil, false
	}

	nextDoc := patchDoc
	if mode == routerConfigMutationMerge {
		nextDoc = mergeConfigDocuments(existingDoc, patchDoc)
	}

	yamlBytes, err := normalizeRouterConfigDocument(nextDoc)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "CONFIG_VALIDATION_ERROR", err.Error())
		return nil, nil, false
	}

	if len(existingData) == 0 {
		logging.Infof("[RouterConfig] No existing config at path=%s, writing %s document", sourceConfigPath, mode)
		return yamlBytes, nil, true
	}

	logging.Infof(
		"[RouterConfig] Existing config found: path=%s, size=%d bytes; applying %s update",
		sourceConfigPath,
		len(existingData),
		mode,
	)
	return yamlBytes, existingData, true
}

func normalizeRouterConfigDocument(doc map[string]any) ([]byte, error) {
	rawYAML, err := yaml.Marshal(doc)
	if err != nil {
		return nil, fmt.Errorf("failed to encode router config update: %w", err)
	}

	parsedCfg, err := config.ParseYAMLBytes(rawYAML)
	if err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	canonicalCfg := config.CanonicalConfigFromRouterConfig(parsedCfg)
	yamlBytes, err := yaml.Marshal(canonicalCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to normalize router config: %w", err)
	}
	return yamlBytes, nil
}

func readConfigDocument(path string) (map[string]any, []byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return map[string]any{}, nil, err
		}
		return nil, nil, err
	}
	doc, err := decodeYAMLDocument(data)
	if err != nil {
		return nil, nil, err
	}
	return doc, data, nil
}

func decodeYAMLDocument(data []byte) (map[string]any, error) {
	var doc map[string]any
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("failed to decode YAML document: %w", err)
	}
	if doc == nil {
		return map[string]any{}, nil
	}
	return doc, nil
}

func mergeConfigDocuments(base map[string]any, patch map[string]any) map[string]any {
	merged, ok := mergeYAMLValue(base, patch).(map[string]any)
	if !ok {
		return map[string]any{}
	}
	return merged
}

func mergeYAMLValue(base any, patch any) any {
	if patch == nil {
		return nil
	}

	patchMap, ok := patch.(map[string]any)
	if !ok {
		return cloneYAMLValue(patch)
	}

	merged := map[string]any{}
	if baseMap, ok := base.(map[string]any); ok {
		for key, value := range baseMap {
			merged[key] = cloneYAMLValue(value)
		}
	}

	for key, value := range patchMap {
		if value == nil {
			delete(merged, key)
			continue
		}
		merged[key] = mergeYAMLValue(merged[key], value)
	}
	return merged
}

func cloneYAMLValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		cloned := make(map[string]any, len(typed))
		for key, nested := range typed {
			cloned[key] = cloneYAMLValue(nested)
		}
		return cloned
	case []any:
		cloned := make([]any, len(typed))
		for i, nested := range typed {
			cloned[i] = cloneYAMLValue(nested)
		}
		return cloned
	default:
		return typed
	}
}

func (s *ClassificationAPIServer) recordRouterConfigArtifacts(sourceConfigPath string, existingData []byte, dsl string) (string, string) {
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

func (s *ClassificationAPIServer) writeRouterConfigFiles(w http.ResponseWriter, paths configPersistencePaths, yamlBytes []byte) bool {
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
	cleanPath := filepath.Clean(configPath)
	tmpConfigFile := cleanPath + ".tmp"
	if err := os.WriteFile(tmpConfigFile, yamlBytes, 0o644); err != nil { //nolint:gosec // G703: path cleaned above
		return err
	}
	if err := os.Rename(tmpConfigFile, cleanPath); err != nil {
		if writeErr := os.WriteFile(cleanPath, yamlBytes, 0o644); writeErr != nil { //nolint:gosec // G703: path cleaned above
			return writeErr
		}
	}
	return nil
}
