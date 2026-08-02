//go:build !windows && cgo

package apiserver

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
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
	if !checkConfigPrecondition(w, r, existingData, false) {
		return
	}

	s.commitRouterConfigDocument(
		w,
		paths,
		existingData,
		yamlBytes,
		req.DSL,
		http.StatusOK,
		fmt.Sprintf("config.%s", mode),
		routerConfigMutationMessage(mode),
	)
}

func routerConfigMutationMessage(mode routerConfigMutationMode) string {
	if mode == routerConfigMutationMerge {
		return "Router config merged successfully."
	}
	return "Router config replaced successfully."
}

func (s *ClassificationAPIServer) parseRouterConfigUpdateRequest(
	w http.ResponseWriter,
	r *http.Request,
) (RouterConfigUpdateRequest, map[string]any, bool) {
	var req RouterConfigUpdateRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
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

	yamlBytes, err := validateAndEncodeRouterConfigDocument(nextDoc)
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

func validateAndEncodeRouterConfigDocument(doc map[string]any) ([]byte, error) {
	rawYAML, err := yaml.Marshal(doc)
	if err != nil {
		return nil, fmt.Errorf("failed to encode router config update: %w", err)
	}

	if _, err := config.ParseYAMLBytes(rawYAML); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}
	// The submitted document is already canonical v0.3 because ParseYAMLBytes
	// rejects legacy layouts. Persist that document rather than exporting the
	// runtime representation back to YAML: runtime endpoint names and explicit
	// zero values are implementation details and must not rewrite user intent.
	return rawYAML, nil
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

	version := nextConfigVersion(backupDir, time.Now())
	source := configVersionSourceAPI
	if strings.TrimSpace(dsl) != "" {
		source = configVersionSourceDSL
	}
	recordConfigBackup(backupDir, version, existingData, source)

	if strings.TrimSpace(dsl) != "" {
		dslDir := filepath.Join(configDir, ".vllm-sr")
		dslFile := filepath.Join(dslDir, "config.dsl")
		if err := os.WriteFile(dslFile, []byte(dsl), 0o644); err != nil {
			logging.Warnf("Failed to archive DSL source: %v", err)
		}
	}

	return version, backupDir
}

func (s *ClassificationAPIServer) writeRouterConfigFiles(
	w http.ResponseWriter,
	paths configPersistencePaths,
	previousData []byte,
	yamlBytes []byte,
) bool {
	if err := writeConfigAtomically(paths.sourcePath, yamlBytes); err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "WRITE_ERROR", fmt.Sprintf("Failed to write source config: %v", err))
		return false
	}

	if !paths.usesRuntimeOverride() {
		return true
	}

	if err := syncRuntimeConfigOrRestore(paths, previousData); err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "RUNTIME_SYNC_ERROR", err.Error())
		return false
	}
	return true
}

func (s *ClassificationAPIServer) commitRouterConfigDocument(
	w http.ResponseWriter,
	paths configPersistencePaths,
	previousData []byte,
	yamlBytes []byte,
	dsl string,
	statusCode int,
	action string,
	message string,
) bool {
	version, backupDir := s.recordRouterConfigArtifacts(paths.sourcePath, previousData, dsl)
	if !s.writeRouterConfigFiles(w, paths, previousData, yamlBytes) {
		return false
	}

	etag := configDocumentETag(yamlBytes)
	runtimeHash, runtimeStatus := s.waitForRuntimeConfigActivation(paths.runtimePath)
	responseStatus := "success"
	responseCode := statusCode
	switch runtimeStatus {
	case "pending":
		responseStatus = "accepted"
		responseCode = http.StatusAccepted
		message += " The config is persisted but runtime activation is still pending; poll /config/hash until status is active."
	case "active":
		message += " Runtime activation is complete."
	default:
		message += " Router reload will continue asynchronously."
	}
	w.Header().Set("ETag", etag)
	logging.Infof(
		"Router config mutation committed via API: action=%s, version=%s, size=%d bytes, sourceConfigPath=%s, runtimeConfigPath=%s",
		action,
		version,
		len(yamlBytes),
		paths.sourcePath,
		paths.runtimePath,
	)
	configCleanupBackups(backupDir)
	s.writeJSONResponse(w, responseCode, RouterConfigUpdateResponse{
		Status:        responseStatus,
		Version:       version,
		ETag:          etag,
		RuntimeStatus: runtimeStatus,
		RuntimeHash:   runtimeHash,
		Message:       message,
	})
	return true
}

func configFileHash(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}

func (s *ClassificationAPIServer) activeConfigDocumentHash() string {
	if s.runtimeRegistry != nil {
		if cfg := s.runtimeRegistry.CurrentConfig(); cfg != nil {
			return cfg.DocumentHash
		}
		return ""
	}
	if cfg := s.currentConfig(); cfg != nil {
		return cfg.DocumentHash
	}
	return ""
}

// waitForRuntimeConfigActivation closes the read-after-write gap for the
// shared runtime used by the production Router. The file watcher performs all
// expensive preparation off to the side and publishes the matching document
// hash only after the new router and classification service are atomically
// available. Legacy/test servers without a runtime registry keep their
// asynchronous behavior.
func (s *ClassificationAPIServer) waitForRuntimeConfigActivation(runtimePath string) (string, string) {
	runtimeHash, err := configFileHash(runtimePath)
	if err != nil {
		return "", "unknown"
	}
	if s.runtimeRegistry == nil {
		return runtimeHash, "unknown"
	}

	const activationTimeout = 20 * time.Second
	deadline := time.Now().Add(activationTimeout)
	for {
		if s.activeConfigDocumentHash() == runtimeHash {
			return runtimeHash, "active"
		}
		if time.Now().After(deadline) {
			return runtimeHash, "pending"
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func syncRuntimeConfigOrRestore(paths configPersistencePaths, previousData []byte) error {
	_, syncErr := runtimeConfigSyncRunner(paths.sourcePath)
	if syncErr == nil {
		return nil
	}
	if restoreErr := restoreSourceConfig(paths.sourcePath, previousData); restoreErr != nil {
		return errors.Join(syncErr, fmt.Errorf("failed to restore source config: %w", restoreErr))
	}
	if len(previousData) == 0 {
		return syncErr
	}
	if _, restoreSyncErr := runtimeConfigSyncRunner(paths.sourcePath); restoreSyncErr != nil {
		return errors.Join(syncErr, fmt.Errorf("source config restored but runtime resync failed: %w", restoreSyncErr))
	}
	return syncErr
}

func restoreSourceConfig(sourcePath string, previousData []byte) error {
	if len(previousData) > 0 {
		return writeConfigAtomically(sourcePath, previousData)
	}
	if err := os.Remove(sourcePath); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
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
