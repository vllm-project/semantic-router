//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	yamlv2 "gopkg.in/yaml.v2"
	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// deployMu ensures only one deploy operation at a time
var deployMu sync.Mutex

// maxBackups is the maximum number of config backups to keep
const maxBackups = 10

// ConfigDeployRequest is the JSON body for a config deploy request
type ConfigDeployRequest struct {
	// YAML is the compiled config YAML (user-friendly format)
	YAML string `json:"yaml"`
	// DSL is the original DSL source (archived for audit trail)
	DSL string `json:"dsl,omitempty"`
}

// ConfigDeployResponse is the JSON response for a deploy operation
type ConfigDeployResponse struct {
	Status  string `json:"status"`
	Version string `json:"version"`
	Message string `json:"message,omitempty"`
}

// ConfigVersionEntry represents a backup version entry
type ConfigVersionEntry struct {
	Version   string `json:"version"`
	Timestamp string `json:"timestamp"`
	Source    string `json:"source"` // "dsl" or "manual"
	Filename  string `json:"filename"`
}

// handleConfigDeploy handles POST /config/deploy
// The Router writes its own config file and triggers hot-reload via fsnotify.
func (s *ClassificationAPIServer) handleConfigDeploy(w http.ResponseWriter, r *http.Request) {
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return
	}

	// Acquire deploy lock (only one deploy at a time)
	if !deployMu.TryLock() {
		s.writeErrorResponse(w, http.StatusConflict, "DEPLOY_IN_PROGRESS", "Another deploy operation is in progress. Please try again.")
		return
	}
	defer deployMu.Unlock()

	var req ConfigDeployRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	if strings.TrimSpace(req.YAML) == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "YAML content is required")
		return
	}

	// Step 1: Validate YAML syntax
	yamlBytes := []byte(req.YAML)
	var yamlMap interface{}
	if err := yaml.Unmarshal(yamlBytes, &yamlMap); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "YAML_PARSE_ERROR", fmt.Sprintf("Invalid YAML syntax: %v", err))
		return
	}

	// Step 2: Validate using router's config parser (authoritative validation)
	tempFile := filepath.Join(os.TempDir(), fmt.Sprintf("deploy_validate_%d.yaml", time.Now().UnixNano()))
	if err := os.WriteFile(tempFile, yamlBytes, 0o644); err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "TEMP_FILE_ERROR", fmt.Sprintf("Failed to create temp file: %v", err))
		return
	}
	defer func() { _ = os.Remove(tempFile) }()

	parsedCfg, err := config.Parse(tempFile)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "CONFIG_VALIDATION_ERROR", fmt.Sprintf("Config validation failed: %v", err))
		return
	}

	// Log decisions after parse
	logging.Infof("[Deploy] After config.Parse: decisions=%d", len(parsedCfg.Decisions))
	for i, d := range parsedCfg.Decisions {
		logging.Infof("[Deploy]   parsed decision[%d]: name=%q, modelRefs=%d, priority=%d", i, d.Name, len(d.ModelRefs), d.Priority)
	}

	// Step 2b: Normalize to flat RouterConfig YAML format.
	flatYAML, err := yamlv2.Marshal(parsedCfg)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NORMALIZE_ERROR", fmt.Sprintf("Failed to normalize config: %v", err))
		return
	}

	logging.Infof("[Deploy] After yamlv2.Marshal (normalize): flatYAML size=%d bytes", len(flatYAML))

	// Step 2c: Deep merge with existing config.
	// DSL only covers signals/decisions/models — infrastructure fields must be preserved.
	existingData, err := os.ReadFile(s.configPath)
	if err == nil && len(existingData) > 0 {
		logging.Infof("[Deploy] Existing config found: path=%s, size=%d bytes", s.configPath, len(existingData))
		existingMap := make(map[string]interface{})
		if err := yamlv2.Unmarshal(existingData, &existingMap); err == nil {
			newMap := make(map[string]interface{})
			if err := yamlv2.Unmarshal(flatYAML, &newMap); err == nil {
				// Log decisions in newMap before merge
				if decisionsRaw, ok := newMap["decisions"]; ok {
					if decisionsSlice, ok := decisionsRaw.([]interface{}); ok {
						logging.Infof("[Deploy] newMap decisions before merge: count=%d", len(decisionsSlice))
						for i, d := range decisionsSlice {
							if dm, ok := d.(map[interface{}]interface{}); ok {
								logging.Infof("[Deploy]   newMap decision[%d]: name=%v", i, dm["name"])
							} else if dm, ok := d.(map[string]interface{}); ok {
								logging.Infof("[Deploy]   newMap decision[%d]: name=%v", i, dm["name"])
							}
						}
					} else {
						logging.Infof("[Deploy] newMap decisions is not a slice: type=%T", decisionsRaw)
					}
				} else {
					logging.Warnf("[Deploy] newMap has NO 'decisions' key!")
				}

				merged := configDeepMerge(existingMap, newMap)

				// Log decisions in merged map
				if decisionsRaw, ok := merged["decisions"]; ok {
					if decisionsSlice, ok := decisionsRaw.([]interface{}); ok {
						logging.Infof("[Deploy] After merge: decisions count=%d", len(decisionsSlice))
						for i, d := range decisionsSlice {
							if dm, ok := d.(map[interface{}]interface{}); ok {
								logging.Infof("[Deploy]   merged decision[%d]: name=%v", i, dm["name"])
							} else if dm, ok := d.(map[string]interface{}); ok {
								logging.Infof("[Deploy]   merged decision[%d]: name=%v", i, dm["name"])
							}
						}
					}
				}

				if mergedYAML, err := yamlv2.Marshal(merged); err == nil {
					flatYAML = mergedYAML
				}
			}
		}
	} else {
		logging.Infof("[Deploy] No existing config at path=%s (err=%v), using new config as-is", s.configPath, err)
	}
	yamlBytes = flatYAML

	// Step 3: Create backup of current config
	configDir := filepath.Dir(s.configPath)
	backupDir := filepath.Join(configDir, ".vllm-sr", "config-backups")
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		logging.Warnf("Failed to create backup directory: %v", err)
	}

	version := time.Now().Format("20060102-150405")

	existingData, readErr := os.ReadFile(s.configPath)
	if readErr == nil && len(existingData) > 0 {
		backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version))
		if err := os.WriteFile(backupFile, existingData, 0o644); err != nil {
			logging.Warnf("Failed to create backup: %v", err)
		} else {
			logging.Infof("Config backup created: %s", backupFile)
		}
	}

	// Step 4: Archive DSL source (for audit trail)
	if req.DSL != "" {
		dslDir := filepath.Join(configDir, ".vllm-sr")
		dslFile := filepath.Join(dslDir, "config.dsl")
		if err := os.WriteFile(dslFile, []byte(req.DSL), 0o644); err != nil {
			logging.Warnf("Failed to archive DSL source: %v", err)
		}
	}

	// Step 5: Atomic write — write to temp file then rename
	// This triggers fsnotify and the Router hot-reloads automatically.
	tmpConfigFile := s.configPath + ".tmp"
	if err := os.WriteFile(tmpConfigFile, yamlBytes, 0o644); err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "WRITE_ERROR", fmt.Sprintf("Failed to write config: %v", err))
		return
	}
	if err := os.Rename(tmpConfigFile, s.configPath); err != nil {
		// Fallback to direct write if rename fails (e.g., cross-device)
		if writeErr := os.WriteFile(s.configPath, yamlBytes, 0o644); writeErr != nil {
			s.writeErrorResponse(w, http.StatusInternalServerError, "WRITE_ERROR", fmt.Sprintf("Failed to write config: %v", writeErr))
			return
		}
	}

	logging.Infof("Config deployed via API: version=%s, size=%d bytes, configPath=%s", version, len(yamlBytes), s.configPath)

	// Step 6: Clean up old backups
	configCleanupBackups(backupDir)

	s.writeJSONResponse(w, http.StatusOK, ConfigDeployResponse{
		Status:  "success",
		Version: version,
		Message: "Config deployed successfully. Router will reload automatically via fsnotify.",
	})
}

// handleConfigRollback handles POST /config/rollback
func (s *ClassificationAPIServer) handleConfigRollback(w http.ResponseWriter, r *http.Request) {
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return
	}

	if !deployMu.TryLock() {
		s.writeErrorResponse(w, http.StatusConflict, "DEPLOY_IN_PROGRESS", "Another deploy operation is in progress.")
		return
	}
	defer deployMu.Unlock()

	var req struct {
		Version string `json:"version"`
	}
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}
	if req.Version == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "version is required")
		return
	}

	configDir := filepath.Dir(s.configPath)
	backupDir := filepath.Join(configDir, ".vllm-sr", "config-backups")
	backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", req.Version))

	backupData, err := os.ReadFile(backupFile)
	if err != nil {
		s.writeErrorResponse(w, http.StatusNotFound, "VERSION_NOT_FOUND", fmt.Sprintf("Backup version %s not found", req.Version))
		return
	}

	// Validate backup config
	tempFile := filepath.Join(os.TempDir(), fmt.Sprintf("rollback_validate_%d.yaml", time.Now().UnixNano()))
	if err := os.WriteFile(tempFile, backupData, 0o644); err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "TEMP_FILE_ERROR", fmt.Sprintf("Failed to validate backup: %v", err))
		return
	}
	defer func() { _ = os.Remove(tempFile) }()

	if _, err := config.Parse(tempFile); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "BACKUP_INVALID", fmt.Sprintf("Backup config is invalid: %v", err))
		return
	}

	// Back up current config before rollback
	currentVersion := time.Now().Format("20060102-150405")
	existingData, err := os.ReadFile(s.configPath)
	if err == nil && len(existingData) > 0 {
		preRollbackFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", currentVersion))
		_ = os.WriteFile(preRollbackFile, existingData, 0o644)
	}

	// Atomic write
	tmpConfigFile := s.configPath + ".tmp"
	if err := os.WriteFile(tmpConfigFile, backupData, 0o644); err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "WRITE_ERROR", fmt.Sprintf("Failed to write config: %v", err))
		return
	}
	if err := os.Rename(tmpConfigFile, s.configPath); err != nil {
		if writeErr := os.WriteFile(s.configPath, backupData, 0o644); writeErr != nil {
			s.writeErrorResponse(w, http.StatusInternalServerError, "WRITE_ERROR", fmt.Sprintf("Failed to write config: %v", writeErr))
			return
		}
	}

	logging.Infof("Config rolled back to version %s via API", req.Version)

	s.writeJSONResponse(w, http.StatusOK, ConfigDeployResponse{
		Status:  "success",
		Version: req.Version,
		Message: fmt.Sprintf("Rolled back to version %s. Router will reload automatically.", req.Version),
	})
}

// handleConfigVersions handles GET /config/versions
func (s *ClassificationAPIServer) handleConfigVersions(w http.ResponseWriter, _ *http.Request) {
	if s.configPath == "" {
		s.writeJSONResponse(w, http.StatusOK, []ConfigVersionEntry{})
		return
	}

	configDir := filepath.Dir(s.configPath)
	backupDir := filepath.Join(configDir, ".vllm-sr", "config-backups")

	versions := []ConfigVersionEntry{}

	entries, err := os.ReadDir(backupDir)
	if err != nil {
		s.writeJSONResponse(w, http.StatusOK, versions)
		return
	}

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasPrefix(entry.Name(), "config.") || !strings.HasSuffix(entry.Name(), ".yaml") {
			continue
		}
		name := entry.Name()
		versionStr := strings.TrimPrefix(name, "config.")
		versionStr = strings.TrimSuffix(versionStr, ".yaml")

		t, err := time.Parse("20060102-150405", versionStr)
		timestamp := versionStr
		if err == nil {
			timestamp = t.Format("2006-01-02 15:04:05")
		}

		versions = append(versions, ConfigVersionEntry{
			Version:   versionStr,
			Timestamp: timestamp,
			Source:    "dsl",
			Filename:  name,
		})
	}

	sort.Slice(versions, func(i, j int) bool {
		return versions[i].Version > versions[j].Version
	})

	s.writeJSONResponse(w, http.StatusOK, versions)
}

// handleConfigGet handles GET /config/router — returns current config as JSON
func (s *ClassificationAPIServer) handleConfigGet(w http.ResponseWriter, _ *http.Request) {
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return
	}

	data, err := os.ReadFile(s.configPath)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "READ_ERROR", fmt.Sprintf("Failed to read config: %v", err))
		return
	}

	var cfgMap interface{}
	if err := yaml.Unmarshal(data, &cfgMap); err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "PARSE_ERROR", fmt.Sprintf("Failed to parse config: %v", err))
		return
	}

	s.writeJSONResponse(w, http.StatusOK, cfgMap)
}

// configDeepMerge recursively merges src into dst.
func configDeepMerge(dst, src map[string]interface{}) map[string]interface{} {
	for key, srcVal := range src {
		if dstVal, exists := dst[key]; exists {
			if dstMap, ok := dstVal.(map[string]interface{}); ok {
				if srcMap, ok := srcVal.(map[string]interface{}); ok {
					dst[key] = configDeepMerge(dstMap, srcMap)
					continue
				}
			}
			if dstMap, ok := configToStringKeyMap(dstVal); ok {
				if srcMap, ok := configToStringKeyMap(srcVal); ok {
					dst[key] = configDeepMerge(dstMap, srcMap)
					continue
				}
			}
		}
		dst[key] = srcVal
	}
	return dst
}

func configToStringKeyMap(v interface{}) (map[string]interface{}, bool) {
	switch m := v.(type) {
	case map[string]interface{}:
		return m, true
	case map[interface{}]interface{}:
		result := make(map[string]interface{}, len(m))
		for k, val := range m {
			result[fmt.Sprintf("%v", k)] = val
		}
		return result, true
	}
	return nil, false
}

func configCleanupBackups(backupDir string) {
	entries, err := os.ReadDir(backupDir)
	if err != nil {
		return
	}

	var backups []os.DirEntry
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasPrefix(entry.Name(), "config.") && strings.HasSuffix(entry.Name(), ".yaml") {
			backups = append(backups, entry)
		}
	}

	if len(backups) <= maxBackups {
		return
	}

	sort.Slice(backups, func(i, j int) bool {
		return backups[i].Name() < backups[j].Name()
	})

	toRemove := len(backups) - maxBackups
	for i := 0; i < toRemove; i++ {
		path := filepath.Join(backupDir, backups[i].Name())
		if err := os.Remove(path); err != nil {
			logging.Warnf("Failed to remove old backup %s: %v", path, err)
		} else {
			logging.Infof("Removed old backup: %s", backups[i].Name())
		}
	}
}
