//go:build !windows && cgo

package apiserver

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// deployMu ensures only one deploy operation at a time
var deployMu sync.Mutex

// maxBackups is the maximum number of config backups to keep
const maxBackups = 10

// RouterConfigUpdateRequest is the JSON body for a router config update request.
type RouterConfigUpdateRequest struct {
	// YAML is the router config YAML payload.
	YAML string `json:"yaml"`
	// DSL is the original DSL source (archived for audit trail).
	DSL string `json:"dsl,omitempty"`
}

// RouterConfigUpdateResponse is the JSON response for a router config mutation.
type RouterConfigUpdateResponse struct {
	Status  string `json:"status"`
	Version string `json:"version"`
	Message string `json:"message,omitempty"`
}

// RouterConfigVersionEntry represents a backup version entry.
type RouterConfigVersionEntry struct {
	Version   string `json:"version"`
	Timestamp string `json:"timestamp"`
	Source    string `json:"source"` // "dsl" or "manual"
	Filename  string `json:"filename"`
}

// handleConfigRollback handles POST /config/router/rollback.
//
//nolint:cyclop,funlen // Legacy rollback flow still owns validation, backup, and atomic write in one handler.
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

	paths := resolveConfigPersistencePaths(s.configPath)
	configDir := filepath.Dir(paths.sourcePath)
	backupDir := filepath.Join(configDir, ".vllm-sr", "config-backups")
	backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", req.Version))

	backupData, err := os.ReadFile(backupFile)
	if err != nil {
		s.writeErrorResponse(w, http.StatusNotFound, "VERSION_NOT_FOUND", fmt.Sprintf("Backup version %s not found", req.Version))
		return
	}

	// Validate backup config
	tempFile := filepath.Clean(filepath.Join(os.TempDir(), fmt.Sprintf("rollback_validate_%d.yaml", time.Now().UnixNano())))
	if writeErr := os.WriteFile(tempFile, backupData, 0o644); writeErr != nil { //nolint:gosec // G703: path is constructed from os.TempDir() + timestamp
		s.writeErrorResponse(w, http.StatusInternalServerError, "TEMP_FILE_ERROR", fmt.Sprintf("Failed to validate backup: %v", writeErr))
		return
	}
	defer func() { _ = os.Remove(tempFile) }()

	if _, parseErr := config.Parse(tempFile); parseErr != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "BACKUP_INVALID", fmt.Sprintf("Backup config is invalid: %v", parseErr))
		return
	}

	// Back up current config before rollback
	currentVersion := time.Now().Format("20060102-150405")
	existingData, err := os.ReadFile(paths.sourcePath)
	if err == nil && len(existingData) > 0 {
		preRollbackFile := filepath.Clean(filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", currentVersion)))
		_ = os.WriteFile(preRollbackFile, existingData, 0o644) //nolint:gosec // G703: path is constructed from validated backup dir + timestamp
	}

	if err := writeConfigAtomically(paths.sourcePath, backupData); err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "WRITE_ERROR", fmt.Sprintf("Failed to write config: %v", err))
		return
	}

	if paths.usesRuntimeOverride() {
		if _, err := runtimeConfigSyncRunner(paths.sourcePath); err != nil {
			s.writeErrorResponse(w, http.StatusInternalServerError, "RUNTIME_SYNC_ERROR", err.Error())
			return
		}
	}

	logging.Infof(
		"Config rolled back to version %s via API: sourceConfigPath=%s, runtimeConfigPath=%s",
		req.Version,
		paths.sourcePath,
		paths.runtimePath,
	)

	s.writeJSONResponse(w, http.StatusOK, RouterConfigUpdateResponse{
		Status:  "success",
		Version: req.Version,
		Message: fmt.Sprintf("Rolled back to version %s. Router will reload automatically.", req.Version),
	})
}

// handleConfigVersions handles GET /config/router/versions.
func (s *ClassificationAPIServer) handleConfigVersions(w http.ResponseWriter, _ *http.Request) {
	if s.configPath == "" {
		s.writeJSONResponse(w, http.StatusOK, []RouterConfigVersionEntry{})
		return
	}

	paths := resolveConfigPersistencePaths(s.configPath)
	configDir := filepath.Dir(paths.sourcePath)
	backupDir := filepath.Join(configDir, ".vllm-sr", "config-backups")

	versions := []RouterConfigVersionEntry{}

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

		versions = append(versions, RouterConfigVersionEntry{
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

// handleConfigGet handles GET /config/router and returns the current router config as JSON.
func (s *ClassificationAPIServer) handleConfigGet(w http.ResponseWriter, _ *http.Request) {
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return
	}

	paths := resolveConfigPersistencePaths(s.configPath)
	data, err := os.ReadFile(paths.sourcePath)
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

// handleConfigHash handles GET /config/hash — returns SHA256 of the active config file.
// Used by the tuning pipeline to confirm a reload completed.
func (s *ClassificationAPIServer) handleConfigHash(w http.ResponseWriter, _ *http.Request) {
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return
	}

	paths := resolveConfigPersistencePaths(s.configPath)
	data, err := os.ReadFile(paths.sourcePath)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "READ_ERROR", fmt.Sprintf("Failed to read config: %v", err))
		return
	}

	hash := sha256.Sum256(data)
	s.writeJSONResponse(w, http.StatusOK, map[string]string{
		"hash": hex.EncodeToString(hash[:]),
	})
}
