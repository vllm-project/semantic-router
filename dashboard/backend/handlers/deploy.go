package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

// deployMu ensures only one deploy operation at a time
var deployMu sync.Mutex

// maxBackups is the maximum number of config backups to keep
const maxBackups = 10

// DeployRequest is the JSON body for a DSL deploy request
type DeployRequest struct {
	// YAML is the compiled config YAML from the DSL compiler (user-friendly format)
	YAML string `json:"yaml"`
	// DSL is the original DSL source (archived for audit trail)
	DSL string `json:"dsl,omitempty"`
}

// DeployResponse is the JSON response for a deploy operation
type DeployResponse struct {
	Status  string `json:"status"`
	Version string `json:"version"`
	Message string `json:"message,omitempty"`
}

// ConfigVersion represents a backup version entry
type ConfigVersion struct {
	Version   string `json:"version"`
	Timestamp string `json:"timestamp"`
	Source    string `json:"source"` // "dsl" or "manual"
	Filename  string `json:"filename"`
}

// DeployPreviewResponse contains the before/after YAML for diff comparison
type DeployPreviewResponse struct {
	Current string `json:"current"` // Current config.yaml content
	Preview string `json:"preview"` // What config.yaml will look like after deploy
}

// DeployPreviewHandler returns the current config and the merged preview
// so the frontend can show a side-by-side diff before confirming deploy.
// POST /api/router/config/deploy/preview
func DeployPreviewHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req DeployRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		if strings.TrimSpace(req.YAML) == "" {
			http.Error(w, "YAML content is required", http.StatusBadRequest)
			return
		}

		// Validate the incoming YAML
		yamlBytes := []byte(req.YAML)
		var yamlMap interface{}
		if err := yaml.Unmarshal(yamlBytes, &yamlMap); err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"error":   "yaml_parse_error",
				"message": fmt.Sprintf("Invalid YAML syntax: %v", err),
			})
			return
		}

		// Read current config
		currentData, err := os.ReadFile(configPath)
		if err != nil {
			currentData = []byte("# No existing config\n")
		}

		// Compute merged preview (same logic as deployDirectWrite)
		previewBytes := yamlBytes
		if len(currentData) > 0 {
			existingMap := make(map[string]interface{})
			if err := yaml.Unmarshal(currentData, &existingMap); err == nil {
				newMap := make(map[string]interface{})
				if err := yaml.Unmarshal(yamlBytes, &newMap); err == nil {
					merged := deepMerge(existingMap, newMap)
					if mergedYAML, err := yaml.Marshal(merged); err == nil {
						previewBytes = mergedYAML
					}
				}
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(DeployPreviewResponse{
			Current: string(currentData),
			Preview: string(previewBytes),
		})
	}
}

// DeployHandler handles DSL config deployment.
// It follows the same pattern as UpdateConfigHandler:
//  1. Write YAML to config.yaml (the user-facing config)
//  2. Call regenerateRouterConfig (Python CLI) to generate .vllm-sr/router-config.yaml
//  3. Router's fsnotify watcher detects the change and hot-reloads
//
// POST /api/router/config/deploy
func DeployHandler(configPath string, readonlyMode bool, configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if readonlyMode {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusForbidden)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"error":   "readonly_mode",
				"message": "Dashboard is in read-only mode. Deploy is disabled.",
			})
			return
		}

		var req DeployRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		if strings.TrimSpace(req.YAML) == "" {
			http.Error(w, "YAML content is required", http.StatusBadRequest)
			return
		}

		log.Printf("[Deploy] Received: YAML=%d bytes, DSL=%d bytes", len(req.YAML), len(req.DSL))

		deployDirectWrite(w, configPath, configDir, req)
	}
}

// RollbackHandler rolls back to a specific backup version.
// After writing, it calls regenerateRouterConfig to regenerate .vllm-sr/router-config.yaml.
// POST /api/router/config/rollback
func RollbackHandler(configPath string, readonlyMode bool, configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if readonlyMode {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusForbidden)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"error":   "readonly_mode",
				"message": "Dashboard is in read-only mode. Rollback is disabled.",
			})
			return
		}

		var rollbackReq struct {
			Version string `json:"version"`
		}
		if err := json.NewDecoder(r.Body).Decode(&rollbackReq); err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}
		if rollbackReq.Version == "" {
			http.Error(w, "version is required", http.StatusBadRequest)
			return
		}

		rollbackDirectWrite(w, configPath, configDir, rollbackReq.Version)
	}
}

// ConfigVersionsHandler lists available backup versions from local backup directory.
// GET /api/router/config/versions
func ConfigVersionsHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		versionsLocalList(w, configPath)
	}
}

// ==================== Deploy: write config.yaml + regenerate router-config.yaml ====================

func deployDirectWrite(w http.ResponseWriter, configPath string, configDir string, req DeployRequest) {
	// Acquire deploy lock (only one deploy at a time)
	if !deployMu.TryLock() {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "deploy_in_progress",
			"message": "Another deploy operation is in progress. Please try again.",
		})
		return
	}
	defer deployMu.Unlock()

	// Step 1: Validate the YAML parses correctly
	yamlBytes := []byte(req.YAML)
	var yamlMap interface{}
	if err := yaml.Unmarshal(yamlBytes, &yamlMap); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "yaml_parse_error",
			"message": fmt.Sprintf("Invalid YAML syntax: %v", err),
		})
		return
	}

	// Step 2: Deep merge with existing config (same as UpdateConfigHandler)
	existingData, err := os.ReadFile(configPath)
	if err == nil && len(existingData) > 0 {
		existingMap := make(map[string]interface{})
		if err := yaml.Unmarshal(existingData, &existingMap); err == nil {
			newMap := make(map[string]interface{})
			if err := yaml.Unmarshal(yamlBytes, &newMap); err == nil {
				merged := deepMerge(existingMap, newMap)
				if mergedYAML, err := yaml.Marshal(merged); err == nil {
					yamlBytes = mergedYAML
				}
			}
		}
	}

	// Step 3: Create backup of current config
	backupDir := filepath.Join(configDir, ".vllm-sr", "config-backups")
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		log.Printf("Warning: failed to create backup directory: %v", err)
	}

	version := time.Now().Format("20060102-150405")

	if len(existingData) > 0 {
		backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version))
		if err := os.WriteFile(backupFile, existingData, 0o644); err != nil {
			log.Printf("Warning: failed to create backup: %v", err)
		} else {
			log.Printf("[Deploy] Config backup created: %s", backupFile)
		}
	}

	// Step 4: Archive DSL source (for audit trail)
	if req.DSL != "" {
		dslDir := filepath.Join(configDir, ".vllm-sr")
		dslFile := filepath.Join(dslDir, "config.dsl")
		if err := os.WriteFile(dslFile, []byte(req.DSL), 0o644); err != nil {
			log.Printf("Warning: failed to archive DSL source: %v", err)
		}
	}

	// Step 5: Atomic write to config.yaml
	tmpConfigFile := configPath + ".tmp"
	if err := os.WriteFile(tmpConfigFile, yamlBytes, 0o644); err != nil {
		http.Error(w, fmt.Sprintf("Failed to write config: %v", err), http.StatusInternalServerError)
		return
	}
	if err := os.Rename(tmpConfigFile, configPath); err != nil {
		if writeErr := os.WriteFile(configPath, yamlBytes, 0o644); writeErr != nil {
			http.Error(w, fmt.Sprintf("Failed to write config: %v", writeErr), http.StatusInternalServerError)
			return
		}
	}

	log.Printf("[Deploy] Config written to %s: version=%s, size=%d bytes", configPath, version, len(yamlBytes))

	// Step 6: Regenerate .vllm-sr/router-config.yaml (same as UpdateConfigHandler)
	// This calls the Python CLI to convert nested user config to flat Router format.
	// The Router's fsnotify watcher will detect the change and hot-reload.
	go regenerateRouterConfig(configPath, configDir)

	// Step 7: Clean up old backups (keep only maxBackups most recent)
	cleanupBackups(backupDir)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(DeployResponse{
		Status:  "success",
		Version: version,
		Message: "Config deployed successfully. Router will reload automatically via fsnotify.",
	})
}

func rollbackDirectWrite(w http.ResponseWriter, configPath string, configDir string, version string) {
	// Acquire deploy lock
	if !deployMu.TryLock() {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "deploy_in_progress",
			"message": "Another deploy operation is in progress.",
		})
		return
	}
	defer deployMu.Unlock()

	// Find backup file
	backupDir := filepath.Join(configDir, ".vllm-sr", "config-backups")
	backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version))

	backupData, err := os.ReadFile(backupFile)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "version_not_found",
			"message": fmt.Sprintf("Backup version %s not found", version),
		})
		return
	}

	// Validate backup YAML syntax
	var yamlCheck interface{}
	if err := yaml.Unmarshal(backupData, &yamlCheck); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "backup_invalid",
			"message": fmt.Sprintf("Backup config has invalid YAML: %v", err),
		})
		return
	}

	// Back up current config before rollback
	currentVersion := time.Now().Format("20060102-150405")
	existingData, err := os.ReadFile(configPath)
	if err == nil && len(existingData) > 0 {
		preRollbackFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", currentVersion))
		_ = os.WriteFile(preRollbackFile, existingData, 0o644)
	}

	// Atomic write to config.yaml
	tmpConfigFile := configPath + ".tmp"
	if err := os.WriteFile(tmpConfigFile, backupData, 0o644); err != nil {
		http.Error(w, fmt.Sprintf("Failed to write config: %v", err), http.StatusInternalServerError)
		return
	}
	if err := os.Rename(tmpConfigFile, configPath); err != nil {
		if writeErr := os.WriteFile(configPath, backupData, 0o644); writeErr != nil {
			http.Error(w, fmt.Sprintf("Failed to write config: %v", writeErr), http.StatusInternalServerError)
			return
		}
	}

	log.Printf("[Rollback] Config rolled back to version %s, written to %s", version, configPath)

	// Regenerate .vllm-sr/router-config.yaml (same as UpdateConfigHandler)
	go regenerateRouterConfig(configPath, configDir)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(DeployResponse{
		Status:  "success",
		Version: version,
		Message: fmt.Sprintf("Rolled back to version %s. Router will reload automatically.", version),
	})
}

func versionsLocalList(w http.ResponseWriter, configPath string) {
	configDir := filepath.Dir(configPath)
	backupDir := filepath.Join(configDir, ".vllm-sr", "config-backups")

	versions := []ConfigVersion{}

	entries, err := os.ReadDir(backupDir)
	if err != nil {
		// No backups yet, return empty list
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(versions)
		return
	}

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasPrefix(entry.Name(), "config.") || !strings.HasSuffix(entry.Name(), ".yaml") {
			continue
		}

		// Extract version from filename: config.20060102-150405.yaml
		name := entry.Name()
		versionStr := strings.TrimPrefix(name, "config.")
		versionStr = strings.TrimSuffix(versionStr, ".yaml")

		// Parse timestamp for display
		t, err := time.Parse("20060102-150405", versionStr)
		timestamp := versionStr
		if err == nil {
			timestamp = t.Format("2006-01-02 15:04:05")
		}

		versions = append(versions, ConfigVersion{
			Version:   versionStr,
			Timestamp: timestamp,
			Source:    "dsl",
			Filename:  name,
		})
	}

	// Sort by version descending (newest first)
	sort.Slice(versions, func(i, j int) bool {
		return versions[i].Version > versions[j].Version
	})

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(versions)
}

// deepMerge recursively merges src into dst. For map values, it recurses;
// for slices and scalars, src overwrites dst. dst is modified in place.
func deepMerge(dst, src map[string]interface{}) map[string]interface{} {
	for key, srcVal := range src {
		if dstVal, exists := dst[key]; exists {
			// Both sides are maps → recurse
			if dstMap, ok := dstVal.(map[string]interface{}); ok {
				if srcMap, ok := srcVal.(map[string]interface{}); ok {
					dst[key] = deepMerge(dstMap, srcMap)
					continue
				}
			}
			// Also handle map[interface{}]interface{} from yaml.v2
			if dstMap, ok := toStringKeyMap(dstVal); ok {
				if srcMap, ok := toStringKeyMap(srcVal); ok {
					dst[key] = deepMerge(dstMap, srcMap)
					continue
				}
			}
		}
		// Non-map or new key → overwrite
		dst[key] = srcVal
	}
	return dst
}

// toStringKeyMap converts map[interface{}]interface{} (yaml.v2) to map[string]interface{}
func toStringKeyMap(v interface{}) (map[string]interface{}, bool) {
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

// cleanupBackups removes old backups beyond maxBackups
func cleanupBackups(backupDir string) {
	entries, err := os.ReadDir(backupDir)
	if err != nil {
		return
	}

	// Filter yaml backup files
	var backups []os.DirEntry
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasPrefix(entry.Name(), "config.") && strings.HasSuffix(entry.Name(), ".yaml") {
			backups = append(backups, entry)
		}
	}

	if len(backups) <= maxBackups {
		return
	}

	// Sort by name ascending (oldest first since names contain timestamps)
	sort.Slice(backups, func(i, j int) bool {
		return backups[i].Name() < backups[j].Name()
	})

	// Remove oldest backups
	toRemove := len(backups) - maxBackups
	for i := 0; i < toRemove; i++ {
		path := filepath.Join(backupDir, backups[i].Name())
		if err := os.Remove(path); err != nil {
			log.Printf("Warning: failed to remove old backup %s: %v", path, err)
		} else {
			log.Printf("Removed old backup: %s", backups[i].Name())
		}
	}
}
