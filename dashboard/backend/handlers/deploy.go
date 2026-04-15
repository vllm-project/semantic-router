package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	// BaseYAML is the full canonical config imported into the builder, if any.
	// Routing-only fragments are ignored as merge bases.
	BaseYAML string `json:"baseYaml,omitempty"`
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

		if _, err := decodeYAMLTaggedBytes[routingFragmentDocument]([]byte(req.YAML)); err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"error":   "yaml_parse_error",
				"message": fmt.Sprintf("Invalid YAML syntax: %v", err),
			})
			return
		}

		currentData, err := os.ReadFile(configPath)
		currentForDiffBytes := currentData
		if err != nil {
			currentForDiffBytes = []byte("# No existing config\n")
		}

		previewBytes, err := mergeDeployPayload(currentData, req)
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"error":   "deploy_preview_error",
				"message": err.Error(),
			})
			return
		}

		currentForDiff := canonicalizeYAMLForDiff(currentForDiffBytes)
		previewForDiff := canonicalizeYAMLForDiff(previewBytes)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(DeployPreviewResponse{
			Current: currentForDiff,
			Preview: previewForDiff,
		})
	}
}

// DeployHandler handles DSL config deployment.
// It writes the user-facing config.yaml, then synchronously propagates the
// change to Router and Envoy before returning success.
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

// RollbackHandler rolls back to a specific backup version and synchronously
// propagates the restored config to Router and Envoy.
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

// ==================== Deploy: write canonical config.yaml ====================

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

	fragmentBytes := []byte(req.YAML)
	if _, err := decodeYAMLTaggedBytes[routingFragmentDocument](fragmentBytes); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "yaml_parse_error",
			"message": fmt.Sprintf("Invalid YAML syntax: %v", err),
		})
		return
	}

	existingData, err := os.ReadFile(configPath)
	if err != nil {
		existingData = nil
	}

	// Step 2: Deep merge the routing fragment into the deploy base.
	yamlBytes, err := mergeDeployPayload(existingData, req)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "config_merge_error",
			"message": err.Error(),
		})
		return
	}

	if _, err := routerconfig.ParseYAMLBytes(yamlBytes); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "config_validation_error",
			"message": fmt.Sprintf("Merged config validation failed: %v", err),
		})
		return
	}

	// Step 3: Create backup of current config
	version := createConfigBackup(configDir, existingData)

	// Step 4: Archive DSL source (for audit trail)
	archiveDeployDSL(configDir, req.DSL)

	// Step 5: Atomic write to config.yaml
	if err := writeConfigAtomically(configPath, yamlBytes); err != nil {
		http.Error(w, fmt.Sprintf("Failed to write config: %v", err), http.StatusInternalServerError)
		return
	}

	log.Printf("[Deploy] Config written to %s: version=%s, size=%d bytes", configPath, version, len(yamlBytes))

	// Step 6: Propagate the new config to the managed runtime before returning.
	if err := applyWrittenConfig(configPath, configDir, existingData, true); err != nil {
		http.Error(w, formatRuntimeApplyError("Failed to apply deployed config to runtime", err), http.StatusInternalServerError)
		return
	}

	// Step 7: Clean up old backups (keep only maxBackups most recent)
	cleanupBackups(configBackupDir(configDir))

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(DeployResponse{
		Status:  "success",
		Version: version,
		Message: "Config deployed successfully. Router and Envoy have been updated.",
	})
}

func mergeDeployPayload(currentData []byte, req DeployRequest) ([]byte, error) {
	fragmentBytes := []byte(req.YAML)
	baseData, err := resolveDeployBaseYAML(currentData, req.BaseYAML)
	if err != nil {
		return nil, err
	}
	if len(baseData) == 0 {
		return fragmentBytes, nil
	}

	baseConfig, err := decodeYAMLTaggedBytes[routerconfig.CanonicalConfig](baseData)
	if err != nil {
		return nil, fmt.Errorf("failed to parse deploy base config: %w", err)
	}

	fragmentConfig, err := decodeYAMLTaggedBytes[routingFragmentDocument](fragmentBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to parse compiled routing fragment: %w", err)
	}

	baseConfig.Routing = mergeCanonicalRouting(baseConfig.Routing, fragmentConfig.Routing)
	mergeFragmentGlobal(&baseConfig, fragmentConfig.Global)
	mergedYAML, err := marshalYAMLBytes(baseConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal merged config: %w", err)
	}
	return mergedYAML, nil
}

func mergeCanonicalRouting(base, patch routerconfig.CanonicalRouting) routerconfig.CanonicalRouting {
	merged := base
	if len(patch.ModelCards) > 0 {
		merged.ModelCards = patch.ModelCards
	}
	merged.Signals = mergeCanonicalSignals(base.Signals, patch.Signals)
	if len(patch.Decisions) > 0 {
		merged.Decisions = patch.Decisions
	}
	return merged
}

func mergeCanonicalSignals(base, patch routerconfig.CanonicalSignals) routerconfig.CanonicalSignals {
	merged := base
	if len(patch.Keywords) > 0 {
		merged.Keywords = patch.Keywords
	}
	if len(patch.Embeddings) > 0 {
		merged.Embeddings = patch.Embeddings
	}
	if len(patch.Domains) > 0 {
		merged.Domains = patch.Domains
	}
	if len(patch.FactCheck) > 0 {
		merged.FactCheck = patch.FactCheck
	}
	if len(patch.UserFeedbacks) > 0 {
		merged.UserFeedbacks = patch.UserFeedbacks
	}
	if len(patch.Preferences) > 0 {
		merged.Preferences = patch.Preferences
	}
	if len(patch.Language) > 0 {
		merged.Language = patch.Language
	}
	if len(patch.Context) > 0 {
		merged.Context = patch.Context
	}
	if len(patch.Complexity) > 0 {
		merged.Complexity = patch.Complexity
	}
	if len(patch.Modality) > 0 {
		merged.Modality = patch.Modality
	}
	if len(patch.RoleBindings) > 0 {
		merged.RoleBindings = patch.RoleBindings
	}
	if len(patch.Jailbreak) > 0 {
		merged.Jailbreak = patch.Jailbreak
	}
	if len(patch.PII) > 0 {
		merged.PII = patch.PII
	}
	return merged
}

func mergeFragmentGlobal(base *routerconfig.CanonicalConfig, frag *globalFragment) {
	if frag == nil || frag.Services == nil || frag.Services.RateLimit == nil {
		return
	}
	if base.Global == nil {
		base.Global = &routerconfig.CanonicalGlobal{}
	}
	base.Global.Services.RateLimit = *frag.Services.RateLimit
}

func resolveDeployBaseYAML(currentData []byte, providedBase string) ([]byte, error) {
	if strings.TrimSpace(providedBase) == "" {
		return currentData, nil
	}

	providedBaseBytes := []byte(providedBase)
	if _, err := parseYAMLDocument(providedBaseBytes); err != nil {
		return nil, fmt.Errorf("invalid imported base config: %w", err)
	}
	looksLikeFullBase, err := looksLikeFullCanonicalDeployBase(providedBaseBytes)
	if err != nil {
		return nil, fmt.Errorf("invalid imported base config: %w", err)
	}
	if !looksLikeFullBase {
		return currentData, nil
	}
	return providedBaseBytes, nil
}

func looksLikeFullCanonicalDeployBase(raw []byte) (bool, error) {
	doc, err := parseYAMLDocument(raw)
	if err != nil {
		return false, err
	}
	root, err := documentMappingNode(doc)
	if err != nil {
		return false, err
	}
	if mappingValueNode(root, "routing") == nil {
		return false, nil
	}

	for _, key := range []string{"version", "listeners", "providers", "global"} {
		if mappingValueNode(root, key) != nil {
			return true, nil
		}
	}
	return false, nil
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
	backupData, err := readConfigBackup(configDir, version)
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
	if _, unmarshalErr := parseYAMLDocument(backupData); unmarshalErr != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "backup_invalid",
			"message": fmt.Sprintf("Backup config has invalid YAML: %v", unmarshalErr),
		})
		return
	}

	// Back up current config before rollback
	existingData := snapshotCurrentConfigBeforeRollback(configPath, configDir)

	// Atomic write to config.yaml
	if err := writeConfigAtomically(configPath, backupData); err != nil {
		http.Error(w, fmt.Sprintf("Failed to write config: %v", err), http.StatusInternalServerError)
		return
	}

	log.Printf("[Rollback] Config rolled back to version %s, written to %s", version, configPath)

	if err := applyWrittenConfig(configPath, configDir, existingData, true); err != nil {
		http.Error(w, formatRuntimeApplyError("Failed to apply rolled back config to runtime", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(DeployResponse{
		Status:  "success",
		Version: version,
		Message: fmt.Sprintf("Rolled back to version %s. Router and Envoy have been updated.", version),
	})
}

// canonicalizeYAMLForDiff converts YAML into a normalized representation so
// order-only key changes do not produce noisy diffs in the preview modal.
func canonicalizeYAMLForDiff(raw []byte) string {
	text := string(raw)
	if strings.TrimSpace(text) == "" {
		return text
	}
	if strings.Contains(text, "# No existing config") {
		return text
	}

	if looksLikeFullBase, err := looksLikeFullCanonicalDeployBase(raw); err == nil && looksLikeFullBase {
		if configData, decodeErr := decodeYAMLTaggedBytes[routerconfig.CanonicalConfig](raw); decodeErr == nil {
			if canonical, marshalErr := marshalYAMLBytes(configData); marshalErr == nil {
				return string(canonical)
			}
		}
	}
	if fragment, err := decodeYAMLTaggedBytes[routingFragmentDocument](raw); err == nil {
		if canonical, marshalErr := marshalYAMLBytes(fragment); marshalErr == nil {
			return string(canonical)
		}
	}
	return text
}
