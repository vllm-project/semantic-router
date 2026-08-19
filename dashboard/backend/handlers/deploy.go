package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/configprojection"
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
	// Mode controls how the DSL-owned config surface is applied. Merge is the
	// backward-compatible default for partial API fragments. Replace atomically
	// replaces routing, entrypoints, and recipes while preserving the static
	// deployment config outside that surface.
	Mode DeployMode `json:"mode,omitempty"`
}

// DeployMode defines the update semantics for a compiled routing fragment.
type DeployMode string

const (
	DeployModeMerge   DeployMode = "merge"
	DeployModeReplace DeployMode = "replace"
)

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
	release, err := beginOrdinaryRuntimeConfigMutation(configDir)
	if err != nil {
		writeRuntimeConfigMutationError(w, err)
		return
	}
	defer release()

	fragmentBytes := []byte(req.YAML)
	if _, decodeErr := decodeYAMLTaggedBytes[routingFragmentDocument](fragmentBytes); decodeErr != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "yaml_parse_error",
			"message": fmt.Sprintf("Invalid YAML syntax: %v", decodeErr),
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

	refreshConfigProjection(configprojection.RefreshInput{
		Version:     newActivationVersion(),
		Source:      configprojection.SourceDSL,
		YAMLBytes:   yamlBytes,
		DSLSnapshot: req.DSL,
	})

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
	if req.Mode != "" && req.Mode != DeployModeMerge && req.Mode != DeployModeReplace {
		return nil, fmt.Errorf("unsupported deploy mode %q", req.Mode)
	}

	baseDoc, err := parseYAMLDocument(baseData)
	if err != nil {
		return nil, fmt.Errorf("failed to parse deploy base config: %w", err)
	}
	baseRoot, err := documentMappingNode(baseDoc)
	if err != nil {
		return nil, fmt.Errorf("failed to parse deploy base config: %w", err)
	}

	fragmentDoc, err := parseYAMLDocument(fragmentBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to parse compiled routing fragment: %w", err)
	}
	fragmentRoot, err := documentMappingNode(fragmentDoc)
	if err != nil {
		return nil, fmt.Errorf("failed to parse compiled routing fragment: %w", err)
	}
	fragmentConfig, err := decodeYAMLTaggedBytes[routingFragmentDocument](fragmentBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to parse compiled routing fragment: %w", err)
	}

	if mergeErr := mergeDSLOwnedNodes(baseRoot, fragmentRoot, req.Mode); mergeErr != nil {
		return nil, mergeErr
	}
	if mergeErr := mergeFragmentGlobalNode(baseRoot, fragmentConfig.Global); mergeErr != nil {
		return nil, mergeErr
	}

	mergedYAML, err := marshalYAMLDocument(baseDoc)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal merged config: %w", err)
	}
	return mergedYAML, nil
}

// mergeDSLOwnedNodes applies only the configuration surface owned by the DSL.
// Node-level merging makes the transport forward-compatible: a new signal or
// projection is carried without adding another field-by-field merge branch.
func mergeDSLOwnedNodes(baseRoot, fragmentRoot *yaml.Node, mode DeployMode) error {
	fragmentRouting := mappingValueNode(fragmentRoot, "routing")
	if mode == DeployModeReplace {
		if fragmentRouting == nil {
			return fmt.Errorf("compiled routing fragment must contain routing")
		}
		setMappingValueNode(baseRoot, "routing", fragmentRouting)
		for _, section := range []string{"entrypoints", "recipes"} {
			if value := mappingValueNode(fragmentRoot, section); value != nil {
				setMappingValueNode(baseRoot, section, value)
			} else {
				deleteMappingValueNode(baseRoot, section)
			}
		}
		return nil
	}

	if fragmentRouting != nil {
		baseRouting := mappingValueNode(baseRoot, "routing")
		if baseRouting == nil {
			setMappingValueNode(baseRoot, "routing", fragmentRouting)
		} else if err := mergeMappingNodes(baseRouting, fragmentRouting); err != nil {
			return fmt.Errorf("failed to merge routing fragment: %w", err)
		}
	}

	// Scoped lists have identity and ordering semantics, so a supplied list is
	// replaced as one unit. An omitted list remains untouched in merge mode.
	for _, section := range []string{"entrypoints", "recipes"} {
		if value := mappingValueNode(fragmentRoot, section); value != nil {
			setMappingValueNode(baseRoot, section, value)
		}
	}
	return nil
}

func mergeFragmentGlobalNode(baseRoot *yaml.Node, fragment *globalFragment) error {
	if fragment == nil || fragment.Services == nil || fragment.Services.RateLimit == nil {
		return nil
	}

	global, err := ensureMappingNode(baseRoot, "global")
	if err != nil {
		return err
	}
	services, err := ensureMappingNode(global, "services")
	if err != nil {
		return err
	}
	rateLimit := &yaml.Node{}
	if err := rateLimit.Encode(fragment.Services.RateLimit); err != nil {
		return fmt.Errorf("failed to encode global.services.ratelimit: %w", err)
	}
	setMappingValueNode(services, "ratelimit", rateLimit)
	return nil
}

func ensureMappingNode(parent *yaml.Node, key string) (*yaml.Node, error) {
	value := mappingValueNode(parent, key)
	if value == nil {
		setMappingValueNode(parent, key, &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"})
		value = mappingValueNode(parent, key)
	}
	if value.Kind != yaml.MappingNode {
		return nil, fmt.Errorf("config %s block must be a YAML mapping", key)
	}
	return value, nil
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
	release, err := beginOrdinaryRuntimeConfigMutation(configDir)
	if err != nil {
		writeRuntimeConfigMutationError(w, err)
		return
	}
	defer release()

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

	refreshConfigProjection(configprojection.RefreshInput{
		Version:     version,
		Source:      configprojection.SourceRollback,
		YAMLBytes:   backupData,
		DSLSnapshot: readArchivedDSL(configDir),
	})

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
