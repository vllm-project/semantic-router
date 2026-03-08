package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/configlifecycle"
)

// maxBackups is the maximum number of config backups to keep.
const maxBackups = configlifecycle.MaxBackups

// DeployRequest is the JSON body for a DSL deploy request.
type DeployRequest struct {
	YAML string `json:"yaml"`
	DSL  string `json:"dsl,omitempty"`
}

// DeployResponse is the JSON response for a deploy operation.
type DeployResponse struct {
	Status  string `json:"status"`
	Version string `json:"version"`
	Message string `json:"message,omitempty"`
}

// ConfigVersion represents a backup version entry.
type ConfigVersion struct {
	Version   string `json:"version"`
	Timestamp string `json:"timestamp"`
	Source    string `json:"source"`
	Filename  string `json:"filename"`
}

// DeployPreviewResponse contains the before/after YAML for diff comparison.
type DeployPreviewResponse struct {
	Current string `json:"current"`
	Preview string `json:"preview"`
}

// DeployPreviewHandler returns the current config and the merged preview.
func DeployPreviewHandler(configPath string) http.HandlerFunc {
	return DeployPreviewHandlerWithService(configlifecycle.New(configPath, ""))
}

func DeployPreviewHandlerWithService(service *configlifecycle.Service) http.HandlerFunc {
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

		preview, err := service.DeployPreview(configlifecycle.DeployRequest{YAML: req.YAML, DSL: req.DSL})
		if err != nil {
			writeLifecycleError(w, err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(DeployPreviewResponse{
			Current: preview.Current,
			Preview: preview.Preview,
		})
	}
}

// DeployHandler handles DSL config deployment.
func DeployHandler(configPath string, readonlyMode bool, configDir string) http.HandlerFunc {
	return DeployHandlerWithService(configlifecycle.New(configPath, configDir), readonlyMode)
}

func DeployHandlerWithService(service *configlifecycle.Service, readonlyMode bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if readonlyMode {
			writeReadonlyResponse(w, "Dashboard is in read-only mode. Deploy is disabled.")
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
		result, err := service.Deploy(configlifecycle.DeployRequest{YAML: req.YAML, DSL: req.DSL})
		if err != nil {
			writeLifecycleError(w, err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(DeployResponse{
			Status:  "success",
			Version: result.Version,
			Message: result.Message,
		})
	}
}

// RollbackHandler rolls back to a specific backup version.
func RollbackHandler(configPath string, readonlyMode bool, configDir string) http.HandlerFunc {
	return RollbackHandlerWithService(configlifecycle.New(configPath, configDir), readonlyMode)
}

func RollbackHandlerWithService(service *configlifecycle.Service, readonlyMode bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if readonlyMode {
			writeReadonlyResponse(w, "Dashboard is in read-only mode. Rollback is disabled.")
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

		result, err := service.Rollback(rollbackReq.Version)
		if err != nil {
			writeLifecycleError(w, err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(DeployResponse{
			Status:  "success",
			Version: result.Version,
			Message: result.Message,
		})
	}
}

// ConfigVersionsHandler lists available backup versions.
func ConfigVersionsHandler(configPath string) http.HandlerFunc {
	return ConfigVersionsHandlerWithService(configlifecycle.New(configPath, ""))
}

func ConfigVersionsHandlerWithService(service *configlifecycle.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		versions, err := service.ListVersions()
		if err != nil {
			writeLifecycleError(w, err)
			return
		}

		response := make([]ConfigVersion, 0, len(versions))
		for _, version := range versions {
			response = append(response, ConfigVersion(version))
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}
}

func deepMerge(dst, src map[string]interface{}) map[string]interface{} {
	return configlifecycle.DeepMerge(dst, src)
}

func toStringKeyMap(v interface{}) (map[string]interface{}, bool) {
	return configlifecycle.ToStringKeyMap(v)
}

func canonicalizeYAMLForDiff(raw []byte) string {
	return configlifecycle.CanonicalizeYAMLForDiff(raw)
}

func cleanupBackups(backupDir string) {
	configlifecycle.CleanupBackups(backupDir)
}
