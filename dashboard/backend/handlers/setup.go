package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	routercontract "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routercontract"
	routerprojection "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerprojection"
)

type SetupStateResponse struct {
	SetupMode    bool `json:"setupMode"`
	ListenerPort int  `json:"listenerPort"`
	Models       int  `json:"models"`
	Decisions    int  `json:"decisions"`
	HasModels    bool `json:"hasModels"`
	HasDecisions bool `json:"hasDecisions"`
	CanActivate  bool `json:"canActivate"`
}

type SetupConfigRequest struct {
	Config json.RawMessage `json:"config"`
}

type SetupValidateResponse struct {
	Valid       bool            `json:"valid"`
	Config      json.RawMessage `json:"config,omitempty"`
	Models      int             `json:"models"`
	Decisions   int             `json:"decisions"`
	Signals     int             `json:"signals"`
	CanActivate bool            `json:"canActivate"`
}

type SetupActivateResponse struct {
	Status    string `json:"status"`
	SetupMode bool   `json:"setupMode"`
	Message   string `json:"message,omitempty"`
}

type SetupImportRemoteRequest struct {
	URL string `json:"url"`
}

type SetupImportRemoteResponse struct {
	Config      json.RawMessage `json:"config"`
	Models      int             `json:"models"`
	Decisions   int             `json:"decisions"`
	Signals     int             `json:"signals"`
	CanActivate bool            `json:"canActivate"`
	SourceURL   string          `json:"sourceUrl"`
}

type setupConfigSummary struct {
	Models    int
	Decisions int
	Signals   int
}

func SetupStateHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		configFile, err := readSetupConfigFile(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read config: %v", err), http.StatusInternalServerError)
			return
		}

		summary := summarizeSetupConfig(&configFile.CanonicalConfig)
		if projected, ok := summarizeProjectedConfig(configPath); ok {
			summary = projected
		}
		resp := SetupStateResponse{
			SetupMode:    hasSetupMode(configFile),
			ListenerPort: firstListenerPort(configFile),
			Models:       summary.Models,
			Decisions:    summary.Decisions,
			HasModels:    summary.Models > 0,
			HasDecisions: summary.Decisions > 0,
			CanActivate:  summary.Models > 0 && summary.Decisions > 0,
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
}

func summarizeProjectedConfig(configPath string) (setupConfigSummary, bool) {
	configDir := filepath.Dir(configPath)
	projection, err := readActiveConfigProjection(configDir)
	if err != nil || projection == nil || !projection.Validation.Valid {
		return setupConfigSummary{}, false
	}
	configData, err := os.ReadFile(configPath)
	if err != nil || projection.ConfigHash != routerprojection.HashConfigBytes(configData) {
		return setupConfigSummary{}, false
	}

	return setupConfigSummary{
		Models:    len(projection.Models),
		Decisions: len(projection.Decisions),
		Signals:   len(projection.Signals),
	}, true
}

func SetupValidateHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		candidate, err := buildSetupCandidateConfig(configPath, r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if validationErr := validateSetupCandidate(configPath, candidate); validationErr != nil {
			http.Error(w, fmt.Sprintf("Setup validation failed: %v", validationErr), http.StatusBadRequest)
			return
		}

		summary := summarizeSetupConfig(&candidate.CanonicalConfig)
		configJSON, err := rawJSONMessage(candidate.CanonicalConfig)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to encode validated config: %v", err), http.StatusInternalServerError)
			return
		}
		resp := SetupValidateResponse{
			Valid:       true,
			Config:      configJSON,
			Models:      summary.Models,
			Decisions:   summary.Decisions,
			Signals:     summary.Signals,
			CanActivate: summary.Models > 0 && summary.Decisions > 0,
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
}

func SetupActivateHandler(configPath string, readonlyMode bool, configDir string) http.HandlerFunc {
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
				"message": "Dashboard is in read-only mode. Setup activation is disabled.",
			})
			return
		}

		candidate, err := buildSetupCandidateConfig(configPath, r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if validationErr := validateSetupCandidate(configPath, candidate); validationErr != nil {
			http.Error(w, fmt.Sprintf("Setup activation validation failed: %v", validationErr), http.StatusBadRequest)
			return
		}

		ensureSetupGlobalDefaults(candidate)

		if !deployMu.TryLock() {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusConflict)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"error":   "deploy_in_progress",
				"message": "Another config operation is in progress. Please try again.",
			})
			return
		}
		defer deployMu.Unlock()

		yamlData, err := marshalYAMLBytes(candidate.CanonicalConfig)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to convert config to YAML: %v", err), http.StatusInternalServerError)
			return
		}

		if backupErr := backupCurrentConfig(configPath, configDir); backupErr != nil {
			log.Printf("Warning: failed to back up current config before setup activation: %v", backupErr)
		}

		tmpConfigFile := configPath + ".tmp"
		if writeErr := os.WriteFile(tmpConfigFile, yamlData, 0o644); writeErr != nil {
			http.Error(w, fmt.Sprintf("Failed to write config: %v", writeErr), http.StatusInternalServerError)
			return
		}
		if renameErr := os.Rename(tmpConfigFile, configPath); renameErr != nil {
			if fallbackWriteErr := os.WriteFile(configPath, yamlData, 0o644); fallbackWriteErr != nil {
				http.Error(w, fmt.Sprintf("Failed to write config: %v", fallbackWriteErr), http.StatusInternalServerError)
				return
			}
		}

		if _, parseErr := routercontract.Parse(configPath); parseErr != nil {
			http.Error(w, fmt.Sprintf("Failed to validate activated config: %v", parseErr), http.StatusInternalServerError)
			return
		}

		effectiveConfigPath, err := syncRuntimeConfigForCurrentRuntime(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to sync runtime config: %v", err), http.StatusInternalServerError)
			return
		}

		if err := restartSetupRuntimeServices(configPath, effectiveConfigPath); err != nil {
			log.Printf("Warning: failed to restart router/envoy after activation: %v", err)
		}
		if err := persistActiveConfigProjection(configPath, configDir); err != nil {
			http.Error(w, fmt.Sprintf("Failed to persist config projection: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(SetupActivateResponse{
			Status:    "success",
			SetupMode: false,
			Message:   "Setup activated successfully. Router and Envoy are starting.",
		}); err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
}

func ensureSetupGlobalDefaults(configFile *setupConfigFile) {
	if configFile == nil || configFile.Global != nil {
		return
	}

	defaults := routercontract.DefaultCanonicalGlobal()
	configFile.Global = &defaults
}

func SetupImportRemoteHandler(configPath string) http.HandlerFunc {
	client := &http.Client{Timeout: 10 * time.Second}

	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if _, err := loadBootstrapConfig(configPath); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		var req SetupImportRemoteRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		importURL, err := normalizeRemoteConfigURL(req.URL)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		remoteReq, err := http.NewRequestWithContext(r.Context(), http.MethodGet, importURL, nil)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to create remote import request: %v", err), http.StatusBadRequest)
			return
		}

		resp, err := client.Do(remoteReq)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to fetch remote config: %v", err), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
			http.Error(w, fmt.Sprintf("remote config request failed: HTTP %d", resp.StatusCode), http.StatusBadGateway)
			return
		}

		body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to read remote config: %v", err), http.StatusBadGateway)
			return
		}

		remoteConfig, err := parseSetupCanonicalConfig(body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if validationErr := validateSetupCandidate(configPath, remoteConfig); validationErr != nil {
			http.Error(w, fmt.Sprintf("remote config validation failed: %v", validationErr), http.StatusBadRequest)
			return
		}

		summary := summarizeSetupConfig(&remoteConfig.CanonicalConfig)
		configJSON, err := rawJSONMessage(remoteConfig.CanonicalConfig)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to encode remote config: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(SetupImportRemoteResponse{
			Config:      configJSON,
			Models:      summary.Models,
			Decisions:   summary.Decisions,
			Signals:     summary.Signals,
			CanActivate: summary.Models > 0 && summary.Decisions > 0,
			SourceURL:   importURL,
		}); err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
}

func hasSetupMode(configFile *setupConfigFile) bool {
	return configFile != nil && configFile.Setup != nil && configFile.Setup.Mode
}

func summarizeSetupConfig(configData *routercontract.CanonicalConfig) setupConfigSummary {
	return setupConfigSummary{
		Models:    countConfiguredModelsFallback(configData),
		Decisions: countConfiguredDecisionsFallback(configData),
		Signals:   countConfiguredSignalsFallback(configData),
	}
}

func countConfiguredModelsFallback(configData *routercontract.CanonicalConfig) int {
	if configData == nil {
		return 0
	}
	if len(configData.Routing.ModelCards) > 0 {
		return len(configData.Routing.ModelCards)
	}
	return len(configData.Providers.Models)
}

func countConfiguredDecisionsFallback(configData *routercontract.CanonicalConfig) int {
	if configData == nil {
		return 0
	}
	return len(configData.Routing.Decisions)
}

func countConfiguredSignalsFallback(configData *routercontract.CanonicalConfig) int {
	if configData == nil {
		return 0
	}
	return countCanonicalSignals(configData.Routing.Signals)
}

func countCanonicalSignals(signals routercontract.CanonicalSignals) int {
	return len(signals.Keywords) +
		len(signals.Embeddings) +
		len(signals.Domains) +
		len(signals.FactCheck) +
		len(signals.UserFeedbacks) +
		len(signals.Preferences) +
		len(signals.Language) +
		len(signals.Context) +
		len(signals.Complexity) +
		len(signals.Modality) +
		len(signals.RoleBindings) +
		len(signals.Jailbreak) +
		len(signals.PII)
}

func firstListenerPort(configFile *setupConfigFile) int {
	if configFile == nil || len(configFile.Listeners) == 0 {
		return 0
	}
	return configFile.Listeners[0].Port
}

func buildSetupCandidateConfig(configPath string, bodyReader io.Reader) (*setupConfigFile, error) {
	configFile, err := loadBootstrapConfig(configPath)
	if err != nil {
		return nil, err
	}

	var req SetupConfigRequest
	if decodeErr := json.NewDecoder(bodyReader).Decode(&req); decodeErr != nil {
		return nil, fmt.Errorf("invalid request body: %w", decodeErr)
	}
	if len(req.Config) == 0 {
		return nil, fmt.Errorf("config is required")
	}

	requestConfig, err := decodeYAMLTaggedBytes[routercontract.CanonicalConfig](req.Config)
	if err != nil {
		return nil, fmt.Errorf("invalid config payload: %w", err)
	}

	merged := *configFile
	merged.CanonicalConfig = mergeSetupCanonicalConfig(configFile.CanonicalConfig, requestConfig)
	merged.Setup = nil
	return &merged, nil
}

func loadBootstrapConfig(configPath string) (*setupConfigFile, error) {
	configFile, err := readSetupConfigFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read existing config: %w", err)
	}
	if !hasSetupMode(configFile) {
		return nil, fmt.Errorf("setup mode is not active for this workspace")
	}
	return configFile, nil
}

func normalizeRemoteConfigURL(rawValue string) (string, error) {
	trimmed := strings.TrimSpace(rawValue)
	if trimmed == "" {
		return "", fmt.Errorf("remote config URL is required")
	}

	parsed, err := url.ParseRequestURI(trimmed)
	if err != nil {
		return "", fmt.Errorf("invalid remote config URL: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", fmt.Errorf("remote config URL must use http or https")
	}
	if parsed.Host == "" {
		return "", fmt.Errorf("remote config URL must include a host")
	}

	return parsed.String(), nil
}

func parseSetupCanonicalConfig(raw []byte) (*setupConfigFile, error) {
	parsed, err := decodeYAMLTaggedBytes[setupConfigFile](raw)
	if err != nil {
		return nil, fmt.Errorf("failed to parse remote config: %w", err)
	}
	if parsed.Version == "" &&
		len(parsed.Listeners) == 0 &&
		len(parsed.Providers.Models) == 0 &&
		parsed.Global == nil &&
		len(parsed.Routing.ModelCards) == 0 &&
		len(parsed.Routing.Decisions) == 0 &&
		countCanonicalSignals(parsed.Routing.Signals) == 0 {
		return nil, fmt.Errorf("remote config is empty")
	}
	parsed.Setup = nil
	return &parsed, nil
}

func validateSetupCandidate(configPath string, configData *setupConfigFile) error {
	if configData == nil {
		return fmt.Errorf("config is required")
	}
	if err := validateCanonicalEndpointRefs(configData.CanonicalConfig); err != nil {
		return err
	}

	yamlData, err := marshalYAMLBytes(configData.CanonicalConfig)
	if err != nil {
		return err
	}
	if _, err := routercontract.ParseYAMLBytes(yamlData); err != nil {
		return err
	}

	return nil
}

func mergeSetupCanonicalConfig(base, patch routercontract.CanonicalConfig) routercontract.CanonicalConfig {
	merged := base
	if patch.Version != "" {
		merged.Version = patch.Version
	}
	if len(patch.Listeners) > 0 {
		merged.Listeners = patch.Listeners
	}
	if len(patch.Providers.Models) > 0 ||
		patch.Providers.Defaults.DefaultModel != "" ||
		len(patch.Providers.Defaults.ReasoningFamilies) > 0 ||
		patch.Providers.Defaults.DefaultReasoningEffort != "" {
		merged.Providers = patch.Providers
	}
	if len(patch.Routing.ModelCards) > 0 ||
		len(patch.Routing.Decisions) > 0 ||
		countCanonicalSignals(patch.Routing.Signals) > 0 {
		merged.Routing = patch.Routing
	}
	if patch.Global != nil {
		merged.Global = patch.Global
	}
	return merged
}

func backupCurrentConfig(configPath string, configDir string) error {
	existingData, err := os.ReadFile(configPath)
	if err != nil || len(existingData) == 0 {
		return err
	}

	backupDir := filepath.Join(configDir, ".vllm-sr", "config-backups")
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		return err
	}

	version := time.Now().Format("20060102-150405")
	backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version))
	if err := os.WriteFile(backupFile, existingData, 0o644); err != nil {
		return err
	}
	cleanupBackups(backupDir)
	return nil
}

func restartSetupManagedServices(effectiveConfigPath string) error {
	if err := refreshManagedSplitEnvoyConfig(effectiveConfigPath); err != nil {
		return err
	}

	for _, service := range []string{"router", "envoy"} {
		if err := restartManagedService(service, 20*time.Second); err != nil {
			return err
		}
	}

	return nil
}

func restartSetupRuntimeServices(configPath string, effectiveConfigPath string) error {
	if isRunningInContainer() && isManagedContainerConfigPath(configPath) {
		return restartSetupManagedServices(effectiveConfigPath)
	}

	if getDockerContainerStatus(managedContainerNameForService("router")) == "not found" &&
		getDockerContainerStatus(managedContainerNameForService("envoy")) == "not found" {
		return nil
	}

	return restartSetupManagedServices(effectiveConfigPath)
}
