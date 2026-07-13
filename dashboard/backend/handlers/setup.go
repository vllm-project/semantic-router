package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

const setupRemoteConfigMaxSize = 1 * 1024 * 1024

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

func SetupValidateHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req SetupConfigRequest
		if status, err := decodeBoundedJSON(w, r, documentRequestBodyLimit, &req); err != nil {
			http.Error(w, "invalid request body", status)
			return
		}
		candidate, err := buildSetupCandidateConfig(configPath, req)
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

		var req SetupConfigRequest
		if status, err := decodeBoundedJSON(w, r, documentRequestBodyLimit, &req); err != nil {
			http.Error(w, "invalid request body", status)
			return
		}
		candidate, err := buildSetupCandidateConfig(configPath, req)
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
		previousData, err := os.ReadFile(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read current config: %v", err), http.StatusInternalServerError)
			return
		}
		previous := existingConfigFileSnapshot(previousData)

		if backupErr := backupCurrentConfig(configPath, configDir); backupErr != nil {
			log.Printf("Warning: failed to back up current config before setup activation: %v", backupErr)
		}

		if writeErr := writeConfigAtomically(configPath, yamlData); writeErr != nil {
			http.Error(w, fmt.Sprintf("Failed to write config: %v", writeErr), http.StatusInternalServerError)
			return
		}

		if _, parseErr := routerconfig.Parse(configPath); parseErr != nil {
			restoreErr := restoreConfigFileSnapshot(configPath, previous)
			http.Error(w, formatSetupActivationFailure("Failed to validate activated config", parseErr, restoreErr), http.StatusInternalServerError)
			return
		}

		effectiveConfigPath, err := syncRuntimeConfigForCurrentRuntime(configPath)
		if err != nil {
			restoreErr := restoreSetupRuntimeAfterFailure(configPath, previous)
			http.Error(w, formatSetupActivationFailure("Failed to sync runtime config", err, restoreErr), http.StatusInternalServerError)
			return
		}

		if err := restartSetupRuntimeServices(configPath, effectiveConfigPath); err != nil {
			restoreErr := restoreSetupRuntimeAfterFailure(configPath, previous)
			http.Error(w, formatSetupActivationFailure("Failed to restart router/envoy after activation", err, restoreErr), http.StatusInternalServerError)
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

	defaults := routerconfig.DefaultCanonicalGlobal()
	configFile.Global = &defaults
}

func SetupImportRemoteHandler(configPath string) http.HandlerFunc {
	return setupImportRemoteHandlerWithClient(
		configPath,
		newPublicOutboundHTTPClient(10*time.Second),
	)
}

func setupImportRemoteHandlerWithClient(
	configPath string,
	client outboundHTTPClient,
) http.HandlerFunc {
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
		if status, err := decodeBoundedJSON(w, r, smallJSONRequestBodyLimit, &req); err != nil {
			http.Error(w, "invalid request body", status)
			return
		}

		importURL, err := normalizeRemoteConfigURL(req.URL)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if validationErr := client.ValidateURL(r.Context(), importURL); validationErr != nil {
			http.Error(w, "remote config URL must resolve to a public destination", http.StatusBadRequest)
			return
		}

		remoteReq, err := http.NewRequestWithContext(r.Context(), http.MethodGet, importURL, nil)
		if err != nil {
			http.Error(w, "failed to create remote import request", http.StatusBadRequest)
			return
		}

		resp, err := client.Do(remoteReq)
		if err != nil {
			http.Error(w, "failed to fetch remote config", http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
			http.Error(w, fmt.Sprintf("remote config request failed: HTTP %d", resp.StatusCode), http.StatusBadGateway)
			return
		}

		body, err := readBoundedOutboundBody(resp.Body, setupRemoteConfigMaxSize)
		if err != nil {
			http.Error(w, "failed to read remote config", http.StatusBadGateway)
			return
		}

		remoteConfig, err := parseSetupCanonicalConfig(body)
		if err != nil {
			http.Error(w, "remote config is invalid", http.StatusBadRequest)
			return
		}

		if validationErr := validateSetupCandidate(configPath, remoteConfig); validationErr != nil {
			http.Error(w, "remote config validation failed", http.StatusBadRequest)
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

func summarizeSetupConfig(configData *routerconfig.CanonicalConfig) setupConfigSummary {
	cfg, err := parseSetupRouterConfig(configData)
	if err != nil {
		return summarizeSetupConfigFallback(configData)
	}

	routing := routerconfig.CanonicalRoutingFromRouterConfig(cfg)
	return setupConfigSummary{
		Models:    len(routing.ModelCards),
		Decisions: len(routing.Decisions),
		Signals:   countCanonicalSignals(routing.Signals),
	}
}

func parseSetupRouterConfig(configData *routerconfig.CanonicalConfig) (*routerconfig.RouterConfig, error) {
	yamlData, err := marshalYAMLBytes(configData)
	if err != nil {
		return nil, err
	}
	return routerconfig.ParseYAMLBytes(yamlData)
}

func summarizeSetupConfigFallback(configData *routerconfig.CanonicalConfig) setupConfigSummary {
	return setupConfigSummary{
		Models:    countConfiguredModelsFallback(configData),
		Decisions: countConfiguredDecisionsFallback(configData),
		Signals:   countConfiguredSignalsFallback(configData),
	}
}

func countConfiguredModelsFallback(configData *routerconfig.CanonicalConfig) int {
	if configData == nil {
		return 0
	}
	if len(configData.Routing.ModelCards) > 0 {
		return len(configData.Routing.ModelCards)
	}
	return len(configData.Providers.Models)
}

func countConfiguredDecisionsFallback(configData *routerconfig.CanonicalConfig) int {
	if configData == nil {
		return 0
	}
	return len(configData.Routing.Decisions)
}

func countConfiguredSignalsFallback(configData *routerconfig.CanonicalConfig) int {
	if configData == nil {
		return 0
	}
	return countCanonicalSignals(configData.Routing.Signals)
}

func countCanonicalSignals(signals routerconfig.CanonicalSignals) int {
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

func buildSetupCandidateConfig(configPath string, req SetupConfigRequest) (*setupConfigFile, error) {
	configFile, err := loadBootstrapConfig(configPath)
	if err != nil {
		return nil, err
	}

	if len(req.Config) == 0 {
		return nil, fmt.Errorf("config is required")
	}

	requestConfig, err := decodeYAMLTaggedBytes[routerconfig.CanonicalConfig](req.Config)
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
	parsed, err := parseOutboundHTTPURL(rawValue)
	if err != nil {
		return "", fmt.Errorf("invalid remote config URL")
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

	tempDir := filepath.Dir(filepath.Clean(configPath))
	if tempDir == "." || tempDir == "" {
		tempDir = ""
	}
	tempConfigFile, err := os.CreateTemp(tempDir, "vllm-sr-setup-*.yaml")
	if err != nil {
		return err
	}
	tempConfigPath := tempConfigFile.Name()
	if closeErr := tempConfigFile.Close(); closeErr != nil {
		return closeErr
	}
	defer func() {
		_ = os.Remove(tempConfigPath)
	}()

	if writeErr := os.WriteFile(tempConfigPath, yamlData, 0o644); writeErr != nil {
		return writeErr
	}

	parsedConfig, err := routerconfig.Parse(tempConfigPath)
	if err != nil {
		return err
	}
	if len(parsedConfig.VLLMEndpoints) > 0 {
		for _, endpoint := range parsedConfig.VLLMEndpoints {
			if endpoint.ProviderProfileName != "" && endpoint.Address == "" {
				continue
			}
			if endpointErr := validateEndpointAddress(endpoint.Address); endpointErr != nil {
				return endpointErr
			}
		}
	}

	return nil
}

func mergeSetupCanonicalConfig(base, patch routerconfig.CanonicalConfig) routerconfig.CanonicalConfig {
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
	if err := ensurePrivateStateDirectory(backupDir); err != nil {
		return err
	}

	version := time.Now().Format("20060102-150405")
	backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version))
	if err := writePrivateStateFile(backupFile, existingData); err != nil {
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

	routerStatus := getDockerContainerStatus(managedContainerNameForService("router"))
	if routerStatus == "unknown" {
		return fmt.Errorf("managed router status probe is unavailable")
	}
	envoyStatus := getDockerContainerStatus(managedContainerNameForService("envoy"))
	if envoyStatus == "unknown" {
		return fmt.Errorf("managed Envoy status probe is unavailable")
	}
	if routerStatus == "not found" && envoyStatus == "not found" {
		return nil
	}

	return restartSetupManagedServices(effectiveConfigPath)
}
