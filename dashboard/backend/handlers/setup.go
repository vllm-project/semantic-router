package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const setupModeKey = "setup"

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
	Config map[string]interface{} `json:"config"`
}

type SetupValidateResponse struct {
	Valid       bool                   `json:"valid"`
	Config      map[string]interface{} `json:"config,omitempty"`
	Models      int                    `json:"models"`
	Decisions   int                    `json:"decisions"`
	Signals     int                    `json:"signals"`
	CanActivate bool                   `json:"canActivate"`
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
	Config      map[string]interface{} `json:"config"`
	Models      int                    `json:"models"`
	Decisions   int                    `json:"decisions"`
	Signals     int                    `json:"signals"`
	CanActivate bool                   `json:"canActivate"`
	SourceURL   string                 `json:"sourceUrl"`
}

func SetupStateHandler(configPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		configMap, err := loadConfigMap(configPath)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read config: %v", err), http.StatusInternalServerError)
			return
		}

		models := countConfiguredModels(configMap)
		decisions := countConfiguredDecisions(configMap)
		resp := SetupStateResponse{
			SetupMode:    hasSetupMode(configMap),
			ListenerPort: firstListenerPort(configMap),
			Models:       models,
			Decisions:    decisions,
			HasModels:    models > 0,
			HasDecisions: decisions > 0,
			CanActivate:  models > 0 && decisions > 0,
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

		candidate, err := buildSetupCandidateConfig(configPath, r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if err := validateSetupCandidate(candidate); err != nil {
			http.Error(w, fmt.Sprintf("Setup validation failed: %v", err), http.StatusBadRequest)
			return
		}

		models := countConfiguredModels(candidate)
		decisions := countConfiguredDecisions(candidate)
		resp := SetupValidateResponse{
			Valid:       true,
			Config:      candidate,
			Models:      models,
			Decisions:   decisions,
			Signals:     countConfiguredSignals(candidate),
			CanActivate: models > 0 && decisions > 0,
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

		if validationErr := validateSetupCandidate(candidate); validationErr != nil {
			http.Error(w, fmt.Sprintf("Setup activation validation failed: %v", validationErr), http.StatusBadRequest)
			return
		}

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

		yamlData, err := yaml.Marshal(candidate)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to convert config to YAML: %v", err), http.StatusInternalServerError)
			return
		}

		if err := backupCurrentConfig(configPath, configDir); err != nil {
			log.Printf("Warning: failed to back up current config before setup activation: %v", err)
		}

		tmpConfigFile := configPath + ".tmp"
		if err := os.WriteFile(tmpConfigFile, yamlData, 0o644); err != nil {
			http.Error(w, fmt.Sprintf("Failed to write config: %v", err), http.StatusInternalServerError)
			return
		}
		if err := os.Rename(tmpConfigFile, configPath); err != nil {
			if writeErr := os.WriteFile(configPath, yamlData, 0o644); writeErr != nil {
				http.Error(w, fmt.Sprintf("Failed to write config: %v", writeErr), http.StatusInternalServerError)
				return
			}
		}

		outputDir := filepath.Join(configDir, ".vllm-sr")
		if err := os.MkdirAll(outputDir, 0o755); err != nil {
			http.Error(w, fmt.Sprintf("Failed to create output directory: %v", err), http.StatusInternalServerError)
			return
		}

		if _, err := generateRouterConfigWithPython(configPath, outputDir); err != nil {
			http.Error(w, fmt.Sprintf("Failed to generate router config during activation: %v", err), http.StatusInternalServerError)
			return
		}

		if err := restartSetupManagedServices(); err != nil {
			log.Printf("Warning: failed to restart router/envoy after activation: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(SetupActivateResponse{
			Status:    "success",
			SetupMode: false,
			Message:   "Setup activated successfully. Router and Envoy are restarting.",
		}); err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
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

		remoteConfig, err := parseSetupConfigMap(body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		models := countConfiguredModels(remoteConfig)
		decisions := countConfiguredDecisions(remoteConfig)
		signals := countConfiguredSignals(remoteConfig)

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(SetupImportRemoteResponse{
			Config:      remoteConfig,
			Models:      models,
			Decisions:   decisions,
			Signals:     signals,
			CanActivate: models > 0 && decisions > 0,
			SourceURL:   importURL,
		}); err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
}

func loadConfigMap(configPath string) (map[string]interface{}, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}

	result := make(map[string]interface{})
	if err := yaml.Unmarshal(data, &result); err != nil {
		return nil, err
	}

	return result, nil
}

func hasSetupMode(configMap map[string]interface{}) bool {
	setupData, ok := configMap[setupModeKey].(map[string]interface{})
	if !ok {
		return false
	}
	enabled, _ := setupData["mode"].(bool)
	return enabled
}

func countConfiguredModels(configMap map[string]interface{}) int {
	if providers, ok := configMap["providers"].(map[string]interface{}); ok {
		if models, ok := providers["models"].([]interface{}); ok {
			return len(models)
		}
	}
	if modelConfig, ok := configMap["model_config"].(map[string]interface{}); ok {
		return len(modelConfig)
	}
	return 0
}

func countConfiguredDecisions(configMap map[string]interface{}) int {
	decisions, ok := configMap["decisions"].([]interface{})
	if !ok {
		return 0
	}
	return len(decisions)
}

func countConfiguredSignals(configMap map[string]interface{}) int {
	signals, ok := configMap["signals"].(map[string]interface{})
	if !ok {
		return 0
	}

	total := 0
	for _, raw := range signals {
		if entries, ok := raw.([]interface{}); ok {
			total += len(entries)
		}
	}

	return total
}

func firstListenerPort(configMap map[string]interface{}) int {
	listeners, ok := configMap["listeners"].([]interface{})
	if !ok || len(listeners) == 0 {
		return 0
	}

	listener, ok := listeners[0].(map[string]interface{})
	if !ok {
		return 0
	}

	switch port := listener["port"].(type) {
	case int:
		return port
	case int64:
		return int(port)
	case float64:
		return int(port)
	default:
		return 0
	}
}

func buildSetupCandidateConfig(configPath string, bodyReader interface{ Read([]byte) (int, error) }) (map[string]interface{}, error) {
	configMap, err := loadBootstrapConfig(configPath)
	if err != nil {
		return nil, err
	}

	var req SetupConfigRequest
	if err := json.NewDecoder(bodyReader).Decode(&req); err != nil {
		return nil, fmt.Errorf("invalid request body: %w", err)
	}
	if len(req.Config) == 0 {
		return nil, fmt.Errorf("config is required")
	}

	merged := deepMerge(configMap, req.Config)
	delete(merged, setupModeKey)
	return merged, nil
}

func loadBootstrapConfig(configPath string) (map[string]interface{}, error) {
	configMap, err := loadConfigMap(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read existing config: %w", err)
	}
	if !hasSetupMode(configMap) {
		return nil, fmt.Errorf("setup mode is not active for this workspace")
	}
	return configMap, nil
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

func parseSetupConfigMap(raw []byte) (map[string]interface{}, error) {
	parsed := make(map[string]interface{})
	if err := yaml.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("failed to parse remote config: %w", err)
	}
	if len(parsed) == 0 {
		return nil, fmt.Errorf("remote config is empty")
	}
	delete(parsed, setupModeKey)
	return parsed, nil
}

func validateSetupCandidate(configMap map[string]interface{}) error {
	yamlData, err := yaml.Marshal(configMap)
	if err != nil {
		return err
	}

	tempConfigFile, err := os.CreateTemp("", "vllm-sr-setup-*.yaml")
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
			if endpointErr := validateEndpointAddress(endpoint.Address); endpointErr != nil {
				return endpointErr
			}
		}
	}

	tempOutputDir, err := os.MkdirTemp("", "vllm-sr-setup-out-*")
	if err != nil {
		return err
	}
	defer func() {
		_ = os.RemoveAll(tempOutputDir)
	}()

	if _, generateErr := generateRouterConfigWithPython(tempConfigPath, tempOutputDir); generateErr != nil {
		return generateErr
	}

	return nil
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

func restartSetupManagedServices() error {
	if _, err := exec.LookPath("supervisorctl"); err != nil {
		return nil
	}

	for _, service := range []string{"router", "envoy"} {
		cmd := exec.Command("supervisorctl", "restart", service)
		if output, err := cmd.CombinedOutput(); err != nil {
			startCmd := exec.Command("supervisorctl", "start", service)
			if startOutput, startErr := startCmd.CombinedOutput(); startErr != nil {
				return fmt.Errorf("%s restart failed: %s / start failed: %s", service, string(output), string(startOutput))
			}
		}
	}

	return nil
}
