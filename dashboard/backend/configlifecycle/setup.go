package configlifecycle

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const setupModeKey = "setup"

type SetupState struct {
	SetupMode    bool
	ListenerPort int
	Models       int
	Decisions    int
	HasModels    bool
	HasDecisions bool
	CanActivate  bool
}

type SetupValidation struct {
	Config      map[string]interface{}
	Models      int
	Decisions   int
	CanActivate bool
}

type SetupActivation struct {
	SetupMode bool
	Message   string
}

func (s *Service) SetupState() (SetupState, error) {
	configMap, err := s.loadConfigMap()
	if err != nil {
		return SetupState{}, fmt.Errorf("failed to read config: %w", err)
	}

	models := countConfiguredModels(configMap)
	decisions := countConfiguredDecisions(configMap)
	return SetupState{
		SetupMode:    hasSetupMode(configMap),
		ListenerPort: firstListenerPort(configMap),
		Models:       models,
		Decisions:    decisions,
		HasModels:    models > 0,
		HasDecisions: decisions > 0,
		CanActivate:  models > 0 && decisions > 0,
	}, nil
}

func (s *Service) ValidateSetup(configPatch map[string]interface{}) (SetupValidation, error) {
	candidate, err := s.buildSetupCandidateConfig(configPatch)
	if err != nil {
		return SetupValidation{}, err
	}
	if err := s.validateSetupCandidate(candidate); err != nil {
		return SetupValidation{}, &Error{StatusCode: 400, Message: fmt.Sprintf("Setup validation failed: %v", err)}
	}

	models := countConfiguredModels(candidate)
	decisions := countConfiguredDecisions(candidate)
	return SetupValidation{
		Config:      candidate,
		Models:      models,
		Decisions:   decisions,
		CanActivate: models > 0 && decisions > 0,
	}, nil
}

func (s *Service) ActivateSetup(configPatch map[string]interface{}) (SetupActivation, error) {
	candidate, err := s.buildSetupCandidateConfig(configPatch)
	if err != nil {
		return SetupActivation{}, err
	}
	if validationErr := s.validateSetupCandidate(candidate); validationErr != nil {
		return SetupActivation{}, &Error{StatusCode: 400, Message: fmt.Sprintf("Setup activation validation failed: %v", validationErr)}
	}
	if !deployMu.TryLock() {
		return SetupActivation{}, &Error{
			StatusCode: 409,
			Code:       "deploy_in_progress",
			Message:    "Another config operation is in progress. Please try again.",
		}
	}
	defer deployMu.Unlock()

	yamlData, err := yaml.Marshal(candidate)
	if err != nil {
		return SetupActivation{}, fmt.Errorf("failed to convert config to YAML: %w", err)
	}
	if err := s.backupCurrentConfig(); err != nil {
		log.Printf("Warning: failed to back up current config before setup activation: %v", err)
	}
	if err := writeConfigAtomically(s.ConfigPath, yamlData); err != nil {
		return SetupActivation{}, fmt.Errorf("failed to write config: %w", err)
	}

	outputDir := filepath.Join(s.ConfigDir, ".vllm-sr")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return SetupActivation{}, fmt.Errorf("failed to create output directory: %w", err)
	}
	if _, err := generateRouterConfigWithPython(s.ConfigPath, outputDir); err != nil {
		return SetupActivation{}, fmt.Errorf("failed to generate router config during activation: %w", err)
	}
	if err := restartSetupManagedServices(); err != nil {
		log.Printf("Warning: failed to restart router/envoy after activation: %v", err)
	}
	s.recordSuccessfulCompatibilityChange(yamlData, revisionPersistenceOptions{
		source:         "compat_setup_activate",
		summary:        "Activated setup workflow via compatibility API",
		action:         "config.activate",
		triggerSource:  "setup_api",
		revisionStatus: "active",
		previousStatus: "superseded",
		deployStatus:   "succeeded",
		message:        "Setup activated successfully. Router and Envoy are restarting.",
		metadata: map[string]interface{}{
			"operation": "setup_activate",
		},
	})
	return SetupActivation{
		SetupMode: false,
		Message:   "Setup activated successfully. Router and Envoy are restarting.",
	}, nil
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

func (s *Service) buildSetupCandidateConfig(configPatch map[string]interface{}) (map[string]interface{}, error) {
	configMap, err := s.loadConfigMap()
	if err != nil {
		return nil, newBadRequestError(fmt.Sprintf("failed to read existing config: %v", err))
	}
	if !hasSetupMode(configMap) {
		return nil, newBadRequestError("setup mode is not active for this workspace")
	}
	if len(configPatch) == 0 {
		return nil, newBadRequestError("config is required")
	}

	merged := DeepMerge(configMap, configPatch)
	delete(merged, setupModeKey)
	return merged, nil
}

func (s *Service) validateSetupCandidate(configMap map[string]interface{}) error {
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
	for _, endpoint := range parsedConfig.VLLMEndpoints {
		if validationErr := ValidateEndpointAddress(endpoint.Address); validationErr != nil {
			return validationErr
		}
	}

	tempOutputDir, err := os.MkdirTemp("", "vllm-sr-setup-out-*")
	if err != nil {
		return err
	}
	defer func() {
		_ = os.RemoveAll(tempOutputDir)
	}()

	if _, err := generateRouterConfigWithPython(tempConfigPath, tempOutputDir); err != nil {
		return err
	}
	return nil
}

func (s *Service) backupCurrentConfig() error {
	existingData, err := os.ReadFile(s.ConfigPath)
	if err != nil || len(existingData) == 0 {
		return err
	}

	backupDir := s.backupDir()
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		return err
	}
	version := time.Now().Format("20060102-150405")
	backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version))
	if err := os.WriteFile(backupFile, existingData, 0o644); err != nil {
		return err
	}
	CleanupBackups(backupDir)
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
