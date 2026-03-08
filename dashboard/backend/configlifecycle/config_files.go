package configlifecycle

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (s *Service) ConfigJSON() (interface{}, error) {
	data, err := s.currentConfigYAML()
	if err != nil {
		return nil, err
	}

	var config interface{}
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}
	return config, nil
}

func (s *Service) ConfigYAML() ([]byte, error) {
	return s.currentConfigYAML()
}

func (s *Service) UpdateConfig(configData map[string]interface{}) error {
	existingData, err := s.currentConfigYAML()
	if err != nil {
		return fmt.Errorf("failed to read existing config: %w", err)
	}

	existingMap := make(map[string]interface{})
	if unmarshalErr := yaml.Unmarshal(existingData, &existingMap); unmarshalErr != nil {
		return fmt.Errorf("failed to parse existing config: %w", unmarshalErr)
	}

	originalKeyCount := len(existingMap)
	mergedConfig := DeepMerge(existingMap, configData)
	if len(mergedConfig) < originalKeyCount {
		return fmt.Errorf("merge would result in data loss: original had %d keys, merged has %d keys; file: %s", originalKeyCount, len(mergedConfig), s.ConfigPath)
	}

	yamlData, err := yaml.Marshal(mergedConfig)
	if err != nil {
		return fmt.Errorf("failed to convert to YAML: %w", err)
	}
	if err := validateConfigYAML(yamlData); err != nil {
		return &Error{StatusCode: 400, Message: fmt.Sprintf("Config validation failed: %v", err)}
	}
	if s.hasRevisionStore() {
		_, err := s.runCompatibilityRevisionWorkflow(yamlData, compatibilityWorkflowOptions{
			source:         "compat_config_update",
			summary:        "Applied config update via compatibility API",
			triggerSource:  "config_api",
			auditAction:    "config.update",
			successMessage: "Config update applied via compatibility API.",
			metadata: map[string]interface{}{
				"compat_operation": "update",
			},
		})
		return err
	}
	return s.applyConfigUpdateDirect(existingData, yamlData)
}

func (s *Service) applyConfigUpdateDirect(existingData []byte, yamlData []byte) error {
	if err := writeConfigAtomically(s.ConfigPath, yamlData); err != nil {
		return fmt.Errorf("failed to write config: %w", err)
	}
	if err := s.propagateConfigToRuntime(); err != nil {
		if restoreErr := s.restorePreviousRuntimeConfig(existingData); restoreErr != nil {
			return fmt.Errorf("failed to apply config to runtime: %w; failed to restore previous config: %w", err, restoreErr)
		}
		return fmt.Errorf("failed to apply config to runtime: %w; previous config restored", err)
	}
	s.recordSuccessfulCompatibilityChange(yamlData, revisionPersistenceOptions{
		source:         "compat_config_update",
		summary:        "Applied config update via compatibility API",
		action:         "config.update",
		triggerSource:  "config_api",
		revisionStatus: "active",
		previousStatus: "superseded",
		deployStatus:   "succeeded",
		message:        "Config update applied via compatibility API.",
		metadata: map[string]interface{}{
			"operation": "update",
		},
	})
	return nil
}

func validateConfigYAML(yamlData []byte) error {
	tempFile, err := os.CreateTemp("", "config-validate-*.yaml")
	if err != nil {
		return err
	}
	tempPath := tempFile.Name()
	if closeErr := tempFile.Close(); closeErr != nil {
		return closeErr
	}
	defer func() {
		_ = os.Remove(tempPath)
	}()

	if writeErr := os.WriteFile(tempPath, yamlData, 0o644); writeErr != nil {
		return writeErr
	}

	parsedConfig, err := routerconfig.Parse(tempPath)
	if err != nil {
		return err
	}
	for _, endpoint := range parsedConfig.VLLMEndpoints {
		if err := ValidateEndpointAddress(endpoint.Address); err != nil {
			return fmt.Errorf("vLLM endpoint %q address validation failed: %w\n\nSupported formats:\n- IPv4: 192.168.1.1, 127.0.0.1\n- IPv6: ::1, 2001:db8::1\n- DNS names: localhost, example.com, api.example.com\n\nUnsupported formats:\n- Protocol prefixes: http://, https://\n- Paths: /api/v1, /health\n- Ports in address: use 'port' field instead", endpoint.Name, err)
		}
	}
	return nil
}

func (s *Service) RouterDefaults() (interface{}, error) {
	routerDefaultsPath := filepath.Join(s.ConfigDir, ".vllm-sr", "router-defaults.yaml")
	data, err := os.ReadFile(routerDefaultsPath)
	if err != nil {
		if os.IsNotExist(err) {
			return map[string]interface{}{}, nil
		}
		return nil, fmt.Errorf("failed to read router-defaults: %w", err)
	}

	var config interface{}
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse router-defaults: %w", err)
	}
	return config, nil
}

func (s *Service) UpdateRouterDefaults(configData map[string]interface{}) error {
	routerDefaultsPath := filepath.Join(s.ConfigDir, ".vllm-sr", "router-defaults.yaml")
	existingMap := make(map[string]interface{})

	existingData, err := os.ReadFile(routerDefaultsPath)
	if err == nil {
		if unmarshalErr := yaml.Unmarshal(existingData, &existingMap); unmarshalErr != nil {
			existingMap = map[string]interface{}{}
		}
	}
	existingMap = DeepMerge(existingMap, configData)

	yamlData, err := yaml.Marshal(existingMap)
	if err != nil {
		return fmt.Errorf("failed to convert to YAML: %w", err)
	}

	vllmSRDir := filepath.Join(s.ConfigDir, ".vllm-sr")
	if err := os.MkdirAll(vllmSRDir, 0o755); err != nil {
		return fmt.Errorf("failed to create .vllm-sr directory: %w", err)
	}
	if err := os.WriteFile(routerDefaultsPath, yamlData, 0o644); err != nil {
		return fmt.Errorf("failed to write router-defaults: %w", err)
	}
	return nil
}
