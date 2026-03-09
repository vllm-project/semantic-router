package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

func decodeConfigUpdateBody(r *http.Request) (map[string]interface{}, error) {
	var configData map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&configData); err != nil {
		return nil, fmt.Errorf("invalid request body: %w", err)
	}
	return configData, nil
}

func loadRequiredYAMLMap(configPath string) ([]byte, map[string]interface{}, error) {
	existingData, err := os.ReadFile(configPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read existing config: %w", err)
	}

	existingMap := make(map[string]interface{})
	if err := yaml.Unmarshal(existingData, &existingMap); err != nil {
		return nil, nil, fmt.Errorf("failed to parse existing config: %w", err)
	}

	return existingData, existingMap, nil
}

func loadOptionalYAMLMap(configPath string) map[string]interface{} {
	existingMap := make(map[string]interface{})
	existingData, err := os.ReadFile(configPath)
	if err != nil {
		return existingMap
	}

	if err := yaml.Unmarshal(existingData, &existingMap); err != nil {
		log.Printf("Warning: failed to parse existing config at %s, starting fresh: %v", configPath, err)
		return map[string]interface{}{}
	}

	return existingMap
}

func mergeAndValidateConfigUpdate(configPath string, existingMap, configData map[string]interface{}) ([]byte, error) {
	if shouldMergeDashboardConfigCanonically(configData) {
		if isCanonicalDashboardFullConfig(configData) {
			return renderCanonicalDashboardConfigWithPython(configData)
		}
		return mergeCanonicalDashboardConfigWithPython(configPath, configData)
	}

	originalKeyCount := len(existingMap)
	mergedConfig := deepMerge(existingMap, configData)
	if len(mergedConfig) < originalKeyCount {
		return nil, fmt.Errorf("merge would result in data loss: original had %d keys, merged has %d keys. This indicates a bug. File: %s", originalKeyCount, len(mergedConfig), configPath)
	}

	yamlData, err := yaml.Marshal(mergedConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to convert to YAML: %w", err)
	}
	if err := validateMergedDashboardConfig(yamlData, mergedConfig); err != nil {
		return nil, err
	}

	return yamlData, nil
}

func validateMergedDashboardConfig(yamlData []byte, configData map[string]interface{}) error {
	tempFile := filepath.Join(os.TempDir(), "config_validate.yaml")
	if err := os.WriteFile(tempFile, yamlData, 0o644); err != nil {
		return fmt.Errorf("failed to validate: %w", err)
	}
	defer func() {
		if err := os.Remove(tempFile); err != nil {
			log.Printf("Warning: failed to remove temp file: %v", err)
		}
	}()

	return validateDashboardConfig(tempFile, configData)
}

func mergeRouterDefaultsConfig(existingMap, configData map[string]interface{}) ([]byte, error) {
	for key, value := range configData {
		if mergedValue, merged := mergeRouterDefaultsValue(existingMap[key], value); merged {
			existingMap[key] = mergedValue
			continue
		}
		existingMap[key] = value
	}

	yamlData, err := yaml.Marshal(existingMap)
	if err != nil {
		return nil, fmt.Errorf("failed to convert to YAML: %w", err)
	}
	return yamlData, nil
}

func mergeRouterDefaultsValue(existingValue, newValue interface{}) (interface{}, bool) {
	existingMapValue, ok := existingValue.(map[string]interface{})
	if !ok {
		return nil, false
	}

	newMapValue, ok := newValue.(map[string]interface{})
	if !ok {
		return nil, false
	}

	mergedMap := make(map[string]interface{}, len(existingMapValue)+len(newMapValue))
	for existingKey, existingVal := range existingMapValue {
		mergedMap[existingKey] = existingVal
	}
	for newKey, newVal := range newMapValue {
		mergedMap[newKey] = newVal
	}
	return mergedMap, true
}

func ensureReadonlyDisabled(w http.ResponseWriter, readonlyMode bool) bool {
	if !readonlyMode {
		return true
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusForbidden)
	if err := json.NewEncoder(w).Encode(map[string]string{
		"error":   "readonly_mode",
		"message": "Dashboard is in read-only mode. Configuration editing is disabled.",
	}); err != nil {
		log.Printf("Error encoding readonly response: %v", err)
	}

	return false
}

func formatConfigUpdateError(err error) string {
	message := err.Error()
	if strings.HasPrefix(message, "config validation failed") {
		suffix := strings.TrimPrefix(message, "config validation failed")
		return "Config validation failed" + suffix
	}
	return message
}
