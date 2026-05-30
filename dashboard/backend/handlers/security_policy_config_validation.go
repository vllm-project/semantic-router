package handlers

import (
	"fmt"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func validateMergedSecurityConfig(configData []byte) error {
	var canonical routerconfig.CanonicalConfig
	if err := yaml.Unmarshal(configData, &canonical); err != nil {
		return fmt.Errorf("decode merged canonical config: %w", err)
	}
	if err := validateCanonicalEndpointRefs(canonical); err != nil {
		return fmt.Errorf("merged canonical endpoint validation failed: %w", err)
	}

	parsed, err := routerconfig.ParseYAMLBytes(configData)
	if err != nil {
		return fmt.Errorf("merged router config validation failed: %w", err)
	}
	for _, endpoint := range parsed.VLLMEndpoints {
		if endpoint.ProviderProfileName != "" && endpoint.Address == "" {
			continue
		}
		if err := validateEndpointAddress(endpoint.Address); err != nil {
			return fmt.Errorf("merged router endpoint %q address validation failed: %w", endpoint.Name, err)
		}
	}

	return nil
}
