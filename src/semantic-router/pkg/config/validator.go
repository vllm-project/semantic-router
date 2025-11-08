package config

import (
	"fmt"
	"net"
	"regexp"
	"strings"
)

var (
	// Pre-compiled regular expressions for better performance
	protocolRegex = regexp.MustCompile(`^https?://`)
	pathRegex     = regexp.MustCompile(`/`)
	// Pattern to match IPv4 address followed by port number
	ipv4PortRegex = regexp.MustCompile(`^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+$`)
	// Pattern to match IPv6 address followed by port number [::1]:8080
	ipv6PortRegex = regexp.MustCompile(`^\[.*\]:\d+$`)
)

// validateIPAddress validates IP address format
// Supports IPv4 and IPv6 addresses, rejects domain names, protocol prefixes, paths, etc.
func validateIPAddress(address string) error {
	// Check for empty string
	trimmed := strings.TrimSpace(address)
	if trimmed == "" {
		return fmt.Errorf("address cannot be empty")
	}

	// Check for protocol prefixes (http://, https://)
	if protocolRegex.MatchString(trimmed) {
		return fmt.Errorf("protocol prefixes (http://, https://) are not supported, got: %s", address)
	}

	// Check for paths (contains / character)
	if pathRegex.MatchString(trimmed) {
		return fmt.Errorf("paths are not supported, got: %s", address)
	}

	// Check for port numbers (IPv4 address followed by port or IPv6 address followed by port)
	if ipv4PortRegex.MatchString(trimmed) || ipv6PortRegex.MatchString(trimmed) {
		return fmt.Errorf("port numbers in address are not supported, use 'port' field instead, got: %s", address)
	}

	// Use Go standard library to validate IP address format
	ip := net.ParseIP(trimmed)
	if ip == nil {
		return fmt.Errorf("invalid IP address format, got: %s", address)
	}

	return nil
}

// validateVLLMEndpoints validates the address format of all vLLM endpoints
func validateVLLMEndpoints(endpoints []VLLMEndpoint) error {
	for _, endpoint := range endpoints {
		if err := validateIPAddress(endpoint.Address); err != nil {
			return fmt.Errorf("vLLM endpoint '%s' address validation failed: %w\n\nSupported formats:\n- IPv4: 192.168.1.1, 127.0.0.1\n- IPv6: ::1, 2001:db8::1\n\nUnsupported formats:\n- Domain names: example.com, localhost\n- Protocol prefixes: http://, https://\n- Paths: /api/v1, /health\n- Ports in address: use 'port' field instead", endpoint.Name, err)
		}
	}
	return nil
}

// isValidIPv4 checks if the address is a valid IPv4 address
func isValidIPv4(address string) bool {
	ip := net.ParseIP(address)
	return ip != nil && ip.To4() != nil
}

// isValidIPv6 checks if the address is a valid IPv6 address
func isValidIPv6(address string) bool {
	ip := net.ParseIP(address)
	return ip != nil && ip.To4() == nil
}

// getIPAddressType returns the IP address type information for error messages and debugging
func getIPAddressType(address string) string {
	if isValidIPv4(address) {
		return "IPv4"
	}
	if isValidIPv6(address) {
		return "IPv6"
	}
	return "invalid"
}

// validateConfigStructure performs additional validation on the parsed config
func validateConfigStructure(cfg *RouterConfig) error {
	// Ensure all categories have at least one model with scores
	for _, category := range cfg.Categories {
		if len(category.ModelScores) == 0 {
			return fmt.Errorf("category '%s' has no model_scores defined - each category must have at least one model", category.Name)
		}

		// Validate each model score has the required fields
		for i, modelScore := range category.ModelScores {
			if modelScore.Model == "" {
				return fmt.Errorf("category '%s', model_scores[%d]: model name cannot be empty", category.Name, i)
			}
			if modelScore.Score <= 0 {
				return fmt.Errorf("category '%s', model '%s': score must be greater than 0, got %f", category.Name, modelScore.Model, modelScore.Score)
			}
			if modelScore.UseReasoning == nil {
				return fmt.Errorf("category '%s', model '%s': missing required field 'use_reasoning'", category.Name, modelScore.Model)
			}

			// Validate LoRA name if specified
			if modelScore.LoRAName != "" {
				if err := validateLoRAName(cfg, modelScore.Model, modelScore.LoRAName); err != nil {
					return fmt.Errorf("category '%s', model '%s': %w", category.Name, modelScore.Model, err)
				}
			}
		}
	}

	// Validate vLLM endpoints address formats
	if err := validateVLLMEndpoints(cfg.VLLMEndpoints); err != nil {
		return err
	}

	return nil
}

// validateLoRAName checks if the specified LoRA name is defined in the model's configuration
func validateLoRAName(cfg *RouterConfig, modelName string, loraName string) error {
	// Check if the model exists in model_config
	modelParams, exists := cfg.ModelConfig[modelName]
	if !exists {
		return fmt.Errorf("lora_name '%s' specified but model '%s' is not defined in model_config", loraName, modelName)
	}

	// Check if the model has any LoRAs defined
	if len(modelParams.LoRAs) == 0 {
		return fmt.Errorf("lora_name '%s' specified but model '%s' has no loras defined in model_config", loraName, modelName)
	}

	// Check if the specified LoRA name exists in the model's LoRA list
	for _, lora := range modelParams.LoRAs {
		if lora.Name == loraName {
			return nil // Valid LoRA name found
		}
	}

	// LoRA name not found, provide helpful error message
	availableLoRAs := make([]string, len(modelParams.LoRAs))
	for i, lora := range modelParams.LoRAs {
		availableLoRAs[i] = lora.Name
	}
	return fmt.Errorf("lora_name '%s' is not defined in model '%s' loras. Available LoRAs: %v", loraName, modelName, availableLoRAs)
}
