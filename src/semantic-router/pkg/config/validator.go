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

// validateVLLMClassifierConfig validates vLLM classifier configuration when use_vllm is true
// Note: vLLM configuration is now in external_models, not in PromptGuardConfig
// This function is kept for backward compatibility but does minimal validation
func validateVLLMClassifierConfig(cfg *PromptGuardConfig) error {
	if !cfg.UseVLLM {
		return nil // Skip validation if not using vLLM
	}

	// When use_vllm is true, external_models with model_role="guardrail" is required
	// This will be validated in the main config validation
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

// validateConfigStructure performs additional validation on the parsed config.
func validateConfigStructure(cfg *RouterConfig) error {
	// In Kubernetes mode, decisions and model_config will be loaded from CRDs
	// Skip validation for these fields during initial config parse
	if cfg.ConfigSource == ConfigSourceKubernetes {
		return nil
	}
	if hasLegacyLatencyRoutingConfig(cfg) {
		return fmt.Errorf("legacy latency config is no longer supported; use decision.algorithm.type=latency_aware and remove signals.latency_rules / conditions.type=latency")
	}

	validators := []func(*RouterConfig) error{
		validateDomainContracts,
		validateStructureContracts,
		validateReaskContracts,
		validateProjectionContracts,
		validateKnowledgeBaseContracts,
		validateConversationContracts,
		validateDecisionContracts,
		validateModalityContracts,
		validateModelSelectionConfig,
		validateAdvancedToolFilteringConfig,
	}
	for _, validate := range validators {
		if err := validate(cfg); err != nil {
			return err
		}
	}
	return nil
}

func validateModelSelectionConfig(cfg *RouterConfig) error {
	if err := validateVLLMClassifierConfig(&cfg.PromptGuard); err != nil {
		return err
	}
	if err := validateModelSwitchGate(cfg.ModelSelection.ModelSwitchGate); err != nil {
		return err
	}
	warnModelSwitchGateEnforceWithoutCostSignals(cfg.ModelSelection)
	return nil
}
