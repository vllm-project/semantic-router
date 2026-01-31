package config

import (
	"fmt"
	"net"
	"regexp"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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

// validateConfigStructure performs additional validation on the parsed config
func validateConfigStructure(cfg *RouterConfig) error {
	// In Kubernetes mode, decisions and model_config will be loaded from CRDs
	// Skip validation for these fields during initial config parse
	if cfg.ConfigSource == ConfigSourceKubernetes {
		// Skip validation for decisions and model_config
		return nil
	}

	// File mode: validate decisions have at least one model ref
	for _, decision := range cfg.Decisions {
		if len(decision.ModelRefs) == 0 {
			return fmt.Errorf("decision '%s' has no modelRefs defined - each decision must have at least one model", decision.Name)
		}

		// Validate each model ref has the required fields
		for i, modelRef := range decision.ModelRefs {
			if modelRef.Model == "" {
				return fmt.Errorf("decision '%s', modelRefs[%d]: model name cannot be empty", decision.Name, i)
			}
			if modelRef.UseReasoning == nil {
				return fmt.Errorf("decision '%s', model '%s': missing required field 'use_reasoning'", decision.Name, modelRef.Model)
			}

			// Validate LoRA name if specified
			if modelRef.LoRAName != "" {
				if err := validateLoRAName(cfg, modelRef.Model, modelRef.LoRAName); err != nil {
					return fmt.Errorf("decision '%s', model '%s': %w", decision.Name, modelRef.Model, err)
				}
			}
		}
	}

	// Validate vLLM classifier configurations
	if err := validateVLLMClassifierConfig(&cfg.PromptGuard); err != nil {
		return err
	}

	// Validate advanced tool filtering configuration (opt-in)
	if err := validateAdvancedToolFilteringConfig(cfg); err != nil {
		return err
	}

	// Validate latency rules
	if err := validateLatencyRules(cfg.Signals.LatencyRules); err != nil {
		return err
	}

	return nil
}

// validateLatencyRules validates latency rule configurations
func validateLatencyRules(rules []LatencyRule) error {
	for i, rule := range rules {
		if rule.Name == "" {
			return fmt.Errorf("latency_rules[%d]: name cannot be empty", i)
		}

		// At least one of tpot_percentile or ttft_percentile must be set
		hasTPOTPercentile := rule.TPOTPercentile > 0
		hasTTFTPercentile := rule.TTFTPercentile > 0

		if !hasTPOTPercentile && !hasTTFTPercentile {
			return fmt.Errorf("latency_rules[%d] (%s): must specify at least one of tpot_percentile (1-100) or ttft_percentile (1-100). RECOMMENDED: use both for comprehensive latency evaluation", i, rule.Name)
		}

		// Warn (but don't error) if only one is set - recommend using both
		if hasTPOTPercentile && !hasTTFTPercentile {
			logging.Warnf("latency_rules[%d] (%s): only tpot_percentile is set. RECOMMENDED: also set ttft_percentile for comprehensive latency evaluation (user-perceived latency)", i, rule.Name)
		}
		if !hasTPOTPercentile && hasTTFTPercentile {
			logging.Warnf("latency_rules[%d] (%s): only ttft_percentile is set. RECOMMENDED: also set tpot_percentile for comprehensive latency evaluation (token generation throughput)", i, rule.Name)
		}

		// Validate TPOT percentile if set
		if hasTPOTPercentile && (rule.TPOTPercentile < 1 || rule.TPOTPercentile > 100) {
			return fmt.Errorf("latency_rules[%d] (%s): tpot_percentile must be between 1 and 100, got: %d", i, rule.Name, rule.TPOTPercentile)
		}

		// Validate TTFT percentile if set
		if hasTTFTPercentile && (rule.TTFTPercentile < 1 || rule.TTFTPercentile > 100) {
			return fmt.Errorf("latency_rules[%d] (%s): ttft_percentile must be between 1 and 100, got: %d", i, rule.Name, rule.TTFTPercentile)
		}
	}
	return nil
}

func validateAdvancedToolFilteringConfig(cfg *RouterConfig) error {
	if cfg == nil || cfg.Tools.AdvancedFiltering == nil {
		return nil
	}

	advanced := cfg.Tools.AdvancedFiltering
	if !advanced.Enabled {
		return nil
	}

	if advanced.CandidatePoolSize != nil && *advanced.CandidatePoolSize < 0 {
		return fmt.Errorf("tools.advanced_filtering.candidate_pool_size must be >= 0")
	}

	if advanced.MinLexicalOverlap != nil && *advanced.MinLexicalOverlap < 0 {
		return fmt.Errorf("tools.advanced_filtering.min_lexical_overlap must be >= 0")
	}

	if advanced.MinCombinedScore != nil &&
		(*advanced.MinCombinedScore < 0.0 || *advanced.MinCombinedScore > 1.0) {
		return fmt.Errorf("tools.advanced_filtering.min_combined_score must be between 0.0 and 1.0")
	}

	if advanced.CategoryConfidenceThreshold != nil &&
		(*advanced.CategoryConfidenceThreshold < 0.0 || *advanced.CategoryConfidenceThreshold > 1.0) {
		return fmt.Errorf("tools.advanced_filtering.category_confidence_threshold must be between 0.0 and 1.0")
	}

	weightFields := []struct {
		name  string
		value *float32
	}{
		{"embed", advanced.Weights.Embed},
		{"lexical", advanced.Weights.Lexical},
		{"tag", advanced.Weights.Tag},
		{"name", advanced.Weights.Name},
		{"category", advanced.Weights.Category},
	}
	for _, field := range weightFields {
		if field.value != nil && (*field.value < 0.0 || *field.value > 1.0) {
			return fmt.Errorf("tools.advanced_filtering.weights.%s must be between 0.0 and 1.0", field.name)
		}
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
