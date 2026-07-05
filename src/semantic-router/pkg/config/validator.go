package config

import (
	"fmt"
	"net"
	"regexp"
	"strings"
)

type configValidationScope uint8

const (
	configValidationScopeFile configValidationScope = 1 << iota
	configValidationScopeKubernetes
)

type configContractValidator struct {
	name     string
	validate func(*RouterConfig) error
	scopes   configValidationScope
}

var (
	// Pre-compiled regular expressions for better performance
	protocolRegex = regexp.MustCompile(`^https?://`)
	pathRegex     = regexp.MustCompile(`/`)
	// Pattern to match IPv4 address followed by port number
	ipv4PortRegex = regexp.MustCompile(`^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+$`)
	// Pattern to match IPv6 address followed by port number [::1]:8080
	ipv6PortRegex = regexp.MustCompile(`^\[.*\]:\d+$`)

	sharedConfigContractValidators = []configContractValidator{
		{
			name:     "legacy_latency_routing",
			validate: validateLegacyLatencyRoutingConfig,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "domain",
			validate: validateDomainContracts,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "structure",
			validate: validateStructureContracts,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "reask",
			validate: validateReaskContracts,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "projection",
			validate: validateProjectionContracts,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "knowledge_base",
			validate: validateKnowledgeBaseContracts,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "conversation",
			validate: validateConversationContracts,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "decision",
			validate: validateDecisionContracts,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "semantic_cache",
			validate: validateSemanticCacheContracts,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "memory",
			validate: validateMemoryContracts,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "embedding",
			validate: validateEmbeddingContracts,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "modality",
			validate: validateModalityContracts,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "model_selection",
			validate: validateModelSelectionConfig,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "router_learning",
			validate: validateRouterLearningConfig,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "remom",
			validate: validateReMoMContracts,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "fusion",
			validate: validateFusionContracts,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "flow",
			validate: validateFlowContracts,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "advanced_tool_filtering",
			validate: validateAdvancedToolFilteringConfig,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
		{
			name:     "prompt_compression",
			validate: validatePromptCompressionContracts,
			scopes:   configValidationScopeFile | configValidationScopeKubernetes,
		},
	}
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
	return validateConfigContracts(cfg, configValidationScopeFile)
}

// ValidateKubernetesConfigContracts runs the validators that apply after CRDs
// have been converted into the canonical runtime config. The initial
// Kubernetes static-config parse stays tolerant because routing state is still
// absent there; the reconciler calls this function once the pool and route have
// been merged.
func ValidateKubernetesConfigContracts(cfg *RouterConfig) error {
	return validateConfigContracts(cfg, configValidationScopeKubernetes)
}

func validateConfigContracts(cfg *RouterConfig, scope configValidationScope) error {
	for _, validator := range sharedConfigContractValidators {
		if validator.scopes&scope == 0 {
			continue
		}
		if err := validator.validate(cfg); err != nil {
			return err
		}
	}
	return nil
}

func validateLegacyLatencyRoutingConfig(cfg *RouterConfig) error {
	if hasLegacyLatencyRoutingConfig(cfg) {
		return fmt.Errorf("legacy latency config is no longer supported; use decision.algorithm.type=latency_aware and remove signals.latency_rules / conditions.type=latency")
	}
	return nil
}

func validateModelSelectionConfig(cfg *RouterConfig) error {
	if err := validateVLLMClassifierConfig(&cfg.PromptGuard); err != nil {
		return err
	}
	if isSessionAwareSelectionConfigConfigured(cfg.ModelSelection.SessionAware) {
		return fmt.Errorf("global.router.model_selection.session_aware is no longer supported; use global.router.learning.protection")
	}
	if isModelSwitchGateConfigured(cfg.ModelSelection.ModelSwitchGate) {
		return fmt.Errorf("global.router.model_selection.model_switch_gate is no longer supported; use global.router.learning.protection.tuning")
	}
	if isLookupTableConfigConfigured(cfg.ModelSelection.LookupTables) {
		return fmt.Errorf("global.router.model_selection.lookup_tables has moved to future Router Learning experience; remove lookup_tables from public config")
	}
	if method := strings.TrimSpace(cfg.ModelSelection.Method); removedGlobalLearningSelector(method) {
		return fmt.Errorf("global.router.model_selection.%s is no longer supported; use global.router.learning.adaptation", method)
	}
	if isEloSelectionConfigConfigured(cfg.ModelSelection.Elo) {
		return fmt.Errorf("global.router.model_selection.elo is no longer supported; use global.router.learning.adaptation")
	}
	return nil
}
