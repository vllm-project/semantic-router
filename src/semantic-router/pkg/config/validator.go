package config

import (
	"fmt"
	"net"
	"regexp"
	"strings"
)

type configContractValidator func(*RouterConfig) error

// globalConfigValidatorEntry is one row of globalConfigValidatorRegistry.
type globalConfigValidatorEntry struct {
	validate             configContractValidator
	requiresRoutingState bool
}

// globalConfigValidators returns the registered global validators, in
// registration order. With includeRoutingStateDependent false it filters out
// the entries requiresRoutingState marks, giving the subset safe to run
// before Kubernetes CRD conversion has populated provider and routing state.
func globalConfigValidators(includeRoutingStateDependent bool) []configContractValidator {
	out := make([]configContractValidator, 0, len(globalConfigValidatorRegistry))
	for _, entry := range globalConfigValidatorRegistry {
		if entry.requiresRoutingState && !includeRoutingStateDependent {
			continue
		}
		out = append(out, entry.validate)
	}
	return out
}

var (
	// Pre-compiled regular expressions for better performance
	protocolRegex = regexp.MustCompile(`^https?://`)
	pathRegex     = regexp.MustCompile(`/`)
	// Pattern to match IPv4 address followed by port number
	ipv4PortRegex = regexp.MustCompile(`^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+$`)
	// Pattern to match IPv6 address followed by port number [::1]:8080
	ipv6PortRegex = regexp.MustCompile(`^\[.*\]:\d+$`)

	// globalConfigValidatorRegistry is the one place a global contract
	// validator is registered. requiresRoutingState marks a validator that
	// reads provider or routing state (Providers.Models, Recipes,
	// RoutingScope, decision refs) the Kubernetes source populates only
	// after CRD conversion; validateConfigStructure defers those for that
	// source instead of running them against a document that has not been
	// merged with its CRDs yet. See issue #3758.
	globalConfigValidatorRegistry = []globalConfigValidatorEntry{
		// Reads only cfg.API.RoutingPreview, a static field: safe to run
		// before CRD conversion.
		{validate: validateRoutingPreviewConfig},
		{validate: validateModelPricingContracts},
		{validate: validateReasoningFamilyContracts},
		{validate: validateGlobalSemanticCacheContracts},
		{validate: validateGlobalMemoryContracts},
		{validate: validateEmbeddingModelContracts},
		{validate: validateGlobalModalityContracts},
		// validateModelSelectionConfig, validateCategoryModelBackendContracts
		// and validatePIIModelBackendContracts resolve a remote backend's
		// model name against ExternalModels, a static document field the
		// Kubernetes CRD conversion never touches (only Providers.Models,
		// Routing.Decisions, and Routing.Signals are CRD-populated); see
		// validator_kubernetes_startup_test.go for the case proving this.
		{validate: validateModelSelectionConfig},
		{validate: validateCategoryModelBackendContracts},
		// validateComplexityModelBackendContracts also runs
		// ValidateComplexityRuleBoundaries, which checks rules "that exist
		// only inside a recipe, since routing.signals is replaced wholesale
		// per recipe" (its own doc comment) - genuinely routing-state
		// dependent, unlike the two above.
		{validate: validateComplexityModelBackendContracts, requiresRoutingState: true},
		{validate: validatePIIModelBackendContracts},
		{validate: validateGlobalClassifierRuntimeContracts, requiresRoutingState: true},
		{validate: validateGlobalRouterLearningConfig, requiresRoutingState: true},
		{validate: validateReMoMContracts},
		{validate: validateFusionContracts},
		{validate: validateFlowContracts},
		{validate: validateAdvancedToolFilteringConfig},
		{validate: validatePromptCompressionContracts},
		{validate: validateHallucinationContracts},
		{validate: validateModelAdmissionContracts},
		{validate: validateModelDeploymentContracts, requiresRoutingState: true},
	}

	// globalConfigContractValidators is every registered global validator, in
	// registration order. Kept as a plain slice for existing callers and
	// tests; globalConfigValidators(false) is the filtered view.
	globalConfigContractValidators = globalConfigValidators(true)

	routingProfileContractValidators = []configContractValidator{
		validateRuleOperatorContracts,
		validateRoutingLocalNames,
		validateLanguageContracts,
		validateContextContracts,
		validateRoutingStrategy,
		validateDecisionSignalReferences,
		validateDomainContracts,
		validateStructureContracts,
		validateReaskContracts,
		validateProjectionContracts,
		validateKnowledgeBaseContracts,
		validateConversationContracts,
		validateDecisionContracts,
		validateDecisionSemanticCacheContracts,
		validateDecisionMemoryContracts,
		validateEmbeddingSignalContracts,
		validateRoutingModalityContracts,
		validateComplexityContracts,
		validateJailbreakContracts,
		validateSignalStageContracts,
		validateHallucinationSignalContracts,
		validateDecisionRouterLearningConfig,
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

func validateRoutingStrategy(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	return cfg.Strategy.Validate()
}

// validPromptGuardVariants is the set of recognized PromptGuardConfig.Variant values.
var validPromptGuardVariants = map[string]bool{
	"":                          true, // unset defaults to PromptGuardVariantMmBERT32K under canonical resolution
	PromptGuardVariantCandle:    true,
	PromptGuardVariantMmBERT32K: true,
}

// prompt_guard backend validation lives in validator_prompt_guard.go.

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
	if cfg.ConfigSource == ConfigSourceKubernetes {
		// Routing state (decisions, recipes) and CRD-supplied provider models
		// are not populated yet: they arrive from CRD conversion after this
		// parse. Run the global validators that do not depend on that state
		// now, so a bad static setting fails at load instead of surfacing
		// later; skip the routing-profile validators and the global ones
		// requiresRoutingState marks. ValidateKubernetesConfigContracts runs
		// the full set, including those, once the reconciler has merged CRDs.
		return runConfigContractValidators(cfg, globalConfigValidators(false))
	}
	return validateConfigContracts(cfg)
}

// ValidateKubernetesConfigContracts runs the validators that apply after CRDs
// have been converted into the canonical runtime config. The initial
// Kubernetes static-config parse stays tolerant because routing state is still
// absent there; the reconciler calls this function once the pool and route have
// been merged.
func ValidateKubernetesConfigContracts(cfg *RouterConfig) error {
	return validateConfigContracts(cfg)
}

func validateConfigContracts(cfg *RouterConfig) error {
	if err := runConfigContractValidators(cfg, globalConfigContractValidators); err != nil {
		return err
	}
	return visitRoutingProfileConfigs(cfg, func(profile *RouterConfig) error {
		return runConfigContractValidators(profile, routingProfileContractValidators)
	})
}

func runConfigContractValidators(cfg *RouterConfig, validators []configContractValidator) error {
	for _, validator := range validators {
		if err := validator(cfg); err != nil {
			return err
		}
	}
	return nil
}

func validateModelSelectionConfig(cfg *RouterConfig) error {
	if err := validatePromptGuardBackend(cfg); err != nil {
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
