package config

import "fmt"

type configContractValidator func(*RouterConfig) error

type configValidationStage uint8

const (
	staticConfigValidation configValidationStage = iota
	completeConfigValidation
)

// Register contract validators here once. Global validators run at every load,
// before runtime resources can be created. Only checks requiring CRD routing
// state belong in the routing groups; each family still owns its own rules.
var (
	globalConfigContractValidators = []configContractValidator{
		validateListenerContracts,
		validateRoutingPreviewConfig,
		validateModelPricingContracts,
		validateReasoningFamilyContracts,
		validateGlobalMemoryContracts,
		validateEmbeddingStores,
		validateModelSelectionConfig,
		validateGlobalRouterLearningConfig,
		validateLossyTranslationContracts,
		validateRoutingInputContracts,
		validateContextWindowContracts,
		validateRouterHeadersContracts,
		validateFlowContracts,
		validateModelAdmissionContracts,
		validateModelSignalTimeoutContracts,
		validateVideoEntryContracts,
		validateBackendTLSFiles,
		validateModelDeploymentContracts,
		validateFallbackForwardRequestHeaders,
		validateForwardedIdentityHeaders,
		validateGlobalModelBindingContracts,
		validateGlobalToolSessionsContracts,
		validateGlobalRouterReplayCaptureContracts,
		validateGlobalPluginContracts,
	}

	// These contracts need the complete routing graph, including all recipes.
	routingConfigContractValidators = []configContractValidator{
		validateKVTransferConfig,
		validateModelBindingContracts,
		validateModelResolutionContracts,
		logConfigWarnings,
	}

	routingProfileContractValidators = []configContractValidator{
		validateRuleOperatorContracts,
		validateRoutingLocalNames,
		validateKeywordContracts,
		validateLanguageContracts,
		validateContextContracts,
		validateContextCapacity,
		validateRoutingStrategy,
		validateDecisionSignalReferences,
		validateDomainContracts,
		validateApplicationSignalContracts,
		validateProjectionContracts,
		validateConversationContracts,
		validateDecisionContracts,
		validateDecisionOutputContracts,
		validateSignalStageContracts,
		validateResponseGuardReferences,
		validateMaskingSignals,
		validateHeaderMutationContracts,
		validateDecisionRouterLearningConfig,
		validateDecisionRouterReplayCaptureContracts,
	}
)

// validateConfigStructure validates the initial document. Kubernetes supplies
// routing state later, but all static global settings are already available.
func validateConfigStructure(cfg *RouterConfig) error {
	stage := completeConfigValidation
	if cfg != nil && cfg.ConfigSource == ConfigSourceKubernetes {
		stage = staticConfigValidation
	}
	return validateConfigContractsAtStage(cfg, stage)
}

// ValidateKubernetesConfigContracts validates the complete candidate after CRDs
// have been merged, before the reconciler publishes it to the runtime.
func ValidateKubernetesConfigContracts(cfg *RouterConfig) error {
	return validateConfigContracts(cfg)
}

func validateConfigContracts(cfg *RouterConfig) error {
	return validateConfigContractsAtStage(cfg, completeConfigValidation)
}

func validateConfigContractsAtStage(cfg *RouterConfig, stage configValidationStage) error {
	if cfg == nil {
		return fmt.Errorf("router configuration is nil")
	}
	effective := *cfg
	effective.ModelBindings = cfg.EffectiveModelBindings(cfg.ModelBindings)
	if effective.scopedBindings == nil {
		effective.scopedBindings = cloneModelMap(cfg.ModelBindings)
		if effective.scopedBindings == nil {
			effective.scopedBindings = map[string]ModelBinding{}
		}
	}
	cfg = &effective
	if err := runConfigContractValidators(cfg, globalConfigContractValidators); err != nil {
		return err
	}
	if stage == staticConfigValidation {
		return nil
	}
	if err := runConfigContractValidators(cfg, routingConfigContractValidators); err != nil {
		return err
	}
	if err := visitRoutingProfileConfigs(cfg, func(profile *RouterConfig) error {
		return runConfigContractValidators(profile, routingProfileContractValidators)
	}); err != nil {
		return err
	}
	// Which calls reach a delegated upstream follows from the whole, valid
	// routing graph.
	return validateDelegatedUpstreams(cfg)
}

func runConfigContractValidators(cfg *RouterConfig, validators []configContractValidator) error {
	for _, validator := range validators {
		if err := validator(cfg); err != nil {
			return err
		}
	}
	return nil
}
