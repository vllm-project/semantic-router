package config

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
		validateModelPricingContracts,
		validateReasoningFamilyContracts,
		validateGlobalSemanticCacheContracts,
		validateGlobalMemoryContracts,
		validateEmbeddingModelContracts,
		validateGlobalModalityContracts,
		validateModelSelectionConfig,
		validateCategoryModelBackendContracts,
		validateComplexityModelBackendContracts,
		validatePIIModelBackendContracts,
		validateGlobalRouterLearningConfig,
		validateReMoMContracts,
		validateFusionContracts,
		validateFlowContracts,
		validateAdvancedToolFilteringConfig,
		validatePromptCompressionContracts,
		validateHallucinationContracts,
		validateModelAdmissionContracts,
	}

	// These contracts need the complete routing graph, including all recipes.
	routingConfigContractValidators = []configContractValidator{
		validateGlobalClassifierRuntimeContracts,
		validateComplexityRoutingContracts,
	}

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

// validateConfigStructure validates the initial document. Kubernetes supplies
// routing state later, but all static global settings are already available.
func validateConfigStructure(cfg *RouterConfig) error {
	stage := completeConfigValidation
	if cfg.ConfigSource == ConfigSourceKubernetes {
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
	if err := runConfigContractValidators(cfg, globalConfigContractValidators); err != nil {
		return err
	}
	if stage == staticConfigValidation {
		return nil
	}
	if err := runConfigContractValidators(cfg, routingConfigContractValidators); err != nil {
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
