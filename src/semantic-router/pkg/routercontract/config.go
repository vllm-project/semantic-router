package routercontract

import (
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// CanonicalConfig and related aliases define the public router config contract
// consumed by control-plane code.
type (
	CanonicalConfig            = config.CanonicalConfig
	CanonicalRouting           = config.CanonicalRouting
	CanonicalSignals           = config.CanonicalSignals
	CanonicalProjections       = config.CanonicalProjections
	CanonicalProviders         = config.CanonicalProviders
	CanonicalProviderDefaults  = config.CanonicalProviderDefaults
	CanonicalProviderModel     = config.CanonicalProviderModel
	CanonicalBackendRef        = config.CanonicalBackendRef
	CanonicalGlobal            = config.CanonicalGlobal
	CanonicalRouterGlobal      = config.CanonicalRouterGlobal
	CanonicalServiceGlobal     = config.CanonicalServiceGlobal
	CanonicalStoreGlobal       = config.CanonicalStoreGlobal
	CanonicalIntegrationGlobal = config.CanonicalIntegrationGlobal
	CanonicalModelCatalog      = config.CanonicalModelCatalog
	CanonicalEmbeddingModels   = config.CanonicalEmbeddingModels
	CanonicalSystemModels      = config.CanonicalSystemModels
	CanonicalModelModules      = config.CanonicalModelModules
	CanonicalPromptGuardModule = config.CanonicalPromptGuardModule
	CanonicalClassifierModule  = config.CanonicalClassifierModule
	CanonicalCategoryModule    = config.CanonicalCategoryModule
	CanonicalPIIModule         = config.CanonicalPIIModule
	Listener                   = config.Listener
	RoutingModel               = config.RoutingModel
	Decision                   = config.Decision
	ComplexityRule             = config.ComplexityRule
	RuleCombination            = config.RuleCombination
	RuleNode                   = config.RuleNode
	ReasoningFamilyConfig      = config.ReasoningFamilyConfig
	LoRAAdapter                = config.LoRAAdapter
	APIConfig                  = config.APIConfig
	ObservabilityConfig        = config.ObservabilityConfig
	SemanticCache              = config.SemanticCache
	ToolsConfig                = config.ToolsConfig
	EmbeddingModels            = config.EmbeddingModels
	Category                   = config.Category
	CategoryMetadata           = config.CategoryMetadata
	ModelRef                   = config.ModelRef
	AlgorithmConfig            = config.AlgorithmConfig
	KeywordRule                = config.KeywordRule
	EmbeddingRule              = config.EmbeddingRule
	FactCheckRule              = config.FactCheckRule
	UserFeedbackRule           = config.UserFeedbackRule
	ReaskRule                  = config.ReaskRule
	PreferenceRule             = config.PreferenceRule
	LanguageRule               = config.LanguageRule
	ContextRule                = config.ContextRule
	StructureRule              = config.StructureRule
	ModalityRule               = config.ModalityRule
	RoleBinding                = config.RoleBinding
	JailbreakRule              = config.JailbreakRule
	PIIRule                    = config.PIIRule
	KBSignalRule               = config.KBSignalRule
	ProjectionPartition        = config.ProjectionPartition
	ProjectionScore            = config.ProjectionScore
	ProjectionMapping          = config.ProjectionMapping
)

// ParseYAMLBytes validates canonical config bytes through the runtime parser
// and re-exports the normalized public contract for control-plane consumers.
func ParseYAMLBytes(data []byte) (*CanonicalConfig, error) {
	parsed, err := config.ParseYAMLBytes(data)
	if err != nil {
		return nil, err
	}
	canonical := config.CanonicalConfigFromRouterConfig(parsed)
	return &canonical, nil
}

// Parse validates a canonical config file and returns the normalized public
// contract view.
func Parse(configPath string) (*CanonicalConfig, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}
	return ParseYAMLBytes(data)
}

// DefaultCanonicalGlobal returns the router-owned canonical defaults exposed to
// control-plane consumers.
func DefaultCanonicalGlobal() CanonicalGlobal {
	return config.DefaultCanonicalGlobal()
}

// SupportedRoutingDomainNames returns the repository-supported routing domain
// labels exposed as part of the public config contract.
func SupportedRoutingDomainNames() []string {
	return config.SupportedRoutingDomainNames()
}

// IsSupportedRoutingDomainName reports whether name matches the public routing
// domain contract.
func IsSupportedRoutingDomainName(name string) bool {
	return config.IsSupportedRoutingDomainName(name)
}

// SuggestSupportedRoutingDomainName suggests the canonical routing domain label
// for a near-match.
func SuggestSupportedRoutingDomainName(name string) string {
	return config.SuggestSupportedRoutingDomainName(name)
}
