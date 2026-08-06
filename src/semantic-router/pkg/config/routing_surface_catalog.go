package config

import "sort"

const (
	DecisionAlgorithmAutoMix      = "automix"
	DecisionAlgorithmConfidence   = "confidence"
	DecisionAlgorithmFusion       = "fusion"
	DecisionAlgorithmHybrid       = "hybrid"
	DecisionAlgorithmKMeans       = "kmeans"
	DecisionAlgorithmKNN          = "knn"
	DecisionAlgorithmLatencyAware = "latency_aware"
	DecisionAlgorithmMLP          = "mlp"
	DecisionAlgorithmMultiFactor  = "multi_factor"
	DecisionAlgorithmRatings      = "ratings"
	DecisionAlgorithmReMoM        = "remom"
	DecisionAlgorithmRouterDC     = "router_dc"
	DecisionAlgorithmStatic       = "static"
	DecisionAlgorithmSVM          = "svm"
	DecisionAlgorithmWorkflows    = "workflows"
	DecisionAlgorithmPrompt       = "prompt"

	DecisionPluginSemanticCache       = "semantic-cache"
	DecisionPluginSystemPrompt        = "system_prompt"
	DecisionPluginHeaderMutation      = "header_mutation"
	DecisionPluginHallucination       = "hallucination"
	DecisionPluginResponseJailbreak   = "response_jailbreak"
	DecisionPluginRouterReplay        = "router_replay"
	DecisionPluginMemory              = "memory"
	DecisionPluginRAG                 = "rag"
	DecisionPluginImageGen            = "image_gen"
	DecisionPluginFastResponse        = "fast_response"
	DecisionPluginRequestParams       = "request_params"
	DecisionPluginToolSelection       = "tool_selection"
	DecisionPluginProviderPromptCache = "provider_prompt_cache"
	DecisionPluginContextCompression  = "context_compression"
)

var supportedSignalTypes = []string{
	SignalTypeAuthz,
	SignalTypeComplexity,
	SignalTypeContext,
	SignalTypeConversation,
	SignalTypeDomain,
	SignalTypeEmbedding,
	SignalTypeFactCheck,
	SignalTypeJailbreak,
	SignalTypeKeyword,
	SignalTypeLanguage,
	SignalTypeModality,
	SignalTypePII,
	SignalTypePreference,
	SignalTypeReask,
	SignalTypeStructure,
	SignalTypeKB,
	SignalTypeUserFeedback,
	SignalTypeEvent,
	SignalTypeMetadata,
	SignalTypeClassifier,
}

var supportedDecisionPluginTypes = []string{
	DecisionPluginFastResponse,
	DecisionPluginHallucination,
	DecisionPluginHeaderMutation,
	DecisionPluginImageGen,
	DecisionPluginMemory,
	DecisionPluginRAG,
	DecisionPluginRequestParams,
	DecisionPluginProviderPromptCache,
	DecisionPluginContextCompression,
	DecisionPluginResponseJailbreak,
	DecisionPluginRouterReplay,
	DecisionPluginSemanticCache,
	DecisionPluginSystemPrompt,
	DecisionPluginToolSelection,
	DecisionPluginTools,
}

// AlgorithmCatalogEntry describes a model-selection algorithm and its tier
type AlgorithmCatalogEntry struct {
	Type string // algorithm type name (e.g., "elo")
	Tier string // "supported" or "experimental"
}

var decisionAlgorithmCatalog = []AlgorithmCatalogEntry{
	{Type: DecisionAlgorithmAutoMix, Tier: "experimental"},
	{Type: DecisionAlgorithmConfidence, Tier: "supported"},
	{Type: DecisionAlgorithmFusion, Tier: "experimental"},
	{Type: DecisionAlgorithmHybrid, Tier: "supported"},
	{Type: DecisionAlgorithmKMeans, Tier: "experimental"},
	{Type: DecisionAlgorithmKNN, Tier: "experimental"},
	{Type: DecisionAlgorithmLatencyAware, Tier: "supported"},
	{Type: DecisionAlgorithmMLP, Tier: "experimental"},
	{Type: DecisionAlgorithmMultiFactor, Tier: "supported"},
	{Type: DecisionAlgorithmRatings, Tier: "supported"},
	{Type: DecisionAlgorithmReMoM, Tier: "supported"},
	{Type: DecisionAlgorithmRouterDC, Tier: "supported"},
	{Type: DecisionAlgorithmStatic, Tier: "supported"},
	{Type: DecisionAlgorithmSVM, Tier: "experimental"},
	{Type: DecisionAlgorithmWorkflows, Tier: "experimental"},
	{Type: DecisionAlgorithmPrompt, Tier: "experimental"},
}

// supportedDecisionAlgorithmTypes is derived from the catalog for backwards compatibility
var supportedDecisionAlgorithmTypes = func() []string {
	types := make([]string, len(decisionAlgorithmCatalog))
	for i, entry := range decisionAlgorithmCatalog {
		types[i] = entry.Type
	}
	return types
}()

var pluginTypeAliases = map[string]string{
	"semantic_cache": DecisionPluginSemanticCache,
}

func SupportedSignalTypes() []string {
	return cloneSortedStrings(supportedSignalTypes)
}

func IsSupportedSignalType(signalType string) bool {
	for _, candidate := range supportedSignalTypes {
		if candidate == signalType {
			return true
		}
	}
	return false
}

func SupportedDecisionPluginTypes() []string {
	return cloneSortedStrings(supportedDecisionPluginTypes)
}

func NormalizeDecisionPluginType(pluginType string) string {
	if normalized, ok := pluginTypeAliases[pluginType]; ok {
		return normalized
	}
	return pluginType
}

func IsSupportedDecisionPluginType(pluginType string) bool {
	normalized := NormalizeDecisionPluginType(pluginType)
	for _, candidate := range supportedDecisionPluginTypes {
		if candidate == normalized {
			return true
		}
	}
	return false
}

func SupportedDecisionAlgorithmTypes() []string {
	return cloneSortedStrings(supportedDecisionAlgorithmTypes)
}

func IsSupportedDecisionAlgorithmType(algorithmType string) bool {
	for _, candidate := range supportedDecisionAlgorithmTypes {
		if candidate == algorithmType {
			return true
		}
	}
	return false
}

// DecisionAlgorithmCatalog returns the full structured catalog of algorithm types and tiers
func DecisionAlgorithmCatalog() []AlgorithmCatalogEntry {
	result := make([]AlgorithmCatalogEntry, len(decisionAlgorithmCatalog))
	copy(result, decisionAlgorithmCatalog)
	return result
}

// GetAlgorithmTier returns the tier for a given algorithm type, or empty string if unknown
func GetAlgorithmTier(algorithmType string) string {
	for _, entry := range decisionAlgorithmCatalog {
		if entry.Type == algorithmType {
			return entry.Tier
		}
	}
	return ""
}

func cloneSortedStrings(values []string) []string {
	cloned := append([]string(nil), values...)
	sort.Strings(cloned)
	return cloned
}
