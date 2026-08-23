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

	DecisionPluginResponseCache      = "response_cache"
	DecisionPluginSystemPrompt       = "system_prompt"
	DecisionPluginHeaderMutation     = "header_mutation"
	DecisionPluginHallucination      = "hallucination"
	DecisionPluginResponseJailbreak  = "response_jailbreak"
	DecisionPluginRouterReplay       = "router_replay"
	DecisionPluginMemory             = "memory"
	DecisionPluginRAG                = "rag"
	DecisionPluginImageGen           = "image_gen"
	DecisionPluginFastResponse       = "fast_response"
	DecisionPluginRequestParams      = "request_params"
	DecisionPluginToolSelection      = "tool_selection"
	DecisionPluginContextCompression = "context_compression"
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
	DecisionPluginContextCompression,
	DecisionPluginResponseJailbreak,
	DecisionPluginRouterReplay,
	DecisionPluginResponseCache,
	DecisionPluginSystemPrompt,
	DecisionPluginToolSelection,
	DecisionPluginTools,
}

// AlgorithmExecution identifies the runtime path for a decision algorithm.
type AlgorithmExecution string

const (
	AlgorithmExecutionLooper   AlgorithmExecution = "looper"
	AlgorithmExecutionSelector AlgorithmExecution = "selector"
)

// AlgorithmDispatchCardinality describes how many inference Models an
// algorithm requires for one matched decision. It is a closed capability of
// the canonical algorithm catalog, not a property inferred from an
// Entrypoint assignment.
type AlgorithmDispatchCardinality string

const (
	AlgorithmDispatchSingle AlgorithmDispatchCardinality = "single"
	AlgorithmDispatchMulti  AlgorithmDispatchCardinality = "multi"
)

// AlgorithmCatalogEntry describes a decision algorithm, its tier, and the
// runtime path that executes it.
type AlgorithmCatalogEntry struct {
	Type      string             // algorithm type name (e.g., "automix")
	Tier      string             // "supported" or "experimental"
	Execution AlgorithmExecution // "selector" or "looper"
}

var decisionAlgorithmCatalog = []AlgorithmCatalogEntry{
	{Type: DecisionAlgorithmAutoMix, Tier: "experimental", Execution: AlgorithmExecutionSelector},
	{Type: DecisionAlgorithmConfidence, Tier: "supported", Execution: AlgorithmExecutionLooper},
	{Type: DecisionAlgorithmFusion, Tier: "experimental", Execution: AlgorithmExecutionLooper},
	{Type: DecisionAlgorithmHybrid, Tier: "supported", Execution: AlgorithmExecutionSelector},
	{Type: DecisionAlgorithmKMeans, Tier: "experimental", Execution: AlgorithmExecutionSelector},
	{Type: DecisionAlgorithmKNN, Tier: "experimental", Execution: AlgorithmExecutionSelector},
	{Type: DecisionAlgorithmLatencyAware, Tier: "supported", Execution: AlgorithmExecutionSelector},
	{Type: DecisionAlgorithmMLP, Tier: "experimental", Execution: AlgorithmExecutionSelector},
	{Type: DecisionAlgorithmMultiFactor, Tier: "supported", Execution: AlgorithmExecutionSelector},
	{Type: DecisionAlgorithmRatings, Tier: "supported", Execution: AlgorithmExecutionLooper},
	{Type: DecisionAlgorithmReMoM, Tier: "supported", Execution: AlgorithmExecutionLooper},
	{Type: DecisionAlgorithmRouterDC, Tier: "supported", Execution: AlgorithmExecutionSelector},
	{Type: DecisionAlgorithmStatic, Tier: "supported", Execution: AlgorithmExecutionSelector},
	{Type: DecisionAlgorithmSVM, Tier: "experimental", Execution: AlgorithmExecutionSelector},
	{Type: DecisionAlgorithmWorkflows, Tier: "experimental", Execution: AlgorithmExecutionLooper},
	{Type: DecisionAlgorithmPrompt, Tier: "experimental", Execution: AlgorithmExecutionSelector},
}

// supportedDecisionAlgorithmTypes is derived from the catalog for backwards compatibility
var supportedDecisionAlgorithmTypes = func() []string {
	types := make([]string, len(decisionAlgorithmCatalog))
	for i, entry := range decisionAlgorithmCatalog {
		types[i] = entry.Type
	}
	return types
}()

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

// SupportedLooperAlgorithmTypes returns the decision algorithms executed by
// the multi-model Looper runtime.
func SupportedLooperAlgorithmTypes() []string {
	types := make([]string, 0)
	for _, entry := range decisionAlgorithmCatalog {
		if entry.Execution == AlgorithmExecutionLooper {
			types = append(types, entry.Type)
		}
	}
	return cloneSortedStrings(types)
}

// IsLooperAlgorithmType reports whether an algorithm is executed by Looper.
func IsLooperAlgorithmType(algorithmType string) bool {
	for _, entry := range decisionAlgorithmCatalog {
		if entry.Type == algorithmType {
			return entry.Execution == AlgorithmExecutionLooper
		}
	}
	return false
}

// DecisionAlgorithmDispatchCardinality returns the declared dispatch
// cardinality for a canonical decision algorithm. Omitting an algorithm uses
// the static selector and therefore dispatches one Model. Unknown algorithm
// identifiers fail closed.
func DecisionAlgorithmDispatchCardinality(algorithmType string) (AlgorithmDispatchCardinality, bool) {
	if algorithmType == "" {
		return AlgorithmDispatchSingle, true
	}
	for _, entry := range decisionAlgorithmCatalog {
		if entry.Type != algorithmType {
			continue
		}
		if entry.Execution == AlgorithmExecutionSelector {
			return AlgorithmDispatchSingle, true
		}
		if entry.Execution == AlgorithmExecutionLooper {
			return AlgorithmDispatchMulti, true
		}
		return "", false
	}
	return "", false
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
