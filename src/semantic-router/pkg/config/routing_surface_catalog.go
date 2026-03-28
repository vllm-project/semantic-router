package config

import "sort"

const (
	DecisionPluginSemanticCache     = "semantic-cache"
	DecisionPluginJailbreak         = "jailbreak"
	DecisionPluginPII               = "pii"
	DecisionPluginSystemPrompt      = "system_prompt"
	DecisionPluginHeaderMutation    = "header_mutation"
	DecisionPluginHallucination     = "hallucination"
	DecisionPluginResponseJailbreak = "response_jailbreak"
	DecisionPluginRouterReplay      = "router_replay"
	DecisionPluginMemory            = "memory"
	DecisionPluginRAG               = "rag"
	DecisionPluginImageGen          = "image_gen"
	DecisionPluginFastResponse      = "fast_response"
	DecisionPluginRequestParams     = "request_params"
)

var supportedSignalTypes = []string{
	SignalTypeAuthz,
	SignalTypeComplexity,
	SignalTypeContext,
	SignalTypeDomain,
	SignalTypeEmbedding,
	SignalTypeFactCheck,
	SignalTypeJailbreak,
	SignalTypeKeyword,
	SignalTypeLanguage,
	SignalTypeModality,
	SignalTypePII,
	SignalTypePreference,
	SignalTypeStructure,
	SignalTypeKB,
	SignalTypeUserFeedback,
}

var supportedDecisionPluginTypes = []string{
	DecisionPluginFastResponse,
	DecisionPluginHallucination,
	DecisionPluginHeaderMutation,
	DecisionPluginImageGen,
	DecisionPluginJailbreak,
	DecisionPluginMemory,
	DecisionPluginPII,
	DecisionPluginRAG,
	DecisionPluginRequestParams,
	DecisionPluginResponseJailbreak,
	DecisionPluginRouterReplay,
	DecisionPluginSemanticCache,
	DecisionPluginSystemPrompt,
	DecisionPluginTools,
}

var supportedDecisionAlgorithmTypes = []string{
	"automix",
	"confidence",
	"elo",
	"gmtrouter",
	"hybrid",
	"kmeans",
	"knn",
	"latency_aware",
	"ratings",
	"remom",
	"rl_driven",
	"router_dc",
	"static",
	"svm",
}

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

func cloneSortedStrings(values []string) []string {
	cloned := append([]string(nil), values...)
	sort.Strings(cloned)
	return cloned
}
