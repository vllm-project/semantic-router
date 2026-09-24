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

	DecisionPluginResponseCache = "response_cache"
	// DecisionPluginSemanticCache is the deprecated public spelling retained
	// for source compatibility. Runtime config is normalized to response_cache.
	DecisionPluginSemanticCache      = "semantic-cache"
	DecisionPluginSystemPrompt       = "system_prompt"
	DecisionPluginHeaderMutation     = "header_mutation"
	DecisionPluginHallucination      = "hallucination"
	DecisionPluginResponseJailbreak  = "response_jailbreak"
	DecisionPluginRouterReplay       = "router_replay"
	DecisionPluginMemory             = "memory"
	DecisionPluginRAG                = "rag"
	DecisionPluginFastResponse       = "fast_response"
	DecisionPluginRequestParams      = "request_params"
	DecisionPluginToolSelection      = "tool_selection"
	DecisionPluginContextCompression = "context_compression"
	DecisionPluginShadowDispatch     = "shadow_dispatch"
)

// SignalReferenceQualifier describes how a signal can add a third component
// to its runtime reference identity.
type SignalReferenceQualifier string

const (
	SignalReferenceQualifierFixedSuffix SignalReferenceQualifier = "fixed_suffix"
	SignalReferenceQualifierLabel       SignalReferenceQualifier = "label"
)

// SignalCatalogEntry is the canonical public identity for one signal family.
// Collection is the YAML key under routing.signals. ObservationKey is the JSON
// field under decision_result matched/used/unmatched signal collections.
// ReferenceSuffixes names fixed derived decision references; label-qualified
// signals discover their suffixes from the configured signal payload.
type SignalCatalogEntry struct {
	Type                  string                   `json:"type"`
	DisplayName           string                   `json:"display_name"`
	Collection            string                   `json:"collection"`
	ObservationKey        string                   `json:"observation_key,omitempty"`
	DecisionReferenceable bool                     `json:"decision_referenceable"`
	ReferenceSuffixes     []string                 `json:"reference_suffixes,omitempty"`
	ReferenceQualifier    SignalReferenceQualifier `json:"reference_qualifier,omitempty"`
}

var signalCatalog = []SignalCatalogEntry{
	{Type: SignalTypeKeyword, DisplayName: "Keywords", Collection: "keywords", ObservationKey: "keywords", DecisionReferenceable: true},
	{Type: SignalTypeEmbedding, DisplayName: "Embeddings", Collection: "embeddings", ObservationKey: "embeddings", DecisionReferenceable: true},
	{Type: SignalTypeDomain, DisplayName: "Domain", Collection: "domains", ObservationKey: "domains", DecisionReferenceable: true},
	{Type: SignalTypeFactCheck, DisplayName: "Fact Check", Collection: "fact_check", ObservationKey: "fact_check", DecisionReferenceable: true},
	{Type: SignalTypeUserFeedback, DisplayName: "User Feedback", Collection: "user_feedbacks", ObservationKey: "user_feedback", DecisionReferenceable: true},
	{Type: SignalTypeReask, DisplayName: "Reask", Collection: "reasks", ObservationKey: "reask", DecisionReferenceable: true},
	{Type: SignalTypePreference, DisplayName: "Preference", Collection: "preferences", ObservationKey: "preferences", DecisionReferenceable: true},
	{Type: SignalTypeLanguage, DisplayName: "Language", Collection: "language", ObservationKey: "language", DecisionReferenceable: true},
	{Type: SignalTypeContext, DisplayName: "Context", Collection: "context", ObservationKey: "context", DecisionReferenceable: true},
	{Type: SignalTypeStructure, DisplayName: "Structure", Collection: "structure", ObservationKey: "structure", DecisionReferenceable: true},
	{Type: SignalTypeComplexity, DisplayName: "Complexity", Collection: "complexity", ObservationKey: "complexity", DecisionReferenceable: true, ReferenceSuffixes: []string{"easy", "medium", "hard"}, ReferenceQualifier: SignalReferenceQualifierFixedSuffix},
	{Type: SignalTypeModality, DisplayName: "Modality", Collection: "modality", ObservationKey: "modality", DecisionReferenceable: true},
	{Type: SignalTypeAuthz, DisplayName: "Authz", Collection: "role_bindings", ObservationKey: "authz", DecisionReferenceable: true},
	{Type: SignalTypeJailbreak, DisplayName: "Jailbreak", Collection: "jailbreak", ObservationKey: "jailbreak", DecisionReferenceable: true},
	{Type: SignalTypeSafety, DisplayName: "Safety", Collection: "safety", ObservationKey: "safety", DecisionReferenceable: true},
	{Type: SignalTypeHallucination, DisplayName: "Hallucination", Collection: "hallucination", DecisionReferenceable: false},
	{Type: SignalTypePII, DisplayName: "PII", Collection: "pii", ObservationKey: "pii", DecisionReferenceable: true},
	{Type: SignalTypeKB, DisplayName: "KB", Collection: "kb", ObservationKey: "kb", DecisionReferenceable: true},
	{Type: SignalTypeConversation, DisplayName: "Conversation", Collection: "conversation", ObservationKey: "conversation", DecisionReferenceable: true},
	{Type: SignalTypeEvent, DisplayName: "Event", Collection: "events", ObservationKey: "event", DecisionReferenceable: true},
	{Type: SignalTypeMetadata, DisplayName: "Metadata", Collection: "metadata", ObservationKey: "metadata", DecisionReferenceable: true},
	{Type: SignalTypeClassifier, DisplayName: "Classifier", Collection: "classifiers", ObservationKey: "classifier", DecisionReferenceable: true, ReferenceQualifier: SignalReferenceQualifierLabel},
	{Type: SignalTypeInputModality, DisplayName: "Input Modality", Collection: "input_modality", ObservationKey: "input_modality", DecisionReferenceable: true},
}

// DecisionPluginCatalogEntry describes one route-local plugin family. Its
// payload schema is resolved from the same factory used for Router validation.
type DecisionPluginCatalogEntry struct {
	Type        string `json:"type"`
	DisplayName string `json:"display_name"`
	Description string `json:"description"`
}

type decisionPluginRegistryEntry struct {
	Catalog    DecisionPluginCatalogEntry
	NewPayload func() interface{}
}

var decisionPluginRegistry = []decisionPluginRegistryEntry{
	{Catalog: DecisionPluginCatalogEntry{Type: DecisionPluginResponseCache, DisplayName: "Response Cache", Description: "Reuse exact or semantically compatible responses."}, NewPayload: func() interface{} { return &ResponseCachePluginConfig{} }},
	{Catalog: DecisionPluginCatalogEntry{Type: DecisionPluginMemory, DisplayName: "Memory", Description: "Retrieve and store persistent conversation memory."}, NewPayload: func() interface{} { return &MemoryPluginConfig{} }},
	{Catalog: DecisionPluginCatalogEntry{Type: DecisionPluginSystemPrompt, DisplayName: "System Prompt", Description: "Insert or replace the system prompt."}, NewPayload: func() interface{} { return &SystemPromptPluginConfig{} }},
	{Catalog: DecisionPluginCatalogEntry{Type: DecisionPluginHeaderMutation, DisplayName: "Header Mutation", Description: "Add, update, or remove provider-bound headers."}, NewPayload: func() interface{} { return &HeaderMutationPluginConfig{} }},
	{Catalog: DecisionPluginCatalogEntry{Type: DecisionPluginHallucination, DisplayName: "Hallucination", Description: "Apply response hallucination handling."}, NewPayload: func() interface{} { return &HallucinationPluginConfig{} }},
	{Catalog: DecisionPluginCatalogEntry{Type: DecisionPluginRouterReplay, DisplayName: "Router Replay", Description: "Capture bounded request and response replay evidence."}, NewPayload: func() interface{} { return &RouterReplayPluginConfig{} }},
	{Catalog: DecisionPluginCatalogEntry{Type: DecisionPluginRAG, DisplayName: "RAG", Description: "Retrieve external context and inject it into the request."}, NewPayload: func() interface{} { return &RAGPluginConfig{} }},
	{Catalog: DecisionPluginCatalogEntry{Type: DecisionPluginFastResponse, DisplayName: "Fast Response", Description: "Return a fixed response without calling an upstream model."}, NewPayload: func() interface{} { return &FastResponsePluginConfig{} }},
	{Catalog: DecisionPluginCatalogEntry{Type: DecisionPluginTools, DisplayName: "Tools", Description: "Apply route-local tool filtering and selection."}, NewPayload: func() interface{} { return &ToolsPluginConfig{} }},
	{Catalog: DecisionPluginCatalogEntry{Type: DecisionPluginToolSelection, DisplayName: "Tool Selection", Description: "Add or filter tools using semantic retrieval."}, NewPayload: func() interface{} { return &ToolSelectionPluginConfig{} }},
	{Catalog: DecisionPluginCatalogEntry{Type: DecisionPluginRequestParams, DisplayName: "Request Parameters", Description: "Constrain or remove provider request parameters."}, NewPayload: func() interface{} { return &RequestParamsPluginConfig{} }},
	{Catalog: DecisionPluginCatalogEntry{Type: DecisionPluginResponseJailbreak, DisplayName: "Response Jailbreak", Description: "Screen generated responses for jailbreak-like output."}, NewPayload: func() interface{} { return &ResponseJailbreakPluginConfig{} }},
	{Catalog: DecisionPluginCatalogEntry{Type: DecisionPluginContextCompression, DisplayName: "Context Compression", Description: "Compress selected context before provider dispatch."}, NewPayload: func() interface{} { return &ContextCompressionPluginConfig{} }},
	{Catalog: DecisionPluginCatalogEntry{Type: DecisionPluginShadowDispatch, DisplayName: "Shadow Dispatch", Description: "Send a bounded asynchronous copy to a secondary model."}, NewPayload: func() interface{} { return &ShadowDispatchPluginConfig{} }},
}

// AlgorithmExecution identifies the runtime path for a decision algorithm.
type AlgorithmExecution string

const (
	AlgorithmExecutionLooper   AlgorithmExecution = "looper"
	AlgorithmExecutionSelector AlgorithmExecution = "selector"
)

// AlgorithmPayloadShape describes whether an algorithm's public DSL fields
// are written directly in the ALGORITHM block or under its config field.
type AlgorithmPayloadShape string

const AlgorithmPayloadNested AlgorithmPayloadShape = "nested"

// AlgorithmCatalogEntry describes a decision algorithm, its tier, and the
// runtime path that executes it.
type AlgorithmCatalogEntry struct {
	Type         string                `json:"type"`         // algorithm type name (e.g., "automix")
	DisplayName  string                `json:"display_name"` // concise user-facing name
	Description  string                `json:"description"`  // request-time behavior
	Tier         string                `json:"tier"`         // "supported" or "experimental"
	Execution    AlgorithmExecution    `json:"execution"`    // "selector" or "looper"
	ConfigField  string                `json:"config_field,omitempty"`
	PayloadShape AlgorithmPayloadShape `json:"payload_shape,omitempty"` // empty/flat or nested in public DSL editors
}

type decisionAlgorithmRegistryEntry struct {
	Catalog      AlgorithmCatalogEntry
	IsConfigured func(*AlgorithmConfig) bool
}

var decisionAlgorithmRegistry = []decisionAlgorithmRegistryEntry{
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmAutoMix, DisplayName: "AutoMix", Description: "Optimize a cost-quality escalation policy.", Tier: "experimental", Execution: AlgorithmExecutionSelector, ConfigField: "automix"}, IsConfigured: func(config *AlgorithmConfig) bool { return config.AutoMix != nil }},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmConfidence, DisplayName: "Confidence", Description: "Escalate across candidate models until confidence is sufficient.", Tier: "supported", Execution: AlgorithmExecutionLooper, ConfigField: "confidence"}, IsConfigured: func(config *AlgorithmConfig) bool { return config.Confidence != nil }},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmFusion, DisplayName: "Fusion", Description: "Run a parallel panel and synthesize a judged final response.", Tier: "experimental", Execution: AlgorithmExecutionLooper, ConfigField: "fusion"}, IsConfigured: func(config *AlgorithmConfig) bool { return config.Fusion != nil }},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmHybrid, DisplayName: "Hybrid", Description: "Combine experience, similarity, AutoMix, and cost signals.", Tier: "supported", Execution: AlgorithmExecutionSelector, ConfigField: "hybrid"}, IsConfigured: func(config *AlgorithmConfig) bool { return config.Hybrid != nil }},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmKMeans, DisplayName: "K-Means", Description: "Select a model with the shared K-Means classifier.", Tier: "experimental", Execution: AlgorithmExecutionSelector}},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmKNN, DisplayName: "KNN", Description: "Select a model with the shared nearest-neighbor classifier.", Tier: "experimental", Execution: AlgorithmExecutionSelector}},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmLatencyAware, DisplayName: "Latency Aware", Description: "Select against configured latency percentiles.", Tier: "supported", Execution: AlgorithmExecutionSelector, ConfigField: "latency_aware"}, IsConfigured: func(config *AlgorithmConfig) bool { return config.LatencyAware != nil }},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmMLP, DisplayName: "MLP", Description: "Select a model with the shared neural classifier.", Tier: "experimental", Execution: AlgorithmExecutionSelector}},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmMultiFactor, DisplayName: "Multi Factor", Description: "Score quality, latency, cost, and load under optional SLOs.", Tier: "supported", Execution: AlgorithmExecutionSelector, ConfigField: "multi_factor"}, IsConfigured: func(config *AlgorithmConfig) bool { return config.MultiFactor != nil }},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmRatings, DisplayName: "Ratings", Description: "Execute a bounded candidate set and return comparable choices.", Tier: "supported", Execution: AlgorithmExecutionLooper, ConfigField: "ratings"}, IsConfigured: func(config *AlgorithmConfig) bool { return config.Ratings != nil }},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmReMoM, DisplayName: "ReMoM", Description: "Run multi-round parallel reasoning and synthesis.", Tier: "supported", Execution: AlgorithmExecutionLooper, ConfigField: "remom"}, IsConfigured: func(config *AlgorithmConfig) bool { return config.ReMoM != nil }},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmRouterDC, DisplayName: "RouterDC", Description: "Match the request to candidate descriptions with dual contrastive embeddings.", Tier: "supported", Execution: AlgorithmExecutionSelector, ConfigField: "router_dc"}, IsConfigured: func(config *AlgorithmConfig) bool { return config.RouterDC != nil }},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmStatic, DisplayName: "Static", Description: "Select from configured candidate order and weights.", Tier: "supported", Execution: AlgorithmExecutionSelector}},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmSVM, DisplayName: "SVM", Description: "Select a model with the shared support-vector classifier.", Tier: "experimental", Execution: AlgorithmExecutionSelector}},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmWorkflows, DisplayName: "Workflows", Description: "Execute a static or dynamically planned Router Flow.", Tier: "experimental", Execution: AlgorithmExecutionLooper, ConfigField: "workflows"}, IsConfigured: func(config *AlgorithmConfig) bool { return config.Workflows != nil }},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmPrompt, DisplayName: "Prompt", Description: "Use a helper model to select one declared candidate.", Tier: "experimental", Execution: AlgorithmExecutionSelector, ConfigField: "prompt", PayloadShape: AlgorithmPayloadNested}, IsConfigured: func(config *AlgorithmConfig) bool { return config.Prompt != nil }},
}

var pluginTypeAliases = map[string]string{
	"semantic-cache": DecisionPluginResponseCache,
	"semantic_cache": DecisionPluginResponseCache,
	"response-cache": DecisionPluginResponseCache,
}

func SupportedSignalTypes() []string {
	types := make([]string, 0, len(signalCatalog))
	for _, entry := range signalCatalog {
		types = append(types, entry.Type)
	}
	return cloneSortedStrings(types)
}

func IsSupportedSignalType(signalType string) bool {
	_, ok := LookupSignalCatalog(signalType)
	return ok
}

// LookupSignalCatalog returns the canonical metadata for one signal type.
func LookupSignalCatalog(signalType string) (SignalCatalogEntry, bool) {
	for _, entry := range signalCatalog {
		if entry.Type == signalType {
			entry.ReferenceSuffixes = append([]string(nil), entry.ReferenceSuffixes...)
			return entry, true
		}
	}
	return SignalCatalogEntry{}, false
}

// SupportedDecisionSignalTypes returns signals that can participate in the
// request-time decision rule tree. Response-only observations remain
// configurable but are consumed by response plugins instead.
func SupportedDecisionSignalTypes() []string {
	types := make([]string, 0, len(signalCatalog))
	for _, entry := range signalCatalog {
		if entry.DecisionReferenceable {
			types = append(types, entry.Type)
		}
	}
	return cloneSortedStrings(types)
}

// SignalCatalog returns the configuration and decision-reference identity for
// every supported signal family.
func SignalCatalog() []SignalCatalogEntry {
	result := make([]SignalCatalogEntry, len(signalCatalog))
	for index, entry := range signalCatalog {
		result[index] = entry
		result[index].ReferenceSuffixes = append([]string(nil), entry.ReferenceSuffixes...)
	}
	return result
}

func SupportedDecisionPluginTypes() []string {
	types := make([]string, 0, len(decisionPluginRegistry))
	for _, entry := range decisionPluginRegistry {
		types = append(types, entry.Catalog.Type)
	}
	return cloneSortedStrings(types)
}

func NormalizeDecisionPluginType(pluginType string) string {
	if normalized, ok := pluginTypeAliases[pluginType]; ok {
		return normalized
	}
	return pluginType
}

func IsSupportedDecisionPluginType(pluginType string) bool {
	normalized := NormalizeDecisionPluginType(pluginType)
	for _, entry := range decisionPluginRegistry {
		if entry.Catalog.Type == normalized {
			return true
		}
	}
	return false
}

// DecisionPluginCatalog returns the public route-local plugin inventory.
func DecisionPluginCatalog() []DecisionPluginCatalogEntry {
	result := make([]DecisionPluginCatalogEntry, len(decisionPluginRegistry))
	for index, entry := range decisionPluginRegistry {
		result[index] = entry.Catalog
	}
	return result
}

func newDecisionPluginPayload(pluginType string) interface{} {
	normalized := NormalizeDecisionPluginType(pluginType)
	for _, entry := range decisionPluginRegistry {
		if entry.Catalog.Type == normalized {
			return entry.NewPayload()
		}
	}
	return nil
}

func SupportedDecisionAlgorithmTypes() []string {
	types := make([]string, 0, len(decisionAlgorithmRegistry))
	for _, entry := range decisionAlgorithmRegistry {
		types = append(types, entry.Catalog.Type)
	}
	return cloneSortedStrings(types)
}

func IsSupportedDecisionAlgorithmType(algorithmType string) bool {
	_, ok := decisionAlgorithmCatalogEntry(algorithmType)
	return ok
}

func decisionAlgorithmCatalogEntry(algorithmType string) (AlgorithmCatalogEntry, bool) {
	for _, entry := range decisionAlgorithmRegistry {
		if entry.Catalog.Type == algorithmType {
			return entry.Catalog, true
		}
	}
	return AlgorithmCatalogEntry{}, false
}

// DecisionAlgorithmConfigField reports the algorithm-specific YAML block for
// a supported type. A supported blockless algorithm returns an empty field and
// true; an unknown type returns false.
func DecisionAlgorithmConfigField(algorithmType string) (string, bool) {
	entry, ok := decisionAlgorithmCatalogEntry(algorithmType)
	if !ok {
		return "", false
	}
	return entry.ConfigField, true
}

func configuredDecisionAlgorithmBlocks(config *AlgorithmConfig) []string {
	if config == nil {
		return nil
	}
	blocks := make([]string, 0)
	for _, entry := range decisionAlgorithmRegistry {
		if entry.Catalog.ConfigField != "" && entry.IsConfigured != nil && entry.IsConfigured(config) {
			blocks = append(blocks, entry.Catalog.ConfigField)
		}
	}
	return blocks
}

// SupportedLooperAlgorithmTypes returns the decision algorithms executed by
// the multi-model Looper runtime.
func SupportedLooperAlgorithmTypes() []string {
	types := make([]string, 0)
	for _, entry := range decisionAlgorithmRegistry {
		if entry.Catalog.Execution == AlgorithmExecutionLooper {
			types = append(types, entry.Catalog.Type)
		}
	}
	return cloneSortedStrings(types)
}

// IsLooperAlgorithmType reports whether an algorithm is executed by Looper.
func IsLooperAlgorithmType(algorithmType string) bool {
	entry, ok := decisionAlgorithmCatalogEntry(algorithmType)
	if ok {
		return entry.Execution == AlgorithmExecutionLooper
	}
	return false
}

// DecisionAlgorithmCatalog returns the full structured catalog of algorithm types and tiers
func DecisionAlgorithmCatalog() []AlgorithmCatalogEntry {
	result := make([]AlgorithmCatalogEntry, len(decisionAlgorithmRegistry))
	for index, entry := range decisionAlgorithmRegistry {
		result[index] = entry.Catalog
	}
	return result
}

// GetAlgorithmTier returns the tier for a given algorithm type, or empty string if unknown
func GetAlgorithmTier(algorithmType string) string {
	entry, ok := decisionAlgorithmCatalogEntry(algorithmType)
	if ok {
		return entry.Tier
	}
	return ""
}

func cloneSortedStrings(values []string) []string {
	cloned := append([]string(nil), values...)
	sort.Strings(cloned)
	return cloned
}
