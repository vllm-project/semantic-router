package config

import (
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extension"
)

const (
	DecisionAlgorithmCascade      = "cascade"
	DecisionAlgorithmPolicy       = "policy"
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
	DecisionAlgorithmDecision     = "decision"

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
	DecisionPluginPromptCache        = "prompt_cache"
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

var builtinSignalCatalog = []SignalCatalogEntry{
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
	{Type: SignalTypeDecision, DisplayName: "Decision Model", Collection: "decision", ObservationKey: "decision", DecisionReferenceable: true, ReferenceQualifier: SignalReferenceQualifierLabel},
}

// DecisionPluginCatalogEntry describes one route-local plugin family. Its
// payload schema is resolved from the same factory used for Router validation.
type DecisionPluginCatalogEntry struct {
	Type        string `json:"type"`
	DisplayName string `json:"display_name"`
	Description string `json:"description"`
}

// AlgorithmExecution identifies the runtime path for a decision algorithm.
type AlgorithmExecution string

const (
	AlgorithmExecutionNative   AlgorithmExecution = "native"
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
	Execution    AlgorithmExecution    `json:"execution"`    // "selector", "looper", or "native"
	ConfigField  string                `json:"config_field,omitempty"`
	PayloadShape AlgorithmPayloadShape `json:"payload_shape,omitempty"` // empty/flat or nested in public DSL editors
}

// DecisionAlgorithmType is a decision algorithm type. A built-in type keeps
// its block in a field of AlgorithmConfig; a type registered outside the
// Router keeps it in AlgorithmConfig.Extensions under its name, and its
// payload's Go type is that block's schema.
type DecisionAlgorithmType struct {
	Catalog      AlgorithmCatalogEntry
	IsConfigured func(*AlgorithmConfig) bool
	// NewPayload returns the empty payload a registered type's block decodes
	// into; nil for a built-in type.
	NewPayload func() interface{}
	// Strict rejects block fields the payload does not declare.
	Strict bool
	// Defaults, when set, fills a decoded payload's unset fields.
	Defaults func(payload interface{})
	// Validate, when set, checks a decoded payload after its defaults.
	Validate func(decision string, payload interface{}) error
}

var builtinDecisionAlgorithms = []DecisionAlgorithmType{
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmCascade, DisplayName: "Native Cascade", Description: "Escalate complete typed responses through declared stages.", Tier: "experimental", Execution: AlgorithmExecutionNative}},
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmPolicy, DisplayName: "Native Policy", Description: "Select declared native stages using immutable learned parameters.", Tier: "experimental", Execution: AlgorithmExecutionNative, ConfigField: "policy", PayloadShape: AlgorithmPayloadNested}, IsConfigured: func(config *AlgorithmConfig) bool { return config.Policy != nil }},
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
	{Catalog: AlgorithmCatalogEntry{Type: DecisionAlgorithmDecision, DisplayName: "Decision Model", Description: "Ask a decision model which declared candidate should answer.", Tier: "supported", Execution: AlgorithmExecutionSelector, ConfigField: "decision", PayloadShape: AlgorithmPayloadNested}, IsConfigured: func(config *AlgorithmConfig) bool { return config.Decision != nil }},
}

// signalTypes and decisionAlgorithms hold the signal families and decision
// algorithms by type, in catalog order.
var (
	signalTypes = newCatalogRegistry("signal", builtinSignalCatalog,
		func(entry SignalCatalogEntry) string { return entry.Type })
	decisionAlgorithms = newCatalogRegistry("decision algorithm", builtinDecisionAlgorithms,
		func(entry DecisionAlgorithmType) string { return entry.Catalog.Type })
)

func newCatalogRegistry[S any](kind string, builtins []S, typeOf func(S) string) *extension.Registry[S] {
	registry := extension.NewRegistry[S](kind)
	for _, spec := range builtins {
		registry.MustRegister(typeOf(spec), spec)
	}
	return registry
}

func registeredSpecs[S any](registry *extension.Registry[S]) []S {
	entries := registry.Entries()
	specs := make([]S, len(entries))
	for i, entry := range entries {
		specs[i] = entry.Spec
	}
	return specs
}

func SupportedSignalTypes() []string {
	types := make([]string, 0, len(signalTypes.Entries()))
	for _, entry := range registeredSpecs(signalTypes) {
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
	entry, ok := signalTypes.Lookup(signalType)
	if !ok {
		return SignalCatalogEntry{}, false
	}
	entry.ReferenceSuffixes = append([]string(nil), entry.ReferenceSuffixes...)
	return entry, true
}

// SupportedDecisionSignalTypes returns signals that can participate in the
// request-time decision rule tree. Response-only observations remain
// configurable but are consumed by response plugins instead.
func SupportedDecisionSignalTypes() []string {
	types := make([]string, 0, len(signalTypes.Entries()))
	for _, entry := range registeredSpecs(signalTypes) {
		if entry.DecisionReferenceable {
			types = append(types, entry.Type)
		}
	}
	return cloneSortedStrings(types)
}

// SignalCatalog returns the configuration and decision-reference identity for
// every supported signal family.
func SignalCatalog() []SignalCatalogEntry {
	result := make([]SignalCatalogEntry, len(signalTypes.Entries()))
	for index, entry := range registeredSpecs(signalTypes) {
		result[index] = entry
		result[index].ReferenceSuffixes = append([]string(nil), entry.ReferenceSuffixes...)
	}
	return result
}

func SupportedDecisionPluginTypes() []string {
	return DecisionPlugins.Types()
}

func NormalizeDecisionPluginType(pluginType string) string {
	return DecisionPlugins.Normalize(pluginType)
}

func IsSupportedDecisionPluginType(pluginType string) bool {
	_, ok := DecisionPlugins.Lookup(pluginType)
	return ok
}

// DecisionPluginCatalog returns the public route-local plugin inventory.
func DecisionPluginCatalog() []DecisionPluginCatalogEntry {
	entries := DecisionPlugins.Entries()
	result := make([]DecisionPluginCatalogEntry, len(entries))
	for index, entry := range entries {
		result[index] = entry.Spec.Catalog
	}
	return result
}

func SupportedDecisionAlgorithmTypes() []string {
	types := make([]string, 0, len(decisionAlgorithms.Entries()))
	for _, entry := range registeredSpecs(decisionAlgorithms) {
		types = append(types, entry.Catalog.Type)
	}
	return cloneSortedStrings(types)
}

func IsSupportedDecisionAlgorithmType(algorithmType string) bool {
	_, ok := decisionAlgorithmCatalogEntry(algorithmType)
	return ok
}

func decisionAlgorithmCatalogEntry(algorithmType string) (AlgorithmCatalogEntry, bool) {
	entry, ok := decisionAlgorithms.Lookup(algorithmType)
	return entry.Catalog, ok
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
	for _, entry := range registeredSpecs(decisionAlgorithms) {
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
	for _, entry := range registeredSpecs(decisionAlgorithms) {
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
	result := make([]AlgorithmCatalogEntry, len(decisionAlgorithms.Entries()))
	for index, entry := range registeredSpecs(decisionAlgorithms) {
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
