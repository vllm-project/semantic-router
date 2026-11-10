package config

import (
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extension"
)

const (
	DecisionAlgorithmStatic      = "static"
	DecisionAlgorithmRandom      = "random"
	DecisionAlgorithmMultiFactor = "multi_factor"
	DecisionAlgorithmSystemOne   = "systemone"
	DecisionAlgorithmEscalate    = "escalate"
	DecisionAlgorithmFusion      = "fusion"
	DecisionAlgorithmReMoM       = "remom"
	DecisionAlgorithmRatings     = "ratings"
	DecisionAlgorithmWorkflows   = "workflows"

	DecisionPluginResponseCache  = "response_cache"
	DecisionPluginSystemPrompt   = "system_prompt"
	DecisionPluginHeaderMutation = "header_mutation"
	DecisionPluginRouterReplay   = "router_replay"
	DecisionPluginFastResponse   = "fast_response"
	DecisionPluginRequestParams  = "request_params"
	DecisionPluginPromptCache    = "prompt_cache"
)

// SignalReferenceQualifier describes how a signal can add a third component
// to its runtime reference identity.
type SignalReferenceQualifier string

// SignalReferenceQualifierLabel adds one of the rule's options:
// <type>:<rule>:<label>; the options come from the configured rule.
const SignalReferenceQualifierLabel SignalReferenceQualifier = "label"

// SignalCatalogEntry is the canonical public identity for one signal family.
// Collection is the YAML key under recipes[].signals. ObservationKey is the JSON
// field under decision_result matched/used/unmatched signal collections.
type SignalCatalogEntry struct {
	Type                  string                   `json:"type"`
	DisplayName           string                   `json:"display_name"`
	Collection            string                   `json:"collection"`
	ObservationKey        string                   `json:"observation_key,omitempty"`
	DecisionReferenceable bool                     `json:"decision_referenceable"`
	ReferenceQualifier    SignalReferenceQualifier `json:"reference_qualifier,omitempty"`
}

var builtinSignalCatalog = []SignalCatalogEntry{
	{Type: SignalTypeKeyword, DisplayName: "Keywords", Collection: "keywords", ObservationKey: "keywords", DecisionReferenceable: true},
	{Type: SignalTypeDomain, DisplayName: "Domain", Collection: "domains", ObservationKey: "domains", DecisionReferenceable: true, ReferenceQualifier: SignalReferenceQualifierLabel},
	{Type: SignalTypeFactCheck, DisplayName: "Fact Check", Collection: "fact_check", ObservationKey: "fact_check", DecisionReferenceable: true, ReferenceQualifier: SignalReferenceQualifierLabel},
	{Type: SignalTypeUserFeedback, DisplayName: "User Feedback", Collection: "user_feedbacks", ObservationKey: "user_feedback", DecisionReferenceable: true, ReferenceQualifier: SignalReferenceQualifierLabel},
	{Type: SignalTypeLanguage, DisplayName: "Language", Collection: "language", ObservationKey: "language", DecisionReferenceable: true},
	{Type: SignalTypeContext, DisplayName: "Context", Collection: "context", ObservationKey: "context", DecisionReferenceable: true},
	{Type: SignalTypeComplexity, DisplayName: "Complexity", Collection: "complexity", ObservationKey: "complexity", DecisionReferenceable: true, ReferenceQualifier: SignalReferenceQualifierLabel},
	{Type: SignalTypeModality, DisplayName: "Modality", Collection: "modality", ObservationKey: "modality", DecisionReferenceable: true, ReferenceQualifier: SignalReferenceQualifierLabel},
	{Type: SignalTypeAuthz, DisplayName: "Authz", Collection: "role_bindings", ObservationKey: "authz", DecisionReferenceable: true},
	{Type: SignalTypeJailbreak, DisplayName: "Jailbreak", Collection: "jailbreak", ObservationKey: "jailbreak", DecisionReferenceable: true, ReferenceQualifier: SignalReferenceQualifierLabel},
	{Type: SignalTypeSafety, DisplayName: "Safety", Collection: "safety", ObservationKey: "safety", DecisionReferenceable: true, ReferenceQualifier: SignalReferenceQualifierLabel},
	{Type: SignalTypeHallucination, DisplayName: "Hallucination", Collection: "hallucination", DecisionReferenceable: false},
	{Type: SignalTypeRefusal, DisplayName: "Refusal", Collection: "refusal", DecisionReferenceable: false},
	{Type: SignalTypeTurnRelation, DisplayName: "Turn Relation", Collection: "turn_relation", ObservationKey: "turn_relation", DecisionReferenceable: true, ReferenceQualifier: SignalReferenceQualifierLabel},
	{Type: SignalTypePII, DisplayName: "PII", Collection: "pii", ObservationKey: "pii", DecisionReferenceable: true, ReferenceQualifier: SignalReferenceQualifierLabel},
	{Type: SignalTypeConversation, DisplayName: "Conversation", Collection: "conversation", ObservationKey: "conversation", DecisionReferenceable: true},
	{Type: SignalTypeMetadata, DisplayName: "Metadata", Collection: "metadata", ObservationKey: "metadata", DecisionReferenceable: true},
	{Type: SignalTypeInputModality, DisplayName: "Input Modality", Collection: "input_modality", ObservationKey: "input_modality", DecisionReferenceable: true},
	{Type: SignalTypeSystemOne, DisplayName: "System One", Collection: "systemone", ObservationKey: "systemone", DecisionReferenceable: true, ReferenceQualifier: SignalReferenceQualifierLabel},
}

// DecisionPluginCatalogEntry describes one route-local plugin family. Its
// payload schema is resolved from the same factory used for Router validation.
type DecisionPluginCatalogEntry struct {
	Type        string `json:"type"`
	DisplayName string `json:"display_name"`
	Description string `json:"description"`
}

// AlgorithmKind is what a decision algorithm does with its candidates: a
// selector chooses one before calling it; an executor calls several and
// returns one response.
type AlgorithmKind string

const (
	AlgorithmKindSelector AlgorithmKind = "selector"
	AlgorithmKindExecutor AlgorithmKind = "executor"
)

// AlgorithmPayloadShape describes whether an algorithm's public DSL fields
// are written directly in the ALGORITHM block or under its config field.
type AlgorithmPayloadShape string

const AlgorithmPayloadNested AlgorithmPayloadShape = "nested"

// AlgorithmCatalogEntry describes a decision algorithm, its tier, its kind
// and the APIs (entrypoint protocols) whose recipes can use it.
type AlgorithmCatalogEntry struct {
	Type         string                `json:"type"`         // algorithm type name (e.g., "escalate")
	DisplayName  string                `json:"display_name"` // concise user-facing name
	Description  string                `json:"description"`  // request-time behavior
	Tier         string                `json:"tier"`         // "supported" or "experimental"
	Kind         AlgorithmKind         `json:"kind"`         // "selector" or "executor"
	APIs         []InferenceAPI        `json:"apis"`         // "chat" and, for escalate, "systemone"
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

var (
	chatAPIs     = []InferenceAPI{ChatAPI}
	escalateAPIs = []InferenceAPI{ChatAPI, SystemOneAPI}
)

func selectorEntry(entry AlgorithmCatalogEntry) AlgorithmCatalogEntry {
	entry.Kind, entry.APIs = AlgorithmKindSelector, chatAPIs
	return entry
}

// executorEntry is an executor's catalog entry; its settings are the block
// named after its type, beside the shared budget and on_error.
func executorEntry(entry AlgorithmCatalogEntry, apis []InferenceAPI) AlgorithmCatalogEntry {
	entry.Kind, entry.APIs, entry.ConfigField = AlgorithmKindExecutor, apis, entry.Type
	return entry
}

var builtinDecisionAlgorithms = []DecisionAlgorithmType{
	{Catalog: selectorEntry(AlgorithmCatalogEntry{Type: DecisionAlgorithmStatic, DisplayName: "Static", Description: "Select from configured candidate order and weights.", Tier: "supported"})},
	{Catalog: selectorEntry(AlgorithmCatalogEntry{Type: DecisionAlgorithmRandom, DisplayName: "Random", Description: "Select uniformly from the eligible candidate models.", Tier: "supported"})},
	{Catalog: selectorEntry(AlgorithmCatalogEntry{Type: DecisionAlgorithmMultiFactor, DisplayName: "Multi Factor", Description: "Score quality, latency, cost, and load under optional SLOs.", Tier: "supported", ConfigField: "multi_factor"}), IsConfigured: func(config *AlgorithmConfig) bool { return config.MultiFactor != nil }},
	{Catalog: selectorEntry(AlgorithmCatalogEntry{Type: DecisionAlgorithmSystemOne, DisplayName: "System One", Description: "Ask a System One model which declared candidate should answer.", Tier: "supported", ConfigField: "systemone", PayloadShape: AlgorithmPayloadNested}), IsConfigured: func(config *AlgorithmConfig) bool { return config.SystemOne != nil }},
	{Catalog: executorEntry(AlgorithmCatalogEntry{Type: DecisionAlgorithmEscalate, DisplayName: "Escalate", Description: "Call the candidates in order until one answer is accepted.", Tier: "supported"}, escalateAPIs), IsConfigured: func(config *AlgorithmConfig) bool { return config.Escalate != nil }},
	{Catalog: executorEntry(AlgorithmCatalogEntry{Type: DecisionAlgorithmFusion, DisplayName: "Fusion", Description: "Run a parallel panel and synthesize a judged final response.", Tier: "experimental"}, chatAPIs), IsConfigured: func(config *AlgorithmConfig) bool { return config.Fusion != nil }},
	{Catalog: executorEntry(AlgorithmCatalogEntry{Type: DecisionAlgorithmReMoM, DisplayName: "ReMoM", Description: "Run multi-round parallel reasoning and synthesis.", Tier: "supported"}, chatAPIs), IsConfigured: func(config *AlgorithmConfig) bool { return config.ReMoM != nil }},
	{Catalog: executorEntry(AlgorithmCatalogEntry{Type: DecisionAlgorithmRatings, DisplayName: "Ratings", Description: "Execute a bounded candidate set and return comparable choices.", Tier: "supported"}, chatAPIs), IsConfigured: func(config *AlgorithmConfig) bool { return config.Ratings != nil }},
	{Catalog: executorEntry(AlgorithmCatalogEntry{Type: DecisionAlgorithmWorkflows, DisplayName: "Workflows", Description: "Execute a static or dynamically planned Router Flow.", Tier: "experimental"}, chatAPIs), IsConfigured: func(config *AlgorithmConfig) bool { return config.Workflows != nil }},
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
	return signalTypes.Lookup(signalType)
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
	return registeredSpecs(signalTypes)
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

// SupportedExecutorAlgorithmTypes returns the decision algorithms that call
// several candidates and return one response.
func SupportedExecutorAlgorithmTypes() []string {
	types := make([]string, 0)
	for _, entry := range registeredSpecs(decisionAlgorithms) {
		if entry.Catalog.Kind == AlgorithmKindExecutor {
			types = append(types, entry.Catalog.Type)
		}
	}
	return cloneSortedStrings(types)
}

// IsExecutorAlgorithmType reports whether an algorithm is an executor. A
// Chat decision's executor runs on the Looper; escalate also runs a System
// One recipe's stages.
func IsExecutorAlgorithmType(algorithmType string) bool {
	entry, ok := decisionAlgorithmCatalogEntry(algorithmType)
	return ok && entry.Kind == AlgorithmKindExecutor
}

// DecisionAlgorithmCatalog returns the full structured catalog of algorithm types and tiers
func DecisionAlgorithmCatalog() []AlgorithmCatalogEntry {
	result := make([]AlgorithmCatalogEntry, len(decisionAlgorithms.Entries()))
	for index, entry := range registeredSpecs(decisionAlgorithms) {
		result[index] = entry.Catalog
		result[index].APIs = append([]InferenceAPI(nil), entry.Catalog.APIs...)
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
