package config

import (
	"fmt"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelauthoring"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// CanonicalConfig is the additive public v0.3 authoring contract. Physical
// Model connections stay under providers.models while connection-free Model
// metadata stays under routing.modelCards.
type CanonicalConfig struct {
	Version     string                `yaml:"version,omitempty"`
	Listeners   []Listener            `yaml:"listeners,omitempty"`
	Providers   CanonicalProviders    `yaml:"providers,omitempty"`
	Routing     CanonicalRouting      `yaml:"routing,omitempty"`
	Entrypoints []CanonicalEntrypoint `yaml:"entrypoints,omitempty"`
	Recipes     []CanonicalRecipe     `yaml:"recipes,omitempty"`
	Global      *CanonicalGlobal      `yaml:"global,omitempty"`

	globalOverrideRaw *StructuredPayload `yaml:"-"`
}

// CanonicalRouting owns the DSL routing surface. ModelCards is accepted only
// on the top-level routing value; Recipe routing values share the other fields.
type CanonicalRouting struct {
	ModelCards  []RoutingModel       `yaml:"modelCards,omitempty"`
	Signals     CanonicalSignals     `yaml:"signals,omitempty"`
	Projections CanonicalProjections `yaml:"projections,omitempty"`
	Decisions   []Decision           `yaml:"decisions,omitempty"`
	Strategy    RoutingStrategy      `yaml:"strategy,omitempty"`
}

// CanonicalSignals groups routing signals under routing.signals.
type CanonicalSignals struct {
	Keywords      []KeywordRule          `yaml:"keywords,omitempty"`
	Embeddings    []EmbeddingRule        `yaml:"embeddings,omitempty"`
	Domains       []Category             `yaml:"domains,omitempty"`
	FactCheck     []FactCheckRule        `yaml:"fact_check,omitempty"`
	UserFeedbacks []UserFeedbackRule     `yaml:"user_feedbacks,omitempty"`
	Reasks        []ReaskRule            `yaml:"reasks,omitempty"`
	Preferences   []PreferenceRule       `yaml:"preferences,omitempty"`
	Language      []LanguageRule         `yaml:"language,omitempty"`
	Context       []ContextRule          `yaml:"context,omitempty"`
	Structure     []StructureRule        `yaml:"structure,omitempty"`
	Complexity    []ComplexityRule       `yaml:"complexity,omitempty"`
	Modality      []ModalityRule         `yaml:"modality,omitempty"`
	RoleBindings  []RoleBinding          `yaml:"role_bindings,omitempty"`
	Jailbreak     []JailbreakRule        `yaml:"jailbreak,omitempty"`
	PII           []PIIRule              `yaml:"pii,omitempty"`
	KB            []KBSignalRule         `yaml:"kb,omitempty"`
	Conversation  []ConversationRule     `yaml:"conversation,omitempty"`
	EventRules    []EventRule            `yaml:"events,omitempty"`
	Metadata      []MetadataRule         `yaml:"metadata,omitempty"`
	Classifiers   []ClassifierSignalRule `yaml:"classifiers,omitempty"`
}

// CanonicalProjections groups derived routing outputs under routing.projections.
type CanonicalProjections struct {
	Partitions []ProjectionPartition `yaml:"partitions,omitempty"`
	Scores     []ProjectionScore     `yaml:"scores,omitempty"`
	Mappings   []ProjectionMapping   `yaml:"mappings,omitempty"`
}

// AuthoringModel is the internal Management compiler input. Public file YAML
// uses the v0.3 providers.models/routing.modelCards split instead.
type AuthoringModel struct {
	Name           string                      `yaml:"name" json:"name"`
	Card           AuthoringModelCard          `yaml:"card" json:"card"`
	Connections    []modelauthoring.Connection `yaml:"connections" json:"connections"`
	Execution      ModelExecutionSettings      `yaml:"runtime,omitempty" json:"runtime,omitempty"`
	RuntimePricing ModelRuntimePricing         `yaml:"pricing,omitempty" json:"pricing,omitempty"`
}

// AuthoringModelCard is semantic Model metadata. It is intentionally separate
// from connections so one readable card can describe every physical replica.
type AuthoringModelCard struct {
	Aliases           []string                        `yaml:"aliases,omitempty" json:"aliases,omitempty"`
	ParamSize         string                          `yaml:"param_size,omitempty" json:"param_size,omitempty"`
	ContextWindowSize int                             `yaml:"context_window_size,omitempty" json:"context_window_size,omitempty"`
	Description       string                          `yaml:"description,omitempty" json:"description,omitempty"`
	Capabilities      []string                        `yaml:"capabilities,omitempty" json:"capabilities,omitempty"`
	Reasoning         routingsnapshot.ReasoningFamily `yaml:"reasoning,omitempty" json:"reasoning,omitempty"`
	LoRAs             []string                        `yaml:"loras,omitempty" json:"loras,omitempty"`
	QualityScore      float64                         `yaml:"quality_score,omitempty" json:"quality_score,omitempty"`
	Modality          string                          `yaml:"modality,omitempty" json:"modality,omitempty"`
	Tags              []string                        `yaml:"tags,omitempty" json:"tags,omitempty"`
}

func isCanonicalConfig(raw map[string]interface{}) bool {
	_, hasVersion := raw["version"]
	return hasVersion
}

func normalizeCanonicalConfig(
	canonical *CanonicalConfig,
	connectionCompiler modelauthoring.ConnectionCompiler,
) (*RouterConfig, error) {
	if err := validateCanonicalContract(canonical); err != nil {
		return nil, err
	}

	global, err := resolveCanonicalGlobal(canonical.Global, canonical.globalOverrideRaw)
	if err != nil {
		return nil, err
	}

	cfg := DefaultGlobalConfig()
	cfg.CanonicalVersion = canonical.Version
	if applyErr := applyCanonicalGlobal(&cfg, &global); applyErr != nil {
		return nil, applyErr
	}
	cfg.Listeners = append([]Listener(nil), canonical.Listeners...)

	if canonicalHasInlineRouting(canonical) {
		bundle, bundleErr := buildFileAuthoringBundle(canonical)
		if bundleErr != nil {
			return nil, bundleErr
		}
		if mergeErr := mergeGeneratedBackendCredentials(&cfg.BackendCredentials, bundle.Credentials); mergeErr != nil {
			return nil, mergeErr
		}
		snapshot, snapshotErr := compileFileAuthoringBundle(bundle, canonical, connectionCompiler)
		if snapshotErr != nil {
			return nil, snapshotErr
		}
		if applyErr := applyRoutingSnapshotState(&cfg, snapshot); applyErr != nil {
			return nil, applyErr
		}
		applyPublicProviderRuntimeMetadata(&cfg, canonical.Providers)
	}

	if cfg.VectorStore != nil {
		cfg.VectorStore.ApplyDefaults()
	}
	fileAuthoring, cloneErr := cloneCanonicalConfig(canonical)
	if cloneErr != nil {
		return nil, cloneErr
	}
	cfg.fileAuthoring = fileAuthoring

	return &cfg, nil
}

func reasoningParameter(reasoningType string) string {
	switch reasoningType {
	case ReasoningFamilyTypeChatTemplateKwargs:
		return "enable_thinking"
	case ReasoningFamilyTypeReasoningEffort, ReasoningFamilyTypeTopLevelReasoningEffort:
		return "reasoning_effort"
	default:
		return ""
	}
}

func validateCanonicalContract(canonical *CanonicalConfig) error {
	if canonical.Version != "v0.3" {
		return fmt.Errorf("version must be v0.3, got %q", canonical.Version)
	}
	if err := validateCanonicalStores(canonical); err != nil {
		return err
	}
	if err := validateCanonicalBilling(canonical); err != nil {
		return err
	}
	if !canonicalHasInlineRouting(canonical) {
		// A durable Management store may start without a file routing baseline;
		// its active Namespace publication becomes the routing authority.
		if canonicalHasDynamicRoutingAuthority(canonical) {
			return nil
		}
		return fmt.Errorf("public routing requires providers.models, routing.modelCards, a routing profile or Recipe, and entrypoints")
	}
	hasEntrypoint := len(canonical.Entrypoints) != 0 || len(canonicalImplicitAutoModelNames(canonical)) != 0
	if len(canonical.Providers.Models) == 0 || len(canonical.Routing.ModelCards) == 0 ||
		(!canonicalRoutingHasProfile(canonical.Routing) && len(canonical.Recipes) == 0) ||
		!hasEntrypoint {
		return fmt.Errorf("public routing requires providers.models, routing.modelCards, a routing profile or Recipe, and entrypoints")
	}
	_, err := buildFileAuthoringBundle(canonical)
	return err
}

func validateCanonicalStores(canonical *CanonicalConfig) error {
	if canonical == nil || canonical.Global == nil {
		return nil
	}
	stores := canonical.Global.Stores
	if stores.Management != nil && stores.Management.Postgres == nil {
		return fmt.Errorf("global.stores.management requires postgres")
	}
	if stores.Runtime != nil && stores.Runtime.Redis == nil {
		return fmt.Errorf("global.stores.runtime requires redis")
	}
	return nil
}

func canonicalHasDynamicRoutingAuthority(canonical *CanonicalConfig) bool {
	return canonical != nil && canonical.Global != nil &&
		canonical.Global.Stores.Management != nil && canonical.Global.Stores.Management.Postgres != nil
}

func canonicalHasInlineRouting(canonical *CanonicalConfig) bool {
	return canonical != nil && (len(canonical.Providers.Models) != 0 || len(canonical.Routing.ModelCards) != 0 ||
		canonicalRoutingHasProfile(canonical.Routing) || len(canonical.Recipes) != 0 || len(canonical.Entrypoints) != 0)
}

func applyPublicProviderRuntimeMetadata(cfg *RouterConfig, providers CanonicalProviders) {
	if cfg == nil {
		return
	}
	defaults := canonicalProviderDefaults(providers)
	cfg.DefaultModel = defaults.DefaultModel
	cfg.DefaultReasoningEffort = defaults.DefaultReasoningEffort
	cfg.ReasoningFamilies = copyReasoningFamilies(defaults.ReasoningFamilies)
	for _, source := range providers.Models {
		params, found := cfg.ModelConfig[source.Name]
		if !found {
			continue
		}
		params.ReasoningFamily = source.ReasoningFamily
		params.ExternalModelIDs = copyStringMap(source.ExternalModelIDs)
		if len(params.ExternalModelIDs) == 0 && source.ProviderModelID != "" {
			params.ExternalModelIDs = map[string]string{"default": source.ProviderModelID}
		}
		cfg.ModelConfig[source.Name] = params
	}
}

func normalizeSignals(signals CanonicalSignals, decisions []Decision) Signals {
	result := Signals{
		KeywordRules:      append([]KeywordRule(nil), signals.Keywords...),
		EmbeddingRules:    append([]EmbeddingRule(nil), signals.Embeddings...),
		Categories:        append([]Category(nil), signals.Domains...),
		FactCheckRules:    append([]FactCheckRule(nil), signals.FactCheck...),
		UserFeedbackRules: append([]UserFeedbackRule(nil), signals.UserFeedbacks...),
		ReaskRules:        append([]ReaskRule(nil), signals.Reasks...),
		PreferenceRules:   append([]PreferenceRule(nil), signals.Preferences...),
		LanguageRules:     append([]LanguageRule(nil), signals.Language...),
		ContextRules:      append([]ContextRule(nil), signals.Context...),
		StructureRules:    append([]StructureRule(nil), signals.Structure...),
		ComplexityRules:   append([]ComplexityRule(nil), signals.Complexity...),
		ModalityRules:     append([]ModalityRule(nil), signals.Modality...),
		RoleBindings:      append([]RoleBinding(nil), signals.RoleBindings...),
		JailbreakRules:    append([]JailbreakRule(nil), signals.Jailbreak...),
		PIIRules:          append([]PIIRule(nil), signals.PII...),
		KBRules:           append([]KBSignalRule(nil), signals.KB...),
		ConversationRules: append([]ConversationRule(nil), signals.Conversation...),
		EventRules:        append([]EventRule(nil), signals.EventRules...),
		MetadataRules:     append([]MetadataRule(nil), signals.Metadata...),
		ClassifierRules:   append([]ClassifierSignalRule(nil), signals.Classifiers...),
	}

	if len(result.Categories) == 0 {
		result.Categories = autoGenerateCategoriesFromDecisions(decisions)
	}

	return result
}

func normalizeProjections(projections CanonicalProjections) Projections {
	return Projections{
		Partitions: append([]ProjectionPartition(nil), projections.Partitions...),
		Scores:     append([]ProjectionScore(nil), projections.Scores...),
		Mappings:   append([]ProjectionMapping(nil), projections.Mappings...),
	}
}

func autoGenerateCategoriesFromDecisions(decisions []Decision) []Category {
	names := map[string]bool{}
	for _, decision := range decisions {
		collectRuleNames(decision.Rules, SignalTypeDomain, names)
	}
	if len(names) == 0 {
		return nil
	}

	categories := make([]Category, 0, len(names))
	keys := make([]string, 0, len(names))
	for name := range names {
		keys = append(keys, name)
	}
	sort.Strings(keys)
	for _, name := range keys {
		categories = append(categories, Category{
			CategoryMetadata: CategoryMetadata{
				Name:           name,
				Description:    name,
				MMLUCategories: []string{"other"},
			},
		})
	}
	return categories
}

func collectRuleNames(node RuleCombination, signalType string, out map[string]bool) {
	if node.Type == signalType && node.Name != "" {
		out[node.Name] = true
	}
	for _, child := range node.Conditions {
		collectRuleNames(child, signalType, out)
	}
}

func ensureModelRefDefaults(decisions []Decision) {
	for i := range decisions {
		for j := range decisions[i].ModelRefs {
			if decisions[i].ModelRefs[j].UseReasoning == nil {
				defaultReasoning := false
				decisions[i].ModelRefs[j].UseReasoning = &defaultReasoning
			}
		}
	}
}

func copyDecisions(input []Decision) []Decision {
	if len(input) == 0 {
		return nil
	}
	output := make([]Decision, len(input))
	copy(output, input)
	return output
}

func routingModelHasLoRA(model AuthoringModel, loraName string) bool {
	for _, name := range model.Card.LoRAs {
		if name == loraName {
			return true
		}
	}
	return false
}
