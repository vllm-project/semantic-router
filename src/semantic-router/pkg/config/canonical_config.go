package config

import (
	"fmt"
	"reflect"
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokencount/tokenizerspec"
)

// CanonicalConfigVersion identifies the steady-state public configuration
// contract accepted by the Router and published through configschema.
const CanonicalConfigVersion = "v0.5"

// CanonicalConfig is the public v0.5 config contract.
type CanonicalConfig struct {
	// Version selects the canonical configuration contract understood by the Router.
	Version string `yaml:"version,omitempty"`
	// Listeners expose named public request entry points through Envoy.
	Listeners []Listener `yaml:"listeners,omitempty"`
	// Providers define each model alias once: its endpoints, credentials and facts.
	Providers CanonicalProviders `yaml:"providers,omitempty"`
	// Global holds the defaults every recipe shares (model catalog, plugins) and
	// where the Router's stores and services are.
	Global *CanonicalGlobal `yaml:"global,omitempty"`
	// Recipes are the routing profiles. One recipe serves every request-facing
	// model; several are selected through entrypoints.
	Recipes []CanonicalRecipe `yaml:"recipes,omitempty"`
	// Entrypoints map public model names to recipes. Several recipes require
	// them, with exactly one entrypoint marked default.
	Entrypoints []CanonicalEntrypoint `yaml:"entrypoints,omitempty"`
	// Evaluation defines operator-owned benchmarks, indices, and model measurements.
	Evaluation *CanonicalEvaluation `yaml:"evaluation,omitempty"`

	globalOverrideRaw *StructuredPayload `yaml:"-"`
}

// CanonicalConfigDocument is the complete product configuration document. Its
// canonical Router payload is embedded so every consumer shares the same Go
// field contract, while setup remains explicit control-plane metadata rather
// than Router runtime state.
type CanonicalConfigDocument struct {
	CanonicalConfig `yaml:",inline"`
	Setup           *CanonicalSetup `yaml:"setup,omitempty"`
}

// CanonicalSetup is the product bootstrap state. The Dashboard removes this
// block when setup is activated.
type CanonicalSetup struct {
	Mode bool `yaml:"mode,omitempty"`
}

// CanonicalEvaluation is the single operator-owned evaluation surface. It
// keeps benchmark and index definitions beside model-linked measurement
// records so user configuration matches the built-in catalog data model.
type CanonicalEvaluation struct {
	Calibrations []CalibrationArtifact              `yaml:"calibrations,omitempty"`
	Benchmarks   []modelcatalog.BenchmarkDefinition `yaml:"benchmarks,omitempty"`
	Indices      []modelcatalog.IndexDefinition     `yaml:"indices,omitempty"`
	Records      []CanonicalEvaluationRecord        `yaml:"records,omitempty"`
}

// CanonicalEvaluationRecord is a compact operator-authored measurement. Model
// names a canonical Model Card identity; the compiler assigns internal record
// identity and provenance without asking users to duplicate repository fields.
type CanonicalEvaluationRecord struct {
	Model            string             `yaml:"model"`
	Benchmark        string             `yaml:"benchmark"`
	BenchmarkProfile string             `yaml:"benchmark_profile,omitempty"`
	ReasoningEffort  string             `yaml:"reasoning_effort,omitempty"`
	Metrics          map[string]float64 `yaml:"metrics"`
	Source           string             `yaml:"source,omitempty"`
	MeasuredAt       string             `yaml:"measured_at,omitempty"`
	Metadata         map[string]any     `yaml:"metadata,omitempty"`
}

// CanonicalSignals groups a recipe's signals under recipes[].signals.
type CanonicalSignals struct {
	Keywords      []KeywordRule       `yaml:"keywords,omitempty"`
	Domains       []Category          `yaml:"domains,omitempty"`
	FactCheck     []FactCheckRule     `yaml:"fact_check,omitempty"`
	UserFeedbacks []UserFeedbackRule  `yaml:"user_feedbacks,omitempty"`
	Language      []LanguageRule      `yaml:"language,omitempty"`
	Context       []ContextRule       `yaml:"context,omitempty"`
	Complexity    []ComplexityRule    `yaml:"complexity,omitempty"`
	Modality      []ModalityRule      `yaml:"modality,omitempty"`
	RoleBindings  []RoleBinding       `yaml:"role_bindings,omitempty"`
	Jailbreak     []JailbreakRule     `yaml:"jailbreak,omitempty"`
	Safety        []SafetyRule        `yaml:"safety,omitempty"`
	Hallucination []HallucinationRule `yaml:"hallucination,omitempty"`
	Refusal       []RefusalRule       `yaml:"refusal,omitempty"`
	TurnRelation  []TurnRelationRule  `yaml:"turn_relation,omitempty"`
	PII           []PIIRule           `yaml:"pii,omitempty"`
	Conversation  []ConversationRule  `yaml:"conversation,omitempty"`
	Metadata      []MetadataRule      `yaml:"metadata,omitempty"`
	InputModality []InputModalityRule `yaml:"input_modality,omitempty"`
	SystemOne     []SystemOneRule     `yaml:"systemone,omitempty"`
}

// CanonicalProjections groups a recipe's derived routing outputs under
// recipes[].projections.
type CanonicalProjections struct {
	Partitions []ProjectionPartition `yaml:"partitions,omitempty"`
	Scores     []ProjectionScore     `yaml:"scores,omitempty"`
	Mappings   []ProjectionMapping   `yaml:"mappings,omitempty"`
}

// RoutingModel is a model card: the catalog identity of a providers.models[]
// entry (its catalog, else its name) and the facts the entry declares.
type RoutingModel struct {
	Name       string `yaml:"name"`
	ModelFacts `yaml:",inline"`
}

// ModelFacts describe what a model is: its capabilities and modalities, its
// limits, release and presentation.
type ModelFacts struct {
	DisplayName       string                             `yaml:"display_name,omitempty"`
	Publisher         string                             `yaml:"publisher,omitempty"`
	Presentation      *modelcatalog.ProviderPresentation `yaml:"presentation,omitempty"`
	Distribution      *modelcatalog.ModelDistribution    `yaml:"distribution,omitempty"`
	Family            string                             `yaml:"family,omitempty"`
	Revision          string                             `yaml:"revision,omitempty"`
	ReleasedAt        string                             `yaml:"released_at,omitempty"`
	KnowledgeCutoff   string                             `yaml:"knowledge_cutoff,omitempty"`
	Lifecycle         string                             `yaml:"lifecycle,omitempty"`
	ParamSize         string                             `yaml:"param_size,omitempty"`
	ContextWindowSize int                                `yaml:"context_window_size,omitempty"`
	MaxOutputTokens   int                                `yaml:"max_output_tokens,omitempty"`
	Description       string                             `yaml:"description,omitempty"`
	Capabilities      []string                           `yaml:"capabilities,omitempty"`
	LoRAs             []LoRAAdapter                      `yaml:"loras,omitempty"`
	Modalities        *modelcatalog.Modalities           `yaml:"modalities,omitempty"`
	Tags              []string                           `yaml:"tags,omitempty"`
}

func isCanonicalConfig(raw map[string]interface{}) bool {
	_, hasGlobal := raw["global"]
	_, hasRecipes := raw["recipes"]
	_, hasEntrypoints := raw["entrypoints"]
	return hasGlobal || hasRecipes || hasEntrypoints
}

func normalizeCanonicalConfig(canonical *CanonicalConfig) (*RouterConfig, error) {
	if err := validateCanonicalContract(canonical); err != nil {
		return nil, err
	}
	if err := validateBackendConnections(canonical); err != nil {
		return nil, err
	}
	if err := checkCanonicalBindingKeys(canonical); err != nil {
		return nil, err
	}

	input, err := canonicalCatalogInput(canonical)
	if err != nil {
		return nil, err
	}
	builtIn, err := modelcatalog.BuiltIn()
	if err != nil {
		return nil, fmt.Errorf("load built-in model catalog: %w", err)
	}
	effective, err := builtIn.Compile(input)
	if err != nil {
		return nil, fmt.Errorf("compile effective model registry: %w", err)
	}

	global, err := resolveCanonicalGlobal(canonical.Global, canonical.globalOverrideRaw)
	if err != nil {
		return nil, err
	}

	cfg := DefaultGlobalConfig()
	if applyErr := applyCanonicalGlobal(&cfg, &global); applyErr != nil {
		return nil, applyErr
	}

	cfg.Listeners = append([]Listener(nil), canonical.Listeners...)
	if err := normalizeListenerAPIs(cfg.Listeners); err != nil {
		return nil, err
	}
	cfg.ModelConfig = make(map[string]ModelParams)
	cfg.GlobalPlugins = copyDecisionPlugins(global.Plugins)
	if err := applyCanonicalRecipeState(&cfg, canonical); err != nil {
		return nil, err
	}
	if err := applyEffectiveModelRegistry(&cfg, effective, canonical.Providers.Models); err != nil {
		return nil, err
	}
	cfg.DefaultBackendTLS = cloneProviderDefaultsTLS(canonical.Providers.Defaults.TLS)
	cfg.EffectiveModelRegistry = effective
	cfg.Evaluation = cloneCanonicalEvaluation(canonical.Evaluation)
	if err := validateNativeRoutingConfig(&cfg); err != nil {
		return nil, err
	}

	if cfg.VectorStore != nil {
		cfg.VectorStore.ApplyDefaults()
	}

	return &cfg, nil
}

func validateCanonicalContract(canonical *CanonicalConfig) error {
	if err := validateCanonicalVersion(canonical); err != nil {
		return err
	}
	if err := validateCanonicalNativeResources(canonical); err != nil {
		return err
	}
	if err := validateCanonicalFallback(canonical); err != nil {
		return err
	}
	if err := validateCanonicalDelegatedBackends(canonical); err != nil {
		return err
	}
	if global := canonical.Global; global != nil && global.Router.StreamedBody.Enabled && global.Router.StreamedBody.MaxBytes <= 0 {
		// The handler accumulates without a bound until this names a ceiling.
		return fmt.Errorf("global.router.streamed_body.max_bytes: an enabled streamed body needs a positive max_bytes")
	}
	cards, err := canonicalModelCardIndex(canonical.Providers.Models)
	if err != nil {
		return err
	}
	aliases, err := canonicalProviderModelIndex(canonical.Providers.Models)
	if err != nil {
		return err
	}
	return validateCanonicalDefaultModel(canonical.Providers, aliases, cards)
}

// validateCanonicalDelegatedBackends checks every backend_ref whose auth is
// credential: delegated: the authz contract that supplies the requesting
// user's credential must be enabled.
func validateCanonicalDelegatedBackends(canonical *CanonicalConfig) error {
	enabled := canonical.Global != nil && canonical.Global.Services.Authz.DelegatedCredential.Enabled
	for _, model := range canonical.Providers.Models {
		for index, backend := range model.BackendRefs {
			if backend.Auth.Delegated() && !enabled {
				return fmt.Errorf("providers.models[%s].backend_refs[%d]: %s", model.Name, index, delegatedContractRequired)
			}
		}
	}
	return nil
}

func validateCanonicalVersion(canonical *CanonicalConfig) error {
	if canonical == nil {
		return fmt.Errorf("config cannot be nil")
	}
	if canonical.Version != "" && canonical.Version != CanonicalConfigVersion {
		return fmt.Errorf("unsupported config version %q: %s is required", canonical.Version, CanonicalConfigVersion)
	}
	return nil
}

func validateCanonicalFallback(canonical *CanonicalConfig) error {
	if canonical == nil {
		return nil
	}
	defaults := RoutingDefaults{}
	if canonical.Global != nil {
		defaults.Strategy = canonical.Global.Router.Strategy
		defaults.Fallback = canonical.Global.Router.Fallback
		if err := defaults.Strategy.Validate(); err != nil {
			return fmt.Errorf("global.router.strategy: %w", err)
		}
		if defaults.Fallback != nil {
			if err := defaults.Fallback.WithDefaults().Validate(); err != nil {
				return fmt.Errorf("global.router.fallback: %w", err)
			}
		}
	}
	for _, recipe := range canonical.Recipes {
		if err := validateRecipeFallback(recipePath(recipe.Name), recipe, defaults); err != nil {
			return err
		}
	}
	return nil
}

func validateRecipeFallback(path string, recipe CanonicalRecipe, defaults RoutingDefaults) error {
	base := fallback.DefaultPolicy()
	if effective := defaults.resolveFallback(recipe.Fallback); effective != nil {
		base = *effective
	}
	if err := base.Validate(); err != nil {
		return fmt.Errorf("%s.fallback: %w", path, err)
	}
	return validateDecisionFallbacks(path, recipe.Decisions, base)
}

// validateDecisionFallbacks checks each decision's fallback override, and the
// policy it makes over its recipe's, as the recipe block is checked.
func validateDecisionFallbacks(path string, decisions []Decision, recipe fallback.FallbackPolicy) error {
	for _, decision := range decisions {
		if decision.Fallback == nil {
			continue
		}
		if err := decision.Fallback.Validate(); err != nil {
			return fmt.Errorf("%s.decisions[%s].fallback: %w", path, decision.Name, err)
		}
		if err := recipe.WithDefaults().Resolve(decision.Fallback).Validate(); err != nil {
			return fmt.Errorf("%s.decisions[%s].fallback: %w", path, decision.Name, err)
		}
	}
	return nil
}

// canonicalModelCardIndex is the model cards the providers.models[] entries
// declare facts for, by card identity. One entry declares a card's facts.
func canonicalModelCardIndex(models []CanonicalProviderModel) (map[string]RoutingModel, error) {
	cards := make(map[string]RoutingModel, len(models))
	declaredBy := make(map[string]string, len(models))
	for _, model := range models {
		if !model.ModelFacts.declared() {
			continue
		}
		cardID := effectiveCanonicalCardID(model)
		if owner, exists := declaredBy[cardID]; exists {
			return nil, fmt.Errorf(
				"providers.models[%s]: the facts of model card %q are already declared by providers.models[%s]",
				model.Name, cardID, owner,
			)
		}
		declaredBy[cardID] = model.Name
		cards[cardID] = RoutingModel{Name: cardID, ModelFacts: model.ModelFacts}
	}
	return cards, nil
}

// canonicalModelCards lists canonicalModelCardIndex in providers.models[] order.
func canonicalModelCards(models []CanonicalProviderModel) []RoutingModel {
	var cards []RoutingModel
	for _, model := range models {
		if model.ModelFacts.declared() {
			cards = append(cards, RoutingModel{Name: effectiveCanonicalCardID(model), ModelFacts: model.ModelFacts})
		}
	}
	return cards
}

// declared reports whether an entry states any fact.
func (f ModelFacts) declared() bool {
	return !reflect.ValueOf(f).IsZero()
}

func canonicalProviderModelIndex(models []CanonicalProviderModel) (map[string]CanonicalProviderModel, error) {
	aliases := make(map[string]CanonicalProviderModel, len(models))
	catalogBackedTargets := make(map[string]bool, len(models))
	for index, model := range models {
		cardID, err := validateCanonicalProviderModel(model, index, aliases)
		if err != nil {
			return nil, err
		}
		catalogBacked := strings.TrimSpace(model.Catalog) != ""
		if existing, duplicate := catalogBackedTargets[cardID]; duplicate && existing != catalogBacked {
			return nil, fmt.Errorf(
				"providers.models[%d] makes model card %q ambiguous: it cannot represent both a catalog-backed and custom model",
				index, cardID,
			)
		}
		aliases[model.Name] = model
		catalogBackedTargets[cardID] = catalogBacked
	}
	return aliases, nil
}

func validateCanonicalProviderModel(
	model CanonicalProviderModel,
	index int,
	aliases map[string]CanonicalProviderModel,
) (string, error) {
	if strings.TrimSpace(model.Name) == "" {
		return "", fmt.Errorf("providers.models.name cannot be empty")
	}
	if _, duplicate := aliases[model.Name]; duplicate {
		return "", fmt.Errorf("providers.models[%d].name %q is duplicated", index, model.Name)
	}
	if err := validateCanonicalProviderModelMetadata(model); err != nil {
		return "", err
	}
	return effectiveCanonicalCardID(model), nil
}

func effectiveCanonicalCardID(model CanonicalProviderModel) string {
	if cardID := strings.TrimSpace(model.Catalog); cardID != "" {
		return cardID
	}
	return model.Name
}

func validateCanonicalProviderModelMetadata(model CanonicalProviderModel) error {
	if err := validateCanonicalReasoning(model.Name, model.Reasoning); err != nil {
		return err
	}
	if err := validateProviderReliability(model.Name, model.Reliability); err != nil {
		return err
	}
	if _, err := llmprotocol.ParseUnsupportedCapabilities(model.UnsupportedCapabilities); err != nil {
		return fmt.Errorf("providers.models[%s].unsupported_capabilities: %w", model.Name, err)
	}
	if model.Tokenizer != "" {
		if _, err := tokenizerspec.Parse(model.Tokenizer); err != nil {
			return fmt.Errorf("providers.models[%s].tokenizer: %w", model.Name, err)
		}
	}
	return validateModelPricing(model.Name, model.Pricing)
}

func validateCanonicalDefaultModel(
	providers CanonicalProviders,
	aliases map[string]CanonicalProviderModel,
	cards map[string]RoutingModel,
) error {
	defaultModel := canonicalProviderDefaults(providers).DefaultModel
	if defaultModel == "" || canonicalDefaultModelExists(defaultModel, aliases, cards) {
		return nil
	}
	return fmt.Errorf(
		"providers.defaults.model %q not found in providers.models or their loras",
		defaultModel,
	)
}

func canonicalDefaultModelExists(
	defaultModel string,
	aliases map[string]CanonicalProviderModel,
	cards map[string]RoutingModel,
) bool {
	if _, ok := aliases[defaultModel]; ok {
		return true
	}
	return aliasLoRAExists(defaultModel, aliases, cards)
}

func validateCanonicalReasoning(modelName string, reasoning *CanonicalReasoning) error {
	if reasoning == nil {
		return nil
	}
	inline := reasoning.Type != "" || reasoning.Parameter != "" || reasoning.ActivationParameter != "" || len(reasoning.EffortFlags) > 0 || len(reasoning.Levels) > 0 || reasoning.Default != "" || len(reasoning.Modes) > 0 || reasoning.DefaultMode != "" || reasoning.Disabled != ""
	if reasoning.Family != "" && inline {
		return fmt.Errorf("providers.models[%s].reasoning must set either family or inline fields, not both", modelName)
	}
	if reasoning.Family == "" && !inline {
		return fmt.Errorf("providers.models[%s].reasoning cannot be empty", modelName)
	}
	if inline && (reasoning.Type == "" || reasoning.Parameter == "") {
		return fmt.Errorf("providers.models[%s].reasoning.type and parameter are required for inline reasoning", modelName)
	}
	return nil
}

func validateCanonicalDecisions(path string, decisions []Decision, aliases map[string]CanonicalProviderModel, cards map[string]RoutingModel) error {
	decisionNames := make(map[string]bool, len(decisions))
	for _, decision := range decisions {
		if decision.Name != "" {
			if decisionNames[decision.Name] {
				return fmt.Errorf("%s.decisions[%s]: duplicate decision name", path, decision.Name)
			}
			decisionNames[decision.Name] = true
		}

		if err := validateCanonicalDecisionModelRefs(path, decision, aliases, cards); err != nil {
			return err
		}
	}

	return nil
}

// validateCanonicalDecisionModelRefs checks the models a decision names: its
// modelRefs and its candidate iterations' models.
func validateCanonicalDecisionModelRefs(path string, decision Decision, aliases map[string]CanonicalProviderModel, cards map[string]RoutingModel) error {
	at := fmt.Sprintf("%s.decisions[%s]", path, decision.Name)
	if err := validateCanonicalModelRefs(at+".modelRefs", decision.ModelRefs, aliases, cards); err != nil {
		return err
	}
	for i, iteration := range decision.CandidateIterations {
		if err := validateCanonicalModelRefs(fmt.Sprintf("%s.candidateIterations[%d].models", at, i), iteration.Models, aliases, cards); err != nil {
			return err
		}
	}
	return nil
}

func validateCanonicalModelRefs(path string, refs []ModelRef, aliases map[string]CanonicalProviderModel, cards map[string]RoutingModel) error {
	for _, modelRef := range refs {
		if modelRef.Model == "" {
			continue
		}
		if len(aliases) > 0 {
			if _, ok := aliases[modelRef.Model]; !ok && !aliasLoRAExists(modelRef.Model, aliases, cards) {
				return fmt.Errorf("%s[%s] references unknown model %q", path, modelRef.Model, modelRef.Model)
			}
		}
		if modelRef.LoRAName == "" {
			continue
		}
		alias, ok := aliases[modelRef.Model]
		if !ok {
			continue
		}
		cardID := alias.Catalog
		if cardID == "" {
			cardID = alias.Name
		}
		card, ok := cards[cardID]
		if !ok {
			continue
		}
		if !routingModelHasLoRA(card, modelRef.LoRAName) {
			return fmt.Errorf("%s[%s].lora_name %q not found in the loras of model card %q", path, modelRef.Model, modelRef.LoRAName, cardID)
		}
	}

	return nil
}

func aliasLoRAExists(name string, aliases map[string]CanonicalProviderModel, cards map[string]RoutingModel) bool {
	for _, alias := range aliases {
		cardID := alias.Catalog
		if cardID == "" {
			cardID = alias.Name
		}
		if card, ok := cards[cardID]; ok && routingModelHasLoRA(card, name) {
			return true
		}
	}
	return false
}

func normalizeSignals(signals CanonicalSignals) Signals {
	return Signals{
		KeywordRules:       append([]KeywordRule(nil), signals.Keywords...),
		Categories:         append([]Category(nil), signals.Domains...),
		FactCheckRules:     append([]FactCheckRule(nil), signals.FactCheck...),
		UserFeedbackRules:  append([]UserFeedbackRule(nil), signals.UserFeedbacks...),
		LanguageRules:      append([]LanguageRule(nil), signals.Language...),
		ContextRules:       append([]ContextRule(nil), signals.Context...),
		ComplexityRules:    append([]ComplexityRule(nil), signals.Complexity...),
		ModalityRules:      append([]ModalityRule(nil), signals.Modality...),
		RoleBindings:       append([]RoleBinding(nil), signals.RoleBindings...),
		JailbreakRules:     append([]JailbreakRule(nil), signals.Jailbreak...),
		SafetyRules:        append([]SafetyRule(nil), signals.Safety...),
		HallucinationRules: append([]HallucinationRule(nil), signals.Hallucination...),
		RefusalRules:       append([]RefusalRule(nil), signals.Refusal...),
		TurnRelationRules:  append([]TurnRelationRule(nil), signals.TurnRelation...),
		PIIRules:           append([]PIIRule(nil), signals.PII...),
		ConversationRules:  append([]ConversationRule(nil), signals.Conversation...),
		MetadataRules:      append([]MetadataRule(nil), signals.Metadata...),
		InputModalityRules: append([]InputModalityRule(nil), signals.InputModality...),
		SystemOneRules:     append([]SystemOneRule(nil), signals.SystemOne...),
	}
}

func normalizeProjections(projections CanonicalProjections) Projections {
	return Projections{
		Partitions: append([]ProjectionPartition(nil), projections.Partitions...),
		Scores:     append([]ProjectionScore(nil), projections.Scores...),
		Mappings:   append([]ProjectionMapping(nil), projections.Mappings...),
	}
}

func canonicalProviderModelHasMetadata(model CanonicalProviderModel) bool {
	if model.Deployment != "" || model.Catalog != "" || model.Reasoning != nil || model.ProviderModelID != "" || model.APIFormat != "" || len(model.ExternalModelIDs) > 0 {
		return true
	}
	return model.Pricing != (ModelPricing{}) || !model.Reliability.IsZero() || model.ModelFacts.declared()
}

func canonicalEndpointName(modelName string, backendRef CanonicalBackendRef, index int) string {
	suffix := strings.TrimSpace(backendRef.Name)
	if suffix == "" {
		if index == 0 {
			suffix = "primary"
		} else {
			suffix = fmt.Sprintf("backend-%d", index+1)
		}
	}
	return modelName + "_" + suffix
}

func defaultProtocol(protocol string) string {
	if protocol == "" {
		return "http"
	}
	return strings.ToLower(protocol)
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
		if decisions[i].Algorithm.IsNative() {
			continue
		}
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
	for index := range output {
		output[index].Algorithm = cloneNativeAlgorithm(input[index].Algorithm)
	}
	return output
}

func copyStringMap(input map[string]string) map[string]string {
	if len(input) == 0 {
		return nil
	}
	output := make(map[string]string, len(input))
	for key, value := range input {
		output[key] = value
	}
	return output
}

func copyLoRAAdapters(input []LoRAAdapter) []LoRAAdapter {
	if len(input) == 0 {
		return nil
	}
	output := make([]LoRAAdapter, len(input))
	copy(output, input)
	return output
}

func routingModelHasLoRA(model RoutingModel, loraName string) bool {
	for _, adapter := range model.LoRAs {
		if adapter.Name == loraName {
			return true
		}
	}
	return false
}
