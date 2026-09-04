package config

import (
	"fmt"
	"sort"
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

// CanonicalConfig is the public v0.3 config contract.
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

// CanonicalRouting contains the DSL-owned routing surface.
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
	InputModality []InputModalityRule    `yaml:"input_modality,omitempty"`
}

// CanonicalProjections groups derived routing outputs under routing.projections.
type CanonicalProjections struct {
	Partitions []ProjectionPartition `yaml:"partitions,omitempty"`
	Scores     []ProjectionScore     `yaml:"scores,omitempty"`
	Mappings   []ProjectionMapping   `yaml:"mappings,omitempty"`
}

// RoutingModel defines the logical model catalog available to routing decisions.
type RoutingModel struct {
	Name              string                             `yaml:"name"`
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
	Modality          string                             `yaml:"modality,omitempty"`
	Tags              []string                           `yaml:"tags,omitempty"`
	Evaluations       []modelcatalog.UserEvaluation      `yaml:"evaluations,omitempty"`
}

func isCanonicalConfig(raw map[string]interface{}) bool {
	_, hasRouting := raw["routing"]
	_, hasGlobal := raw["global"]
	return hasRouting || hasGlobal
}

func normalizeCanonicalConfig(canonical *CanonicalConfig) (*RouterConfig, error) {
	if err := validateCanonicalContract(canonical); err != nil {
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

	applyCanonicalRoutingState(&cfg, canonical)
	if err := applyCanonicalRecipeState(&cfg, canonical); err != nil {
		return nil, err
	}
	if err := applyEffectiveModelRegistry(&cfg, effective, canonical.Providers.Models); err != nil {
		return nil, err
	}
	cfg.EffectiveModelRegistry = effective

	if cfg.VectorStore != nil {
		cfg.VectorStore.ApplyDefaults()
	}

	return &cfg, nil
}

func applyCanonicalRoutingState(cfg *RouterConfig, canonical *CanonicalConfig) {
	cfg.Listeners = append([]Listener(nil), canonical.Listeners...)
	cfg.Decisions = copyDecisions(canonical.Routing.Decisions)
	ensureModelRefDefaults(cfg.Decisions)
	cfg.Signals = normalizeSignals(canonical.Routing.Signals, cfg.Decisions)
	cfg.Projections = normalizeProjections(canonical.Routing.Projections)
	if canonical.Routing.Strategy != "" {
		cfg.Strategy = canonical.Routing.Strategy
	}
	cfg.ModelConfig = make(map[string]ModelParams)
}

func validateCanonicalContract(canonical *CanonicalConfig) error {
	if err := validateCanonicalVersion(canonical); err != nil {
		return err
	}
	modelsByName, err := canonicalModelCardIndex(canonical.Routing)
	if err != nil {
		return err
	}
	aliases, cardTargets, err := canonicalProviderModelIndex(canonical.Providers.Models, modelsByName)
	if err != nil {
		return err
	}
	if err := validateCanonicalDefaultModel(canonical.Providers, aliases, modelsByName); err != nil {
		return err
	}
	if err := validateCanonicalCardTargets(modelsByName, aliases, cardTargets); err != nil {
		return err
	}
	return validateCanonicalDecisions(canonical.Routing.Decisions, aliases, modelsByName)
}

func validateCanonicalVersion(canonical *CanonicalConfig) error {
	if canonical == nil {
		return fmt.Errorf("config cannot be nil")
	}
	if canonical.Version != "" && canonical.Version != "v0.3" {
		return fmt.Errorf("unsupported config version %q: v0.3 is required", canonical.Version)
	}
	return nil
}

func canonicalModelCardIndex(routing CanonicalRouting) (map[string]RoutingModel, error) {
	modelCards := canonicalRoutingModels(routing)
	modelsByName := make(map[string]RoutingModel, len(modelCards))
	for _, model := range modelCards {
		if strings.TrimSpace(model.Name) == "" {
			return nil, fmt.Errorf("routing.modelCards.name cannot be empty")
		}
		if _, exists := modelsByName[model.Name]; exists {
			return nil, fmt.Errorf("routing.modelCards[%s]: duplicate model name", model.Name)
		}
		modelsByName[model.Name] = model
	}
	return modelsByName, nil
}

func canonicalProviderModelIndex(
	models []CanonicalProviderModel,
	cards map[string]RoutingModel,
) (map[string]CanonicalProviderModel, map[string]struct{}, error) {
	aliases := make(map[string]CanonicalProviderModel, len(models))
	cardTargets := make(map[string]struct{}, len(models))
	catalogBackedTargets := make(map[string]bool, len(models))
	for index, model := range models {
		cardID, err := validateCanonicalProviderModel(model, index, cards, aliases)
		if err != nil {
			return nil, nil, err
		}
		catalogBacked := strings.TrimSpace(model.Catalog) != ""
		if existing, duplicate := catalogBackedTargets[cardID]; duplicate && existing != catalogBacked {
			return nil, nil, fmt.Errorf(
				"providers.models[%d] makes model card %q ambiguous: it cannot represent both a catalog-backed and custom model",
				index, cardID,
			)
		}
		aliases[model.Name] = model
		cardTargets[cardID] = struct{}{}
		catalogBackedTargets[cardID] = catalogBacked
	}
	return aliases, cardTargets, nil
}

func validateCanonicalProviderModel(
	model CanonicalProviderModel,
	index int,
	cards map[string]RoutingModel,
	aliases map[string]CanonicalProviderModel,
) (string, error) {
	if strings.TrimSpace(model.Name) == "" {
		return "", fmt.Errorf("providers.models.name cannot be empty")
	}
	if _, duplicate := aliases[model.Name]; duplicate {
		return "", fmt.Errorf("providers.models[%d].name %q is duplicated", index, model.Name)
	}
	cardID := effectiveCanonicalCardID(model)
	if model.Catalog != "" {
		if _, aliasNamedCard := cards[model.Name]; aliasNamedCard && model.Name != model.Catalog {
			return "", fmt.Errorf(
				"routing.modelCards[%s] is ambiguous: built-in overrides must use catalog name %q",
				model.Name, model.Catalog,
			)
		}
	}
	if err := validateCanonicalProviderModelMetadata(model); err != nil {
		return "", err
	}
	if len(canonicalBackendRefs(model)) == 0 && !canonicalProviderModelHasMetadata(model) {
		return "", fmt.Errorf("providers.models[%s] must define backend_refs or model metadata", model.Name)
	}
	return cardID, nil
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
		"providers.defaults.model %q not found in providers.models or routing-only modelCards/loras",
		defaultModel,
	)
}

func validateCanonicalCardTargets(
	cards map[string]RoutingModel,
	aliases map[string]CanonicalProviderModel,
	targets map[string]struct{},
) error {
	for cardID := range cards {
		_, targeted := targets[cardID]
		if !targeted && len(aliases) > 0 && !routingLoRAAliasExists(cardID, cards) {
			return fmt.Errorf(
				"routing.modelCards[%s] is not referenced by providers.models[].catalog or a custom model alias",
				cardID,
			)
		}
	}
	return nil
}

func canonicalDefaultModelExists(
	defaultModel string,
	aliases map[string]CanonicalProviderModel,
	cards map[string]RoutingModel,
) bool {
	if _, ok := aliases[defaultModel]; ok {
		return true
	}
	if len(aliases) == 0 {
		if _, ok := cards[defaultModel]; ok {
			return true
		}
	}
	return aliasLoRAExists(defaultModel, aliases, cards)
}

func routingLoRAAliasExists(name string, cards map[string]RoutingModel) bool {
	for _, card := range cards {
		if routingModelHasLoRA(card, name) {
			return true
		}
	}
	return false
}

func validateCanonicalReasoning(modelName string, reasoning *CanonicalReasoning) error {
	if reasoning == nil {
		return nil
	}
	inline := reasoning.Type != "" || reasoning.Parameter != "" || len(reasoning.Levels) > 0 || reasoning.Default != ""
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

func validateCanonicalDecisions(decisions []Decision, aliases map[string]CanonicalProviderModel, cards map[string]RoutingModel) error {
	decisionNames := make(map[string]bool, len(decisions))
	for _, decision := range decisions {
		if decision.Name != "" {
			if decisionNames[decision.Name] {
				return fmt.Errorf("routing.decisions[%s]: duplicate decision name", decision.Name)
			}
			decisionNames[decision.Name] = true
		}

		if err := validateCanonicalDecisionModelRefs(decision, aliases, cards); err != nil {
			return err
		}
	}

	return nil
}

func validateCanonicalDecisionModelRefs(decision Decision, aliases map[string]CanonicalProviderModel, cards map[string]RoutingModel) error {
	for _, modelRef := range decision.ModelRefs {
		if modelRef.Model == "" {
			continue
		}
		if len(aliases) > 0 {
			if _, ok := aliases[modelRef.Model]; !ok && !aliasLoRAExists(modelRef.Model, aliases, cards) {
				return fmt.Errorf("routing.decisions[%s].modelRefs[%s] references unknown model %q", decision.Name, modelRef.Model, modelRef.Model)
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
			return fmt.Errorf("routing.decisions[%s].modelRefs[%s].lora_name %q not found in routing.modelCards[%s].loras", decision.Name, modelRef.Model, modelRef.LoRAName, cardID)
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

func normalizeSignals(signals CanonicalSignals, decisions []Decision) Signals {
	result := Signals{
		KeywordRules:       append([]KeywordRule(nil), signals.Keywords...),
		EmbeddingRules:     append([]EmbeddingRule(nil), signals.Embeddings...),
		Categories:         append([]Category(nil), signals.Domains...),
		FactCheckRules:     append([]FactCheckRule(nil), signals.FactCheck...),
		UserFeedbackRules:  append([]UserFeedbackRule(nil), signals.UserFeedbacks...),
		ReaskRules:         append([]ReaskRule(nil), signals.Reasks...),
		PreferenceRules:    append([]PreferenceRule(nil), signals.Preferences...),
		LanguageRules:      append([]LanguageRule(nil), signals.Language...),
		ContextRules:       append([]ContextRule(nil), signals.Context...),
		StructureRules:     append([]StructureRule(nil), signals.Structure...),
		ComplexityRules:    append([]ComplexityRule(nil), signals.Complexity...),
		ModalityRules:      append([]ModalityRule(nil), signals.Modality...),
		RoleBindings:       append([]RoleBinding(nil), signals.RoleBindings...),
		JailbreakRules:     append([]JailbreakRule(nil), signals.Jailbreak...),
		PIIRules:           append([]PIIRule(nil), signals.PII...),
		KBRules:            append([]KBSignalRule(nil), signals.KB...),
		ConversationRules:  append([]ConversationRule(nil), signals.Conversation...),
		EventRules:         append([]EventRule(nil), signals.EventRules...),
		MetadataRules:      append([]MetadataRule(nil), signals.Metadata...),
		ClassifierRules:    append([]ClassifierSignalRule(nil), signals.Classifiers...),
		InputModalityRules: append([]InputModalityRule(nil), signals.InputModality...),
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

func canonicalRoutingModels(routing CanonicalRouting) []RoutingModel {
	return routing.ModelCards
}

func canonicalProviderModelHasMetadata(model CanonicalProviderModel) bool {
	if model.Catalog != "" || model.Reasoning != nil || model.ProviderModelID != "" || model.APIFormat != "" || len(model.ExternalModelIDs) > 0 {
		return true
	}
	return model.Pricing != (ModelPricing{}) ||
		model.Reliability != (ProviderReliability{})
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
