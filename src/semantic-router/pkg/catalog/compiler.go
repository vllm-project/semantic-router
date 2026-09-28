package catalog

import (
	"fmt"
	"net/url"
	"regexp"
	"strconv"
	"strings"
)

const (
	SourceBuiltin  = "builtin"
	SourceOperator = "operator"
)

// EffectiveRegistry is the immutable result of joining repository catalog
// facts with operator-owned provider instances, aliases, and extensions.
type EffectiveRegistry struct {
	defaults          Defaults
	providers         map[string]EffectiveProvider
	models            map[string]EffectiveModel
	cards             map[string]EffectiveModelCard
	reasoningFamilies map[string]ReasoningFamilyDefinition
	indices           map[string]IndexDefinition
	digest            string
}

// Compile validates and materializes a user configuration against this
// release's catalog. It never mutates the built-in Registry.
func (registry *Registry) Compile(input CompileInput) (*EffectiveRegistry, error) {
	if registry == nil {
		return nil, fmt.Errorf("catalog registry is required")
	}
	providers, err := registry.compileProviders(input.Providers)
	if err != nil {
		return nil, err
	}
	reasoning, err := registry.compileReasoningFamilies(input.ReasoningFamilies)
	if err != nil {
		return nil, err
	}
	cards, err := registry.compileCards(input.ModelCards, reasoning)
	if err != nil {
		return nil, err
	}
	indices, results, err := registry.compileEvaluations(input.Evaluations, cards, reasoning)
	if err != nil {
		return nil, err
	}
	models, err := registry.compileModels(input.Models, providers, cards, reasoning, results)
	if err != nil {
		return nil, err
	}
	defaults, err := registry.compileDefaults(input.Defaults, models, indices)
	if err != nil {
		return nil, err
	}

	return &EffectiveRegistry{
		defaults:          defaults,
		providers:         providers,
		models:            models,
		cards:             cards,
		reasoningFamilies: reasoning,
		indices:           indices,
		digest:            registry.digest,
	}, nil
}

func (registry *Registry) compileDefaults(
	defaults Defaults,
	models map[string]EffectiveModel,
	indices map[string]IndexDefinition,
) (Defaults, error) {
	if defaults.QualityIndex == "" {
		defaults.QualityIndex = registry.header.DefaultIntelligenceIndex
	}
	if defaults.Model != "" && !effectiveModelOrLoRAExists(models, defaults.Model) {
		return Defaults{}, fmt.Errorf("providers.defaults.model %q does not reference providers.models[].name", defaults.Model)
	}
	if defaults.QualityIndex != "" {
		if _, ok := indices[defaults.QualityIndex]; !ok {
			return Defaults{}, fmt.Errorf("defaults.quality_index %q is unknown", defaults.QualityIndex)
		}
	}
	return defaults, nil
}

func effectiveModelOrLoRAExists(models map[string]EffectiveModel, name string) bool {
	if _, ok := models[name]; ok {
		return true
	}
	return effectiveLoRAAliasExists(models, name)
}

func effectiveLoRAAliasExists(models map[string]EffectiveModel, name string) bool {
	for _, model := range models {
		for _, adapter := range model.Card.LoRAs {
			if adapter.Name == name {
				return true
			}
		}
	}
	return false
}

func (registry *Registry) compileProviders(inputs []ProviderInstance) (map[string]EffectiveProvider, error) {
	result := make(map[string]EffectiveProvider, len(inputs))
	for index, input := range inputs {
		path := fmt.Sprintf("providers[%d]", index)
		effective, err := registry.compileProvider(input, path, result)
		if err != nil {
			return nil, err
		}
		result[effective.Instance.Name] = effective
	}
	return result, nil
}

func (registry *Registry) compileProvider(
	input ProviderInstance,
	path string,
	existing map[string]EffectiveProvider,
) (EffectiveProvider, error) {
	input.Name = strings.TrimSpace(input.Name)
	input.Catalog = strings.TrimSpace(input.Catalog)
	definition, err := registry.validateProviderInstance(input, path, existing)
	if err != nil {
		return EffectiveProvider{}, err
	}
	endpoints, err := normalizeProviderEndpoints(input.Endpoints, path)
	if err != nil {
		return EffectiveProvider{}, err
	}
	input.Headers = cloneMap(input.Headers)
	input.Endpoints = endpoints
	return EffectiveProvider{Instance: input, Definition: cloneProvider(definition)}, nil
}

func (registry *Registry) validateProviderInstance(
	input ProviderInstance,
	path string,
	existing map[string]EffectiveProvider,
) (ProviderDefinition, error) {
	if input.Name == "" {
		return ProviderDefinition{}, fmt.Errorf("%s.name cannot be empty", path)
	}
	if _, exists := existing[input.Name]; exists {
		return ProviderDefinition{}, fmt.Errorf("%s.name %q is duplicated", path, input.Name)
	}
	definition, ok := registry.providers[input.Catalog]
	if !ok {
		return ProviderDefinition{}, fmt.Errorf("%s.catalog %q is unknown", path, input.Catalog)
	}
	if input.BaseURL != "" && len(input.Endpoints) > 0 {
		return ProviderDefinition{}, fmt.Errorf("%s cannot set both base_url and endpoints", path)
	}
	if input.BaseURL == "" && len(input.Endpoints) == 0 && definition.DefaultBaseURL == "" {
		return ProviderDefinition{}, fmt.Errorf("%s requires base_url or endpoints because provider %q has no default", path, input.Catalog)
	}
	if input.BaseURL != "" {
		if err := validateEndpointURL(input.BaseURL); err != nil {
			return ProviderDefinition{}, fmt.Errorf("%s.base_url: %w", path, err)
		}
	}
	return definition, nil
}

func normalizeProviderEndpoints(endpoints []Endpoint, path string) ([]Endpoint, error) {
	result := append([]Endpoint(nil), endpoints...)
	seen := map[string]struct{}{}
	for index := range result {
		endpointPath := fmt.Sprintf("%s.endpoints[%d]", path, index)
		if err := normalizeProviderEndpoint(&result[index], index, endpointPath, seen); err != nil {
			return nil, err
		}
	}
	return result, nil
}

func normalizeProviderEndpoint(
	endpoint *Endpoint,
	index int,
	path string,
	seen map[string]struct{},
) error {
	if endpoint.Name == "" {
		endpoint.Name = fmt.Sprintf("endpoint-%d", index+1)
	}
	if _, exists := seen[endpoint.Name]; exists {
		return fmt.Errorf("%s.name %q is duplicated", path, endpoint.Name)
	}
	seen[endpoint.Name] = struct{}{}
	if err := validateEndpointURL(endpoint.URL); err != nil {
		return fmt.Errorf("%s.url: %w", path, err)
	}
	if endpoint.Weight < 0 {
		return fmt.Errorf("%s.weight cannot be negative", path)
	}
	if endpoint.Weight == 0 {
		endpoint.Weight = 1
	}
	return nil
}

func validateEndpointURL(raw string) error {
	parsed, err := url.Parse(endpointURLForValidation(raw))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return fmt.Errorf("must be an absolute http(s) URL")
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return fmt.Errorf("scheme %q is unsupported", parsed.Scheme)
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return fmt.Errorf("userinfo, query parameters, and fragments are not allowed")
	}
	if err := validateEndpointPort(parsed); err != nil {
		return err
	}
	return nil
}

func validateEndpointPort(parsed *url.URL) error {
	rawPort := parsed.Port()
	if rawPort == "" {
		return nil
	}
	port, err := strconv.Atoi(rawPort)
	if err != nil || port < 1 || port > 65535 {
		return fmt.Errorf("port must be between 1 and 65535")
	}
	return nil
}

var endpointTemplateToken = regexp.MustCompile(`\{\{[A-Za-z_][A-Za-z0-9_.-]*\}\}|\$\{[A-Za-z_][A-Za-z0-9_]*\}`)

func endpointURLForValidation(raw string) string {
	return endpointTemplateToken.ReplaceAllString(strings.TrimSpace(raw), "catalog-placeholder.invalid")
}

func (registry *Registry) compileReasoningFamilies(inputs []ReasoningFamilyDefinition) (map[string]ReasoningFamilyDefinition, error) {
	result := make(map[string]ReasoningFamilyDefinition, len(registry.reasoningFamilies)+len(inputs))
	for id, definition := range registry.reasoningFamilies {
		result[id] = cloneReasoningFamily(definition)
	}
	for index, definition := range inputs {
		path := fmt.Sprintf("reasoning_families[%d]", index)
		if strings.TrimSpace(definition.ID) == "" {
			return nil, fmt.Errorf("%s.id cannot be empty", path)
		}
		if _, exists := result[definition.ID]; exists {
			return nil, fmt.Errorf("%s.id %q conflicts with an existing reasoning family", path, definition.ID)
		}
		if err := validateReasoningFamily(definition, path); err != nil {
			return nil, err
		}
		result[definition.ID] = cloneReasoningFamily(definition)
	}
	return result, nil
}

func validateReasoningFamily(definition ReasoningFamilyDefinition, path string) error {
	if err := validateReasoningFamilyType(definition.Type, path); err != nil {
		return err
	}
	if strings.TrimSpace(definition.Parameter) == "" {
		return fmt.Errorf("%s.parameter cannot be empty", path)
	}
	if err := validateReasoningActivationParameter(definition, path); err != nil {
		return err
	}
	if err := validateReasoningEffortFlags(definition, path); err != nil {
		return err
	}
	return validateReasoningLevels(definition, path)
}

func validateReasoningEffortFlags(definition ReasoningFamilyDefinition, path string) error {
	if len(definition.EffortFlags) == 0 {
		return nil
	}
	if definition.Type != "reasoning_effort" {
		return fmt.Errorf("%s.effort_flags requires reasoning_effort type", path)
	}
	if definition.ActivationParameter == "" {
		return fmt.Errorf("%s.effort_flags requires activation_parameter", path)
	}
	levels := make(map[string]struct{}, len(definition.Levels))
	for _, level := range definition.Levels {
		levels[level] = struct{}{}
	}
	seenParameters := make(map[string]struct{}, len(definition.EffortFlags))
	for effort, parameter := range definition.EffortFlags {
		if err := validateReasoningEffortFlag(definition, levels, seenParameters, effort, parameter, path); err != nil {
			return err
		}
		seenParameters[parameter] = struct{}{}
	}
	activeLevelCount := len(definition.Levels)
	if definition.Disabled != "" {
		if _, ok := levels[definition.Disabled]; ok {
			activeLevelCount--
		}
	}
	if activeLevelCount-len(definition.EffortFlags) > 1 {
		return fmt.Errorf("%s.effort_flags leaves multiple levels indistinguishable by omission", path)
	}
	return nil
}

func validateReasoningEffortFlag(
	definition ReasoningFamilyDefinition,
	levels map[string]struct{},
	seenParameters map[string]struct{},
	effort string,
	parameter string,
	path string,
) error {
	if _, ok := levels[effort]; !ok {
		return fmt.Errorf("%s.effort_flags key %q is not listed in levels", path, effort)
	}
	if strings.TrimSpace(parameter) == "" {
		return fmt.Errorf("%s.effort_flags[%q] cannot be blank", path, effort)
	}
	if parameter == definition.Parameter || parameter == definition.ActivationParameter {
		return fmt.Errorf("%s.effort_flags[%q] must differ from parameter and activation_parameter", path, effort)
	}
	if _, exists := seenParameters[parameter]; exists {
		return fmt.Errorf("%s.effort_flags contains duplicate parameter %q", path, parameter)
	}
	return nil
}

func validateReasoningFamilyType(familyType, path string) error {
	switch familyType {
	case "chat_template_kwargs", "reasoning_effort", "reasoning_mode", "top_level_reasoning_effort":
		return nil
	default:
		return fmt.Errorf("%s.type %q is unsupported", path, familyType)
	}
}

func validateReasoningActivationParameter(definition ReasoningFamilyDefinition, path string) error {
	if definition.ActivationParameter == "" {
		return nil
	}
	if strings.TrimSpace(definition.ActivationParameter) == "" {
		return fmt.Errorf("%s.activation_parameter cannot be blank", path)
	}
	if definition.ActivationParameter == definition.Parameter {
		return fmt.Errorf("%s.activation_parameter must differ from parameter", path)
	}
	if definition.Type != "reasoning_effort" {
		return fmt.Errorf("%s.activation_parameter requires reasoning_effort type", path)
	}
	return nil
}

func validateReasoningLevels(definition ReasoningFamilyDefinition, path string) error {
	if len(definition.Levels) == 0 {
		if definition.Type == "reasoning_effort" || definition.Type == "top_level_reasoning_effort" {
			return fmt.Errorf("%s.levels must be set for effort-based reasoning", path)
		}
		return validateReasoningModes(definition, path)
	}
	seen := map[string]struct{}{}
	for _, level := range definition.Levels {
		if level == "" {
			return fmt.Errorf("%s.levels cannot contain an empty value", path)
		}
		if _, exists := seen[level]; exists {
			return fmt.Errorf("%s.levels contains duplicate %q", path, level)
		}
		seen[level] = struct{}{}
	}
	if definition.Default != "" {
		if _, ok := seen[definition.Default]; !ok {
			return fmt.Errorf("%s.default %q is not listed in levels", path, definition.Default)
		}
	}
	return validateReasoningModes(definition, path)
}

func validateReasoningModes(definition ReasoningFamilyDefinition, path string) error {
	if len(definition.Modes) == 0 {
		// Operator-authored families predate the explicit mode contract. Keep an
		// entirely omitted mode set open-ended; a partially authored contract is
		// still rejected. Built-in resources are validated strictly by the
		// catalog generator before this compiler ever sees them.
		if definition.DefaultMode == "" {
			return nil
		}
		return fmt.Errorf("%s.modes cannot be empty when default_mode is set", path)
	}
	seen := make(map[string]struct{}, len(definition.Modes))
	for _, mode := range definition.Modes {
		switch mode {
		case "enabled", "disabled", "adaptive":
		default:
			return fmt.Errorf("%s.modes contains unsupported mode %q", path, mode)
		}
		if _, exists := seen[mode]; exists {
			return fmt.Errorf("%s.modes contains duplicate %q", path, mode)
		}
		seen[mode] = struct{}{}
	}
	if definition.DefaultMode == "" {
		return fmt.Errorf("%s.default_mode cannot be empty", path)
	}
	if _, ok := seen[definition.DefaultMode]; !ok {
		return fmt.Errorf("%s.default_mode %q is not listed in modes", path, definition.DefaultMode)
	}
	if definition.Disabled != "" {
		if _, ok := seen["disabled"]; !ok {
			return fmt.Errorf("%s.disabled requires disabled to be listed in modes", path)
		}
	}
	return nil
}

func (registry *Registry) compileCards(overlays []ModelCardOverlay, reasoning map[string]ReasoningFamilyDefinition) (map[string]EffectiveModelCard, error) {
	result := registry.builtinEffectiveCards(len(overlays))
	seen := map[string]struct{}{}
	for index, overlay := range overlays {
		path := fmt.Sprintf("model_cards[%d]", index)
		effective, err := compileCardOverlay(result, seen, overlay, path, reasoning)
		if err != nil {
			return nil, err
		}
		result[effective.Card.ID] = effective
	}
	return result, nil
}

func (registry *Registry) builtinEffectiveCards(extra int) map[string]EffectiveModelCard {
	result := make(map[string]EffectiveModelCard, len(registry.models)+extra)
	for id, card := range registry.models {
		result[id] = EffectiveModelCard{
			Card:       cloneModel(card),
			Provenance: builtinCardProvenance(card),
		}
	}
	return result
}

func compileCardOverlay(
	cards map[string]EffectiveModelCard,
	seen map[string]struct{},
	overlay ModelCardOverlay,
	path string,
	reasoning map[string]ReasoningFamilyDefinition,
) (EffectiveModelCard, error) {
	overlay.Name = strings.TrimSpace(overlay.Name)
	if overlay.Name == "" {
		return EffectiveModelCard{}, fmt.Errorf("%s.name cannot be empty", path)
	}
	if _, duplicate := seen[overlay.Name]; duplicate {
		return EffectiveModelCard{}, fmt.Errorf("%s.name %q is duplicated", path, overlay.Name)
	}
	seen[overlay.Name] = struct{}{}
	effective, builtin := cards[overlay.Name]
	if overlay.BuiltIn != nil {
		builtin = *overlay.BuiltIn
		if builtin {
			if _, exists := cards[overlay.Name]; !exists {
				return EffectiveModelCard{}, fmt.Errorf("%s.name %q does not reference a built-in model card", path, overlay.Name)
			}
		}
	}
	if !builtin {
		effective = newCustomEffectiveCard(overlay.Name)
	}
	if err := applyCardOverlay(&effective, overlay, builtin); err != nil {
		return EffectiveModelCard{}, fmt.Errorf("%s: %w", path, err)
	}
	if err := applyInlineReasoning(&effective, overlay, path, reasoning); err != nil {
		return EffectiveModelCard{}, err
	}
	if err := validateCardReasoning(effective, path, reasoning); err != nil {
		return EffectiveModelCard{}, err
	}
	ensureCustomCardVerification(&effective, builtin)
	if err := validateEffectiveCard(effective, path); err != nil {
		return EffectiveModelCard{}, err
	}
	return effective, nil
}

func newCustomEffectiveCard(name string) EffectiveModelCard {
	return EffectiveModelCard{
		Card: ModelCard{
			ID:           name,
			DisplayName:  name,
			Kind:         "physical",
			Lifecycle:    "active",
			Capabilities: []string{"chat"},
			Modalities:   Modalities{Input: []string{"text"}, Output: []string{"text"}},
		},
		// Only the identity is operator-authored. The remaining values are
		// internal defaults that make a sparse custom card executable.
		Provenance: FieldProvenance{"id": SourceOperator},
	}
}

func applyInlineReasoning(
	effective *EffectiveModelCard,
	overlay ModelCardOverlay,
	path string,
	reasoning map[string]ReasoningFamilyDefinition,
) error {
	if overlay.Reasoning == nil {
		return nil
	}
	inlineID := "operator/" + sanitizeID(overlay.Name) + "-reasoning"
	definition := cloneReasoningFamily(*overlay.Reasoning)
	definition.ID = inlineID
	if err := validateReasoningFamily(definition, path+".reasoning"); err != nil {
		return err
	}
	reasoning[inlineID] = definition
	effective.Card.ReasoningFamily = inlineID
	effective.Provenance["reasoning_family"] = SourceOperator
	return nil
}

func validateCardReasoning(
	effective EffectiveModelCard,
	path string,
	reasoning map[string]ReasoningFamilyDefinition,
) error {
	if effective.Card.ReasoningFamily == "" {
		return nil
	}
	if _, ok := reasoning[effective.Card.ReasoningFamily]; !ok {
		return fmt.Errorf("%s.reasoning_family %q is unknown", path, effective.Card.ReasoningFamily)
	}
	return nil
}

func ensureCustomCardVerification(effective *EffectiveModelCard, builtin bool) {
	if builtin || effective.Card.Verification.Status != "" {
		return
	}
	effective.Card.Verification = ModelVerification{Authority: "operator", Status: "claimed"}
	effective.Provenance["verification"] = SourceOperator
}

func applyCardOverlay(effective *EffectiveModelCard, overlay ModelCardOverlay, builtin bool) error {
	applyCardScalarOverlay(effective, overlay)
	if err := applyCardLimitOverlay(effective, overlay); err != nil {
		return err
	}
	if err := applyCardClaimOverlay(effective, overlay, builtin); err != nil {
		return err
	}
	applyCardMetadataOverlay(effective, overlay)
	return nil
}

func applyCardScalarOverlay(effective *EffectiveModelCard, overlay ModelCardOverlay) {
	card := &effective.Card
	setOverlayString(effective, "display_name", overlay.DisplayName, &card.DisplayName)
	setOverlayString(effective, "description", overlay.Description, &card.Description)
	setOverlayString(effective, "publisher", overlay.Publisher, &card.Publisher)
	setOverlayString(effective, "family", overlay.Family, &card.Family)
	setOverlayString(effective, "parameter_size", overlay.ParameterSize, &card.ParameterSize)
	setOverlayString(effective, "revision", overlay.Revision, &card.Revision)
	setOverlayString(effective, "released_at", overlay.ReleasedAt, &card.ReleasedAt)
	setOverlayString(effective, "knowledge_cutoff", overlay.KnowledgeCutoff, &card.KnowledgeCutoff)
	setOverlayString(effective, "lifecycle", overlay.Lifecycle, &card.Lifecycle)
	setOverlayString(effective, "reasoning_family", overlay.ReasoningFamily, &card.ReasoningFamily)
}

func setOverlayString(effective *EffectiveModelCard, name string, value *string, target *string) {
	if value == nil {
		return
	}
	*target = *value
	effective.Provenance[name] = SourceOperator
}

func applyCardLimitOverlay(effective *EffectiveModelCard, overlay ModelCardOverlay) error {
	if overlay.ContextWindowSize != nil {
		if *overlay.ContextWindowSize <= 0 {
			return fmt.Errorf("context_window_size must be positive")
		}
		effective.Card.Limits.ContextWindowSize = *overlay.ContextWindowSize
		effective.Provenance["limits.context_window_size"] = SourceOperator
	}
	if overlay.MaxOutputTokens != nil {
		if *overlay.MaxOutputTokens <= 0 {
			return fmt.Errorf("max_output_tokens must be positive")
		}
		effective.Card.Limits.MaxOutputTokens = *overlay.MaxOutputTokens
		effective.Provenance["limits.max_output_tokens"] = SourceOperator
	}
	return nil
}

func applyCardClaimOverlay(effective *EffectiveModelCard, overlay ModelCardOverlay, builtin bool) error {
	if overlay.Capabilities != nil {
		if builtin && widensStrings(effective.Card.Capabilities, *overlay.Capabilities) && !operatorVerified(overlay.Verification) {
			return fmt.Errorf("capabilities widen a built-in claim without verification.status=reproduced")
		}
		effective.Card.Capabilities = uniqueStrings(*overlay.Capabilities)
		effective.Provenance["capabilities"] = SourceOperator
	}
	if overlay.Modalities != nil {
		widens := widensStrings(effective.Card.Modalities.Input, overlay.Modalities.Input) ||
			widensStrings(effective.Card.Modalities.Output, overlay.Modalities.Output)
		if builtin && widens && !operatorVerified(overlay.Verification) {
			return fmt.Errorf("modalities widen a built-in claim without verification.status=reproduced")
		}
		effective.Card.Modalities = Modalities{Input: uniqueStrings(overlay.Modalities.Input), Output: uniqueStrings(overlay.Modalities.Output)}
		effective.Provenance["modalities"] = SourceOperator
	}
	return nil
}

func applyCardMetadataOverlay(effective *EffectiveModelCard, overlay ModelCardOverlay) {
	if overlay.Presentation != nil {
		effective.Card.Presentation = *overlay.Presentation
		effective.Provenance["presentation"] = SourceOperator
	}
	if overlay.Distribution != nil {
		effective.Card.Distribution = *overlay.Distribution
		effective.Provenance["distribution"] = SourceOperator
	}
	if overlay.Tags != nil {
		effective.Card.Tags = uniqueStrings(*overlay.Tags)
		effective.Provenance["tags"] = SourceOperator
	}
	if overlay.LoRAs != nil {
		effective.LoRAs = append([]LoRAAdapter(nil), (*overlay.LoRAs)...)
		effective.Provenance["loras"] = SourceOperator
	}
	if overlay.RuntimeModality != nil {
		effective.RuntimeModality = *overlay.RuntimeModality
		effective.Provenance["runtime_modality"] = SourceOperator
	}
	if overlay.Verification != nil {
		effective.Card.Verification = *overlay.Verification
		effective.Provenance["verification"] = SourceOperator
	}
}

func validateEffectiveCard(effective EffectiveModelCard, path string) error {
	card := effective.Card
	if strings.TrimSpace(card.DisplayName) == "" {
		return fmt.Errorf("%s.display_name cannot be empty", path)
	}
	if len(card.Capabilities) == 0 {
		return fmt.Errorf("%s.capabilities cannot be empty", path)
	}
	if len(card.Modalities.Input) == 0 || len(card.Modalities.Output) == 0 {
		return fmt.Errorf("%s.modalities.input and output cannot be empty", path)
	}
	if err := validateOperatorPresentation(card, effective.Provenance, path); err != nil {
		return err
	}
	return validateOperatorDistribution(card, effective.Provenance, path)
}

func validateOperatorPresentation(card ModelCard, provenance FieldProvenance, path string) error {
	if provenance["presentation"] != SourceOperator {
		return nil
	}
	if strings.TrimSpace(card.Presentation.Logo) == "" || strings.TrimSpace(card.Presentation.Monogram) == "" {
		return fmt.Errorf("%s.presentation.logo and monogram cannot be empty", path)
	}
	return nil
}

func validateOperatorDistribution(card ModelCard, provenance FieldProvenance, path string) error {
	if provenance["distribution"] != SourceOperator {
		return nil
	}
	switch card.Distribution.Type {
	case "proprietary_api", "router_recipe":
	case "open_weights":
		if strings.TrimSpace(card.Distribution.License) == "" {
			return fmt.Errorf("%s.distribution.license cannot be empty for open_weights", path)
		}
	default:
		return fmt.Errorf("%s.distribution.type must be proprietary_api, open_weights, or router_recipe", path)
	}
	if strings.TrimSpace(card.Distribution.Source) == "" {
		return fmt.Errorf("%s.distribution.source cannot be empty", path)
	}
	return nil
}

func operatorVerified(verification *ModelVerification) bool {
	return verification != nil && verification.Status == "reproduced" && strings.TrimSpace(verification.Authority) != ""
}

func widensStrings(existing, replacement []string) bool {
	set := map[string]struct{}{}
	for _, item := range existing {
		set[item] = struct{}{}
	}
	for _, item := range replacement {
		if _, ok := set[item]; !ok {
			return true
		}
	}
	return false
}

func uniqueStrings(values []string) []string {
	seen := map[string]struct{}{}
	result := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	return result
}

func sanitizeID(value string) string {
	replacer := strings.NewReplacer("/", "-", "@", "-", ":", "-", ".", "-")
	return strings.ToLower(replacer.Replace(value))
}

func builtinCardProvenance(card ModelCard) FieldProvenance {
	result := FieldProvenance{}
	for _, field := range []string{"id", "display_name", "description", "kind", "publisher", "presentation", "distribution", "family", "parameter_size", "revision", "released_at", "knowledge_cutoff", "lifecycle", "limits.context_window_size", "limits.max_output_tokens", "capabilities", "modalities", "reasoning_family", "tags", "verification"} {
		result[field] = SourceBuiltin
	}
	return result
}

func (registry *Registry) compileModels(inputs []ModelAlias, providers map[string]EffectiveProvider, cards map[string]EffectiveModelCard, reasoning map[string]ReasoningFamilyDefinition, results map[string]map[string]map[string]IndexResult) (map[string]EffectiveModel, error) {
	models := make(map[string]EffectiveModel, len(inputs))
	for index, input := range inputs {
		path := fmt.Sprintf("models[%d]", index)
		effective, err := registry.compileModel(input, path, providers, cards, results, reasoning, models)
		if err != nil {
			return nil, err
		}
		models[effective.Alias] = effective
	}
	return models, nil
}

func (registry *Registry) compileModel(
	input ModelAlias,
	path string,
	providers map[string]EffectiveProvider,
	cards map[string]EffectiveModelCard,
	results map[string]map[string]map[string]IndexResult,
	reasoning map[string]ReasoningFamilyDefinition,
	existing map[string]EffectiveModel,
) (EffectiveModel, error) {
	input.Name = strings.TrimSpace(input.Name)
	input.Catalog = strings.TrimSpace(input.Catalog)
	if input.Name == "" || input.Catalog == "" {
		return EffectiveModel{}, fmt.Errorf("%s.name and catalog are required", path)
	}
	if _, exists := existing[input.Name]; exists {
		return EffectiveModel{}, fmt.Errorf("%s.name %q is duplicated", path, input.Name)
	}
	card, ok := cards[input.Catalog]
	if !ok {
		return EffectiveModel{}, fmt.Errorf("%s.catalog %q has no built-in or handwritten model card", path, input.Catalog)
	}
	effectiveProviders, err := registry.compileModelProviders(input.Providers, path, providers, card)
	if err != nil {
		return EffectiveModel{}, err
	}
	indicesByEffort := cloneIndexResultsByEffort(results[input.Catalog])
	preferredEffort := preferredEvaluationEffort(card, indicesByEffort, reasoning)
	return EffectiveModel{
		Alias:           input.Name,
		Catalog:         input.Catalog,
		Card:            cloneEffectiveCard(card),
		Providers:       effectiveProviders,
		Indices:         cloneIndexResults(indicesByEffort[preferredEffort]),
		IndicesByEffort: indicesByEffort,
		BindingDefaults: input.BindingDefaults,
	}, nil
}
