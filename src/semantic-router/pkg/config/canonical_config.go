package config

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// CanonicalConfig is the public v0.3 config contract.
type CanonicalConfig struct {
	Version   string             `yaml:"version,omitempty"`
	Listeners []Listener         `yaml:"listeners,omitempty"`
	Providers CanonicalProviders `yaml:"providers,omitempty"`
	Routing   CanonicalRouting   `yaml:"routing,omitempty"`
	Global    *CanonicalGlobal   `yaml:"global,omitempty"`

	globalOverrideRaw *StructuredPayload `yaml:"-"`
}

// CanonicalRouting contains the DSL-owned routing surface.
type CanonicalRouting struct {
	ModelCards  []RoutingModel       `yaml:"modelCards,omitempty"`
	Signals     CanonicalSignals     `yaml:"signals,omitempty"`
	Projections CanonicalProjections `yaml:"projections,omitempty"`
	Decisions   []Decision           `yaml:"decisions,omitempty"`
}

// CanonicalSignals groups routing signals under routing.signals.
type CanonicalSignals struct {
	Keywords      []KeywordRule      `yaml:"keywords,omitempty"`
	Embeddings    []EmbeddingRule    `yaml:"embeddings,omitempty"`
	Domains       []Category         `yaml:"domains,omitempty"`
	FactCheck     []FactCheckRule    `yaml:"fact_check,omitempty"`
	UserFeedbacks []UserFeedbackRule `yaml:"user_feedbacks,omitempty"`
	Preferences   []PreferenceRule   `yaml:"preferences,omitempty"`
	Language      []LanguageRule     `yaml:"language,omitempty"`
	Context       []ContextRule      `yaml:"context,omitempty"`
	Structure     []StructureRule    `yaml:"structure,omitempty"`
	Complexity    []ComplexityRule   `yaml:"complexity,omitempty"`
	Modality      []ModalityRule     `yaml:"modality,omitempty"`
	RoleBindings  []RoleBinding      `yaml:"role_bindings,omitempty"`
	Jailbreak     []JailbreakRule    `yaml:"jailbreak,omitempty"`
	PII           []PIIRule          `yaml:"pii,omitempty"`
	KB            []KBSignalRule     `yaml:"kb,omitempty"`
}

// CanonicalProjections groups derived routing outputs under routing.projections.
type CanonicalProjections struct {
	Partitions []ProjectionPartition `yaml:"partitions,omitempty"`
	Scores     []ProjectionScore     `yaml:"scores,omitempty"`
	Mappings   []ProjectionMapping   `yaml:"mappings,omitempty"`
}

// RoutingModel defines the logical model catalog available to routing decisions.
type RoutingModel struct {
	Name              string        `yaml:"name"`
	ParamSize         string        `yaml:"param_size,omitempty"`
	ContextWindowSize int           `yaml:"context_window_size,omitempty"`
	Description       string        `yaml:"description,omitempty"`
	Capabilities      []string      `yaml:"capabilities,omitempty"`
	LoRAs             []LoRAAdapter `yaml:"loras,omitempty"`
	QualityScore      float64       `yaml:"quality_score,omitempty"`
	Modality          string        `yaml:"modality,omitempty"`
	Tags              []string      `yaml:"tags,omitempty"`
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

	global, err := resolveCanonicalGlobal(canonical.Global, canonical.globalOverrideRaw)
	if err != nil {
		return nil, err
	}

	cfg := DefaultGlobalConfig()
	if applyErr := applyCanonicalGlobal(&cfg, &global); applyErr != nil {
		return nil, applyErr
	}

	applyCanonicalRoutingState(&cfg, canonical)
	if err := applyCanonicalProviderState(&cfg, canonical.Providers); err != nil {
		return nil, err
	}

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
	cfg.ModelConfig = make(map[string]ModelParams)

	for _, model := range canonicalRoutingModels(canonical.Routing) {
		cfg.ModelConfig[model.Name] = ModelParams{
			ParamSize:         model.ParamSize,
			ContextWindowSize: model.ContextWindowSize,
			Description:       model.Description,
			Capabilities:      append([]string(nil), model.Capabilities...),
			LoRAs:             copyLoRAAdapters(model.LoRAs),
			Tags:              append([]string(nil), model.Tags...),
			QualityScore:      model.QualityScore,
			Modality:          model.Modality,
		}
	}
}

func applyCanonicalProviderState(cfg *RouterConfig, providers CanonicalProviders) error {
	providerDefaults := canonicalProviderDefaults(providers)
	cfg.DefaultModel = providerDefaults.DefaultModel
	cfg.DefaultReasoningEffort = providerDefaults.DefaultReasoningEffort
	cfg.ReasoningFamilies = copyReasoningFamilies(providerDefaults.ReasoningFamilies)

	profiles, endpoints, modelParams, err := normalizeCanonicalProviderModels(providers.Models)
	if err != nil {
		return err
	}
	cfg.ProviderProfiles = profiles
	cfg.VLLMEndpoints = endpoints
	mergeCanonicalProviderModelParams(cfg.ModelConfig, modelParams)
	return nil
}

func mergeCanonicalProviderModelParams(modelConfig map[string]ModelParams, modelParams map[string]ModelParams) {
	for modelName, providerParams := range modelParams {
		params := modelConfig[modelName]
		if params.Pricing == (ModelPricing{}) {
			params.Pricing = providerParams.Pricing
		}
		if params.ReasoningFamily == "" {
			params.ReasoningFamily = providerParams.ReasoningFamily
		}
		if len(providerParams.PreferredEndpoints) > 0 {
			params.PreferredEndpoints = append([]string(nil), providerParams.PreferredEndpoints...)
		}
		if params.AccessKey == "" {
			params.AccessKey = providerParams.AccessKey
		}
		if len(params.ExternalModelIDs) == 0 {
			params.ExternalModelIDs = copyStringMap(providerParams.ExternalModelIDs)
		}
		if params.APIFormat == "" {
			params.APIFormat = providerParams.APIFormat
		}
		modelConfig[modelName] = params
	}
}

func validateCanonicalContract(canonical *CanonicalConfig) error {
	modelCards := canonicalRoutingModels(canonical.Routing)
	modelsByName := make(map[string]RoutingModel, len(modelCards))
	for _, model := range modelCards {
		if model.Name == "" {
			return fmt.Errorf("routing.modelCards.name cannot be empty")
		}
		if _, exists := modelsByName[model.Name]; exists {
			return fmt.Errorf("routing.modelCards[%s]: duplicate model name", model.Name)
		}
		modelsByName[model.Name] = model
	}

	if canonicalProviderDefaults(canonical.Providers).DefaultModel != "" {
		if !canonicalModelOrLoRAExists(
			modelsByName,
			modelCards,
			canonicalProviderDefaults(canonical.Providers).DefaultModel,
		) {
			return fmt.Errorf(
				"providers.defaults.default_model %q not found in routing.modelCards or routing.modelCards[].loras",
				canonicalProviderDefaults(canonical.Providers).DefaultModel,
			)
		}
	}

	for _, model := range canonical.Providers.Models {
		if model.Name == "" {
			return fmt.Errorf("providers.models.name cannot be empty")
		}
		if _, ok := modelsByName[model.Name]; !ok {
			return fmt.Errorf("providers.models[%s] does not match any routing.modelCards entry", model.Name)
		}
		if model.ReasoningFamily != "" {
			if _, ok := canonicalProviderDefaults(canonical.Providers).ReasoningFamilies[model.ReasoningFamily]; !ok {
				return fmt.Errorf("providers.models[%s].reasoning_family %q not found in providers.defaults.reasoning_families", model.Name, model.ReasoningFamily)
			}
		}
		if len(canonicalBackendRefs(model)) == 0 {
			if !canonicalProviderModelHasMetadata(model) {
				return fmt.Errorf("providers.models[%s] must define backend_refs or provider metadata such as reasoning_family, pricing, api_format, external_model_ids, or provider_model_id", model.Name)
			}
			continue
		}
		for _, backendRef := range canonicalBackendRefs(model) {
			if strings.TrimSpace(backendRef.Endpoint) == "" && strings.TrimSpace(backendRef.BaseURL) == "" {
				return fmt.Errorf("providers.models[%s].backend_refs requires endpoint or base_url", model.Name)
			}
		}
	}

	for _, decision := range canonical.Routing.Decisions {
		for _, modelRef := range decision.ModelRefs {
			if modelRef.Model == "" || modelRef.LoRAName == "" {
				continue
			}
			card, ok := modelsByName[modelRef.Model]
			if !ok {
				return fmt.Errorf("routing.decisions[%s].modelRefs[%s] references unknown model %q", decision.Name, modelRef.Model, modelRef.Model)
			}
			if !routingModelHasLoRA(card, modelRef.LoRAName) {
				return fmt.Errorf("routing.decisions[%s].modelRefs[%s].lora_name %q not found in routing.modelCards[%s].loras", decision.Name, modelRef.Model, modelRef.LoRAName, modelRef.Model)
			}
		}
	}

	return nil
}

func normalizeSignals(signals CanonicalSignals, decisions []Decision) Signals {
	result := Signals{
		KeywordRules:      append([]KeywordRule(nil), signals.Keywords...),
		EmbeddingRules:    append([]EmbeddingRule(nil), signals.Embeddings...),
		Categories:        append([]Category(nil), signals.Domains...),
		FactCheckRules:    append([]FactCheckRule(nil), signals.FactCheck...),
		UserFeedbackRules: append([]UserFeedbackRule(nil), signals.UserFeedbacks...),
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
	if model.ReasoningFamily != "" || model.ProviderModelID != "" || model.APIFormat != "" || len(model.ExternalModelIDs) > 0 {
		return true
	}
	return model.Pricing != (ModelPricing{})
}

func normalizeCanonicalProviderModels(models []CanonicalProviderModel) (map[string]ProviderProfile, []VLLMEndpoint, map[string]ModelParams, error) {
	if len(models) == 0 {
		return nil, nil, nil, nil
	}

	profiles := map[string]ProviderProfile{}
	endpoints := []VLLMEndpoint{}
	modelParams := make(map[string]ModelParams, len(models))

	for _, model := range models {
		params := modelParams[model.Name]
		params.ReasoningFamily = model.ReasoningFamily
		params.Pricing = model.Pricing
		params.APIFormat = model.APIFormat
		params.ExternalModelIDs = normalizeExternalModelIDsFromProviderModel(model)

		backendRefs := canonicalBackendRefs(model)
		params.PreferredEndpoints = make([]string, 0, len(backendRefs))
		for index, backendRef := range backendRefs {
			endpointName := canonicalEndpointName(model.Name, backendRef, index)
			endpoint := VLLMEndpoint{
				Name:     endpointName,
				Weight:   backendRef.Weight,
				Type:     backendRef.Type,
				Protocol: defaultProtocol(backendRef.Protocol),
				Model:    model.Name,
				APIKey:   resolveBackendAPIKey(backendRef),
			}
			if endpoint.Weight == 0 {
				endpoint.Weight = 1
			}

			if backendRef.BaseURL != "" || backendRef.Provider != "" || backendRef.AuthHeader != "" || backendRef.AuthPrefix != "" || backendRef.ChatPath != "" || len(backendRef.ExtraHeaders) > 0 || backendRef.APIVersion != "" {
				profiles[endpointName] = ProviderProfile{
					Type:         backendRef.Provider,
					BaseURL:      backendRef.BaseURL,
					AuthHeader:   backendRef.AuthHeader,
					AuthPrefix:   backendRef.AuthPrefix,
					ExtraHeaders: copyStringMap(backendRef.ExtraHeaders),
					APIVersion:   backendRef.APIVersion,
					ChatPath:     backendRef.ChatPath,
				}
				endpoint.ProviderProfileName = endpointName
			}

			if backendRef.Endpoint != "" {
				address, port, err := splitEndpointAddress(backendRef.Endpoint, backendRef.Protocol)
				if err != nil {
					return nil, nil, nil, fmt.Errorf("providers.models[%s].backend_refs[%d]: %w", model.Name, index, err)
				}
				endpoint.Address = address
				endpoint.Port = port
			}

			params.PreferredEndpoints = append(params.PreferredEndpoints, endpointName)
			if params.AccessKey == "" {
				params.AccessKey = endpoint.APIKey
			}
			endpoints = append(endpoints, endpoint)
		}

		modelParams[model.Name] = params
	}

	if len(profiles) == 0 {
		profiles = nil
	}
	return profiles, endpoints, modelParams, nil
}

func normalizeExternalModelIDsFromProviderModel(model CanonicalProviderModel) map[string]string {
	if len(model.ExternalModelIDs) > 0 {
		return copyStringMap(model.ExternalModelIDs)
	}
	if model.ProviderModelID == "" {
		return nil
	}

	result := map[string]string{}
	for _, backendRef := range canonicalBackendRefs(model) {
		key := backendRef.Type
		if key == "" {
			key = backendRef.Provider
		}
		if key == "" {
			key = "vllm"
		}
		result[key] = model.ProviderModelID
	}
	if len(result) == 0 {
		return nil
	}
	return result
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

func splitEndpointAddress(raw string, protocol string) (string, int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", 0, fmt.Errorf("endpoint is required")
	}

	if strings.Contains(raw, "://") {
		parts := strings.SplitN(raw, "://", 2)
		protocol = parts[0]
		raw = parts[1]
	}
	if slash := strings.Index(raw, "/"); slash >= 0 {
		raw = raw[:slash]
	}

	host := raw
	port := 0
	if strings.Count(raw, ":") > 1 && !strings.HasPrefix(raw, "[") {
		// Bare IPv6 literal without explicit port.
		host = raw
	} else if strings.Contains(raw, ":") {
		lastColon := strings.LastIndex(raw, ":")
		if lastColon > 0 && lastColon < len(raw)-1 {
			if parsedPort, err := strconv.Atoi(raw[lastColon+1:]); err == nil {
				host = raw[:lastColon]
				port = parsedPort
			}
		}
	}

	if strings.HasPrefix(host, "[") && strings.HasSuffix(host, "]") {
		host = strings.TrimPrefix(strings.TrimSuffix(host, "]"), "[")
	}

	if port == 0 {
		if strings.EqualFold(protocol, "https") {
			port = 443
		} else {
			port = 80
		}
	}

	if host == "" {
		return "", 0, fmt.Errorf("endpoint host cannot be empty")
	}
	return host, port, nil
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

func copyReasoningFamilies(input map[string]ReasoningFamilyConfig) map[string]ReasoningFamilyConfig {
	if len(input) == 0 {
		return nil
	}
	output := make(map[string]ReasoningFamilyConfig, len(input))
	for key, value := range input {
		output[key] = value
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
