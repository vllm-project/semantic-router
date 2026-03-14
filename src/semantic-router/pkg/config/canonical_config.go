package config

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"gopkg.in/yaml.v2"
)

// CanonicalConfig is the public v0.3 config contract.
type CanonicalConfig struct {
	Version   string             `yaml:"version,omitempty"`
	Listeners []Listener         `yaml:"listeners,omitempty"`
	Providers CanonicalProviders `yaml:"providers,omitempty"`
	Routing   CanonicalRouting   `yaml:"routing,omitempty"`
	Global    *CanonicalGlobal   `yaml:"global,omitempty"`
}

// CanonicalRouting contains the DSL-owned routing surface.
type CanonicalRouting struct {
	ModelCards []RoutingModel   `yaml:"modelCards,omitempty"`
	Signals    CanonicalSignals `yaml:"signals,omitempty"`
	Decisions  []Decision       `yaml:"decisions,omitempty"`
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
	Complexity    []ComplexityRule   `yaml:"complexity,omitempty"`
	Modality      []ModalityRule     `yaml:"modality,omitempty"`
	RoleBindings  []RoleBinding      `yaml:"role_bindings,omitempty"`
	Jailbreak     []JailbreakRule    `yaml:"jailbreak,omitempty"`
	PII           []PIIRule          `yaml:"pii,omitempty"`
}

// RoutingModel defines the logical model catalog available to routing decisions.
type RoutingModel struct {
	Name               string   `yaml:"name"`
	ReasoningFamilyRef string   `yaml:"reasoning_family_ref,omitempty"`
	ParamSize          string   `yaml:"param_size,omitempty"`
	ContextWindowSize  int      `yaml:"context_window_size,omitempty"`
	Description        string   `yaml:"description,omitempty"`
	Capabilities       []string `yaml:"capabilities,omitempty"`
	QualityScore       float64  `yaml:"quality_score,omitempty"`
	Modality           string   `yaml:"modality,omitempty"`
	Tags               []string `yaml:"tags,omitempty"`
}

// CanonicalProviders holds deployment bindings and provider defaults.
type CanonicalProviders struct {
	DefaultModel           string                           `yaml:"default_model,omitempty"`
	ReasoningFamilies      map[string]ReasoningFamilyConfig `yaml:"reasoning_families,omitempty"`
	DefaultReasoningEffort string                           `yaml:"default_reasoning_effort,omitempty"`
	Models                 []CanonicalProviderModel         `yaml:"models,omitempty"`
}

// CanonicalProviderModel binds a logical routing model to concrete access details.
type CanonicalProviderModel struct {
	Name             string                `yaml:"name"`
	ProviderModelID  string                `yaml:"provider_model_id,omitempty"`
	BackendRefs      []CanonicalBackendRef `yaml:"backend_refs,omitempty"`
	Pricing          ModelPricing          `yaml:"pricing,omitempty"`
	APIFormat        string                `yaml:"api_format,omitempty"`
	ExternalModelIDs map[string]string     `yaml:"external_model_ids,omitempty"`
}

// CanonicalBackendRef defines one physical backend target for a provider model.
type CanonicalBackendRef struct {
	Name         string            `yaml:"name,omitempty"`
	Endpoint     string            `yaml:"endpoint,omitempty"`
	Protocol     string            `yaml:"protocol,omitempty"`
	Weight       int               `yaml:"weight,omitempty"`
	Type         string            `yaml:"type,omitempty"`
	BaseURL      string            `yaml:"base_url,omitempty"`
	Provider     string            `yaml:"provider,omitempty"`
	AuthHeader   string            `yaml:"auth_header,omitempty"`
	AuthPrefix   string            `yaml:"auth_prefix,omitempty"`
	ExtraHeaders map[string]string `yaml:"extra_headers,omitempty"`
	APIVersion   string            `yaml:"api_version,omitempty"`
	ChatPath     string            `yaml:"chat_path,omitempty"`
	APIKey       string            `yaml:"api_key,omitempty"`
	APIKeyEnv    string            `yaml:"api_key_env,omitempty"`
}

// CanonicalGlobal contains router-managed runtime defaults plus sparse overrides.
type CanonicalGlobal struct {
	Strategy                  string `yaml:"strategy,omitempty"`
	AutoModelName             string `yaml:"auto_model_name,omitempty"`
	IncludeConfigModelsInList bool   `yaml:"include_config_models_in_list,omitempty"`
	ClearRouteCache           bool   `yaml:"clear_route_cache,omitempty"`
	StreamedBodyMode          bool   `yaml:"streamed_body_mode,omitempty"`
	MaxStreamedBodyBytes      int64  `yaml:"max_streamed_body_bytes,omitempty"`
	StreamedBodyTimeoutSec    int    `yaml:"streamed_body_timeout_sec,omitempty"`
	InlineModels              `yaml:",inline"`
	ExternalModels            []ExternalModelConfig `yaml:"external_models,omitempty"`
	SemanticCache             SemanticCache         `yaml:"semantic_cache,omitempty"`
	Memory                    MemoryConfig          `yaml:"memory,omitempty"`
	VectorStore               *VectorStoreConfig    `yaml:"vector_store,omitempty"`
	ResponseAPI               ResponseAPIConfig     `yaml:"response_api,omitempty"`
	RouterReplay              RouterReplayConfig    `yaml:"router_replay,omitempty"`
	Looper                    LooperConfig          `yaml:"looper,omitempty"`
	API                       APIConfig             `yaml:"api,omitempty"`
	Observability             ObservabilityConfig   `yaml:"observability,omitempty"`
	Tools                     ToolsConfig           `yaml:"tools,omitempty"`
	Authz                     AuthzConfig           `yaml:"authz,omitempty"`
	RateLimit                 RateLimitConfig       `yaml:"ratelimit,omitempty"`
	ModelSelection            ModelSelectionConfig  `yaml:"model_selection,omitempty"`
	SystemModels              CanonicalSystemModels `yaml:"system_models,omitempty"`
}

// CanonicalSystemModels centralizes stable capability bindings for built-in models.
type CanonicalSystemModels struct {
	PromptGuard            string `yaml:"prompt_guard,omitempty"`
	DomainClassifier       string `yaml:"domain_classifier,omitempty"`
	PIIClassifier          string `yaml:"pii_classifier,omitempty"`
	FactCheckClassifier    string `yaml:"fact_check_classifier,omitempty"`
	HallucinationDetector  string `yaml:"hallucination_detector,omitempty"`
	HallucinationExplainer string `yaml:"hallucination_explainer,omitempty"`
	FeedbackDetector       string `yaml:"feedback_detector,omitempty"`
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

	global, err := resolveCanonicalGlobal(canonical.Global)
	if err != nil {
		return nil, err
	}

	cfg := DefaultGlobalConfig()
	if applyErr := applyCanonicalGlobal(&cfg, &global); applyErr != nil {
		return nil, applyErr
	}

	cfg.Listeners = append([]Listener(nil), canonical.Listeners...)
	cfg.Decisions = copyDecisions(canonical.Routing.Decisions)
	ensureModelRefDefaults(cfg.Decisions)
	cfg.Signals = normalizeSignals(canonical.Routing.Signals, cfg.Decisions)

	cfg.DefaultModel = canonical.Providers.DefaultModel
	cfg.DefaultReasoningEffort = canonical.Providers.DefaultReasoningEffort
	cfg.ReasoningFamilies = copyReasoningFamilies(canonical.Providers.ReasoningFamilies)
	cfg.ModelConfig = make(map[string]ModelParams)

	for _, model := range canonicalRoutingModels(canonical.Routing) {
		cfg.ModelConfig[model.Name] = ModelParams{
			ReasoningFamily:   model.ReasoningFamilyRef,
			ParamSize:         model.ParamSize,
			ContextWindowSize: model.ContextWindowSize,
			Description:       model.Description,
			Capabilities:      append([]string(nil), model.Capabilities...),
			Tags:              append([]string(nil), model.Tags...),
			QualityScore:      model.QualityScore,
			Modality:          model.Modality,
		}
	}

	profiles, endpoints, modelParams, err := normalizeCanonicalProviderModels(canonical.Providers.Models)
	if err != nil {
		return nil, err
	}
	cfg.ProviderProfiles = profiles
	cfg.VLLMEndpoints = endpoints
	for modelName, providerParams := range modelParams {
		params := cfg.ModelConfig[modelName]
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
		cfg.ModelConfig[modelName] = params
	}

	if cfg.VectorStore != nil {
		cfg.VectorStore.ApplyDefaults()
	}

	return &cfg, nil
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
		if model.ReasoningFamilyRef != "" {
			if _, ok := canonical.Providers.ReasoningFamilies[model.ReasoningFamilyRef]; !ok {
				return fmt.Errorf("routing.modelCards[%s].reasoning_family_ref %q not found in providers.reasoning_families", model.Name, model.ReasoningFamilyRef)
			}
		}
	}

	if canonical.Providers.DefaultModel != "" {
		if _, ok := modelsByName[canonical.Providers.DefaultModel]; !ok {
			return fmt.Errorf("providers.default_model %q not found in routing.modelCards", canonical.Providers.DefaultModel)
		}
	}

	for _, model := range canonical.Providers.Models {
		if model.Name == "" {
			return fmt.Errorf("providers.models.name cannot be empty")
		}
		if _, ok := modelsByName[model.Name]; !ok {
			return fmt.Errorf("providers.models[%s] does not match any routing.modelCards entry", model.Name)
		}
		if len(canonicalBackendRefs(model)) == 0 {
			return fmt.Errorf("providers.models[%s].backend_refs cannot be empty", model.Name)
		}
		for _, backendRef := range canonicalBackendRefs(model) {
			if strings.TrimSpace(backendRef.Endpoint) == "" && strings.TrimSpace(backendRef.BaseURL) == "" {
				return fmt.Errorf("providers.models[%s].backend_refs requires endpoint or base_url", model.Name)
			}
		}
	}

	return nil
}

func resolveCanonicalGlobal(override *CanonicalGlobal) (CanonicalGlobal, error) {
	defaults := DefaultCanonicalGlobal()
	if override == nil {
		return defaults, nil
	}

	resolved := defaults
	overrideBytes, err := yaml.Marshal(override)
	if err != nil {
		return CanonicalGlobal{}, fmt.Errorf("failed to marshal global override: %w", err)
	}
	if err := yaml.Unmarshal(overrideBytes, &resolved); err != nil {
		return CanonicalGlobal{}, fmt.Errorf("failed to merge global override: %w", err)
	}

	applySystemModelOverrides(&resolved, &defaults)
	return resolved, nil
}

func applyCanonicalGlobal(cfg *RouterConfig, global *CanonicalGlobal) error {
	if global == nil {
		return nil
	}
	data, err := yaml.Marshal(global)
	if err != nil {
		return fmt.Errorf("failed to marshal canonical global config: %w", err)
	}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return fmt.Errorf("failed to apply canonical global config: %w", err)
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
		ComplexityRules:   append([]ComplexityRule(nil), signals.Complexity...),
		ModalityRules:     append([]ModalityRule(nil), signals.Modality...),
		RoleBindings:      append([]RoleBinding(nil), signals.RoleBindings...),
		JailbreakRules:    append([]JailbreakRule(nil), signals.Jailbreak...),
		PIIRules:          append([]PIIRule(nil), signals.PII...),
	}

	if len(result.Categories) == 0 {
		result.Categories = autoGenerateCategoriesFromDecisions(decisions)
	}

	return result
}

func canonicalRoutingModels(routing CanonicalRouting) []RoutingModel {
	return routing.ModelCards
}

func canonicalBackendRefs(model CanonicalProviderModel) []CanonicalBackendRef {
	return model.BackendRefs
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

func resolveBackendAPIKey(ref CanonicalBackendRef) string {
	if ref.APIKey != "" {
		return ref.APIKey
	}
	if ref.APIKeyEnv != "" {
		return os.Getenv(ref.APIKeyEnv)
	}
	return ""
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
