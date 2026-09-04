package config

import (
	"fmt"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

func applyEffectiveModelRegistry(
	cfg *RouterConfig,
	registry *modelcatalog.EffectiveRegistry,
	authoredModels []CanonicalProviderModel,
) error {
	if registry == nil {
		return fmt.Errorf("effective model registry is required")
	}
	defaults := registry.Defaults()
	cfg.DefaultModel = defaults.Model
	cfg.DefaultReasoningEffort = defaults.ReasoningEffort
	cfg.DefaultQualityIndex = defaults.QualityIndex
	cfg.ReasoningFamilies = map[string]ReasoningFamilyConfig{}
	for _, family := range registry.ReasoningFamilies() {
		cfg.ReasoningFamilies[family.ID] = ReasoningFamilyConfig{Type: family.Type, Parameter: family.Parameter}
	}
	cfg.ModelConfig = make(map[string]ModelParams)
	cfg.ProviderProfiles = make(map[string]ProviderProfile)
	cfg.VLLMEndpoints = nil
	authoredByAlias := make(map[string]*CanonicalProviderModel, len(authoredModels))
	for _, model := range authoredModels {
		authoredByAlias[model.Name] = cloneCanonicalProviderModel(&model)
	}

	for _, model := range registry.Models() {
		params := modelParamsFromEffectiveModel(model, defaults.QualityIndex)
		params.AuthoredModel = authoredByAlias[model.Alias]
		if model.BindingDefaults.Protocol != "" {
			apiFormat, err := apiFormatForProtocol(model.BindingDefaults.Protocol)
			if err != nil {
				return fmt.Errorf("providers.models[%s]: %w", model.Alias, err)
			}
			params.APIFormat = apiFormat
		}
		for bindingIndex, provider := range model.Providers {
			if err := appendEffectiveProviderBinding(cfg, model, bindingIndex, provider, &params); err != nil {
				return err
			}
		}
		cfg.ModelConfig[model.Alias] = params
	}
	if len(cfg.ProviderProfiles) == 0 {
		cfg.ProviderProfiles = nil
	}
	return validateEffectiveLoRAReferences(cfg)
}

func modelParamsFromEffectiveModel(model modelcatalog.EffectiveModel, qualityIndex string) ModelParams {
	card := model.Card.Card
	modality := model.Card.RuntimeModality
	if modality == "" {
		modality = runtimeModalityFromCard(card)
	}
	params := ModelParams{
		Catalog:           model.Catalog,
		ParamSize:         card.ParameterSize,
		ContextWindowSize: card.Limits.ContextWindowSize,
		Description:       card.Description,
		Capabilities:      append([]string(nil), card.Capabilities...),
		LoRAs:             loraAdaptersFromEffectiveCard(model.Card),
		Tags:              append([]string(nil), card.Tags...),
		Evaluations:       cloneUserEvaluations(model.Card.Evaluations),
		ReasoningFamily:   card.ReasoningFamily,
		Modality:          modality,
		IndexResults:      cloneCatalogIndexResults(model.Indices),
		QualityIndex:      qualityIndex,
		Pricing:           modelPricingFromCatalog(model.BindingDefaults.Pricing),
		Reliability:       providerReliabilityFromCatalog(model.BindingDefaults.Reliability),
		AccessKeys:        map[string]string{},
		ExternalModelIDs:  copyStringMap(model.BindingDefaults.ExternalModelIDs),
	}
	if params.ExternalModelIDs == nil {
		params.ExternalModelIDs = map[string]string{}
	}
	if model.BindingDefaults.ModelID != "" {
		params.ExternalModelIDs["default"] = model.BindingDefaults.ModelID
	}
	return params
}

func runtimeModalityFromCard(card modelcatalog.ModelCard) string {
	if stringSliceContains(card.Capabilities, "image_generation") {
		return "diffusion"
	}
	for _, input := range card.Modalities.Input {
		if input != "text" {
			return "omni"
		}
	}
	return "ar"
}

func appendEffectiveProviderBinding(
	cfg *RouterConfig,
	model modelcatalog.EffectiveModel,
	bindingIndex int,
	effective modelcatalog.EffectiveModelProvider,
	params *ModelParams,
) error {
	binding := effective.Binding
	provider := effective.Provider
	baseURLs := effectiveProviderURLs(provider)
	if len(baseURLs) == 0 {
		return fmt.Errorf("providers.models[%s].backend_refs[%d] has no endpoint URL", model.Alias, bindingIndex)
	}
	if err := applyBindingAPIFormat(model.Alias, bindingIndex, binding.Protocol, params); err != nil {
		return err
	}
	credential, err := applyBindingMetadata(model.Alias, effective, params)
	if err != nil {
		return err
	}
	for endpointIndex, endpointURL := range baseURLs {
		if err := appendMaterializedEndpoint(
			cfg, model.Alias, bindingIndex, endpointIndex, endpointURL, effective, credential, params,
		); err != nil {
			return err
		}
	}
	return nil
}

func applyBindingAPIFormat(modelAlias string, bindingIndex int, protocol string, params *ModelParams) error {
	apiFormat, err := apiFormatForProtocol(protocol)
	if err != nil {
		return fmt.Errorf("providers.models[%s].backend_refs[%d]: %w", modelAlias, bindingIndex, err)
	}
	if params.APIFormat != "" && params.APIFormat != apiFormat {
		return fmt.Errorf(
			"providers.models[%s] backend_refs resolve to mixed API formats %q and %q",
			modelAlias, params.APIFormat, apiFormat,
		)
	}
	params.APIFormat = apiFormat
	return nil
}

func applyBindingMetadata(
	modelAlias string,
	effective modelcatalog.EffectiveModelProvider,
	params *ModelParams,
) (string, error) {
	binding := effective.Binding
	providerID := effective.Provider.Definition.ID
	params.ExternalModelIDs[providerID] = binding.ModelID
	for key, value := range binding.ExternalModelIDs {
		params.ExternalModelIDs[key] = value
	}
	credential := resolveProviderCredential(effective.Provider.Instance.Credentials)
	if existing := params.AccessKeys[providerID]; existing != "" && credential != "" && existing != credential {
		return "", fmt.Errorf(
			"providers.models[%s] uses multiple %q backends with different static credentials",
			modelAlias, providerID,
		)
	}
	applyBindingCredential(providerID, credential, params)
	applyBindingServiceMetadata(effective, params)
	return credential, nil
}

func applyBindingCredential(providerID, credential string, params *ModelParams) {
	if credential == "" {
		return
	}
	params.AccessKeys[providerID] = credential
	if params.AccessKey == "" {
		params.AccessKey = credential
	}
}

func applyBindingServiceMetadata(effective modelcatalog.EffectiveModelProvider, params *ModelParams) {
	pricing := effective.Binding.Pricing
	if pricing == (modelcatalog.Pricing{}) && effective.Offering != nil {
		pricing = effective.Offering.Pricing
	}
	if params.Pricing == (ModelPricing{}) {
		params.Pricing = modelPricingFromCatalog(pricing)
	}
	if params.Reliability == (ProviderReliability{}) {
		params.Reliability = providerReliabilityFromCatalog(effective.Binding.Reliability)
	}
}

func appendMaterializedEndpoint(
	cfg *RouterConfig,
	modelAlias string,
	bindingIndex int,
	endpointIndex int,
	endpointURL materializedProviderURL,
	effective modelcatalog.EffectiveModelProvider,
	credential string,
	params *ModelParams,
) error {
	provider := effective.Provider
	endpointName := effectiveEndpointName(
		modelAlias, provider.Instance.Name, endpointURL.name, bindingIndex, endpointIndex,
	)
	cfg.ProviderProfiles[endpointName] = materializedProviderProfile(effective, endpointURL.url)
	address, port, scheme, err := endpointAddress(endpointURL.url)
	if err != nil {
		return fmt.Errorf("providers.models[%s].backend_refs[%d]: %w", modelAlias, bindingIndex, err)
	}
	cfg.VLLMEndpoints = append(cfg.VLLMEndpoints, VLLMEndpoint{
		Name: endpointName, Address: address, Port: port, Weight: endpointURL.weight,
		Type: provider.Definition.ID, APIKey: credential, ProviderProfileName: endpointName,
		Model: modelAlias, Protocol: scheme,
	})
	params.PreferredEndpoints = append(params.PreferredEndpoints, endpointName)
	return nil
}

func materializedProviderProfile(
	effective modelcatalog.EffectiveModelProvider,
	baseURL string,
) ProviderProfile {
	provider := effective.Provider
	return ProviderProfile{
		Type: provider.Definition.ID, Protocol: effective.Binding.Protocol, BaseURL: baseURL,
		ExtraHeaders: mergeProviderHeaders(provider.Definition.DefaultHeaders, provider.Instance.Headers),
		APIVersion:   provider.Instance.APIVersion, AuthHeader: provider.Instance.AuthHeader,
		AuthPrefix: provider.Instance.AuthPrefix, ChatPath: provider.Instance.ChatPath,
	}
}

func mergeProviderHeaders(defaults, overrides map[string]string) map[string]string {
	if len(defaults) == 0 && len(overrides) == 0 {
		return nil
	}
	result := make(map[string]string, len(defaults)+len(overrides))
	for key, value := range defaults {
		result[key] = value
	}
	for key, value := range overrides {
		result[key] = value
	}
	return result
}

type materializedProviderURL struct {
	name   string
	url    string
	weight int
}

func effectiveProviderURLs(provider modelcatalog.EffectiveProvider) []materializedProviderURL {
	if len(provider.Instance.Endpoints) > 0 {
		result := make([]materializedProviderURL, 0, len(provider.Instance.Endpoints))
		for _, endpoint := range provider.Instance.Endpoints {
			result = append(result, materializedProviderURL{name: endpoint.Name, url: endpoint.URL, weight: endpoint.Weight})
		}
		return result
	}
	baseURL := provider.Instance.BaseURL
	if baseURL == "" {
		baseURL = provider.Definition.DefaultBaseURL
	}
	if baseURL == "" {
		return nil
	}
	return []materializedProviderURL{{name: "primary", url: baseURL, weight: 1}}
}

func effectiveEndpointName(modelAlias, providerName, endpointName string, bindingIndex, endpointIndex int) string {
	if providerName != "" && (endpointName == "" || endpointName == "primary") {
		return providerName
	}
	name := strings.Trim(strings.Join([]string{modelAlias, providerName, endpointName}, "_"), "_")
	if name == "" {
		return fmt.Sprintf("model_%d_endpoint_%d", bindingIndex+1, endpointIndex+1)
	}
	return name
}

func endpointAddress(raw string) (string, int, string, error) {
	parsed, err := url.Parse(endpointURLForParsing(raw))
	if err != nil || parsed.Hostname() == "" {
		return "", 0, "", fmt.Errorf("invalid provider endpoint URL %q", raw)
	}
	port := 0
	if parsed.Port() != "" {
		port, err = strconv.Atoi(parsed.Port())
		if err != nil {
			return "", 0, "", fmt.Errorf("invalid provider endpoint port in %q", raw)
		}
	} else if parsed.Scheme == "https" {
		port = 443
	} else {
		port = 80
	}
	hostname := parsed.Hostname()
	if endpointTemplateToken.MatchString(raw) {
		hostname = endpointTemplateHostname(raw)
	}
	return hostname, port, parsed.Scheme, nil
}

var endpointTemplateToken = regexp.MustCompile(`\{\{[A-Za-z_][A-Za-z0-9_.-]*\}\}|\$\{[A-Za-z_][A-Za-z0-9_]*\}`)

func endpointURLForParsing(raw string) string {
	return endpointTemplateToken.ReplaceAllString(strings.TrimSpace(raw), "catalog-placeholder.invalid")
}

func endpointTemplateHostname(raw string) string {
	authority := raw
	if separator := strings.Index(authority, "://"); separator >= 0 {
		authority = authority[separator+3:]
	}
	if end := strings.IndexAny(authority, "/?#"); end >= 0 {
		authority = authority[:end]
	}
	if colon := strings.LastIndex(authority, ":"); colon >= 0 {
		authority = authority[:colon]
	}
	return strings.Trim(authority, "[]")
}

func resolveProviderCredential(credentials modelcatalog.CredentialsRef) string {
	if credentials.APIKey != "" {
		return credentials.APIKey
	}
	if credentials.APIKeyEnv != "" {
		return os.Getenv(credentials.APIKeyEnv)
	}
	return ""
}

func apiFormatForProtocol(protocol string) (string, error) {
	switch protocol {
	case "openai/chat-completions@1":
		return APIFormatOpenAI, nil
	case "openai/responses@1":
		return APIFormatResponses, nil
	case "anthropic/messages@1":
		return APIFormatAnthropic, nil
	default:
		return "", fmt.Errorf("protocol %q has no runtime codec", protocol)
	}
}

func modelPricingFromCatalog(pricing modelcatalog.Pricing) ModelPricing {
	return ModelPricing{
		Currency: pricing.Currency, PromptPer1M: pricing.PromptPer1M,
		CompletionPer1M: pricing.CompletionPer1M, CachedInputPer1M: pricing.CachedInputPer1M,
		CacheWritePer1M: pricing.CacheWritePer1M,
	}
}

func providerReliabilityFromCatalog(reliability modelcatalog.Reliability) ProviderReliability {
	return ProviderReliability{
		LBPolicy: reliability.LBPolicy, RetryCount: reliability.RetryCount, RetryOn: reliability.RetryOn,
		Consecutive5xx: reliability.Consecutive5xx, BaseEjectionTime: reliability.BaseEjectionTime,
		MaxEjectionPercent: reliability.MaxEjectionPercent, HealthCheckPath: reliability.HealthCheckPath,
		HealthCheckInterval: reliability.HealthCheckInterval, HealthCheckTimeout: reliability.HealthCheckTimeout,
	}
}

func cloneCatalogIndexResults(source map[string]modelcatalog.IndexResult) map[string]modelcatalog.IndexResult {
	if source == nil {
		return nil
	}
	result := make(map[string]modelcatalog.IndexResult, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func validateEffectiveLoRAReferences(cfg *RouterConfig) error {
	for _, decision := range cfg.Decisions {
		for _, ref := range decision.ModelRefs {
			if ref.LoRAName == "" {
				continue
			}
			params, ok := cfg.ModelConfig[ref.Model]
			if !ok {
				continue
			}
			found := false
			for _, lora := range params.LoRAs {
				if lora.Name == ref.LoRAName {
					found = true
					break
				}
			}
			if !found {
				return fmt.Errorf("routing.decisions[%s].modelRefs[%s].lora_name %q is not declared by model card %q", decision.Name, ref.Model, ref.LoRAName, params.Catalog)
			}
		}
	}
	return nil
}
