package evaluationplane

import (
	"math"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type modelArmResolver struct {
	config          *routerconfig.RouterConfig
	providersByBase map[string]routerconfig.CanonicalProviderModel
	cardsByBase     map[string]routerconfig.RoutingModel
	runtimeRevision string
}

func newModelArmResolver(
	cfg *routerconfig.RouterConfig,
	canonical routerconfig.CanonicalConfig,
	runtimeRevision string,
) modelArmResolver {
	providers := make(map[string]routerconfig.CanonicalProviderModel, len(canonical.Providers.Models))
	for _, provider := range canonical.Providers.Models {
		providers[strings.TrimSpace(provider.Name)] = provider
	}
	cards := make(map[string]routerconfig.RoutingModel, len(canonical.Routing.ModelCards))
	for _, card := range canonical.Routing.ModelCards {
		cards[strings.TrimSpace(card.Name)] = card
	}
	return modelArmResolver{
		config: cfg, providersByBase: providers, cardsByBase: cards,
		runtimeRevision: runtimeRevision,
	}
}

func (resolver modelArmResolver) resolve(binding mixtureModelBinding) (ModelArm, bool) {
	provider, providerExists := resolver.providersByBase[binding.BaseModel]
	card, cardExists := resolver.cardsByBase[binding.BaseModel]
	if binding.EffectiveModel != binding.BaseModel {
		if direct, exists := resolver.providersByBase[binding.EffectiveModel]; exists && len(direct.BackendRefs) > 0 {
			return ModelArm{}, false
		}
	}
	pricing, configured := resolver.config.GetFullModelPricing(binding.EffectiveModel)
	if !configured {
		pricing.Currency = "USD"
	}
	if !providerExists || !cardExists || !validProviderArm(provider, binding, pricing) {
		return ModelArm{}, false
	}

	providerDigest := baseProviderModelDigest(provider, binding.BaseModel)
	if binding.EffectiveModel != binding.BaseModel {
		providerDigest = digestJSON(adapterProviderIdentityFingerprint{
			BaseProviderModelIDDigest: providerDigest,
			AdapterIdentity:           binding.EffectiveModel,
		})
	}
	arm := ModelArm{
		ID:                            portableModelArmID(binding.EffectiveModel),
		Model:                         binding.EffectiveModel,
		ProviderModelIDDigest:         providerDigest,
		InputCostPerMillionTokensUSD:  pricing.PromptPer1M,
		OutputCostPerMillionTokensUSD: pricing.CompletionPer1M,
		Capabilities:                  normalizedCapabilities(card.Capabilities),
		Modalities:                    normalizedModalities(card),
		ContextWindowTokens:           positiveInt(card.ContextWindowSize),
		ParameterSize:                 boundedOptionalString(card.ParamSize, 64),
		RuntimeRevision:               runtimeRevisionPointer(resolver.runtimeRevision),
	}
	arm.ConfigDigest = stringPointer(modelArmConfigDigest(provider, arm, pricing))
	return arm, true
}

type adapterProviderIdentityFingerprint struct {
	BaseProviderModelIDDigest string `json:"base_provider_model_id_digest"`
	AdapterIdentity           string `json:"adapter_identity"`
}

func baseProviderModelDigest(provider routerconfig.CanonicalProviderModel, baseModel string) string {
	providerIdentity := strings.TrimSpace(provider.ProviderModelID)
	if providerIdentity == "" {
		providerIdentity = baseModel
	}
	return digestString(providerIdentity)
}

type armConfigFingerprint struct {
	Model                          string   `json:"model"`
	ProviderModelIDDigest          string   `json:"provider_model_id_digest"`
	ReasoningFamily                string   `json:"reasoning_family,omitempty"`
	APIFormat                      string   `json:"api_format,omitempty"`
	InputCostPerMillionTokensUSD   float64  `json:"input_cost_per_million_tokens_usd"`
	OutputCostPerMillionTokensUSD  float64  `json:"output_cost_per_million_tokens_usd"`
	CachedInputPerMillionTokensUSD float64  `json:"cached_input_per_million_tokens_usd,omitempty"`
	CacheWritePerMillionTokensUSD  *float64 `json:"cache_write_per_million_tokens_usd,omitempty"`
	Capabilities                   []string `json:"capabilities,omitempty"`
	Modalities                     []string `json:"modalities,omitempty"`
	ContextWindowTokens            *int     `json:"context_window_tokens,omitempty"`
	ParameterSize                  *string  `json:"parameter_size,omitempty"`
	ExternalModelIDDigest          string   `json:"external_model_ids_digest,omitempty"`
}

func modelArmConfigDigest(
	provider routerconfig.CanonicalProviderModel,
	arm ModelArm,
	pricing routerconfig.ModelPricing,
) string {
	externalDigest := ""
	if len(provider.ExternalModelIDs) > 0 {
		externalDigest = digestJSON(provider.ExternalModelIDs)
	}
	return digestJSON(armConfigFingerprint{
		Model: arm.Model, ProviderModelIDDigest: arm.ProviderModelIDDigest,
		ReasoningFamily: provider.ReasoningFamily, APIFormat: provider.APIFormat,
		InputCostPerMillionTokensUSD:   arm.InputCostPerMillionTokensUSD,
		OutputCostPerMillionTokensUSD:  arm.OutputCostPerMillionTokensUSD,
		CachedInputPerMillionTokensUSD: pricing.CachedInputPer1M,
		CacheWritePerMillionTokensUSD:  pricing.CacheWritePer1M,
		Capabilities:                   arm.Capabilities, Modalities: arm.Modalities,
		ContextWindowTokens: arm.ContextWindowTokens, ParameterSize: arm.ParameterSize,
		ExternalModelIDDigest: externalDigest,
	})
}

func validProviderArm(
	provider routerconfig.CanonicalProviderModel,
	binding mixtureModelBinding,
	pricing routerconfig.ModelPricing,
) bool {
	if binding.EffectiveModel == "" || len(binding.EffectiveModel) > 512 ||
		binding.BaseModel == "" || len(binding.BaseModel) > 512 ||
		strings.TrimSpace(provider.Name) != binding.BaseModel || len(provider.BackendRefs) == 0 {
		return false
	}
	rates := []float64{pricing.PromptPer1M, pricing.CompletionPer1M, pricing.CachedInputPer1M}
	if pricing.CacheWritePer1M != nil {
		rates = append(rates, *pricing.CacheWritePer1M)
	}
	for _, rate := range rates {
		if rate < 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
			return false
		}
	}
	return strings.EqualFold(strings.TrimSpace(pricing.Currency), "USD")
}

func portableModelArmID(model string) string {
	var base strings.Builder
	lastSeparator := false
	for _, value := range strings.ToLower(model) {
		if (value >= 'a' && value <= 'z') || (value >= '0' && value <= '9') || value == '.' || value == '_' {
			base.WriteRune(value)
			lastSeparator = false
			continue
		}
		if !lastSeparator && base.Len() > 0 {
			base.WriteByte('-')
			lastSeparator = true
		}
	}
	readable := strings.Trim(base.String(), ".-_")
	if readable == "" || !startsWithASCIILetterOrDigit(readable) {
		readable = "model"
	}
	// A full digest makes IDs collision-resistant even when different model
	// names normalize to the same portable prefix. 63+1+64 is the Python
	// contract's 128-character maximum.
	if len(readable) > 63 {
		readable = strings.TrimRight(readable[:63], ".-_")
	}
	return readable + "-" + strings.TrimPrefix(digestString(model), "sha256:")
}

func startsWithASCIILetterOrDigit(value string) bool {
	first := value[0]
	return (first >= 'a' && first <= 'z') || (first >= '0' && first <= '9')
}
