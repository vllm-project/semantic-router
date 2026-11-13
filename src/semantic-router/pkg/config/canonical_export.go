package config

import (
	"fmt"
	"sort"
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

// CanonicalConfigFromRouterConfig exports the canonical v0.5 config surface
// from the internal runtime config.
func CanonicalConfigFromRouterConfig(cfg *RouterConfig) CanonicalConfig {
	if cfg == nil {
		return CanonicalConfig{Version: CanonicalConfigVersion}
	}

	return CanonicalConfig{
		Version:    CanonicalConfigVersion,
		Listeners:  append([]Listener(nil), cfg.Listeners...),
		Evaluation: cloneCanonicalEvaluation(cfg.Evaluation),
		Providers: CanonicalProviders{
			Defaults: CanonicalProviderDefaults{
				DefaultModel:           cfg.DefaultModel,
				DefaultReasoningEffort: cfg.DefaultReasoningEffort,
				TLS:                    cloneProviderDefaultsTLS(cfg.DefaultBackendTLS),
			},
			Models: canonicalProviderModelsFromRouterConfig(cfg),
		},
		Recipes:     canonicalRecipesFromRouterConfig(cfg),
		Entrypoints: canonicalEntrypointsFromRouterConfig(cfg),
		Global:      CanonicalGlobalFromRouterConfig(cfg),
	}
}

// CanonicalStaticConfigFromRouterConfig exports the static canonical base used
// by K8s CRD reconciliation. Dynamic routing state is expected to come from the
// CRDs, so recipes and entrypoints are left empty.
func CanonicalStaticConfigFromRouterConfig(cfg *RouterConfig) CanonicalConfig {
	canonical := CanonicalConfigFromRouterConfig(cfg)
	canonical.Entrypoints = nil
	canonical.Recipes = nil
	return canonical
}

// ModelCardsFromRouterConfig exports the facts the providers.models[] entries
// state, each under its entry's name, in entry order. A catalog-backed entry's
// facts are named after the entry, not its catalog card, so they merge back
// onto it.
func ModelCardsFromRouterConfig(cfg *RouterConfig) []RoutingModel {
	var cards []RoutingModel
	for _, model := range canonicalProviderModelsFromRouterConfig(cfg) {
		if model.ModelFacts.declared() {
			cards = append(cards, RoutingModel{Name: model.Name, ModelFacts: model.ModelFacts})
		}
	}
	return cards
}

func canonicalSignalsFromSignals(signals Signals) CanonicalSignals {
	return CanonicalSignals{
		Keywords:      append([]KeywordRule(nil), signals.KeywordRules...),
		Domains:       append([]Category(nil), signals.Categories...),
		FactCheck:     append([]FactCheckRule(nil), signals.FactCheckRules...),
		UserFeedbacks: append([]UserFeedbackRule(nil), signals.UserFeedbackRules...),
		Language:      append([]LanguageRule(nil), signals.LanguageRules...),
		Context:       append([]ContextRule(nil), signals.ContextRules...),
		Complexity:    append([]ComplexityRule(nil), signals.ComplexityRules...),
		Modality:      append([]ModalityRule(nil), signals.ModalityRules...),
		RoleBindings:  append([]RoleBinding(nil), signals.RoleBindings...),
		Jailbreak:     append([]JailbreakRule(nil), signals.JailbreakRules...),
		Safety:        append([]SafetyRule(nil), signals.SafetyRules...),
		Hallucination: append([]HallucinationRule(nil), signals.HallucinationRules...),
		Refusal:       append([]RefusalRule(nil), signals.RefusalRules...),
		TurnRelation:  append([]TurnRelationRule(nil), signals.TurnRelationRules...),
		PII:           append([]PIIRule(nil), signals.PIIRules...),
		Conversation:  append([]ConversationRule(nil), signals.ConversationRules...),
		Metadata:      append([]MetadataRule(nil), signals.MetadataRules...),
		InputModality: append([]InputModalityRule(nil), signals.InputModalityRules...),
		SystemOne:     append([]SystemOneRule(nil), signals.SystemOneRules...),
	}
}

func canonicalProjectionsFromProjections(projections Projections) CanonicalProjections {
	return CanonicalProjections{
		Partitions: append([]ProjectionPartition(nil), projections.Partitions...),
		Scores:     append([]ProjectionScore(nil), projections.Scores...),
		Mappings:   append([]ProjectionMapping(nil), projections.Mappings...),
	}
}

func loraAdaptersFromEffectiveCard(card modelcatalog.EffectiveModelCard) []LoRAAdapter {
	if len(card.LoRAs) == 0 {
		return nil
	}
	loras := make([]LoRAAdapter, 0, len(card.LoRAs))
	for _, lora := range card.LoRAs {
		loras = append(loras, LoRAAdapter{Name: lora.Name, Description: lora.Description})
	}
	return loras
}

// CanonicalGlobalFromRouterConfig exports the router-wide canonical global
// block from the internal runtime config.
func CanonicalGlobalFromRouterConfig(cfg *RouterConfig) *CanonicalGlobal {
	if cfg == nil {
		return nil
	}

	global := &CanonicalGlobal{
		Router: CanonicalRouterGlobal{
			Enabled:           cfg.RouterEnabled,
			ConfigSource:      normalizedConfigSource(cfg.ConfigSource),
			Strategy:          cfg.RoutingDefaults.Strategy,
			ListBackendModels: cfg.ListBackendModels,
			ClearRouteCache:   cfg.ClearRouteCache,
			StreamedBody: CanonicalStreamedBody{
				Enabled:    cfg.StreamedBodyMode,
				MaxBytes:   cfg.MaxStreamedBodyBytes,
				TimeoutSec: cfg.StreamedBodyTimeoutSec,
			},
			SkipProcessing: cfg.SkipProcessing,
			ModelSelection: cfg.ModelSelection,
			Learning:       cfg.RouterLearning,
			Fallback:       cfg.RoutingDefaults.Fallback.Clone(),

			AllowLossyTranslation: cfg.AllowLossyTranslation,
			RoutingInput:          cfg.RoutingInput,
			Headers:               cfg.Headers,
			ContextWindow:         cfg.ContextWindow,
		},
		Services: CanonicalServiceGlobal{
			API:           cfg.API,
			ResponseAPI:   cfg.ResponseAPI,
			Observability: cfg.Observability,
			Authz:         cfg.Authz,
			RateLimit:     cfg.RateLimit,
			ManagementAPI: cfg.ManagementAPI,
			RouterReplay:  cfg.RouterReplay,
			StartupStatus: cfg.StartupStatus,
		},
		Stores: CanonicalStoreGlobal{
			ResponseCache: cfg.SemanticCache,
			Memory:        cfg.Memory,
			VectorStore:   cloneVectorStoreConfig(cfg.VectorStore),
			ToolSessions:  cloneToolSessionStoreConfig(cfg.ToolSessions),
		},
		Integrations: CanonicalIntegrationGlobal{
			KVTransfer: cfg.KVTransfer,
			Looper:     cfg.Looper,
		},
		ModelCatalog: canonicalModelCatalogFromRouterConfig(cfg),
		Plugins:      copyDecisionPlugins(cfg.GlobalPlugins),
	}

	return global
}

func canonicalModelCatalogFromRouterConfig(cfg *RouterConfig) CanonicalModelCatalog {
	return CanonicalModelCatalog{
		Deployments:      cloneModelMap(cfg.ModelDeployments),
		Bindings:         cloneModelMap(cfg.GlobalModelBindings),
		Embeddings:       CanonicalEmbeddingModels{Default: EmbeddingModelBinding{Deployment: cfg.EmbeddingDeployment}},
		System:           canonicalSystemModelsFromRouterConfig(cfg),
		Admission:        cloneAdmissionMap(cfg.ModelAdmission),
		SignalTimeoutMs:  cfg.ModelSignalTimeoutMs,
		OnSignalOverload: cloneSignalOverloadPolicy(cfg.ModelSignalOverload),
	}
}

func canonicalProviderModelsFromRouterConfig(cfg *RouterConfig) []CanonicalProviderModel {
	if cfg == nil {
		return nil
	}

	modelNames := canonicalProviderModelNames(cfg)
	if len(modelNames) == 0 {
		return nil
	}

	names := sortedCanonicalProviderModelNames(modelNames)
	endpointsByName, endpointsByModel := canonicalEndpointIndexes(cfg.VLLMEndpoints)

	models := make([]CanonicalProviderModel, 0, len(names))
	for _, name := range names {
		providerModel := canonicalProviderModelFromRuntime(
			name,
			cfg.ModelConfig[name],
			endpointsByName,
			endpointsByModel,
			cfg.ProviderProfiles,
		)
		if len(providerModel.BackendRefs) == 0 && !canonicalProviderModelHasMetadata(providerModel) {
			continue
		}
		models = append(models, providerModel)
	}

	return models
}

func canonicalProviderModelNames(cfg *RouterConfig) map[string]bool {
	modelNames := make(map[string]bool, len(cfg.ModelConfig))
	for name := range cfg.ModelConfig {
		modelNames[name] = true
	}
	for _, endpoint := range cfg.VLLMEndpoints {
		if endpoint.Model != "" {
			modelNames[endpoint.Model] = true
		}
	}
	return modelNames
}

func sortedCanonicalProviderModelNames(modelNames map[string]bool) []string {
	names := make([]string, 0, len(modelNames))
	for name := range modelNames {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func canonicalEndpointIndexes(
	endpoints []VLLMEndpoint,
) (map[string]VLLMEndpoint, map[string][]VLLMEndpoint) {
	endpointsByName := make(map[string]VLLMEndpoint, len(endpoints))
	endpointsByModel := make(map[string][]VLLMEndpoint)
	for _, endpoint := range endpoints {
		endpointsByName[endpoint.Name] = endpoint
		if endpoint.Model != "" {
			endpointsByModel[endpoint.Model] = append(endpointsByModel[endpoint.Model], endpoint)
		}
	}
	return endpointsByName, endpointsByModel
}

func canonicalProviderModelFromRuntime(
	name string,
	params ModelParams,
	endpointsByName map[string]VLLMEndpoint,
	endpointsByModel map[string][]VLLMEndpoint,
	profiles map[string]ProviderProfile,
) CanonicalProviderModel {
	if authored := cloneCanonicalProviderModel(params.AuthoredModel); authored != nil {
		authored.Name = name
		return *authored
	}
	providerModel := CanonicalProviderModel{
		Deployment:       params.Deployment,
		Name:             name,
		Catalog:          params.Catalog,
		APIFormat:        params.APIFormat,
		Schema:           params.Schema,
		Pricing:          params.Pricing,
		Reliability:      params.Reliability,
		ExternalModelIDs: copyStringMap(params.ExternalModelIDs),
		Tokenizer:        params.Tokenizer,
		BackendRefs: canonicalProviderBackendRefs(
			name,
			params,
			endpointsByName,
			endpointsByModel,
			profiles,
		),
	}
	if providerModelID := canonicalProviderModelID(params.ExternalModelIDs); providerModelID != "" {
		providerModel.ProviderModelID = providerModelID
	}
	providerModel.UnsupportedCapabilities = append([]string(nil), params.UnsupportedCapabilities...)
	if params.Catalog == "" {
		providerModel.ModelFacts = ModelFacts{
			ParamSize:         params.ParamSize,
			ContextWindowSize: params.ContextWindowSize,
			MaxOutputTokens:   params.MaxOutputTokens,
			Description:       params.Description,
			Capabilities:      append([]string(nil), params.Capabilities...),
			LoRAs:             copyLoRAAdapters(params.LoRAs),
			Tags:              append([]string(nil), params.Tags...),
			Modalities:        declaredModalities(params.Modalities),
		}
	}
	return providerModel
}

// declaredModalities is a model's modalities as its entry states them: none
// when it names neither an input nor an output.
func declaredModalities(modalities modelcatalog.Modalities) *modelcatalog.Modalities {
	if len(modalities.Input) == 0 && len(modalities.Output) == 0 {
		return nil
	}
	clone := cloneModalities(modalities)
	return &clone
}

func cloneCanonicalReasoning(reasoning *CanonicalReasoning) *CanonicalReasoning {
	if reasoning == nil {
		return nil
	}
	clone := *reasoning
	clone.EffortFlags = copyStringMap(reasoning.EffortFlags)
	clone.Levels = append([]string(nil), reasoning.Levels...)
	clone.Modes = append([]string(nil), reasoning.Modes...)
	return &clone
}

func cloneCanonicalProviderModel(model *CanonicalProviderModel) *CanonicalProviderModel {
	if model == nil {
		return nil
	}
	clone := *model
	clone.Reasoning = cloneCanonicalReasoning(model.Reasoning)
	clone.ExternalModelIDs = copyStringMap(model.ExternalModelIDs)
	clone.UnsupportedCapabilities = append([]string(nil), model.UnsupportedCapabilities...)
	clone.BackendRefs = make([]CanonicalBackendRef, len(model.BackendRefs))
	for index, backend := range model.BackendRefs {
		clone.BackendRefs[index] = backend
		clone.BackendRefs[index].Headers = copyStringMap(backend.Headers)
		clone.BackendRefs[index].TLS = cloneDeploymentTLS(backend.TLS)
	}
	return &clone
}

func canonicalProviderModelID(externalModelIDs map[string]string) string {
	if modelID := strings.TrimSpace(externalModelIDs["default"]); modelID != "" {
		return modelID
	}
	var candidate string
	for _, modelID := range externalModelIDs {
		modelID = strings.TrimSpace(modelID)
		if modelID == "" {
			continue
		}
		if candidate != "" && candidate != modelID {
			return ""
		}
		candidate = modelID
	}
	return candidate
}

func canonicalProviderBackendRefs(
	modelName string,
	params ModelParams,
	endpointsByName map[string]VLLMEndpoint,
	endpointsByModel map[string][]VLLMEndpoint,
	profiles map[string]ProviderProfile,
) []CanonicalBackendRef {
	preferred := params.PreferredEndpoints
	if len(preferred) == 0 {
		modelEndpoints := endpointsByModel[modelName]
		if len(modelEndpoints) == 0 {
			return nil
		}
		refs := make([]CanonicalBackendRef, 0, len(modelEndpoints))
		for _, endpoint := range modelEndpoints {
			refs = append(refs, canonicalBackendRefFromRuntime(modelName, endpoint, params.AccessKey, profiles[endpoint.ProviderProfileName]))
		}
		return refs
	}

	refs := make([]CanonicalBackendRef, 0, len(preferred))
	for _, endpointName := range preferred {
		endpoint, ok := endpointsByName[endpointName]
		if !ok {
			continue
		}
		refs = append(refs, canonicalBackendRefFromRuntime(modelName, endpoint, params.AccessKey, profiles[endpoint.ProviderProfileName]))
	}
	return refs
}

func canonicalBackendRefFromRuntime(modelName string, endpoint VLLMEndpoint, fallbackAPIKey string, profile ProviderProfile) CanonicalBackendRef {
	ref := CanonicalBackendRef{
		// Import adds exactly one model namespace; remove only that generated prefix.
		Name:       strings.TrimPrefix(endpoint.Name, modelName+"_"),
		Protocol:   endpoint.Protocol,
		Weight:     endpoint.Weight,
		Provider:   profile.Type,
		APIVersion: profile.APIVersion,
		ChatPath:   profile.ChatPath,
		APIKey:     endpoint.APIKey,
	}
	if profile.AuthHeader != "" || profile.AuthPrefixSet || profile.AuthPrefix != "" || profile.Delegated() || profile.Caller() {
		// A delegated or caller backend has no static key to export; the request supplies it.
		ref.Auth = &DeploymentAuth{Credential: profile.AuthCredential, Header: profile.AuthHeader}
		if profile.AuthPrefixSet || profile.AuthPrefix != "" {
			prefix := profile.AuthPrefix
			ref.Auth.Prefix = &prefix
		}
	}
	if ref.Provider == "" {
		ref.Provider = endpoint.Type
	}
	if endpoint.Address != "" {
		ref.Endpoint = endpoint.Address
		if endpoint.Port > 0 {
			ref.Endpoint = fmt.Sprintf("%s:%d", endpoint.Address, endpoint.Port)
		}
	}
	if profile.BaseURL != "" {
		ref.Endpoint, ref.Protocol = profile.BaseURL, ""
	}
	if ref.APIKey == "" && !profile.Delegated() {
		ref.APIKey = fallbackAPIKey
	}
	if len(profile.ExtraHeaders) > 0 {
		ref.Headers = copyStringMap(profile.ExtraHeaders)
	}
	ref.TLS = cloneDeploymentTLS(profile.TLS)
	return ref
}

func cloneVectorStoreConfig(cfg *VectorStoreConfig) *VectorStoreConfig {
	if cfg == nil {
		return nil
	}
	cloned := *cfg
	return &cloned
}

// cloneToolSessionStoreConfig deep-clones a ToolSessionStoreConfig. Unlike
// cloneVectorStoreConfig's shallow *cfg copy (safe there because
// VectorStoreConfig has no pointer fields of its own), this struct is
// pointer-heavy (five *int fields plus a nested *ToolSessionRedisConfig) —
// a shallow copy would alias those pointers with the original across the
// canonical export boundary, letting a mutation on one side leak into the
// other.
func cloneToolSessionStoreConfig(cfg *ToolSessionStoreConfig) *ToolSessionStoreConfig {
	if cfg == nil {
		return nil
	}
	cloned := *cfg
	cloned.TTLSeconds = cloneIntPtr(cfg.TTLSeconds)
	cloned.MaxSessions = cloneIntPtr(cfg.MaxSessions)
	cloned.MaxSessionsByIdentity = cloneIntPtr(cfg.MaxSessionsByIdentity)
	cloned.MaxStateBytes = cloneIntPtr(cfg.MaxStateBytes)
	cloned.TimeoutMs = cloneIntPtr(cfg.TimeoutMs)
	if cfg.Redis != nil {
		redisClone := *cfg.Redis
		cloned.Redis = &redisClone
	}
	return &cloned
}

func cloneIntPtr(v *int) *int {
	if v == nil {
		return nil
	}
	cloned := *v
	return &cloned
}

func normalizedConfigSource(source ConfigSource) ConfigSource {
	if source == "" {
		return ConfigSourceFile
	}
	return source
}

// canonicalSystemModelsFromRouterConfig writes the decision model; modules
// follow it by name, or the bindings name their deployments.
func canonicalSystemModelsFromRouterConfig(cfg *RouterConfig) CanonicalSystemModels {
	name := cfg.DecisionModel
	if name == "" {
		name = DefaultDecisionDeployment
	}
	return CanonicalSystemModels{DecisionModel: DecisionModelBinding{Deployment: name}}
}
