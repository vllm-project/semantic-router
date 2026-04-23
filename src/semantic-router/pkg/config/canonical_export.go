package config

import (
	"fmt"
	"sort"
)

// CanonicalConfigFromRouterConfig exports the canonical v0.3 config surface
// from the internal runtime config.
func CanonicalConfigFromRouterConfig(cfg *RouterConfig) CanonicalConfig {
	if cfg == nil {
		return CanonicalConfig{Version: "v0.3"}
	}

	return CanonicalConfig{
		Version:   "v0.3",
		Listeners: append([]Listener(nil), cfg.Listeners...),
		Providers: CanonicalProviders{
			Defaults: CanonicalProviderDefaults{
				DefaultModel:           cfg.DefaultModel,
				ReasoningFamilies:      copyReasoningFamilies(cfg.ReasoningFamilies),
				DefaultReasoningEffort: cfg.DefaultReasoningEffort,
			},
			Models: canonicalProviderModelsFromRouterConfig(cfg),
		},
		Routing: CanonicalRoutingFromRouterConfig(cfg),
		Global:  CanonicalGlobalFromRouterConfig(cfg),
	}
}

// CanonicalStaticConfigFromRouterConfig exports the static canonical base used
// by K8s CRD reconciliation. Dynamic routing state is expected to come from the
// CRDs, so the routing block is intentionally left empty.
func CanonicalStaticConfigFromRouterConfig(cfg *RouterConfig) CanonicalConfig {
	canonical := CanonicalConfigFromRouterConfig(cfg)
	canonical.Routing = CanonicalRouting{}
	return canonical
}

// CanonicalRoutingFromRouterConfig exports the routing-owned canonical surface
// from the internal runtime config. Deployment bindings and router-global
// runtime settings intentionally stay outside this view.
func CanonicalRoutingFromRouterConfig(cfg *RouterConfig) CanonicalRouting {
	if cfg == nil {
		return CanonicalRouting{}
	}

	return CanonicalRouting{
		ModelCards:    routingModelsFromRouterConfig(cfg),
		Signals:       canonicalSignalsFromRouterConfig(cfg),
		Projections:   canonicalProjectionsFromRouterConfig(cfg),
		Decisions:     copyDecisions(cfg.Decisions),
		SessionStates: append([]SessionStateConfig(nil), cfg.SessionStates...),
	}
}

func canonicalSignalsFromRouterConfig(cfg *RouterConfig) CanonicalSignals {
	return CanonicalSignals{
		Keywords:      append([]KeywordRule(nil), cfg.KeywordRules...),
		Embeddings:    append([]EmbeddingRule(nil), cfg.EmbeddingRules...),
		Domains:       append([]Category(nil), cfg.Categories...),
		FactCheck:     append([]FactCheckRule(nil), cfg.FactCheckRules...),
		UserFeedbacks: append([]UserFeedbackRule(nil), cfg.UserFeedbackRules...),
		Reasks:        append([]ReaskRule(nil), cfg.ReaskRules...),
		Preferences:   append([]PreferenceRule(nil), cfg.PreferenceRules...),
		Language:      append([]LanguageRule(nil), cfg.LanguageRules...),
		Context:       append([]ContextRule(nil), cfg.ContextRules...),
		Structure:     append([]StructureRule(nil), cfg.StructureRules...),
		Complexity:    append([]ComplexityRule(nil), cfg.ComplexityRules...),
		Modality:      append([]ModalityRule(nil), cfg.ModalityRules...),
		RoleBindings:  append([]RoleBinding(nil), cfg.RoleBindings...),
		Jailbreak:     append([]JailbreakRule(nil), cfg.JailbreakRules...),
		PII:           append([]PIIRule(nil), cfg.PIIRules...),
		KB:            append([]KBSignalRule(nil), cfg.KBRules...),
		Conversation:  append([]ConversationRule(nil), cfg.ConversationRules...),
	}
}

func canonicalProjectionsFromRouterConfig(cfg *RouterConfig) CanonicalProjections {
	return CanonicalProjections{
		Partitions: append([]ProjectionPartition(nil), cfg.Projections.Partitions...),
		Scores:     append([]ProjectionScore(nil), cfg.Projections.Scores...),
		Mappings:   append([]ProjectionMapping(nil), cfg.Projections.Mappings...),
	}
}

func routingModelsFromRouterConfig(cfg *RouterConfig) []RoutingModel {
	modelNames := make(map[string]bool)
	for name := range cfg.ModelConfig {
		modelNames[name] = true
	}
	for _, decision := range cfg.Decisions {
		for _, ref := range decision.ModelRefs {
			if ref.Model != "" {
				modelNames[ref.Model] = true
			}
		}
	}

	if len(modelNames) == 0 {
		return nil
	}

	names := make([]string, 0, len(modelNames))
	for name := range modelNames {
		names = append(names, name)
	}
	sort.Strings(names)

	models := make([]RoutingModel, 0, len(names))
	for _, name := range names {
		params := cfg.ModelConfig[name]
		models = append(models, RoutingModel{
			Name:              name,
			ParamSize:         params.ParamSize,
			ContextWindowSize: params.ContextWindowSize,
			Description:       params.Description,
			Capabilities:      append([]string(nil), params.Capabilities...),
			LoRAs:             copyLoRAAdapters(params.LoRAs),
			Tags:              append([]string(nil), params.Tags...),
			QualityScore:      params.QualityScore,
			Modality:          params.Modality,
		})
	}
	return models
}

// CanonicalGlobalFromRouterConfig exports the router-wide canonical global
// block from the internal runtime config.
func CanonicalGlobalFromRouterConfig(cfg *RouterConfig) *CanonicalGlobal {
	if cfg == nil {
		return nil
	}

	global := &CanonicalGlobal{
		Router: CanonicalRouterGlobal{
			ConfigSource:              normalizedConfigSource(cfg.ConfigSource),
			Strategy:                  cfg.Strategy,
			AutoModelName:             cfg.AutoModelName,
			IncludeConfigModelsInList: cfg.IncludeConfigModelsInList,
			ClearRouteCache:           cfg.ClearRouteCache,
			StreamedBody: CanonicalStreamedBody{
				Enabled:    cfg.StreamedBodyMode,
				MaxBytes:   cfg.MaxStreamedBodyBytes,
				TimeoutSec: cfg.StreamedBodyTimeoutSec,
			},
			ModelSelection: cfg.ModelSelection,
		},
		Services: CanonicalServiceGlobal{
			API:           cfg.API,
			ResponseAPI:   cfg.ResponseAPI,
			Observability: cfg.Observability,
			Authz:         cfg.Authz,
			RateLimit:     cfg.RateLimit,
			RouterReplay:  cfg.RouterReplay,
			StartupStatus: cfg.StartupStatus,
		},
		Stores: CanonicalStoreGlobal{
			SemanticCache: cfg.SemanticCache,
			Memory:        cfg.Memory,
			VectorStore:   cloneVectorStoreConfig(cfg.VectorStore),
		},
		Integrations: CanonicalIntegrationGlobal{
			Tools:  cfg.Tools,
			Looper: cfg.Looper,
		},
		ModelCatalog: CanonicalModelCatalog{
			Embeddings: CanonicalEmbeddingModels{
				Semantic: cfg.EmbeddingModels,
			},
			System: CanonicalSystemModels{
				PromptGuard:            cfg.PromptGuard.ModelID,
				DomainClassifier:       cfg.CategoryModel.ModelID,
				PIIClassifier:          cfg.PIIModel.ModelID,
				FactCheckClassifier:    cfg.HallucinationMitigation.FactCheckModel.ModelID,
				HallucinationDetector:  cfg.HallucinationMitigation.HallucinationModel.ModelID,
				HallucinationExplainer: cfg.HallucinationMitigation.NLIModel.ModelID,
				FeedbackDetector:       cfg.FeedbackDetector.ModelID,
			},
			External: append([]ExternalModelConfig(nil), cfg.ExternalModels...),
			KBs:      append([]KnowledgeBaseConfig(nil), cfg.KnowledgeBases...),
			Modules: CanonicalModelModules{
				PromptCompression: cfg.PromptCompression,
				PromptGuard: CanonicalPromptGuardModule{
					PromptGuardConfig: cfg.PromptGuard,
					ModelRef:          "prompt_guard",
				},
				Classifier: CanonicalClassifierModule{
					Domain: CanonicalCategoryModule{
						CategoryModel: cfg.CategoryModel,
						ModelRef:      "domain_classifier",
					},
					MCP: cfg.MCPCategoryModel,
					PII: CanonicalPIIModule{
						PIIModel: cfg.PIIModel,
						ModelRef: "pii_classifier",
					},
					Preference: cfg.PreferenceModel.WithDefaults(),
				},
				Complexity: cfg.ComplexityModel.WithDefaults(),
				HallucinationMitigation: CanonicalHallucinationModule{
					Enabled:                 cfg.HallucinationMitigation.Enabled,
					OnHallucinationDetected: cfg.HallucinationMitigation.OnHallucinationDetected,
					FactCheck: CanonicalFactCheckModule{
						FactCheckModelConfig: cfg.HallucinationMitigation.FactCheckModel,
						ModelRef:             "fact_check_classifier",
					},
					Detector: CanonicalHallucinationDetector{
						HallucinationModelConfig: cfg.HallucinationMitigation.HallucinationModel,
						ModelRef:                 "hallucination_detector",
					},
					Explainer: CanonicalExplainerModule{
						NLIModelConfig: cfg.HallucinationMitigation.NLIModel,
						ModelRef:       "hallucination_explainer",
					},
				},
				FeedbackDetector: CanonicalFeedbackDetectorModule{
					FeedbackDetectorConfig: cfg.FeedbackDetector,
					ModelRef:               "feedback_detector",
				},
				ModalityDetector: cfg.ModalityDetector,
			},
		},
	}

	return global
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
	providerModel := CanonicalProviderModel{
		Name:             name,
		ReasoningFamily:  params.ReasoningFamily,
		APIFormat:        params.APIFormat,
		Pricing:          params.Pricing,
		ExternalModelIDs: copyStringMap(params.ExternalModelIDs),
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
	return providerModel
}

func canonicalProviderModelID(externalModelIDs map[string]string) string {
	if len(externalModelIDs) != 1 {
		return ""
	}
	for _, modelID := range externalModelIDs {
		return modelID
	}
	return ""
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
			refs = append(refs, canonicalBackendRefFromRuntime(endpoint, params.AccessKey, profiles[endpoint.ProviderProfileName]))
		}
		return refs
	}

	refs := make([]CanonicalBackendRef, 0, len(preferred))
	for _, endpointName := range preferred {
		endpoint, ok := endpointsByName[endpointName]
		if !ok {
			continue
		}
		refs = append(refs, canonicalBackendRefFromRuntime(endpoint, params.AccessKey, profiles[endpoint.ProviderProfileName]))
	}
	return refs
}

func canonicalBackendRefFromRuntime(endpoint VLLMEndpoint, fallbackAPIKey string, profile ProviderProfile) CanonicalBackendRef {
	ref := CanonicalBackendRef{
		Name:       endpoint.Name,
		Protocol:   endpoint.Protocol,
		Weight:     endpoint.Weight,
		Type:       endpoint.Type,
		Provider:   profile.Type,
		BaseURL:    profile.BaseURL,
		AuthHeader: profile.AuthHeader,
		AuthPrefix: profile.AuthPrefix,
		APIVersion: profile.APIVersion,
		ChatPath:   profile.ChatPath,
		APIKey:     endpoint.APIKey,
	}
	if endpoint.Address != "" {
		ref.Endpoint = endpoint.Address
		if endpoint.Port > 0 {
			ref.Endpoint = fmt.Sprintf("%s:%d", endpoint.Address, endpoint.Port)
		}
	}
	if ref.APIKey == "" {
		ref.APIKey = fallbackAPIKey
	}
	if len(profile.ExtraHeaders) > 0 {
		ref.ExtraHeaders = copyStringMap(profile.ExtraHeaders)
	}
	return ref
}

func cloneVectorStoreConfig(cfg *VectorStoreConfig) *VectorStoreConfig {
	if cfg == nil {
		return nil
	}
	cloned := *cfg
	return &cloned
}

func normalizedConfigSource(source ConfigSource) ConfigSource {
	if source == "" {
		return ConfigSourceFile
	}
	return source
}
