package config

import (
	"fmt"
	"sort"
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
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
				DefaultReasoningEffort: cfg.DefaultReasoningEffort,
			},
			Models: canonicalProviderModelsFromRouterConfig(cfg),
		},
		Routing:     CanonicalRoutingFromRouterConfig(cfg),
		Entrypoints: canonicalEntrypointsFromRouterConfig(cfg),
		Recipes:     canonicalRecipesFromRouterConfig(cfg),
		Global:      CanonicalGlobalFromRouterConfig(cfg),
	}
}

// CanonicalStaticConfigFromRouterConfig exports the static canonical base used
// by K8s CRD reconciliation. Dynamic routing state is expected to come from the
// CRDs, so the routing block, entrypoints, and recipes are left empty.
func CanonicalStaticConfigFromRouterConfig(cfg *RouterConfig) CanonicalConfig {
	canonical := CanonicalConfigFromRouterConfig(cfg)
	canonical.Routing = CanonicalRouting{}
	canonical.Entrypoints = nil
	canonical.Recipes = nil
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
		ModelCards:  routingModelsFromRouterConfig(cfg),
		Signals:     canonicalSignalsFromSignals(cfg.RoutingProfileSignals()),
		Projections: canonicalProjectionsFromProjections(cfg.RoutingProfileProjections()),
		Decisions:   copyDecisions(cfg.Decisions),
		Strategy:    cfg.Strategy,
	}
}

func canonicalSignalsFromSignals(signals Signals) CanonicalSignals {
	return CanonicalSignals{
		Keywords:      append([]KeywordRule(nil), signals.KeywordRules...),
		Embeddings:    append([]EmbeddingRule(nil), signals.EmbeddingRules...),
		Domains:       append([]Category(nil), signals.Categories...),
		FactCheck:     append([]FactCheckRule(nil), signals.FactCheckRules...),
		UserFeedbacks: append([]UserFeedbackRule(nil), signals.UserFeedbackRules...),
		Reasks:        append([]ReaskRule(nil), signals.ReaskRules...),
		Preferences:   append([]PreferenceRule(nil), signals.PreferenceRules...),
		Language:      append([]LanguageRule(nil), signals.LanguageRules...),
		Context:       append([]ContextRule(nil), signals.ContextRules...),
		Structure:     append([]StructureRule(nil), signals.StructureRules...),
		Complexity:    append([]ComplexityRule(nil), signals.ComplexityRules...),
		Modality:      append([]ModalityRule(nil), signals.ModalityRules...),
		RoleBindings:  append([]RoleBinding(nil), signals.RoleBindings...),
		Jailbreak:     append([]JailbreakRule(nil), signals.JailbreakRules...),
		PII:           append([]PIIRule(nil), signals.PIIRules...),
		KB:            append([]KBSignalRule(nil), signals.KBRules...),
		Conversation:  append([]ConversationRule(nil), signals.ConversationRules...),
		EventRules:    append([]EventRule(nil), signals.EventRules...),
		Metadata:      append([]MetadataRule(nil), signals.MetadataRules...),
		Classifiers:   append([]ClassifierSignalRule(nil), signals.ClassifierRules...),
		InputModality: append([]InputModalityRule(nil), signals.InputModalityRules...),
	}
}

func canonicalProjectionsFromProjections(projections Projections) CanonicalProjections {
	return CanonicalProjections{
		Partitions: append([]ProjectionPartition(nil), projections.Partitions...),
		Scores:     append([]ProjectionScore(nil), projections.Scores...),
		Mappings:   append([]ProjectionMapping(nil), projections.Mappings...),
	}
}

func routingModelsFromRouterConfig(cfg *RouterConfig) []RoutingModel {
	if cfg.EffectiveModelRegistry != nil {
		return routingModelOverridesFromEffectiveRegistry(cfg)
	}
	return routingModelsFromRuntimeConfig(cfg)
}

func routingModelOverridesFromEffectiveRegistry(cfg *RouterConfig) []RoutingModel {
	models := make([]RoutingModel, 0)
	for _, effective := range cfg.EffectiveModelRegistry.Models() {
		if !hasOperatorModelCardData(effective.Card) {
			continue
		}
		models = append(models, routingModelFromEffectiveModel(effective))
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Name < models[j].Name })
	return models
}

func hasOperatorModelCardData(card modelcatalog.EffectiveModelCard) bool {
	if len(card.LoRAs) > 0 || len(card.Evaluations) > 0 {
		return true
	}
	for _, source := range card.Provenance {
		if source == "operator" {
			return true
		}
	}
	return false
}

func routingModelFromEffectiveModel(effective modelcatalog.EffectiveModel) RoutingModel {
	card := effective.Card.Card
	provenance := effective.Card.Provenance
	model := RoutingModel{
		Name:        effective.Catalog,
		Evaluations: cloneUserEvaluations(effective.Card.Evaluations),
		LoRAs:       routingLoRAsFromEffectiveCard(effective.Card),
	}
	if provenance["display_name"] == "operator" {
		model.DisplayName = card.DisplayName
	}
	if provenance["family"] == "operator" {
		model.Family = card.Family
	}
	if provenance["parameter_size"] == "operator" {
		model.ParamSize = card.ParameterSize
	}
	if provenance["description"] == "operator" {
		model.Description = card.Description
	}
	applyOperatorModelCardLimits(&model, card, provenance)
	applyOperatorModelCardCollections(&model, card, provenance)
	if provenance["runtime_modality"] == "operator" {
		model.Modality = effective.Card.RuntimeModality
	}
	return model
}

func applyOperatorModelCardLimits(
	model *RoutingModel,
	card modelcatalog.ModelCard,
	provenance modelcatalog.FieldProvenance,
) {
	if provenance["limits.context_window_size"] == "operator" {
		model.ContextWindowSize = card.Limits.ContextWindowSize
	}
	if provenance["limits.max_output_tokens"] == "operator" {
		model.MaxOutputTokens = card.Limits.MaxOutputTokens
	}
}

func applyOperatorModelCardCollections(
	model *RoutingModel,
	card modelcatalog.ModelCard,
	provenance modelcatalog.FieldProvenance,
) {
	if provenance["capabilities"] == "operator" {
		model.Capabilities = append([]string(nil), card.Capabilities...)
	}
	if provenance["modalities"] == "operator" {
		modalities := card.Modalities
		model.Modalities = &modalities
	}
	if provenance["tags"] == "operator" {
		model.Tags = append([]string(nil), card.Tags...)
	}
}

func routingLoRAsFromEffectiveCard(card modelcatalog.EffectiveModelCard) []LoRAAdapter {
	return loraAdaptersFromEffectiveCard(card)
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

func routingModelsFromRuntimeConfig(cfg *RouterConfig) []RoutingModel {
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
		cardName := params.Catalog
		if cardName == "" {
			cardName = name
		}
		models = append(models, RoutingModel{
			Name:              cardName,
			ParamSize:         params.ParamSize,
			ContextWindowSize: params.ContextWindowSize,
			Description:       params.Description,
			Capabilities:      append([]string(nil), params.Capabilities...),
			LoRAs:             copyLoRAAdapters(params.LoRAs),
			Tags:              append([]string(nil), params.Tags...),
			Evaluations:       cloneUserEvaluations(params.Evaluations),
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
			AutoModelNames:            canonicalAutoModelNames(cfg.AutoModelNames),
			IncludeConfigModelsInList: cfg.IncludeConfigModelsInList,
			ClearRouteCache:           cfg.ClearRouteCache,
			StreamedBody: CanonicalStreamedBody{
				Enabled:    cfg.StreamedBodyMode,
				MaxBytes:   cfg.MaxStreamedBodyBytes,
				TimeoutSec: cfg.StreamedBodyTimeoutSec,
			},
			SkipProcessing: cfg.SkipProcessing,
			ModelSelection: cfg.ModelSelection,
			Learning:       cfg.RouterLearning,
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
		},
		Integrations: CanonicalIntegrationGlobal{
			Tools:  cfg.Tools,
			Looper: cfg.Looper,
		},
		ModelCatalog: canonicalModelCatalogFromRouterConfig(cfg),
	}

	return global
}

func canonicalAutoModelNames(names []string) *[]string {
	if names == nil {
		return nil
	}
	cloned := append([]string{}, names...)
	return &cloned
}

func canonicalModelCatalogFromRouterConfig(cfg *RouterConfig) CanonicalModelCatalog {
	categoryModel := cfg.CategoryModel
	if err := normalizeCanonicalCategoryVariant(&categoryModel); err != nil {
		// Export is intentionally non-validating. Preserve an invalid runtime
		// value so the normal configuration validator reports the actionable
		// error instead of silently changing it during serialization.
		categoryModel = cfg.CategoryModel
	}

	return CanonicalModelCatalog{
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
		External:  append([]ExternalModelConfig(nil), cfg.ExternalModels...),
		KBs:       append([]KnowledgeBaseConfig(nil), cfg.KnowledgeBases...),
		Admission: cloneAdmissionMap(cfg.ModelAdmission),
		Modules: CanonicalModelModules{
			PromptCompression: cfg.PromptCompression,
			PromptGuard: CanonicalPromptGuardModule{
				PromptGuardConfig: cfg.PromptGuard,
				ModelRef:          "prompt_guard",
			},
			Classifier: CanonicalClassifierModule{
				Domain: CanonicalCategoryModule{
					CategoryModel: categoryModel,
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
				Enabled: cfg.HallucinationMitigation.Enabled,
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
		Name:             name,
		Catalog:          params.Catalog,
		APIFormat:        params.APIFormat,
		Pricing:          params.Pricing,
		Reliability:      params.Reliability,
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

func cloneCanonicalReasoning(reasoning *CanonicalReasoning) *CanonicalReasoning {
	if reasoning == nil {
		return nil
	}
	clone := *reasoning
	clone.Levels = append([]string(nil), reasoning.Levels...)
	return &clone
}

func cloneCanonicalProviderModel(model *CanonicalProviderModel) *CanonicalProviderModel {
	if model == nil {
		return nil
	}
	clone := *model
	clone.Reasoning = cloneCanonicalReasoning(model.Reasoning)
	clone.ExternalModelIDs = copyStringMap(model.ExternalModelIDs)
	clone.BackendRefs = make([]CanonicalBackendRef, len(model.BackendRefs))
	for index, backend := range model.BackendRefs {
		clone.BackendRefs[index] = backend
		clone.BackendRefs[index].ExtraHeaders = copyStringMap(backend.ExtraHeaders)
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
		Provider:   profile.Type,
		BaseURL:    profile.BaseURL,
		AuthHeader: profile.AuthHeader,
		AuthPrefix: profile.AuthPrefix,
		APIVersion: profile.APIVersion,
		ChatPath:   profile.ChatPath,
		APIKey:     endpoint.APIKey,
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
