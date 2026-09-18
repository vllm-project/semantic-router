//go:build !windows && cgo

package apiserver

import (
	"fmt"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func classificationAvailabilityForService(service classificationService) classifierModelAvailability {
	if service == nil {
		return classifierModelAvailability{}
	}

	availability := classifierModelAvailability{
		core:                   service.HasClassifier(),
		factCheck:              service.HasFactCheckClassifier(),
		hallucination:          service.HasHallucinationDetector(),
		hallucinationExplainer: service.HasHallucinationExplainer(),
		feedback:               service.HasFeedbackDetector(),
	}
	if inventory, ok := service.(classificationInventoryReadinessService); ok {
		availability.factCheck = inventory.HasAnyFactCheckClassifier()
		availability.hallucination = inventory.HasAnyHallucinationDetector()
		availability.hallucinationExplainer = inventory.HasAnyHallucinationExplainer()
		availability.feedback = inventory.HasAnyFeedbackDetector()
	}
	return availability
}

// getClassifierModelsInfo returns information about configured classifier models.
func (s *ClassificationAPIServer) getClassifierModelsInfo(
	cfg *routerconfig.RouterConfig,
	availability classifierModelAvailability,
	runtimeState *startupstatus.State,
) []ModelInfo {
	if cfg == nil {
		return s.getPlaceholderModelsInfo(runtimeState)
	}

	models := appendConfiguredModels(nil, cfg, availability)

	for i := range models {
		models[i] = enrichModelInfo(models[i], runtimeState)
	}

	return models
}

func appendConfiguredModels(
	models []ModelInfo,
	cfg *routerconfig.RouterConfig,
	availability classifierModelAvailability,
) []ModelInfo {
	models = append(models, buildRoutingClassifierModels(cfg, availability)...)
	models = append(models, buildHallucinationModels(cfg, availability)...)
	models = append(models, buildFeedbackAndSimilarityModels(cfg, availability)...)
	return models
}

func buildRoutingClassifierModels(
	cfg *routerconfig.RouterConfig,
	availability classifierModelAvailability,
) []ModelInfo {
	var models []ModelInfo
	categoryModel := cfg.CategoryModel
	if cfg.IsCategoryClassifierEnabled() {
		models = append(models, ModelInfo{
			Name:       "category_classifier",
			Type:       "intent_classification",
			Loaded:     availability.core,
			ModelPath:  categoryModel.ModelID,
			Categories: configuredCategoryNames(cfg),
			Metadata: map[string]string{
				"mapping_path": categoryModel.CategoryMappingPath,
				"model_type":   categoryModelInfoType(categoryModel),
				"threshold":    fmt.Sprintf("%.2f", categoryModel.Threshold),
			},
		})
	}

	piiModel := cfg.PIIModel
	if cfg.IsPIIClassifierEnabled() {
		models = append(models, ModelInfo{
			Name:      "pii_classifier",
			Type:      "pii_detection",
			Loaded:    availability.core,
			ModelPath: piiModel.ModelID,
			Metadata: map[string]string{
				"mapping_path": piiModel.PIIMappingPath,
				"model_type":   resolveInlineModelType(piiModel.UseMmBERT32K, false, true),
				"threshold":    fmt.Sprintf("%.2f", piiModel.Threshold),
			},
		})
	}

	promptGuard := cfg.PromptGuard
	if cfg.IsPromptGuardEnabled() {
		backend := promptGuard.Protocol
		if backend == "" {
			backend = promptGuard.Variant
		}
		if backend == "" {
			backend = routerconfig.PromptGuardVariantCandle
		}
		models = append(models, ModelInfo{
			Name:      "jailbreak_classifier",
			Type:      "security_detection",
			Loaded:    availability.core,
			ModelPath: promptGuard.ModelID,
			Metadata: map[string]string{
				"enabled":                "true",
				"jailbreak_mapping_path": promptGuard.JailbreakMappingPath,
				"backend":                backend,
			},
		})
	}

	return models
}

func categoryModelInfoType(model routerconfig.CategoryModel) string {
	if model.Backend != nil {
		// Match the existing prompt_guard convention: a remote classifier reports
		// its effective transport rather than pretending to be a local model.
		return model.Backend.Protocol
	}
	if variant, err := model.EffectiveVariant(); err == nil && variant != "" {
		return variant
	}
	return resolveInlineModelType(model.UseMmBERT32K, model.UseModernBERT, false)
}

func buildHallucinationModels(
	cfg *routerconfig.RouterConfig,
	availability classifierModelAvailability,
) []ModelInfo {
	var models []ModelInfo
	factCheckModel := cfg.HallucinationMitigation.FactCheckModel
	if cfg.IsFactCheckClassifierEnabled() {
		models = append(models, ModelInfo{
			Name:      "fact_check_classifier",
			Type:      "fact_check_classification",
			Loaded:    availability.factCheck,
			ModelPath: factCheckModel.ModelID,
			Metadata: map[string]string{
				"model_type": resolveInlineModelType(factCheckModel.UseMmBERT32K, false, false),
				"threshold":  fmt.Sprintf("%.2f", factCheckModel.Threshold),
				"use_cpu":    fmt.Sprintf("%t", factCheckModel.UseCPU),
			},
		})
	}

	if !cfg.IsHallucinationModelEnabled() {
		return models
	}

	hallucinationModel := cfg.HallucinationMitigation.HallucinationModel
	hallucinationBackend := hallucinationModel.NormalizedBackend()
	metadata := map[string]string{
		"backend":    hallucinationBackend,
		"model_type": "modernbert",
		"lifecycle":  "router_local",
	}
	if remote, ok := remoteHallucinationBinding(cfg); ok {
		// The binding plan is the source of truth for a remote detector: the
		// legacy scalar desugars into it, and a recipe may bind a token_spans
		// service (http_classify) that is not an OpenAI-compatible endpoint.
		metadata = map[string]string{
			"backend":    routerconfig.HallucinationBackendEndpoint,
			"model_type": "openai_compatible_endpoint",
			"lifecycle":  "external",
			"adapter":    remote.Binding.Adapter,
			"contract":   remote.Binding.Contract,
		}
		if remote.Binding.Adapter == routerconfig.RemoteClassifierProtocolHTTPClassify {
			metadata["model_type"] = "token_spans_endpoint"
		} else {
			metadata["include_explanation"] = fmt.Sprintf("%t", hallucinationModel.IncludeExplanation)
		}
	} else {
		metadata["threshold"] = fmt.Sprintf("%.2f", hallucinationModel.Threshold)
		metadata["min_span_length"] = fmt.Sprintf("%d", hallucinationModel.MinSpanLength)
		metadata["min_span_confidence"] = fmt.Sprintf("%.2f", hallucinationModel.MinSpanConfidence)
		metadata["context_window_size"] = fmt.Sprintf("%d", hallucinationModel.ContextWindowSize)
		metadata["nli_filtering_enabled"] = fmt.Sprintf("%t", hallucinationModel.EnableNLIFiltering)
		metadata["use_cpu"] = fmt.Sprintf("%t", hallucinationModel.UseCPU)
	}
	models = append(models, ModelInfo{
		Name:      "hallucination_detector",
		Type:      "hallucination_detection",
		Loaded:    availability.hallucination,
		ModelPath: hallucinationModel.ModelID,
		Metadata:  metadata,
	})

	nliModel := cfg.HallucinationMitigation.NLIModel
	if cfg.NeedsLocalHallucinationNLIForAPI() ||
		cfg.NeedsLocalHallucinationNLIForRouting() ||
		cfg.NeedsLocalNLIForSemanticCache() {
		models = append(models, ModelInfo{
			Name:      "hallucination_explainer",
			Type:      "nli_explainer",
			Loaded:    availability.hallucinationExplainer,
			ModelPath: nliModel.ModelID,
			Metadata: map[string]string{
				"model_type": "modernbert_nli",
				"threshold":  fmt.Sprintf("%.2f", nliModel.Threshold),
				"use_cpu":    fmt.Sprintf("%t", nliModel.UseCPU),
			},
		})
	}

	return models
}

func buildFeedbackAndSimilarityModels(
	cfg *routerconfig.RouterConfig,
	availability classifierModelAvailability,
) []ModelInfo {
	var models []ModelInfo
	feedbackModel := cfg.FeedbackDetector
	if cfg.IsFeedbackDetectorEnabled() {
		models = append(models, ModelInfo{
			Name:      "feedback_detector",
			Type:      "feedback_detection",
			Loaded:    availability.feedback,
			ModelPath: feedbackModel.ModelID,
			Metadata: map[string]string{
				"model_type": resolveInlineModelType(feedbackModel.UseMmBERT32K, feedbackModel.UseModernBERT, false),
				"threshold":  fmt.Sprintf("%.2f", feedbackModel.Threshold),
				"use_cpu":    fmt.Sprintf("%t", feedbackModel.UseCPU),
			},
		})
	}

	bertModelPath := cfg.BertModelPath
	if bertModelPath != "" {
		models = append(models, ModelInfo{
			Name:      "bert_similarity_model",
			Type:      "similarity",
			Loaded:    availability.core,
			ModelPath: bertModelPath,
			Metadata: map[string]string{
				"model_type": "sentence_transformer",
				"threshold":  fmt.Sprintf("%.2f", cfg.MinSimilarityThreshold()),
				"use_cpu":    fmt.Sprintf("%t", cfg.UseCPU),
			},
		})
	}

	return models
}

func configuredCategoryNames(cfg *routerconfig.RouterConfig) []string {
	categories := make([]string, 0, len(cfg.Categories))
	for _, cat := range cfg.Categories {
		categories = append(categories, cat.Name)
	}
	return categories
}

// getPlaceholderModelsInfo returns placeholder model information.
func (s *ClassificationAPIServer) getPlaceholderModelsInfo(runtimeState *startupstatus.State) []ModelInfo {
	models := []ModelInfo{
		placeholderModelInfo("category_classifier", "intent_classification"),
		placeholderModelInfo("pii_classifier", "pii_detection"),
		placeholderModelInfo("jailbreak_classifier", "security_detection"),
		placeholderModelInfo("fact_check_classifier", "fact_check_classification"),
		placeholderModelInfo("hallucination_detector", "hallucination_detection"),
		placeholderModelInfo("hallucination_explainer", "nli_explainer"),
		placeholderModelInfo("feedback_detector", "feedback_detection"),
	}

	for i := range models {
		models[i] = enrichModelInfo(models[i], runtimeState)
	}

	return models
}

func placeholderModelInfo(name, modelType string) ModelInfo {
	return ModelInfo{
		Name:   name,
		Type:   modelType,
		Loaded: false,
		Metadata: map[string]string{
			"status": "not_initialized",
		},
	}
}

func resolveInlineModelType(useMmBERT32K, useModernBERT, tokenLevel bool) string {
	switch {
	case useMmBERT32K && tokenLevel:
		return "mmbert_32k_token"
	case useMmBERT32K:
		return "mmbert_32k"
	case useModernBERT && tokenLevel:
		return "modernbert_token"
	case useModernBERT:
		return "modernbert"
	case tokenLevel:
		return "bert_token"
	default:
		return "bert"
	}
}

// remoteHallucinationBinding reports the hallucination detector's remote
// binding when the compiled plan has one for the default recipe. A config that
// does not compile falls back to the legacy scalar so model info still
// answers.
func remoteHallucinationBinding(cfg *routerconfig.RouterConfig) (routerconfig.ResolvedModelBinding, bool) {
	recipe := cfg.RoutingScope
	if recipe == "" {
		recipe = routerconfig.DefaultRecipeName
	}
	plan, err := routerconfig.CompileModelBindings(cfg)
	if err != nil {
		if cfg.HallucinationMitigation.HallucinationModel.NormalizedBackend() == routerconfig.HallucinationBackendEndpoint {
			return routerconfig.ResolvedModelBinding{Binding: routerconfig.ModelBinding{Adapter: routerconfig.RemoteClassifierProtocolHTTPChat, Contract: routerconfig.RemoteClassifierContractTokenSpans}}, true
		}
		return routerconfig.ResolvedModelBinding{}, false
	}
	spec, ok := plan.Lookup(recipe, "hallucination_detector")
	if !ok || spec.Deployment.Provider != "http" {
		return routerconfig.ResolvedModelBinding{}, false
	}
	return spec, true
}
