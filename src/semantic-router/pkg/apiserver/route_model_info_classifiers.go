//go:build !windows && cgo

package apiserver

import (
	"fmt"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func (s *ClassificationAPIServer) classifierModelAvailability() classifierModelAvailability {
	if s == nil || s.classificationSvc == nil {
		return classifierModelAvailability{}
	}

	return classifierModelAvailability{
		core:                   s.classificationSvc.HasClassifier(),
		factCheck:              s.classificationSvc.HasFactCheckClassifier(),
		hallucination:          s.classificationSvc.HasHallucinationDetector(),
		hallucinationExplainer: s.classificationSvc.HasHallucinationExplainer(),
		feedback:               s.classificationSvc.HasFeedbackDetector(),
	}
}

// getClassifierModelsInfo returns information about configured classifier models.
func (s *ClassificationAPIServer) getClassifierModelsInfo(
	availability classifierModelAvailability,
	runtimeState *startupstatus.State,
) []ModelInfo {
	cfg := s.currentConfig()
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
				"model_type":   resolveInlineModelType(categoryModel.UseMmBERT32K, categoryModel.UseModernBERT, false),
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
		models = append(models, ModelInfo{
			Name:      "jailbreak_classifier",
			Type:      "security_detection",
			Loaded:    availability.core,
			ModelPath: promptGuard.ModelID,
			Metadata: map[string]string{
				"enabled":                "true",
				"jailbreak_mapping_path": promptGuard.JailbreakMappingPath,
				"model_type":             resolveInlineModelType(promptGuard.UseMmBERT32K, promptGuard.UseModernBERT, false),
			},
		})
	}

	return models
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
	models = append(models, ModelInfo{
		Name:      "hallucination_detector",
		Type:      "hallucination_detection",
		Loaded:    availability.hallucination,
		ModelPath: hallucinationModel.ModelID,
		Metadata: map[string]string{
			"model_type":            "modernbert",
			"threshold":             fmt.Sprintf("%.2f", hallucinationModel.Threshold),
			"min_span_length":       fmt.Sprintf("%d", hallucinationModel.MinSpanLength),
			"min_span_confidence":   fmt.Sprintf("%.2f", hallucinationModel.MinSpanConfidence),
			"context_window_size":   fmt.Sprintf("%d", hallucinationModel.ContextWindowSize),
			"nli_filtering_enabled": fmt.Sprintf("%t", hallucinationModel.EnableNLIFiltering),
			"use_cpu":               fmt.Sprintf("%t", hallucinationModel.UseCPU),
		},
	})

	nliModel := cfg.HallucinationMitigation.NLIModel
	if nliModel.ModelID != "" {
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
