//go:build !windows

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
		core:          service.HasClassifier(),
		factCheck:     service.HasFactCheckClassifier(),
		hallucination: service.HasHallucinationDetector(),
		feedback:      service.HasFeedbackDetector(),
	}
	if inventory, ok := service.(classificationInventoryReadinessService); ok {
		availability.factCheck = inventory.HasAnyFactCheckClassifier()
		availability.hallucination = inventory.HasAnyHallucinationDetector()
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
				"model_type":   localModelType(categoryModel.Backend),
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
				"model_type":   localModelType(piiModel.Backend),
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
				"backend":                localModelType(promptGuard.Backend),
			},
		})
	}

	return models
}

// localModelType names how a classifier module runs: the built-in model
// runtime, or a remote classifier's transport.
func localModelType(backend *routerconfig.RemoteClassifierBackend) string {
	if backend != nil {
		return backend.Protocol
	}
	return routerconfig.ModelRuntimeProvider
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
				"model_type": routerconfig.ModelRuntimeProvider,
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
		// The binding plan is the source of truth for a remote detector: a
		// recipe may bind a chat service (http_chat) or a token_spans service
		// (http_classify) that is not an OpenAI-compatible endpoint.
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
		metadata["use_cpu"] = fmt.Sprintf("%t", hallucinationModel.UseCPU)
	}
	models = append(models, ModelInfo{
		Name:      "hallucination_detector",
		Type:      "hallucination_detection",
		Loaded:    availability.hallucination,
		ModelPath: hallucinationModel.ModelID,
		Metadata:  metadata,
	})

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
				"model_type": routerconfig.ModelRuntimeProvider,
				"threshold":  fmt.Sprintf("%.2f", feedbackModel.Threshold),
				"use_cpu":    fmt.Sprintf("%t", feedbackModel.UseCPU),
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
		return routerconfig.ResolvedModelBinding{}, false
	}
	spec, ok := plan.Lookup(recipe, "hallucination_detector")
	if !ok || spec.Deployment.Provider != "http" {
		return routerconfig.ResolvedModelBinding{}, false
	}
	return spec, true
}
