package modelresearch

import (
	"fmt"
	"strings"

	modelinventory "github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelinventory"
)

type recipeDefinition struct {
	Key                       string
	Label                     string
	GoalTemplates             []GoalTemplate
	DefaultDataset            string
	DatasetHint               string
	DefaultSuccessThresholdPP float64
	PrimaryMetric             string
	RuntimeNames              []string
	FallbackModelID           string
	FallbackDescription       string
	SupportsDatasetOverride   bool
}

var recipeCatalog = map[string]recipeDefinition{
	"feedback": {
		Key:                       "feedback",
		Label:                     "Improve feedback classifier accuracy",
		GoalTemplates:             []GoalTemplate{GoalImproveAccuracy},
		DefaultDataset:            "llm-semantic-router/feedback-detector-dataset",
		DatasetHint:               "Hugging Face dataset id or local path when the feedback trainer supports it.",
		DefaultSuccessThresholdPP: 0.5,
		PrimaryMetric:             "accuracy",
		RuntimeNames:              []string{"feedback_detector"},
		FallbackModelID:           "llm-semantic-router/mmbert-feedback-detector-merged",
		FallbackDescription:       "Current MoM feedback detector baseline.",
		SupportsDatasetOverride:   true,
	},
	"fact-check": {
		Key:                       "fact-check",
		Label:                     "Improve fact-check classifier accuracy",
		GoalTemplates:             []GoalTemplate{GoalImproveAccuracy},
		DefaultDataset:            "llm-semantic-router/fact-check-classification-dataset",
		DatasetHint:               "Offline eval accepts a Hugging Face dataset id or local JSON/CSV path.",
		DefaultSuccessThresholdPP: 0.5,
		PrimaryMetric:             "accuracy",
		RuntimeNames:              []string{"fact_check_classifier"},
		FallbackModelID:           "llm-semantic-router/mmbert-fact-check-merged",
		FallbackDescription:       "Current MoM fact-check baseline.",
		SupportsDatasetOverride:   true,
	},
	"jailbreak": {
		Key:                       "jailbreak",
		Label:                     "Improve jailbreak classifier accuracy",
		GoalTemplates:             []GoalTemplate{GoalImproveAccuracy},
		DefaultDataset:            "llm-semantic-router/jailbreak-detection-dataset",
		DatasetHint:               "Offline eval accepts a Hugging Face dataset id or local JSON/CSV path.",
		DefaultSuccessThresholdPP: 0.5,
		PrimaryMetric:             "accuracy",
		RuntimeNames:              []string{"jailbreak_classifier"},
		FallbackModelID:           "llm-semantic-router/mmbert-jailbreak-detector-merged",
		FallbackDescription:       "Current MoM jailbreak detector baseline.",
		SupportsDatasetOverride:   true,
	},
	"intent": {
		Key:                       "intent",
		Label:                     "Improve intent classifier accuracy",
		GoalTemplates:             []GoalTemplate{GoalImproveAccuracy},
		DefaultDataset:            "TIGER-Lab/MMLU-Pro",
		DatasetHint:               "Offline eval accepts a local JSON/CSV override. Training continues to use the built-in MMLU-Pro intent corpus.",
		DefaultSuccessThresholdPP: 0.5,
		PrimaryMetric:             "accuracy",
		RuntimeNames:              []string{"category_classifier"},
		FallbackModelID:           "llm-semantic-router/mmbert-intent-classifier-merged",
		FallbackDescription:       "Current MoM intent classifier baseline.",
		SupportsDatasetOverride:   true,
	},
	"pii": {
		Key:                       "pii",
		Label:                     "Improve PII classifier accuracy",
		GoalTemplates:             []GoalTemplate{GoalImproveAccuracy},
		DefaultDataset:            "presidio",
		DatasetHint:               "Offline eval accepts a local JSON/CSV override. Training defaults to Presidio plus AI4Privacy unless advanced hints disable it.",
		DefaultSuccessThresholdPP: 0.5,
		PrimaryMetric:             "accuracy",
		RuntimeNames:              []string{"pii_classifier"},
		FallbackModelID:           "llm-semantic-router/mmbert-pii-detector-merged",
		FallbackDescription:       "Current MoM PII detector baseline.",
		SupportsDatasetOverride:   true,
	},
	"domain": {
		Key:                       "domain",
		Label:                     "Explore a new signal classifier",
		GoalTemplates:             []GoalTemplate{GoalExploreSignal},
		DefaultDataset:            "mmlu-prox-en",
		DatasetHint:               "Provide a signal hypothesis, then optionally override the evaluation dataset id from signal_eval.py, for example mmlu-prox-zh or mmlu-pro-en.",
		DefaultSuccessThresholdPP: 0.5,
		PrimaryMetric:             "accuracy",
		RuntimeNames:              []string{"category_classifier"},
		FallbackModelID:           "llm-semantic-router/mmbert-intent-classifier-merged",
		FallbackDescription:       "Current MoM signal-classifier baseline.",
		SupportsDatasetOverride:   true,
	},
}

func recipesResponse(
	defaultAPIBase string,
	defaultRequestModel string,
	defaultPlatform string,
	runtimeModels *modelinventory.ModelsInfoResponse,
) RecipesResponse {
	items := make([]RecipeSummary, 0, len(recipeCatalog))
	for _, key := range []string{"feedback", "fact-check", "jailbreak", "intent", "pii", "domain"} {
		def := recipeCatalog[key]
		items = append(items, RecipeSummary{
			Key:                         def.Key,
			Label:                       def.Label,
			GoalTemplates:               def.GoalTemplates,
			DefaultDataset:              def.DefaultDataset,
			DatasetHint:                 def.DatasetHint,
			DefaultSuccessThresholdPP:   def.DefaultSuccessThresholdPP,
			PrimaryMetric:               def.PrimaryMetric,
			SupportsDatasetOverride:     def.SupportsDatasetOverride,
			SupportsHyperparameterHints: true,
			Baseline:                    resolveBaseline(def, runtimeModels, defaultRequestModel),
		})
	}

	return RecipesResponse{
		DefaultAPIBase:      defaultAPIBase,
		DefaultRequestModel: defaultRequestModel,
		DefaultPlatform:     defaultPlatform,
		RuntimeModels:       runtimeModels,
		Recipes:             items,
	}
}

func resolveRecipe(target string, goal GoalTemplate) (recipeDefinition, error) {
	def, ok := recipeCatalog[target]
	if !ok {
		return recipeDefinition{}, fmt.Errorf("unsupported target %q", target)
	}
	for _, allowed := range def.GoalTemplates {
		if allowed == goal {
			return def, nil
		}
	}
	return recipeDefinition{}, fmt.Errorf("target %q does not support goal template %q", target, goal)
}

func resolveBaseline(
	def recipeDefinition,
	runtimeModels *modelinventory.ModelsInfoResponse,
	requestModel string,
) Baseline {
	baseline := Baseline{
		Label:        def.Label,
		Source:       "fallback",
		ModelID:      def.FallbackModelID,
		Description:  def.FallbackDescription,
		RequestModel: requestModel,
	}

	if runtimeModels == nil {
		return baseline
	}

	for _, candidate := range runtimeModels.Models {
		for _, runtimeName := range def.RuntimeNames {
			if !strings.EqualFold(candidate.Name, runtimeName) {
				continue
			}
			baseline.Source = "runtime"
			baseline.RuntimeName = candidate.Name
			baseline.ModelPath = firstNonEmpty(candidate.ResolvedModelPath, candidate.ModelPath)
			baseline.ModelID = firstNonEmpty(baseline.ModelPath, def.FallbackModelID)
			baseline.State = candidate.State
			baseline.Categories = append([]string(nil), candidate.Categories...)
			if baseline.Description == "" && candidate.Registry != nil && candidate.Registry.Description != "" {
				baseline.Description = candidate.Registry.Description
			}
			if baseline.Description == "" {
				baseline.Description = def.FallbackDescription
			}
			return baseline
		}
	}

	return baseline
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed != "" {
			return trimmed
		}
	}
	return ""
}
