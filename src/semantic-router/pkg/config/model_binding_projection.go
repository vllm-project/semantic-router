package config

import (
	"fmt"
	"slices"
	"strings"
)

// ProjectRecipeModelBindings returns a preparation-only view of the exact recipe
// deployment declarations. The source config stays immutable for export/reload.
func ProjectRecipeModelBindings(cfg *RouterConfig, plan *ModelBindingPlan, recipe RecipeName) (*RouterConfig, error) {
	if cfg == nil {
		return nil, fmt.Errorf("model binding projection requires configuration")
	}
	scoped := *cfg
	scoped.ClassifierRules = slices.Clone(cfg.ClassifierRules)
	for name := range scoped.ModelBindings {
		spec, ok := plan.Lookup(recipe, name)
		if !ok {
			return nil, fmt.Errorf("model binding %q is absent from recipe %q", name, recipe)
		}
		if strings.HasPrefix(name, "classifier.") {
			if rule := classifierSignalRuleByName(scoped.ClassifierRules, strings.TrimPrefix(name, "classifier.")); rule != nil {
				*rule = projectGenericClassifierRule(*rule, spec.Deployment)
			}
			continue
		}
		var remote *RemoteClassifierBackend
		if spec.Deployment.Provider == "http" {
			remote = &RemoteClassifierBackend{Model: spec.Deployment.ExternalModel, Protocol: spec.Binding.Adapter, Contract: spec.Binding.Contract}
		}
		artifact := ResolveModelPath(spec.Deployment.Artifact)
		mapping := spec.Binding.MappingPath
		switch name {
		case "domain_classifier":
			scoped.CategoryModel.ModelID = artifact
			scoped.CategoryModel.Backend = remote
			// The provider adapter is already validated/resolved separately. These
			// legacy family flags must not override an explicit recipe binding.
			scoped.CategoryModel.Variant = ""
			scoped.CategoryModel.UseModernBERT = false
			scoped.CategoryModel.UseMmBERT32K = false
			if mapping != "" {
				scoped.CategoryMappingPath = mapping
			}
		case "pii_classifier":
			scoped.PIIModel.ModelID = artifact
			scoped.PIIModel.Backend = remote
			if mapping != "" {
				scoped.PIIMappingPath = mapping
			}
		case "prompt_guard":
			scoped.PromptGuard.ModelID = artifact
			scoped.PromptGuard.Backend = remote
			scoped.PromptGuard.Protocol = ""
			scoped.PromptGuard.Variant = ""
			if mapping != "" {
				scoped.PromptGuard.JailbreakMappingPath = mapping
			}
		case "complexity":
			if remote == nil {
				return nil, fmt.Errorf("complexity binding requires a remote score or distribution adapter")
			}
			scoped.ComplexityModel.Backend = remote
		case "fact_check_classifier":
			scoped.HallucinationMitigation.FactCheckModel.ModelID = artifact
		case "feedback_detector":
			scoped.FeedbackDetector.ModelID = artifact
			if mapping != "" {
				scoped.FeedbackDetector.FeedbackMappingPath = mapping
			}
		case "hallucination_detector":
			scoped.HallucinationMitigation.HallucinationModel.ModelID = artifact
			if remote != nil {
				external, err := ResolveRemoteClassifierBackend(&scoped, remote, ModelRoleClassification, RemoteClassifierContractTokenSpans)
				if err != nil {
					return nil, err
				}
				scoped.HallucinationMitigation.HallucinationModel.ModelID = external.ModelName
				scoped.HallucinationMitigation.HallucinationModel.Backend = HallucinationBackendEndpoint
			} else {
				scoped.HallucinationMitigation.HallucinationModel.Backend = "candle"
				scoped.HallucinationMitigation.HallucinationModel.Endpoint = ""
			}
		case "modality_detector":
			if scoped.ModalityDetector.Classifier != nil {
				classifier := *scoped.ModalityDetector.Classifier
				classifier.ModelPath = artifact
				scoped.ModalityDetector.Classifier = &classifier
			}
		case "hallucination_explainer":
			scoped.HallucinationMitigation.NLIModel.ModelID = artifact
		}
	}
	return &scoped, nil
}
