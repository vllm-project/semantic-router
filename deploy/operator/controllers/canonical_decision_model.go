package controllers

import (
	"fmt"
	"strings"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// applyOperatorDecisionModel writes the decision model and lets the modules
// follow it. The ConfigMap names no system line, so every module runs the
// decision model's model; and a module the resource leaves at its defaults,
// or a prompt guard that names no model or threshold, takes the thresholds
// calibrated for that model. The operator writes every module field, so the
// Router would otherwise read the default thresholds as set.
func applyOperatorDecisionModel(canonical *routerconfig.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
	decision, err := routerconfig.LookupDecisionModel(spec.DecisionModel)
	if err != nil {
		return fmt.Errorf("config.%w", err)
	}
	catalog := &canonical.Global.ModelCatalog
	catalog.System = routerconfig.CanonicalSystemModels{}
	if decision.Name != routerconfig.DefaultDecisionModel {
		catalog.System.DecisionModel = decision.Name
	}
	modules := &catalog.Modules
	if spec.PromptGuard == nil || strings.TrimSpace(spec.PromptGuard.Threshold) == "" {
		model := modules.PromptGuard.ModelID
		if model == "" {
			model = decision.System.PromptGuard
		}
		if modules.PromptGuard.Backend == nil {
			modules.PromptGuard.Threshold = routerconfig.ModuleThresholdsOf(model).PromptGuard
		}
	}
	if spec.Classifier == nil {
		modules.Classifier.Domain.Threshold = routerconfig.ModuleThresholdsOf(decision.System.DomainClassifier).Domain
		modules.Classifier.PII.Threshold = routerconfig.ModuleThresholdsOf(decision.System.PIIClassifier).PII
	}
	modules.HallucinationMitigation.FactCheck.Threshold = routerconfig.ModuleThresholdsOf(decision.System.FactCheckClassifier).FactCheck
	modules.FeedbackDetector.Threshold = routerconfig.ModuleThresholdsOf(decision.System.FeedbackDetector).Feedback
	return nil
}
