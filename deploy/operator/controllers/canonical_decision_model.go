package controllers

import (
	"fmt"
	"strings"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// applyOperatorDecisionModel writes a deployment reference; the Router owns
// capability validation and task interpretation independently of model families.
func applyOperatorDecisionModel(canonical *routerconfig.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
	catalog := &canonical.Global.ModelCatalog
	selected := catalog.System.DecisionModel
	if spec.DecisionModel != nil {
		selected.Deployment = spec.DecisionModel.Deployment
	}
	if selected.Deployment == "" || strings.TrimSpace(selected.Deployment) != selected.Deployment {
		return fmt.Errorf("config.decision_model.deployment must name a declared deployment")
	}
	deployment, ok := catalog.Deployments[selected.Deployment]
	if !ok || deployment.Provider != routerconfig.ModelRuntimeProvider {
		return fmt.Errorf("config.decision_model.deployment %q must name a declared model_runtime deployment", selected.Deployment)
	}
	catalog.System = routerconfig.CanonicalSystemModels{DecisionModel: selected}
	cfg := &routerconfig.RouterConfig{}
	cfg.DecisionModel, cfg.ModelDeployments = selected.Deployment, catalog.Deployments
	decision := cfg.DecisionModelSpec()
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
