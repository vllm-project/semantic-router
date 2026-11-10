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
	if !ok {
		return fmt.Errorf("config.decision_model.deployment %q must name a declared deployment", selected.Deployment)
	}
	if !deployment.IsModelRuntime() && !deployment.IsSystemOne() {
		return fmt.Errorf("config.decision_model.deployment %q must use provider %s or %s, not %q",
			selected.Deployment, routerconfig.ModelRuntimeProvider, routerconfig.SystemOneProvider, deployment.Provider)
	}
	catalog.System.DecisionModel = selected
	return nil
}
