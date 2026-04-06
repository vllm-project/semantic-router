package controllers

import (
	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routercontract "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routercontract"
)

func (r *SemanticRouterReconciler) applyOperatorProviderDefaults(
	canonical *routercontract.CanonicalConfig,
	spec vllmv1alpha1.ConfigSpec,
) error {
	if spec.ReasoningFamilies != nil {
		canonical.Providers.Defaults.ReasoningFamilies = convertReasoningFamilies(spec.ReasoningFamilies)
	}
	if spec.DefaultReasoningEffort != "" {
		canonical.Providers.Defaults.DefaultReasoningEffort = spec.DefaultReasoningEffort
	}
	return nil
}

func convertReasoningFamilies(
	spec map[string]vllmv1alpha1.ReasoningFamily,
) map[string]routercontract.ReasoningFamilyConfig {
	if len(spec) == 0 {
		return nil
	}
	result := make(map[string]routercontract.ReasoningFamilyConfig, len(spec))
	for name, family := range spec {
		result[name] = routercontract.ReasoningFamilyConfig{
			Type:      family.Type,
			Parameter: family.Parameter,
		}
	}
	return result
}
