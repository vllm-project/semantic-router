package controllers

import (
	"fmt"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// applyOperatorComplexityModel carries spec.complexity_model onto
// global.model_catalog.modules.complexity. The CRD block mirrors the router's
// field for field, so the generic typed conversion is enough; the router's
// own validator then decides whether the backend resolves, which keeps one
// source of truth for what a valid backend is.
func (r *SemanticRouterReconciler) applyOperatorComplexityModel(
	canonical *routerconfig.CanonicalConfig,
	spec vllmv1alpha1.ConfigSpec,
) error {
	if spec.ComplexityModel == nil {
		return nil
	}
	complexity, err := convertToTypedConfig[routerconfig.ComplexityModelConfig](r, spec.ComplexityModel)
	if err != nil {
		return fmt.Errorf("config.complexity_model: %w", err)
	}
	canonical.Global.ModelCatalog.Modules.Complexity = complexity
	return nil
}
