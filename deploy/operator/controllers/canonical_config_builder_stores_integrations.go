package controllers

import (
	"fmt"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routercontract "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routercontract"
)

func (r *SemanticRouterReconciler) applyOperatorStoresAndIntegrations(
	canonical *routercontract.CanonicalConfig,
	spec vllmv1alpha1.ConfigSpec,
) error {
	if spec.SemanticCache != nil {
		semanticCache, err := convertToTypedConfig[routercontract.SemanticCache](r, spec.SemanticCache)
		if err != nil {
			return fmt.Errorf("config.semantic_cache: %w", err)
		}
		canonical.Global.Stores.SemanticCache = semanticCache
	}
	if spec.Tools != nil {
		tools, err := convertToTypedConfig[routercontract.ToolsConfig](r, spec.Tools)
		if err != nil {
			return fmt.Errorf("config.tools: %w", err)
		}
		canonical.Global.Integrations.Tools = tools
	}
	return nil
}
