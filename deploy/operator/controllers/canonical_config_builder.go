package controllers

import (
	"context"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routercontract "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routercontract"
)

func (r *SemanticRouterReconciler) buildCanonicalConfig(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) (*routercontract.CanonicalConfig, error) {
	canonical := &routercontract.CanonicalConfig{
		Version: "v0.3",
		Listeners: []routercontract.Listener{
			{
				Name:    "grpc-50051",
				Address: "0.0.0.0",
				Port:    50051,
				Timeout: "300s",
			},
		},
		Routing: routercontract.CanonicalRouting{
			Signals: routercontract.CanonicalSignals{
				Domains: []routercontract.Category{
					{
						CategoryMetadata: routercontract.CategoryMetadata{
							Name:           "general",
							Description:    "General queries",
							MMLUCategories: []string{"other"},
						},
					},
				},
			},
		},
		Providers: routercontract.CanonicalProviders{
			Defaults: routercontract.CanonicalProviderDefaults{},
			Models:   []routercontract.CanonicalProviderModel{},
		},
		Global: &routercontract.CanonicalGlobal{
			Services:     routercontract.CanonicalServiceGlobal{},
			Stores:       routercontract.CanonicalStoreGlobal{},
			Integrations: routercontract.CanonicalIntegrationGlobal{},
			ModelCatalog: routercontract.CanonicalModelCatalog{
				Embeddings: routercontract.CanonicalEmbeddingModels{},
				System:     routercontract.CanonicalSystemModels{},
				Modules:    routercontract.CanonicalModelModules{},
			},
		},
	}

	if err := r.applyDiscoveredBackends(ctx, canonical, sr); err != nil {
		return nil, err
	}
	if err := r.applyOperatorConfigSpec(canonical, sr.Spec.Config); err != nil {
		return nil, err
	}

	return canonical, nil
}

func (r *SemanticRouterReconciler) applyOperatorConfigSpec(canonical *routercontract.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
	if err := r.applyOperatorModelCatalog(canonical, spec); err != nil {
		return err
	}
	if err := r.applyOperatorStoresAndIntegrations(canonical, spec); err != nil {
		return err
	}
	if err := r.applyOperatorProviderDefaults(canonical, spec); err != nil {
		return err
	}
	if err := r.applyOperatorServices(canonical, spec); err != nil {
		return err
	}
	if err := r.applyOperatorRouting(canonical, spec); err != nil {
		return err
	}
	if spec.Strategy != "" {
		canonical.Global.Router.Strategy = spec.Strategy
	}
	return nil
}
