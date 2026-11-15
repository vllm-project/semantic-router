package controllers

import (
	"context"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (r *SemanticRouterReconciler) buildCanonicalConfig(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) (*routerconfig.CanonicalConfig, error) {
	// Zero-valued fields serialize as explicit false/zero overrides, so start
	// from the Router's defaults: what the Operator does not set keeps the
	// Router's value. Tracing and the Response API stay off: the Operator
	// deploys neither the trace collector nor the Redis their defaults name.
	defaults := routerconfig.DefaultCanonicalGlobal()
	defaults.Services.Observability.Tracing.Enabled = false
	defaults.Services.ResponseAPI.Enabled = false
	canonical := &routerconfig.CanonicalConfig{
		Version: routerconfig.CanonicalConfigVersion,
		Listeners: []routerconfig.Listener{
			{
				Name:    DefaultListenerName,
				Address: "0.0.0.0",
				Port:    int(DefaultListenerPort),
				Timeout: "300s",
			},
		},
		Recipes: []routerconfig.CanonicalRecipe{{
			Name: string(routerconfig.DefaultRecipeName),
			Signals: routerconfig.CanonicalSignals{
				Domains: []routerconfig.Category{
					{
						Name:        "general",
						Description: "General queries",
						Labels:      []string{"other"},
					},
				},
			},
		}},
		Providers: routerconfig.CanonicalProviders{
			Defaults: routerconfig.CanonicalProviderDefaults{},
			Models:   []routerconfig.CanonicalProviderModel{},
		},
		Global: &defaults,
	}

	if err := r.applyDiscoveredBackends(ctx, canonical, sr); err != nil {
		return nil, err
	}
	if err := r.applyOperatorConfigSpec(canonical, sr.Spec.Config); err != nil {
		return nil, err
	}

	return canonical, nil
}
