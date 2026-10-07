package config

import (
	"fmt"
	"strings"
)

// validateDecisionLooperBackends refuses a decision that makes model calls of
// its own (a Looper algorithm, a prompt helper or context recovery) to a model
// without a backend. The Router makes those calls in process, through the
// model's providers.models[].backend_refs, in either gateway mode.
func validateDecisionLooperBackends(cfg *RouterConfig, decision Decision) error {
	if cfg.RoutingFragmentOnly {
		return nil
	}
	for _, model := range decisionCalledModels(decision) {
		if len(cfg.GetEndpointsForModel(model)) == 0 {
			return fmt.Errorf(
				"decision %q: the Router calls model %q in process, but it has no backend; give it providers.models[].backend_refs",
				decision.Name, model,
			)
		}
	}
	return nil
}

// decisionCalledModels lists the models a decision calls in process, beyond
// the one it routes the request to.
func decisionCalledModels(decision Decision) []string {
	var models []string
	refs := func() {
		for _, ref := range decision.ModelRefs {
			models = append(models, ref.Model)
		}
	}
	if algorithm := decision.Algorithm; algorithm != nil {
		switch {
		case IsLooperAlgorithmType(algorithm.Type):
			refs()
			if algorithm.Fusion != nil {
				models = append(models, algorithm.Fusion.AnalysisModels...)
				if algorithm.Fusion.Model != "" {
					models = append(models, algorithm.Fusion.Model)
				}
			}
			if algorithm.Workflows != nil {
				if planner := strings.TrimSpace(algorithm.Workflows.Planner.Model); planner != "" {
					models = append(models, planner)
				}
			}
		case algorithm.Prompt != nil:
			models = append(models, algorithm.Prompt.Model)
		}
	}
	if compression := decision.GetContextCompressionConfig(); compression != nil &&
		compression.Recovery != nil && compression.Recovery.Enabled {
		refs()
	}
	return models
}
