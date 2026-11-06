package controllers

import (
	"fmt"

	"gopkg.in/yaml.v3"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (r *SemanticRouterReconciler) applyOperatorConfigSpec(canonical *routerconfig.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
	if err := r.applyOperatorModelCatalog(canonical, spec); err != nil {
		return err
	}
	if err := applyOperatorDecisionModel(canonical, spec); err != nil {
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
		canonical.Global.Router.Strategy = routerconfig.RoutingStrategy(spec.Strategy)
	}
	if spec.StreamedBody != nil {
		canonical.Global.Router.StreamedBody = routerconfig.CanonicalStreamedBody{
			Enabled:    spec.StreamedBody.Enabled,
			MaxBytes:   spec.StreamedBody.MaxBytes,
			TimeoutSec: spec.StreamedBody.TimeoutSec,
		}
	}
	return nil
}

func (r *SemanticRouterReconciler) applyOperatorModelCatalog(canonical *routerconfig.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
	if err := applyOperatorModelDeployments(canonical, spec); err != nil {
		return err
	}
	if spec.EmbeddingModels != nil {
		embeddings, err := convertToTypedConfig[routerconfig.EmbeddingModels](r, spec.EmbeddingModels)
		if err != nil {
			return fmt.Errorf("config.embedding_models: %w", err)
		}
		canonical.Global.ModelCatalog.Embeddings.Semantic = embeddings
	}

	if spec.PromptGuard != nil {
		return fmt.Errorf("config.prompt_guard is removed: the decision model answers jailbreak rules (a threshold is a jailbreak rule's predicate)")
	}
	if spec.Classifier != nil {
		return fmt.Errorf("config.classifier is removed: the decision model answers domain and pii rules (a threshold is the rule's predicate)")
	}
	if spec.ComplexityModel != nil {
		return fmt.Errorf("config.complexity_model is removed: the decision model answers complexity rules")
	}
	if len(spec.ExternalModels) > 0 {
		// The CRD block mirrors the router's external catalog entry field for
		// field, so the generic typed conversion is enough; backend blocks
		// resolve their model against this list at router config load.
		external, err := convertToTypedConfig[[]routerconfig.ExternalModelConfig](r, spec.ExternalModels)
		if err != nil {
			return fmt.Errorf("config.external_models: %w", err)
		}
		canonical.Global.ModelCatalog.External = external
	}
	return nil
}

func (r *SemanticRouterReconciler) applyOperatorStoresAndIntegrations(canonical *routerconfig.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
	responseCache, err := operatorResponseCacheConfig(spec)
	if err != nil {
		return err
	}
	if responseCache != nil {
		typedCache, err := convertToTypedConfig[routerconfig.ResponseCacheStoreConfig](r, responseCache)
		if err != nil {
			return fmt.Errorf("config.response_cache: %w", err)
		}
		canonical.Global.Stores.ResponseCache = typedCache
	}
	return nil
}

func operatorResponseCacheConfig(
	spec vllmv1alpha1.ConfigSpec,
) (*vllmv1alpha1.SemanticCacheConfig, error) {
	if spec.ResponseCache != nil && spec.SemanticCache != nil {
		return nil, fmt.Errorf("config.response_cache conflicts with deprecated config.semantic_cache")
	}
	if spec.ResponseCache != nil {
		return spec.ResponseCache, nil
	}
	return spec.SemanticCache, nil
}

func (r *SemanticRouterReconciler) applyOperatorProviderDefaults(canonical *routerconfig.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
	if spec.ReasoningEffort != "" {
		canonical.Providers.Defaults.DefaultReasoningEffort = spec.ReasoningEffort
	}
	return nil
}

func (r *SemanticRouterReconciler) applyOperatorServices(canonical *routerconfig.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
	if spec.API != nil {
		api, err := convertToTypedConfig[routerconfig.APIConfig](r, spec.API)
		if err != nil {
			return fmt.Errorf("config.api: %w", err)
		}
		canonical.Global.Services.API = api
	}
	if spec.Observability != nil {
		observability, err := convertToTypedConfig[routerconfig.ObservabilityConfig](r, spec.Observability)
		if err != nil {
			return fmt.Errorf("config.observability: %w", err)
		}
		canonical.Global.Services.Observability = observability
	}
	return nil
}

func (r *SemanticRouterReconciler) applyOperatorRouting(canonical *routerconfig.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
	if spec.Routing != nil {
		routing, fields, err := canonicalRoutingFromKubernetesJSON(spec.Routing)
		if err != nil {
			return formatCanonicalRoutingOverrideError(err)
		}
		applyCanonicalRoutingOverrides(canonical, routing, fields)
	}
	if len(spec.ComplexityRules) > 0 {
		complexity, err := r.convertComplexityRules(spec.ComplexityRules)
		if err != nil {
			return fmt.Errorf("config.complexity_rules: %w", err)
		}
		canonical.Routing.Signals.Complexity = complexity
	}
	if len(spec.Decisions) > 0 {
		decisions, err := convertToTypedConfig[[]routerconfig.Decision](r, spec.Decisions)
		if err != nil {
			return fmt.Errorf("config.decisions: %w", err)
		}
		canonical.Routing.Decisions = decisions
	}
	return nil
}

func (r *SemanticRouterReconciler) convertComplexityRules(spec []vllmv1alpha1.ComplexityRulesConfig) ([]routerconfig.ComplexityRule, error) {
	rules := make([]routerconfig.ComplexityRule, 0, len(spec))
	for _, entry := range spec {
		if entry.Hard != nil || entry.Easy != nil || entry.Composer != nil || entry.PrototypeScoring != nil ||
			entry.Threshold != "" || entry.HardAbove != "" || entry.EasyBelow != "" || entry.HardBelow != "" || entry.EasyAbove != "" {
			return nil, fmt.Errorf("%s: a complexity rule no longer scores example banks; the decision model answers it (write labels)", entry.Name)
		}
		rule, err := convertToTypedConfig[routerconfig.ComplexityRule](r, entry)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", entry.Name, err)
		}
		rules = append(rules, rule)
	}
	return rules, nil
}

func convertToTypedConfig[T any](r *SemanticRouterReconciler, value interface{}) (T, error) {
	var result T

	normalized := r.convertToConfigMap(value)
	data, err := yaml.Marshal(normalized)
	if err != nil {
		return result, err
	}
	if err := yaml.Unmarshal(data, &result); err != nil {
		return result, err
	}

	return result, nil
}
