package controllers

import (
	"fmt"

	"gopkg.in/yaml.v3"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// applyOperatorConfigSpec maps v1alpha1 ConfigSpec (CRD) into the router canonical
// v0.3 surface. Family-specific branches live in applyOperator* helpers below so
// canonical_config_builder.go stays orchestration-only.
func (r *SemanticRouterReconciler) applyOperatorConfigSpec(canonical *routerconfig.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
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

func (r *SemanticRouterReconciler) applyOperatorModelCatalog(canonical *routerconfig.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
	if spec.EmbeddingModels != nil {
		embeddings, err := convertToTypedConfig[routerconfig.EmbeddingModels](r, spec.EmbeddingModels)
		if err != nil {
			return fmt.Errorf("config.embedding_models: %w", err)
		}
		canonical.Global.ModelCatalog.Embeddings.Semantic = embeddings
	}

	if spec.PromptGuard != nil {
		promptGuard, err := convertToTypedConfig[routerconfig.CanonicalPromptGuardModule](r, spec.PromptGuard)
		if err != nil {
			return fmt.Errorf("config.prompt_guard: %w", err)
		}
		canonical.Global.ModelCatalog.Modules.PromptGuard = promptGuard
	}

	if spec.Classifier != nil {
		classifier, err := r.convertClassifierModule(spec.Classifier)
		if err != nil {
			return fmt.Errorf("config.classifier: %w", err)
		}
		canonical.Global.ModelCatalog.Modules.Classifier = classifier
	}
	return nil
}

func (r *SemanticRouterReconciler) applyOperatorStoresAndIntegrations(canonical *routerconfig.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
	if spec.SemanticCache != nil {
		semanticCache, err := convertToTypedConfig[routerconfig.SemanticCache](r, spec.SemanticCache)
		if err != nil {
			return fmt.Errorf("config.semantic_cache: %w", err)
		}
		canonical.Global.Stores.SemanticCache = semanticCache
	}
	if spec.Tools != nil {
		tools, err := convertToTypedConfig[routerconfig.ToolsConfig](r, spec.Tools)
		if err != nil {
			return fmt.Errorf("config.tools: %w", err)
		}
		canonical.Global.Integrations.Tools = tools
	}
	return nil
}

func (r *SemanticRouterReconciler) applyOperatorProviderDefaults(canonical *routerconfig.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
	if spec.ReasoningFamilies != nil {
		canonical.Providers.Defaults.ReasoningFamilies = convertReasoningFamilies(spec.ReasoningFamilies)
	}
	if spec.DefaultReasoningEffort != "" {
		canonical.Providers.Defaults.DefaultReasoningEffort = spec.DefaultReasoningEffort
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

func (r *SemanticRouterReconciler) convertClassifierModule(spec *vllmv1alpha1.ClassifierConfig) (routerconfig.CanonicalClassifierModule, error) {
	if spec == nil {
		return routerconfig.CanonicalClassifierModule{}, nil
	}

	var classifier routerconfig.CanonicalClassifierModule

	if spec.CategoryModel != nil {
		domain, err := convertToTypedConfig[routerconfig.CanonicalCategoryModule](r, spec.CategoryModel)
		if err != nil {
			return routerconfig.CanonicalClassifierModule{}, fmt.Errorf("domain: %w", err)
		}
		classifier.Domain = domain
	}

	if spec.PIIModel != nil {
		pii, err := convertToTypedConfig[routerconfig.CanonicalPIIModule](r, spec.PIIModel)
		if err != nil {
			return routerconfig.CanonicalClassifierModule{}, fmt.Errorf("pii: %w", err)
		}
		classifier.PII = pii
	}

	return classifier, nil
}

func (r *SemanticRouterReconciler) convertComplexityRules(spec []vllmv1alpha1.ComplexityRulesConfig) ([]routerconfig.ComplexityRule, error) {
	rules := make([]routerconfig.ComplexityRule, 0, len(spec))
	for _, entry := range spec {
		rule, err := convertToTypedConfig[routerconfig.ComplexityRule](r, entry)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", entry.Name, err)
		}
		if entry.Composer != nil {
			rule.Composer = &routerconfig.RuleCombination{
				Operator:   entry.Composer.Operator,
				Conditions: convertCompositionConditions(entry.Composer.Conditions),
			}
		}
		rules = append(rules, rule)
	}
	return rules, nil
}

func convertCompositionConditions(conditions []vllmv1alpha1.CompositionCondition) []routerconfig.RuleNode {
	result := make([]routerconfig.RuleNode, 0, len(conditions))
	for _, condition := range conditions {
		result = append(result, routerconfig.RuleNode{
			Type: condition.Type,
			Name: condition.Name,
		})
	}
	return result
}

func convertReasoningFamilies(spec map[string]vllmv1alpha1.ReasoningFamily) map[string]routerconfig.ReasoningFamilyConfig {
	if len(spec) == 0 {
		return nil
	}
	result := make(map[string]routerconfig.ReasoningFamilyConfig, len(spec))
	for name, family := range spec {
		result[name] = routerconfig.ReasoningFamilyConfig{
			Type:      family.Type,
			Parameter: family.Parameter,
		}
	}
	return result
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
