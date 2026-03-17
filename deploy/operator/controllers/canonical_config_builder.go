package controllers

import (
	"context"
	"fmt"
	"sort"

	"gopkg.in/yaml.v3"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (r *SemanticRouterReconciler) buildCanonicalConfig(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) (*routerconfig.CanonicalConfig, error) {
	canonical := &routerconfig.CanonicalConfig{
		Version: "v0.3",
		Listeners: []routerconfig.Listener{
			{
				Name:    "grpc-50051",
				Address: "0.0.0.0",
				Port:    50051,
				Timeout: "300s",
			},
		},
		Routing: routerconfig.CanonicalRouting{
			Signals: routerconfig.CanonicalSignals{
				Domains: []routerconfig.Category{
					{
						CategoryMetadata: routerconfig.CategoryMetadata{
							Name:           "general",
							Description:    "General queries",
							MMLUCategories: []string{"other"},
						},
					},
				},
			},
		},
		Providers: routerconfig.CanonicalProviders{
			Defaults: routerconfig.CanonicalProviderDefaults{},
			Models:   []routerconfig.CanonicalProviderModel{},
		},
		Global: &routerconfig.CanonicalGlobal{
			Services:     routerconfig.CanonicalServiceGlobal{},
			Stores:       routerconfig.CanonicalStoreGlobal{},
			Integrations: routerconfig.CanonicalIntegrationGlobal{},
			ModelCatalog: routerconfig.CanonicalModelCatalog{
				Embeddings: routerconfig.CanonicalEmbeddingModels{},
				System:     routerconfig.CanonicalSystemModels{},
				Modules:    routerconfig.CanonicalModelModules{},
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

func (r *SemanticRouterReconciler) applyDiscoveredBackends(ctx context.Context, canonical *routerconfig.CanonicalConfig, sr *vllmv1alpha1.SemanticRouter) error {
	if len(sr.Spec.VLLMEndpoints) == 0 {
		return nil
	}

	discoveredModels, err := discoverVLLMBackends(ctx, r.Client, sr.Spec.VLLMEndpoints, sr.Namespace)
	if err != nil {
		return fmt.Errorf("failed to generate vLLM endpoints config: %w", err)
	}
	if len(discoveredModels) == 0 {
		return nil
	}

	modelNames := make([]string, 0, len(discoveredModels))
	for modelName := range discoveredModels {
		modelNames = append(modelNames, modelName)
	}
	sort.Strings(modelNames)

	for index, modelName := range modelNames {
		discovered := discoveredModels[modelName]

		modelCard := routerconfig.RoutingModel{
			Name:  modelName,
			LoRAs: convertLoRAAdapters(discovered.LoRAs),
		}
		canonical.Routing.ModelCards = append(canonical.Routing.ModelCards, modelCard)
		canonical.Providers.Models = append(canonical.Providers.Models, routerconfig.CanonicalProviderModel{
			Name:            modelName,
			ReasoningFamily: discovered.ReasoningFamily,
			BackendRefs:     append([]routerconfig.CanonicalBackendRef(nil), discovered.BackendRefs...),
		})

		if index == 0 {
			canonical.Providers.Defaults.DefaultModel = modelName
		}
	}

	return nil
}

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

func convertLoRAAdapters(spec []vllmv1alpha1.LoRAAdapterSpec) []routerconfig.LoRAAdapter {
	if len(spec) == 0 {
		return nil
	}
	loras := make([]routerconfig.LoRAAdapter, 0, len(spec))
	for _, adapter := range spec {
		loras = append(loras, routerconfig.LoRAAdapter{
			Name:        adapter.Name,
			Description: adapter.Description,
		})
	}
	return loras
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
