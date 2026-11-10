package controllers

import (
	"encoding/json"
	"fmt"
	"strconv"

	"gopkg.in/yaml.v3"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (r *SemanticRouterReconciler) applyOperatorConfigSpec(canonical *routerconfig.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
	resource := &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{Config: spec}}
	if err := resource.ValidateDecisionRuleLimits(); err != nil {
		return err
	}
	if spec.DecisionRuleLimits != nil {
		limits := routerconfig.DecisionRuleLimits{
			MaxDepth: spec.DecisionRuleLimits.MaxDepth,
			MaxNodes: spec.DecisionRuleLimits.MaxNodes,
		}
		if _, _, err := limits.Effective(); err != nil {
			return err
		}
		canonical.Global.Router.DecisionRuleLimits = limits
	}
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
	if err := applyOperatorResponseCacheSwitch(canonical, spec); err != nil {
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
		return fmt.Errorf("config.embedding_models is removed: an embedding model is a model deployment, and global.model_catalog.embeddings.default names the one stores embed with")
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
	if spec.ExternalModels != nil {
		return fmt.Errorf("config.external_models is removed: a remote model is a model deployment with provider openai_chat, openai_embeddings or systemone")
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
	cache, field := spec.ResponseCache, "config.response_cache"
	if cache == nil {
		cache, field = spec.SemanticCache, "config.semantic_cache"
	}
	if cache != nil && cache.EmbeddingModel != "" {
		return nil, fmt.Errorf("%s.embedding_model is removed: embedding.deployment names the deployment semantic reuse embeds with, embeddings.default without one", field)
	}
	return cache, nil
}

// applyOperatorResponseCacheSwitch keeps what config.response_cache's enabled
// and similarity_threshold do on a router whose store has no switch: on, a
// configuration without decisions caches every request in a response_cache
// plugin of global.plugins, bounded by the threshold; off, the decisions'
// response_cache plugins are off.
func applyOperatorResponseCacheSwitch(canonical *routerconfig.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
	cache, err := operatorResponseCacheConfig(spec)
	if err != nil || cache == nil {
		return err
	}
	decisions := 0
	for index := range canonical.Recipes {
		recipe := &canonical.Recipes[index]
		decisions += len(recipe.Decisions)
		if cache.Enabled {
			continue
		}
		for decision := range recipe.Decisions {
			if disableErr := disableResponseCache(recipe.Decisions[decision].Plugins); disableErr != nil {
				return fmt.Errorf("config.response_cache: recipes[%s].decisions[%s]: %w", recipe.Name, recipe.Decisions[decision].Name, disableErr)
			}
		}
	}
	if !cache.Enabled || decisions > 0 {
		return nil
	}
	for _, plugin := range canonical.Global.Plugins {
		if routerconfig.NormalizeDecisionPluginType(plugin.Type) == routerconfig.DecisionPluginResponseCache {
			return nil
		}
	}
	configuration := map[string]interface{}{"enabled": true, "scope": "global"}
	if cache.SimilarityThreshold != "" {
		threshold, parseErr := strconv.ParseFloat(cache.SimilarityThreshold, 64)
		if parseErr != nil {
			return fmt.Errorf("config.response_cache.similarity_threshold: %w", parseErr)
		}
		configuration["semantic"] = map[string]interface{}{"match": map[string]interface{}{
			"recall": map[string]interface{}{"top_k": 1, "min_similarity": threshold},
		}}
	}
	payload, err := routerconfig.NewStructuredPayload(configuration)
	if err != nil {
		return fmt.Errorf("config.response_cache: %w", err)
	}
	canonical.Global.Plugins = append(canonical.Global.Plugins, routerconfig.DecisionPlugin{
		Type: routerconfig.DecisionPluginResponseCache, Configuration: payload,
	})
	return nil
}

func disableResponseCache(plugins []routerconfig.DecisionPlugin) error {
	for index := range plugins {
		if routerconfig.NormalizeDecisionPluginType(plugins[index].Type) != routerconfig.DecisionPluginResponseCache {
			continue
		}
		var configuration map[string]interface{}
		if plugins[index].Configuration != nil {
			if err := json.Unmarshal(plugins[index].Configuration.Raw, &configuration); err != nil {
				return err
			}
		}
		if configuration == nil {
			configuration = map[string]interface{}{}
		}
		configuration["enabled"] = false
		payload, err := routerconfig.NewStructuredPayload(configuration)
		if err != nil {
			return err
		}
		plugins[index].Configuration = payload
	}
	return nil
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
		return fmt.Errorf("config.routing is removed: the router configuration has no top-level routing block; write its fields on the recipe named default under config.recipes")
	}
	if err := applyOperatorRecipes(canonical, configRoutingSpec{
		recipes: spec.Recipes, entrypoints: spec.Entrypoints, plugins: spec.Plugins, models: spec.Models,
	}); err != nil {
		return err
	}
	if len(spec.ComplexityRules) > 0 {
		complexity, err := r.convertComplexityRules(spec.ComplexityRules)
		if err != nil {
			return fmt.Errorf("config.complexity_rules: %w", err)
		}
		defaultRecipe(canonical).Signals.Complexity = complexity
	}
	if len(spec.Decisions) > 0 {
		decisions, err := convertToTypedConfig[[]routerconfig.Decision](r, spec.Decisions)
		if err != nil {
			return fmt.Errorf("config.decisions: %w", err)
		}
		defaultRecipe(canonical).Decisions = decisions
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
