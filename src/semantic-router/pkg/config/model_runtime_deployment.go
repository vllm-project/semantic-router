package config

import (
	"fmt"
	"net/url"
	"path/filepath"
	"regexp"
	"strings"
)

// ModelRuntimeProvider serves a deployment through the built-in model runtime
// (src/model-runtime). Without an endpoint the Router starts and supervises
// the runtime process on a private Unix socket; with one it attaches to an
// engine it does not manage.
const ModelRuntimeProvider = "model_runtime"

var (
	// The runtime owns profile and accelerator names, plugins' included: the
	// Router checks their shape and leaves an unknown name to the runtime.
	modelRuntimeProfile  = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)
	modelRuntimeDevice   = regexp.MustCompile(`^[a-z][a-z0-9_]*(:[0-9]+)?$`)
	modelRuntimeRevision = regexp.MustCompile(`^[0-9a-f]{40}$`)
	hubRepositoryID      = regexp.MustCompile(`^[A-Za-z0-9][\w.-]*/[\w.-]+$`)
)

// IsModelRuntime reports whether the deployment is served by the built-in model runtime.
func (d ModelDeployment) IsModelRuntime() bool {
	return d.Provider == ModelRuntimeProvider
}

// Managed reports whether the Router starts and supervises the runtime process.
func (d ModelDeployment) Managed() bool {
	if !d.IsModelRuntime() {
		return false
	}
	for _, placement := range d.Placements() {
		if placement.Endpoint == "" {
			return true
		}
	}
	return false
}

// ServedModel names the model a consumer of the deployment called name runs:
// its artifact, or, for an attached runtime without one, the model the runtime
// serves under served_name or the deployment's name.
func (d ModelDeployment) ServedModel(name string) string {
	if d.Artifact != "" || !d.IsModelRuntime() || d.Managed() {
		return d.Artifact
	}
	if d.ServedName != "" {
		return d.ServedName
	}
	return name
}

// ModelRuntimeDeploymentsInUse returns the model_runtime deployments that the
// configuration uses, with defaults applied: those a decision signal or a
// decision algorithm names, the decision model's for a decision signal or
// algorithm that names none, and those an active task binding names in the
// top-level routing surface, a recipe, or the global service catalog.
// Declared but unused deployments are never started.
func ModelRuntimeDeploymentsInUse(cfg *RouterConfig) map[string]ModelDeployment {
	used := make(map[string]ModelDeployment)
	if cfg == nil {
		return used
	}
	mark := func(name string) {
		if name == "" {
			if shared, deployment, ok, err := cfg.DecisionModelDeployment(); ok && err == nil {
				used[shared] = deployment.WithDefaults()
			}
			return
		}
		if deployment, ok := cfg.ModelDeployments[name]; ok && deployment.IsModelRuntime() {
			used[name] = deployment.WithDefaults()
		}
	}
	for _, listener := range cfg.Listeners {
		if listener.SystemOne == nil {
			continue
		}
		for _, publicName := range listener.SystemOne.Models {
			if name, _, err := cfg.ResolveSystemOneDeployment(publicName); err == nil {
				mark(name)
			}
		}
	}
	if !cfg.RoutingEnabled() {
		mark("")
		return used
	}
	scan := func(signals Signals, decisions []Decision) {
		for _, rule := range signals.DecisionRules {
			mark(rule.Deployment)
		}
		for _, decision := range decisions {
			if decision.Algorithm != nil && decision.Algorithm.Decision != nil &&
				strings.EqualFold(strings.TrimSpace(decision.Algorithm.Type), DecisionAlgorithmDecision) {
				mark(decision.Algorithm.Decision.Deployment)
			}
		}
	}
	scan(cfg.Signals, cfg.Decisions)
	for _, recipe := range cfg.Recipes {
		scan(recipe.Profile.Signals, recipe.Profile.Decisions)
	}
	for name, deployment := range modelRuntimeTaskDeploymentsInUse(cfg) {
		used[name] = deployment
	}
	return used
}

// modelRuntimeTaskDeploymentsInUse resolves the binding plan and keeps the
// model_runtime deployments whose consumer is active in its own scope.
// An invalid plan contributes nothing: validation reports it separately.
func modelRuntimeTaskDeploymentsInUse(cfg *RouterConfig) map[string]ModelDeployment {
	used := make(map[string]ModelDeployment)
	plan, err := compileModelBindings(cfg)
	if err != nil {
		return used
	}
	for name, spec := range plan.global {
		if spec.Deployment.IsModelRuntime() && TaskConsumerInUse(cfg, GlobalModelScope, name) {
			used[spec.Binding.Deployment] = spec.Deployment
		}
	}
	scopes := cfg.consumerScopes()
	for recipe, bindings := range plan.recipes {
		scoped, ok := scopes[recipe]
		if !ok {
			continue
		}
		for name, spec := range bindings {
			if spec.Deployment.IsModelRuntime() && TaskConsumerInUse(scoped, recipe, name) {
				used[spec.Binding.Deployment] = spec.Deployment
			}
		}
	}
	for name, deployment := range implicitTaskDeploymentsInUse(cfg, plan) {
		used[name] = deployment
	}
	for name, deployment := range implicitEmbeddingDeploymentsInUse(cfg, plan) {
		used[name] = deployment
	}
	return used
}

// consumerScopes returns the scoped configuration of every recipe that may
// prepare model consumers: the request-reachable recipes and the default
// recipe, which also owns the public APIs.
func (c *RouterConfig) consumerScopes() map[RecipeName]*RouterConfig {
	scopes := make(map[RecipeName]*RouterConfig)
	if len(c.Recipes) == 0 {
		recipe := c.RoutingScope
		if recipe == "" {
			recipe = DefaultRecipeName
		}
		scopes[recipe] = c
		return scopes
	}
	for i := range c.Recipes {
		recipe := &c.Recipes[i]
		if recipe.Name == DefaultRecipeName || c.IsRecipeReachableForRouting(recipe.Name) {
			scopes[recipe.Name] = c.ConfigForRecipe(recipe)
		}
	}
	return scopes
}

// TaskConsumerInUse reports whether the named task consumer runs in a scope:
// a signal or plugin of a reachable routing profile reads it, or a default
// public API serves it. scoped is the recipe-scoped configuration (the root
// configuration for the global service scope).
func TaskConsumerInUse(scoped *RouterConfig, scope RecipeName, name string) bool {
	if scoped == nil {
		return false
	}
	if scope == GlobalModelScope {
		if name != "embedding" {
			return false
		}
		// Recipe bindings may override the global default. Only shared services
		// can activate the global owner directly; inherited recipe bindings are
		// accounted for in their own scopes.
		services := scoped.ConfigForGlobalModelServices()
		primary := services.primaryEmbeddingModel()
		return EmbeddingModelsNeeded(services, primary, true)[primary]
	}
	switch name {
	case "domain_classifier":
		return scoped.UsesSignalTypeInReachableRouting(SignalTypeDomain)
	case "prompt_guard":
		return scoped.UsesJailbreakClassifierInReachableRouting()
	case "pii_classifier":
		return scoped.UsesSignalTypeInReachableRouting(SignalTypePII)
	case "fact_check_classifier":
		return scoped.UsesSignalTypeInReachableRouting(SignalTypeFactCheck) ||
			(scoped.ownsDefaultAPIConsumer() && (scoped.HallucinationMitigation.Enabled || len(scoped.RoutingProfileSignals().FactCheckRules) > 0))
	case "feedback_detector":
		return scoped.UsesSignalTypeInReachableRouting(SignalTypeUserFeedback) ||
			(scoped.ownsDefaultAPIConsumer() && len(scoped.RoutingProfileSignals().UserFeedbackRules) > 0)
	case "modality_detector":
		return scoped.UsesSignalTypeInReachableRouting(SignalTypeModality)
	case "preference":
		return scoped.UsesSignalTypeInReachableRouting(SignalTypePreference) && scoped.PreferenceUsesDecisionTask()
	case "reask":
		return scoped.UsesSignalTypeInReachableRouting(SignalTypeReask) && scoped.ReaskUsesDecisionTask()
	case "complexity":
		_, bound := scoped.EffectiveModelBindings(scoped.Signals, scoped.ModelBindings)["complexity"]
		if !scoped.UsesSignalTypeInReachableRouting(SignalTypeComplexity) {
			return false
		}
		if bound {
			return true
		}
		if scoped.ComplexityModel.Backend == nil {
			for _, rule := range scoped.ComplexityRules {
				if !scoped.ComplexityRuleUsesPrototypes(rule) {
					return true
				}
			}
		}
		return false
	case "hallucination_detector":
		return scoped.NeedsHallucinationDetectorForRouting() ||
			(scoped.ownsDefaultAPIConsumer() && scoped.HallucinationMitigation.Enabled)
	case "embedding":
		primary := scoped.primaryEmbeddingModel()
		return EmbeddingModelsNeeded(scoped, primary, false)[primary]
	case RAGRerankerConsumer:
		return scoped.NeedsRAGReranker()
	}
	switch {
	case strings.HasPrefix(name, "classifier."):
		return scoped.UsesSignalTypeInReachableRouting(SignalTypeClassifier)
	case strings.HasPrefix(name, "safety."):
		return scoped.UsesSignalTypeInReachableRouting(SignalTypeSafety)
	}
	return false
}

func (c *RouterConfig) primaryEmbeddingModel() string {
	if model := strings.ToLower(strings.TrimSpace(c.EmbeddingConfig.ModelType)); model != "" {
		return model
	}
	return "qwen3"
}

func (d ModelDeployment) validateModelRuntime() error {
	if d.ExternalModel != "" {
		return fmt.Errorf("model_runtime deployments cannot set external_model")
	}
	if d.Input.MaxTokens < 0 {
		return fmt.Errorf("input.max_tokens must not be negative")
	}
	switch d.Input.Overflow {
	case "reject", "truncate", "window":
	default:
		return fmt.Errorf("unsupported input.overflow %q", d.Input.Overflow)
	}
	if !modelRuntimeProfile.MatchString(d.Profile) {
		return fmt.Errorf("profile must be a profile name, such as exact or batching")
	}
	if d.Revision != "" && !modelRuntimeRevision.MatchString(d.Revision) {
		return fmt.Errorf("revision must be a 40-hex commit")
	}
	if err := d.validateReplicas(); err != nil {
		return err
	}
	if !d.Managed() && len(d.Replicas) == 0 {
		return nil
	}
	artifact := strings.TrimSpace(d.Artifact)
	if artifact == "" {
		return fmt.Errorf("a managed model_runtime deployment requires artifact (a Hub repository or an absolute package path)")
	}
	if !filepath.IsAbs(artifact) && !hubRepositoryID.MatchString(artifact) {
		return fmt.Errorf("artifact must be a Hub repository ID or an absolute package path")
	}
	if filepath.IsAbs(artifact) && d.Revision != "" {
		return fmt.Errorf("revision applies only to Hub repositories")
	}
	return nil
}

func validateModelRuntimeEndpoint(endpoint string) error {
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return fmt.Errorf("endpoint: %w", err)
	}
	switch parsed.Scheme {
	case "unix":
		if !filepath.IsAbs(parsed.Path) || parsed.Host != "" {
			return fmt.Errorf("endpoint unix:// needs an absolute socket path (unix:///run/vllm-sr/runtime.sock)")
		}
	case "http", "https":
		if parsed.Host == "" {
			return fmt.Errorf("endpoint needs a host")
		}
	default:
		return fmt.Errorf("endpoint must use unix://, http:// or https://")
	}
	return nil
}
