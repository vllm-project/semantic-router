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

// modelRuntimeDtypes are the precisions a model_runtime backbone computes in.
var modelRuntimeDtypes = map[string]bool{"auto": true, "float32": true, "bfloat16": true, "float16": true}

var (
	// The runtime owns accelerator names, plugins' included: the Router checks
	// their shape and leaves an unknown name to the runtime.
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
	if d.Model != "" {
		return d.Model
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
	cfg = cfg.withModelResources()
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
			if cfg.IsSystemOneBackend(publicName) {
				if deployment := cfg.ModelConfig[publicName].Deployment; deployment != "" {
					mark(deployment)
				}
			}
			if entrypoint, ok := cfg.ResolveEntrypoint(SystemOneAPI, publicName); ok && cfg.RoutingEnabled() {
				if recipe := findRecipe(cfg.Recipes, entrypoint.Recipe); recipe != nil {
					for _, decision := range recipe.Profile.Decisions {
						for _, alias := range decision.Algorithm.NativeStageModels() {
							if deployment := cfg.ModelConfig[alias].Deployment; deployment != "" {
								mark(deployment)
							}
						}
					}
				}
			}
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
		for _, rule := range signals.SystemOneRules {
			mark(rule.Deployment)
		}
		for _, decision := range decisions {
			if decision.Algorithm != nil && decision.Algorithm.SystemOne != nil &&
				strings.EqualFold(strings.TrimSpace(decision.Algorithm.Type), DecisionAlgorithmSystemOne) {
				mark(decision.Algorithm.SystemOne.Deployment)
			}
			if decision.Algorithm.EscalateChecksAnswers() {
				deployment := ""
				if check := decision.Algorithm.Escalate.Accept.SystemOne; check != nil {
					deployment = check.Deployment
				}
				mark(deployment)
			}
			for _, judge := range decision.MatchJudges() {
				mark(judge.Deployment)
			}
		}
	}
	scan(cfg.Signals, cfg.Decisions)
	for _, recipe := range cfg.Recipes {
		scan(recipe.Profile.Signals, recipe.Profile.Decisions)
	}
	for _, scoped := range cfg.consumerScopes() {
		for _, rule := range scoped.ApplicationRules() {
			if scoped.AsksApplicationRule(rule) {
				mark(rule.Rule.Deployment)
			}
		}
		if _, grounds := scoped.GroundingFusionDecision(); grounds {
			deployment, _ := scoped.ApplicationTypeDeployment(SignalTypeHallucination)
			mark(deployment)
		}
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
		if spec.Deployment.IsModelRuntime() && spec.Deployment.MissingLocalModel() == nil && TaskConsumerInUse(cfg, GlobalModelScope, name) {
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
			if spec.Deployment.IsModelRuntime() && spec.Deployment.MissingLocalModel() == nil && TaskConsumerInUse(scoped, recipe, name) {
				used[spec.Binding.Deployment] = spec.Deployment
			}
		}
	}
	for name, deployment := range resolvedTaskDeploymentsInUse(cfg) {
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
	case "embedding":
		primary := scoped.primaryEmbeddingModel()
		return EmbeddingModelsNeeded(scoped, primary, false)[primary]
	case RAGRerankerConsumer:
		return scoped.NeedsRAGReranker()
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
	if err := d.validateEngineArgs(); err != nil {
		return err
	}
	if d.Revision != "" && !modelRuntimeRevision.MatchString(d.Revision) {
		return fmt.Errorf("revision must be a 40-hex commit")
	}
	if err := d.validateReplicas(); err != nil {
		return err
	}
	if err := d.validateAttachedConnection(); err != nil {
		return err
	}
	if d.Generation != nil {
		return fmt.Errorf("generation applies to openai_chat deployments")
	}
	if d.OperatingPoint != nil {
		if !d.Managed() {
			return fmt.Errorf("operating_point pins the policy a managed deployment's package serves; an attached runtime reports its own")
		}
		if err := d.OperatingPoint.Validate(); err != nil {
			return err
		}
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

// validateAttachedConnection checks the connection block of a model_runtime
// deployment: it applies to its attached endpoints, never to a worker the
// Router starts.
func (d ModelDeployment) validateAttachedConnection() error {
	if d.Capabilities != nil || len(d.QuestionTypes) > 0 {
		return fmt.Errorf("a model_runtime deployment's capabilities come from its model card; capabilities and question_types apply to remote deployments")
	}
	attached := false
	var endpoint *url.URL
	for _, placement := range d.Placements() {
		if placement.Endpoint == "" {
			continue
		}
		attached = true
		if parsed, err := url.Parse(placement.Endpoint); err == nil && parsed.Scheme != "unix" {
			endpoint = parsed
		}
	}
	if !attached {
		if d.hasConnectionBlock() {
			return fmt.Errorf("model, auth, headers, timeout_ms, max_retries, tls and limits apply to an attached endpoint, not to a worker the Router starts")
		}
		return nil
	}
	// The runtime contract answers each call once and bounds its own
	// bodies; a retry could repeat a batched call.
	if d.MaxRetries != nil || d.Limits != nil {
		return fmt.Errorf("max_retries and limits apply to remote providers; an attached model_runtime takes endpoint, model, auth, headers, timeout_ms and tls")
	}
	if d.Model != "" && d.ServedName != "" && d.Model != d.ServedName {
		return fmt.Errorf("model and served_name name the same served model; set one")
	}
	if d.TLS != nil && endpoint == nil {
		return fmt.Errorf("tls applies to an https:// endpoint")
	}
	return d.validateConnection(endpoint, MaxRemoteTimeoutMs)
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

// HasEngineArgs reports whether the deployment sets any engine argument.
func (d ModelDeployment) HasEngineArgs() bool {
	return d.MaxNumSeqs != nil || d.MaxNumBatchedTokens != nil || d.BatchWindowMS != nil || d.Dtype != "" || d.EnableSharedContext != nil || d.MaxModelLen != nil
}

func (d ModelDeployment) validateEngineArgs() error {
	if d.MaxNumSeqs != nil && *d.MaxNumSeqs < 1 {
		return fmt.Errorf("max_num_seqs must be at least 1")
	}
	if d.MaxNumBatchedTokens != nil && *d.MaxNumBatchedTokens < 8 {
		return fmt.Errorf("max_num_batched_tokens must be at least 8")
	}
	if d.BatchWindowMS != nil && (*d.BatchWindowMS < 0 || *d.BatchWindowMS > 1000) {
		return fmt.Errorf("batch_window_ms must be between 0 and 1000")
	}
	if d.Dtype != "" && !modelRuntimeDtypes[d.Dtype] {
		return fmt.Errorf("dtype must be one of auto, float32, bfloat16 or float16")
	}
	if d.MaxModelLen != nil && *d.MaxModelLen < 1 {
		return fmt.Errorf("max_model_len must be at least 1")
	}
	if d.MaxModelLen != nil && (d.Endpoint != "" || len(d.Replicas) > 0) {
		return fmt.Errorf("max_model_len applies only to a managed worker, not to an attached endpoint or replicas")
	}
	return nil
}
