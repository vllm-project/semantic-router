package config

import (
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sort"
	"strings"
)

// The deployment that answers an application signal type's rules that name
// none is resolved in one order (binding_levels.go):
//
//  1. the recipe's binding for its signal type;
//  2. the global binding for its signal type;
//  3. the decision model.
//
// A rule that names its own deployment, and every systemone rule, asks it;
// embeddings resolve where they are used.

// ResolutionSource says which step resolved a consumer's deployment.
type ResolutionSource string

const (
	ResolvedByRecipeBinding ResolutionSource = "recipe_binding"
	ResolvedByGlobalBinding ResolutionSource = "global_binding"
	ResolvedByDecisionModel ResolutionSource = "decision_model"
)

// TaskResolution is the deployment a consumer resolves to.
type TaskResolution struct {
	Consumer   string
	Name       string
	Deployment ModelDeployment
	Source     ResolutionSource
}

// Explicit reports whether the configuration names the deployment for this
// consumer rather than leaving it to the decision model.
func (r TaskResolution) Explicit() bool {
	return r.Source != ResolvedByDecisionModel
}

// recipeBindings are the bindings the scope's recipe authors. A recipe's
// scoped configuration also carries, in ModelBindings, the global bindings
// that apply to it; its own are those it declares or that differ from them.
func (c *RouterConfig) recipeBindings() map[string]ModelBinding {
	if c.scopedBindings == nil {
		return c.ModelBindings
	}
	own := make(map[string]ModelBinding, len(c.scopedBindings))
	for key, binding := range c.ModelBindings {
		_, authored := c.scopedBindings[key]
		if global, inherited := c.GlobalModelBindings[key]; authored || !inherited || !reflect.DeepEqual(global, binding) {
			own[key] = binding
		}
	}
	return own
}

// ResolveTask resolves the deployment that answers an application signal
// type in this scope (see the order above). ok is false when nothing answers
// it: no binding and no decision model.
func (c *RouterConfig) ResolveTask(consumer string) (TaskResolution, bool, error) {
	if c == nil {
		return TaskResolution{}, false, nil
	}
	return c.resolveTask(consumer)
}

// declaredDeployments lists the deployment names a configuration declares,
// for an error that names one it does not.
func declaredDeployments(deployments map[string]ModelDeployment) string {
	names := sortedModelKeys(deployments)
	if len(names) == 0 {
		return "global.model_catalog.deployments declares none"
	}
	return "declared deployments: " + strings.Join(names, ", ")
}

func (c *RouterConfig) resolveTask(consumer string) (TaskResolution, bool, error) {
	pick := func(name string, source ResolutionSource) (TaskResolution, bool, error) {
		deployment, ok := c.ModelDeployments[name]
		if !ok {
			return TaskResolution{}, false, fmt.Errorf("%s: unknown deployment %q; %s", consumer, name, declaredDeployments(c.ModelDeployments))
		}
		return TaskResolution{Consumer: consumer, Name: name, Deployment: deployment.WithDefaults(), Source: source}, true, nil
	}
	if binding, ok := c.recipeBindings()[consumer]; ok {
		return pick(binding.Deployment, ResolvedByRecipeBinding)
	}
	if binding, ok := c.GlobalModelBindings[consumer]; ok {
		return pick(binding.Deployment, ResolvedByGlobalBinding)
	}
	name := c.decisionModelName()
	if name == "" {
		return TaskResolution{}, false, nil
	}
	return pick(name, ResolvedByDecisionModel)
}

// decisionModelName is the decision model's deployment, empty for a
// configuration built in code that declares none.
func (c *RouterConfig) decisionModelName() string {
	name := c.DecisionModel
	if name == "" {
		name = DefaultDecisionDeployment
	}
	if _, ok := c.ModelDeployments[name]; !ok {
		return ""
	}
	return name
}

// providerAnswers reports whether a deployment's provider can answer an
// application signal type; the error says why not.
func providerAnswers(name string, deployment ModelDeployment) error {
	provider := deployment.Provider
	switch provider {
	case ModelRuntimeProvider:
		return builtInPurposeAnswers(name, deployment)
	case SystemOneProvider:
		return nil
	case OpenAIChatProvider:
		return fmt.Errorf("%s: an openai_chat deployment answers no signal; bind a model_runtime or systemone deployment", name)
	case OpenAIEmbeddingsProvider:
		return fmt.Errorf("%s: an openai_embeddings deployment serves embeddings only; name it in global.model_catalog.embeddings.default or a store's embedding.deployment", name)
	}
	return fmt.Errorf("%s: unsupported provider %q", name, provider)
}

// builtInPurposeAnswers checks a model_runtime deployment of a built-in
// model against what the consumer asks: an embedding model answers no
// question. A model the registry does not know is checked by its card when
// the runtime serves it.
func builtInPurposeAnswers(name string, deployment ModelDeployment) error {
	spec := registeredModel(deployment.Artifact)
	if spec == nil || !deployment.Managed() {
		return nil
	}
	if spec.Purpose == PurposeEmbedding || spec.Purpose == PurposeSemanticSimilarity {
		return fmt.Errorf("%s: deployment runs %s, a %s model that answers no question; bind a model that answers %s", name, spec.RepoID, spec.Purpose, name)
	}
	return nil
}

// scopeConsumers lists the consumers a scope's modules and rules may run.
func (c *RouterConfig) scopeConsumers() []string {
	consumers := c.implicitConsumers()
	sort.Strings(consumers)
	return consumers
}

// validateModelResolutionContracts checks, for every scope that may run
// consumers, that each one it runs resolves to a deployment that can answer
// it.
func validateModelResolutionContracts(cfg *RouterConfig) error {
	if err := validateMemoryQueryRewrite(cfg); err != nil {
		return err
	}
	for _, name := range sortedModelKeys(cfg.ModelDeployments) {
		if generation := cfg.ModelDeployments[name].Generation; generation != nil && generation.Reasoning != nil {
			if err := validateReasoningSetting(cfg, "global.model_catalog.deployments."+name+".generation.reasoning", generation.Reasoning); err != nil {
				return err
			}
		}
	}
	for _, scoped := range cfg.consumerScopes() {
		if err := validateMaskingSpans(scoped); err != nil {
			return err
		}
	}
	return nil
}

// spanPresets are the presets a systemone API answers a span signal with.
var spanPresets = map[string]string{SignalTypePII: "pii", SignalTypeHallucination: "halu"}

// declaresNoSpans reports whether consumer resolves to a systemone deployment
// whose declared capabilities rule out spans for it: neither its preset nor
// span questions. A deployment that declares nothing is told by its API.
func (c *RouterConfig) declaresNoSpans(consumer string) (string, bool) {
	resolved, ok, err := c.ResolveTask(consumer)
	if !ok || err != nil || !resolved.Deployment.IsSystemOne() || resolved.Deployment.Capabilities == nil {
		return "", false
	}
	if slices.Contains(resolved.Deployment.DeclaredPresets(), spanPresets[consumer]) || slices.Contains(resolved.Deployment.DeclaredQuestionTypes(), "span") {
		return "", false
	}
	return resolved.Name, true
}

// spanlessDeployment names the deployment a consumer resolves to when the
// configuration shows it cannot answer span questions: a systemone deployment
// whose capabilities declare neither the consumer's preset nor span, or a
// built-in decision model the catalog lists without span.
func (c *RouterConfig) spanlessDeployment(consumer string) (string, bool) {
	if name, noSpans := c.declaresNoSpans(consumer); noSpans {
		return name, true
	}
	resolved, ok, err := c.ResolveTask(consumer)
	if !ok || err != nil || !resolved.Deployment.IsModelRuntime() {
		return "", false
	}
	if questionTypes, _ := c.deploymentAnswers(resolved.Name); questionTypes == nil || slices.Contains(questionTypes, SystemOneQuestionSpan) {
		return "", false
	}
	return resolved.Name, true
}

// validateMaskingSpans refuses a masking plugin whose pii rule resolves to a
// deployment that cannot answer span questions: masking replaces personal
// data at the positions the rule's spans give.
func validateMaskingSpans(scoped *RouterConfig) error {
	signals := scoped.MaskingSignals()
	if len(signals) == 0 {
		return nil
	}
	name, spanless := scoped.spanlessDeployment(SignalTypePII)
	if !spanless {
		return nil
	}
	failures := make([]error, 0, len(signals))
	for _, signal := range signals {
		failures = append(failures, CheckError(scoped.ScopePath("signals.pii["+signal+"]"), CodeSpanRequired,
			fmt.Sprintf("resolves to deployment %q, which cannot answer span questions; bind pii to a span-capable deployment (recipes[].model_bindings.pii or global.model_catalog.bindings.pii)", name)))
	}
	return errors.Join(failures...)
}

// modelResolutionWarnings are the findings of model resolution that leave a
// configuration valid: modules on uncalibrated thresholds.
func modelResolutionWarnings(cfg *RouterConfig) []ConfigWarning {
	var warnings []ConfigWarning
	seen := map[string]bool{}
	scopes := cfg.consumerScopes()
	names := make([]string, 0, len(scopes))
	for scope := range scopes {
		names = append(names, string(scope))
	}
	sort.Strings(names)
	for _, name := range names {
		scoped := scopes[RecipeName(name)]
		for _, warning := range scoped.uncalibratedPresetWarnings() {
			if !seen[warning.Field+warning.Message] {
				seen[warning.Field+warning.Message] = true
				warnings = append(warnings, warning)
			}
		}
	}
	return warnings
}
