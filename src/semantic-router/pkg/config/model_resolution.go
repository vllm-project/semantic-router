package config

import (
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sort"
	"strings"
)

// The deployment that answers a consumer is resolved by name, in one order:
//
//  1. the rule's own deployment (decision, classifier, safety and its hazard
//     part, complexity: rules that each ask a question of their own);
//  2. the recipe's binding for its signal type (binding_levels.go);
//  3. the global binding for its signal type;
//  4. for a question the decision model answers, the decision model.
//
// Embeddings, reranking and the hazard head ask no question: they resolve
// where they are used, and an unnamed hazard part uses the pre-declared
// hazard deployment. Deprecated fields are read as rule deployments and
// bindings before resolution (model_resource_translation.go).

// ResolutionSource says which step resolved a consumer's deployment.
type ResolutionSource string

const (
	ResolvedByRule          ResolutionSource = "rule"
	ResolvedByRecipeBinding ResolutionSource = "recipe_binding"
	ResolvedByGlobalBinding ResolutionSource = "global_binding"
	ResolvedByDecisionModel ResolutionSource = "decision_model"
	// ResolvedByDefault is the pre-declared deployment of a signal that asks
	// no question: the hazard deployment for a hazard part.
	ResolvedByDefault ResolutionSource = "default"
)

// TaskResolution is the deployment a consumer resolves to.
type TaskResolution struct {
	Consumer   string
	Name       string
	Deployment ModelDeployment
	Binding    ModelBinding
	Source     ResolutionSource
}

// Explicit reports whether the configuration names the deployment for this
// consumer rather than leaving it to the decision model.
func (r TaskResolution) Explicit() bool {
	return r.Source != ResolvedByDecisionModel && r.Source != ResolvedByDefault
}

// questionConsumer reports whether the decision model answers a consumer
// that names no deployment.
func questionConsumer(consumer string) bool {
	switch consumer {
	case "prompt_guard", "pii_classifier", "hallucination_detector":
		return true
	}
	return strings.HasPrefix(consumer, "systemone.")
}

// bindingKeysOf lists the binding keys a consumer reads, most specific
// first: its rule key, then its signal type's.
func bindingKeysOf(consumer string) []string {
	if strings.HasPrefix(consumer, "systemone.") {
		return []string{consumer, bindingKeySystemOne}
	}
	return []string{consumer}
}

// recipeBindings are the bindings the scope's own recipe declares;
// globalBindings the global ones.
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

func (c *RouterConfig) globalBindings() map[string]ModelBinding {
	return c.GlobalModelBindings
}

// ruleDeployment is the deployment a consumer's rule names, empty when it
// names none.
func (c *RouterConfig) ruleDeployment(consumer string) string {
	if strings.HasPrefix(consumer, "systemone.") {
		for _, rule := range c.SystemOneRules {
			if consumer == "systemone."+rule.Name {
				return rule.Deployment
			}
		}
	}
	return ""
}

// ResolveTask resolves the deployment that answers consumer in this scope
// (see the order above). ok is false when nothing answers it: an unbound
// embedding, reranker or reask, or no decision model.
func (c *RouterConfig) ResolveTask(consumer string) (TaskResolution, bool, error) {
	if c == nil {
		return TaskResolution{}, false, nil
	}
	resolved := c.withModelResources()
	return resolved.resolveTask(consumer)
}

// DeferredModelError is a model a deprecated field names that cannot be
// resolved. It fails the consumer that runs the model, not the
// configuration, as such a model did before. Path is the model path the
// field names.
type DeferredModelError struct {
	Err  error
	Path string
}

func (e *DeferredModelError) Error() string { return e.Err.Error() }
func (e *DeferredModelError) Unwrap() error { return e.Err }

// translationError reports whether err is the error of a deprecated model
// field that could not be translated, which fails only its consumer.
func (c *RouterConfig) translationError(err error) bool {
	for _, recorded := range c.modelResourceErrors {
		if errors.Is(err, recorded) {
			return true
		}
	}
	return false
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

// IsDeferredModelError reports whether err is a DeferredModelError.
func IsDeferredModelError(err error) bool {
	var deferred *DeferredModelError
	return errors.As(err, &deferred)
}

func (c *RouterConfig) resolveTask(consumer string) (TaskResolution, bool, error) {
	pick := func(name string, binding ModelBinding, source ResolutionSource) (TaskResolution, bool, error) {
		deployment, ok := c.ModelDeployments[name]
		if !ok {
			return TaskResolution{}, false, fmt.Errorf("%s: unknown deployment %q; %s", BindingKey(consumer), name, declaredDeployments(c.ModelDeployments))
		}
		return TaskResolution{Consumer: consumer, Name: name, Deployment: deployment.WithDefaults(), Binding: binding, Source: source}, true, nil
	}
	if consumer == RAGRerankerConsumer {
		if rerank := c.ragRerank(); rerank != nil && rerank.Deployment != "" {
			binding := ModelBinding{Deployment: rerank.Deployment}
			if rerank.Layer != 0 || rerank.Dimension != 0 {
				binding.PairScorer = &PairScorerSelection{Layer: rerank.Layer, Dimension: rerank.Dimension}
			}
			return pick(rerank.Deployment, binding, ResolvedByRule)
		}
	}
	if name := strings.TrimSpace(c.ruleDeployment(consumer)); name != "" {
		return pick(name, ModelBinding{Deployment: name}, ResolvedByRule)
	}
	for _, scope := range []struct {
		bindings map[string]ModelBinding
		source   ResolutionSource
	}{{c.recipeBindings(), ResolvedByRecipeBinding}, {c.globalBindings(), ResolvedByGlobalBinding}} {
		for _, key := range bindingKeysOf(consumer) {
			// A deprecated module or rule model is read as a global binding,
			// so the error of one that cannot be translated takes its place.
			if scope.source == ResolvedByGlobalBinding {
				if err := c.modelResourceErrors[key]; err != nil {
					return TaskResolution{}, false, err
				}
			}
			if binding, ok := scope.bindings[key]; ok {
				if scope.source == ResolvedByGlobalBinding && !globalBindingApplies(key, c.Signals) {
					continue
				}
				return pick(binding.Deployment, binding, scope.source)
			}
		}
	}
	if !questionConsumer(consumer) {
		return TaskResolution{}, false, nil
	}
	name := c.decisionModelName()
	if name == "" {
		return TaskResolution{}, false, nil
	}
	return pick(name, ModelBinding{Deployment: name}, ResolvedByDecisionModel)
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

// boundTask resolves consumer only when a rule or binding names its
// deployment.
func (c *RouterConfig) boundTask(consumer string) (TaskResolution, bool) {
	resolved, ok, err := c.ResolveTask(consumer)
	if !ok || err != nil || !resolved.Explicit() {
		return TaskResolution{}, false
	}
	return resolved, true
}

// ModuleModelConfigured reports whether a built-in module's consumer
// resolves to a deployment in this scope.
func (c *RouterConfig) ModuleModelConfigured(consumer string) bool {
	_, ok, err := c.ResolveTask(consumer)
	return ok || err != nil
}

// remoteTaskModule reports whether consumer resolves to a remote classify
// or chat deployment, which reads its labels from a mapping file and has no
// local token budget or window.
func (c *RouterConfig) remoteTaskModule(consumer string) bool {
	switch c.ResolvedProvider(consumer) {
	case ClassifyProvider, OpenAIChatProvider, LegacyHTTPProvider:
		return true
	}
	return false
}

// ResolvedProvider is the provider of the deployment consumer resolves to,
// empty when none answers it.
func (c *RouterConfig) ResolvedProvider(consumer string) string {
	resolved, ok, err := c.ResolveTask(consumer)
	if !ok || err != nil {
		return ""
	}
	return resolved.Deployment.Provider
}

// providerAnswers reports whether a provider can serve a consumer; the
// error says why not.
func providerAnswers(consumer string, deployment ModelDeployment, binding ModelBinding) error {
	provider := deployment.Provider
	name := BindingKey(consumer)
	consumer = ruleConsumer(consumer)
	switch provider {
	case ModelRuntimeProvider:
		return builtInPurposeAnswers(name, consumer, deployment)
	case SystemOneProvider:
		switch {
		case consumer == "embedding" || consumer == RAGRerankerConsumer:
			return fmt.Errorf("%s: a systemone deployment answers questions; embeddings and reranking need a model_runtime, openai_embeddings or rerank deployment", name)
		case binding.Head != "" || binding.OperatingPoint != nil:
			return fmt.Errorf("%s: a systemone deployment answers questions and serves no classify head", name)
		case binding.Contract != "" && binding.Contract != DecisionTaskContract && binding.Contract != RemoteClassifierContractTokenSpans && binding.Contract != RemoteClassifierContractLabelDistribution:
			return fmt.Errorf("%s: a systemone deployment answers questions, not the %s contract", name, binding.Contract)
		}
		return nil
	case ClassifyProvider:
		switch {
		case isApplicationConsumer(consumer) || consumer == "embedding" || consumer == RAGRerankerConsumer:
			return fmt.Errorf("%s: the classify API has no %s task; bind it to a model_runtime or systemone deployment", name, name)
		case binding.Head != "":
			return fmt.Errorf("%s: a remote classify deployment cannot select a local head", name)
		}
		return nil
	case OpenAIChatProvider:
		if consumer == "prompt_guard" || consumer == "hallucination_detector" {
			return nil
		}
		return fmt.Errorf("%s: an openai_chat deployment answers the jailbreak guard and the hallucination detector only", name)
	case OpenAIEmbeddingsProvider:
		if consumer == "embedding" {
			return nil
		}
		return fmt.Errorf("%s: an openai_embeddings deployment serves embeddings only", name)
	case RerankProvider:
		if consumer == RAGRerankerConsumer {
			if binding.PairScorer != nil && (binding.PairScorer.Layer != 0 || binding.PairScorer.Dimension != 0) {
				return fmt.Errorf("%s: rerank.layer and rerank.dimension select an early exit of a model_runtime reranker; a rerank deployment answers at its own depth, so remove them", name)
			}
			return nil
		}
		return fmt.Errorf("%s: a rerank deployment serves RAG reranking only", name)
	case LegacyHTTPProvider:
		return nil
	}
	return fmt.Errorf("%s: unsupported provider %q", name, provider)
}

// builtInPurposeAnswers checks a model_runtime deployment of a built-in
// model against what the consumer asks: an embedding or reranking model
// answers no question, and only an embedding model serves embeddings and a
// reranker reranks. A model the registry does not know is checked by its
// card when the runtime serves it.
func builtInPurposeAnswers(name, consumer string, deployment ModelDeployment) error {
	spec := registeredModel(deployment.Artifact)
	if spec == nil || !deployment.Managed() {
		return nil
	}
	embeds := spec.Purpose == PurposeEmbedding || spec.Purpose == PurposeSemanticSimilarity
	switch {
	case consumer == "embedding":
		if !embeds {
			return fmt.Errorf("%s: deployment runs %s, a %s model that serves no embeddings; name an embedding model in embeddings.default", name, spec.RepoID, spec.Purpose)
		}
	case consumer == RAGRerankerConsumer:
		if spec.Purpose != PurposeReranking {
			return fmt.Errorf("%s: deployment runs %s, a %s model that does not rerank; name a reranker in rerank.deployment", name, spec.RepoID, spec.Purpose)
		}
	case embeds || spec.Purpose == PurposeReranking:
		return fmt.Errorf("%s: deployment runs %s, a %s model that answers no question; bind a model that answers %s", name, spec.RepoID, spec.Purpose, name)
	}
	return nil
}

// scopeConsumers lists the consumers a scope's modules and rules may run.
func (c *RouterConfig) scopeConsumers() []string {
	consumers := append(c.implicitConsumers(), "embedding", RAGRerankerConsumer)
	sort.Strings(consumers)
	return consumers
}

// validateTaskCapabilities checks, at load, that every consumer a scope runs
// resolves to a deployment whose provider can answer it.
func validateTaskCapabilities(scope RecipeName, scoped *RouterConfig) error {
	for _, consumer := range scoped.scopeConsumers() {
		if !TaskConsumerInUse(scoped, scope, consumer) {
			continue
		}
		resolved, ok, err := scoped.resolveTask(consumer)
		if err != nil {
			continue
		}
		if !ok {
			continue
		}
		if err := providerAnswers(consumer, resolved.Deployment, resolved.Binding); err != nil {
			return fmt.Errorf("recipe %q: %w", scope, err)
		}
	}
	return nil
}

// validateModelResolutionContracts checks, for every scope that may run
// consumers, that each one it runs resolves to a deployment that can answer
// it.
func validateModelResolutionContracts(cfg *RouterConfig) error {
	resolved := cfg.withModelResources()
	if err := validateMemoryQueryRewrite(resolved); err != nil {
		return err
	}
	for _, name := range sortedModelKeys(resolved.ModelDeployments) {
		if generation := resolved.ModelDeployments[name].Generation; generation != nil && generation.Reasoning != nil {
			if err := validateReasoningSetting(resolved, "global.model_catalog.deployments."+name+".generation.reasoning", generation.Reasoning); err != nil {
				return err
			}
		}
	}
	for scope, scoped := range resolved.consumerScopes() {
		if err := validateRAGRerankDeployments(scope, scoped); err != nil {
			return err
		}
		if err := validateTaskCapabilities(scope, scoped); err != nil {
			return err
		}
		if err := validateNonQuestionResources(scoped); err != nil {
			return fmt.Errorf("recipe %q: %w", scope, err)
		}
		if err := validateMaskingSpans(scoped); err != nil {
			return err
		}
	}
	return nil
}

// spanPresets are the presets a systemone API answers a span signal with.
var spanPresets = map[string]string{"pii_classifier": "pii", "hallucination_detector": "halu"}

// declaresNoSpans reports whether consumer resolves to a systemone deployment
// whose declared capabilities rule out spans for it: neither its preset nor
// span questions. A deployment that declares nothing is told by its API.
func (c *RouterConfig) declaresNoSpans(consumer string) (string, bool) {
	resolved, ok, err := c.ResolveTask(consumer)
	if !ok || err != nil || !resolved.Deployment.IsSystemOne() || (resolved.Deployment.Capabilities == nil && len(resolved.Deployment.QuestionTypes) == 0) {
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
	name, spanless := scoped.spanlessDeployment("pii_classifier")
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

// validateNonQuestionResources checks that the deployments named where
// embeddings, reranking and the multimodal embedding are used serve them.
func validateNonQuestionResources(scoped *RouterConfig) error {
	if name := scoped.EmbeddingDeployment; name != "" {
		deployment := scoped.ModelDeployments[name]
		if dimensions, target := deployment.DeclaredDimensions(), scoped.EmbeddingConfig.TargetDimension; deployment.Provider == OpenAIEmbeddingsProvider && dimensions > 0 && target > 0 && dimensions != target {
			return fmt.Errorf("global.model_catalog.embeddings.default: deployment %q answers with %d dimensions (capabilities.dimensions), but embeddings.semantic.embedding_config.target_dimension asks for %d; make them match, or set target_dimension: 0 for the model's own", name, dimensions, target)
		}
	}
	if name := scoped.MultimodalEmbeddingDeployment; name != "" {
		deployment, ok := scoped.ModelDeployments[name]
		if !ok {
			return fmt.Errorf("global.model_catalog.embeddings.multimodal: unknown deployment %q; %s", name, declaredDeployments(scoped.ModelDeployments))
		}
		if err := providerAnswers("embedding", deployment.WithDefaults(), ModelBinding{Deployment: name}); err != nil {
			return fmt.Errorf("global.model_catalog.embeddings.multimodal: %w", err)
		}
	}
	return nil
}

// modelResolutionWarnings are the findings of model resolution that leave a
// configuration valid: modules on uncalibrated thresholds.
func modelResolutionWarnings(cfg *RouterConfig) []ConfigWarning {
	resolved := cfg.withModelResources()
	var warnings []ConfigWarning
	seen := map[string]bool{}
	scopes := resolved.consumerScopes()
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

// ragRerank is the rerank block the scope's RAG plugins name a deployment
// in; validateRAGRerankDeployments refuses a scope whose plugins disagree.
func (c *RouterConfig) ragRerank() *RAGRerankConfig {
	for i := range c.Decisions {
		if rag := c.Decisions[i].GetRAGConfig(); rag != nil && rag.Enabled && rag.Rerank != nil && rag.Rerank.Deployment != "" {
			return rag.Rerank
		}
	}
	return nil
}

func validateRAGRerankDeployments(scope RecipeName, scoped *RouterConfig) error {
	var first *RAGRerankConfig
	for i := range scoped.Decisions {
		rag := scoped.Decisions[i].GetRAGConfig()
		if rag == nil || !rag.Enabled || rag.Rerank == nil || rag.Rerank.Deployment == "" {
			continue
		}
		if first == nil {
			first = rag.Rerank
			continue
		}
		if rag.Rerank.Deployment != first.Deployment || rag.Rerank.Layer != first.Layer || rag.Rerank.Dimension != first.Dimension {
			return fmt.Errorf("recipe %q: RAG plugins name different rerank deployments or exits (%q, %q); a recipe has one reranker", scope, first.Deployment, rag.Rerank.Deployment)
		}
	}
	return nil
}
