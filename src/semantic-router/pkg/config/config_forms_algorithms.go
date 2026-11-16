package config

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// removedAlgorithmTypes are the algorithm types the selectors and executors
// replaced; no Router build may register them again.
var removedAlgorithmTypes = []string{
	"confidence", "cascade", "prompt", "router_dc", "automix", "hybrid", "latency_aware", "knn", "kmeans", "svm", "mlp",
}

// defaultSystemOneSelectorInstructions is the question a selector migrated
// from one that asked none puts to the decision model.
const defaultSystemOneSelectorInstructions = "Which candidate model best fits this request?"

// algorithmForms are the algorithm types and fields the selectors and
// executors replaced: the Looper confidence algorithm and the native cascade
// (escalate), the prompt, router_dc and ML selectors (systemone), automix,
// hybrid and latency_aware (multi_factor), the executors' own on_error, and
// the global blocks of the removed selectors. Decisions are rewritten before
// the global blocks they read.
var algorithmForms = []ConfigForm{
	removedAlgorithm("confidence", "type: escalate, with its method and threshold under escalate.accept", migrateConfidenceAlgorithm),
	removedAlgorithm("cascade", "type: escalate, with its stages and quality under escalate", migrateCascadeAlgorithm),
	removedAlgorithm("prompt", "type: systemone, which asks the decision model with the same instructions, over the first modelRef of each model; static when every candidate is one model", migratePromptAlgorithm),
	removedAlgorithm("router_dc", "type: systemone, which asks the decision model to choose by the candidates' descriptions, over the first modelRef of each model; static when every candidate is one model", migrateSystemOneSelector("router_dc's embedding similarity")),
	removedAlgorithm("automix", "type: multi_factor, weighted {quality: 1-t, cost: t} by cost_quality_tradeoff t", migrateAutoMixAlgorithm),
	removedAlgorithm("hybrid", "type: multi_factor, weighted by hybrid's automix and cost weights", migrateHybridAlgorithm),
	removedAlgorithm("latency_aware", "type: multi_factor, weighted on latency", migrateLatencyAwareAlgorithm),
	removedAlgorithm("knn", "type: systemone over the first modelRef of each model, or static when every candidate is one model; the trained model is not used", migrateSystemOneSelector("the trained knn model")),
	removedAlgorithm("kmeans", "type: systemone over the first modelRef of each model, or static when every candidate is one model; the trained model is not used", migrateSystemOneSelector("the trained kmeans model")),
	removedAlgorithm("svm", "type: systemone over the first modelRef of each model, or static when every candidate is one model; the trained model is not used", migrateSystemOneSelector("the trained svm model")),
	removedAlgorithm("mlp", "type: systemone over the first modelRef of each model, or static when every candidate is one model; the trained model is not used", migrateSystemOneSelector("the trained mlp model")),
	{
		Code:  CodeRemovedField,
		Paths: RoutingScopes("decisions[].algorithm"),
		Match: func(node *yaml.Node, _ string) bool { return nestedExecutorOnError(node) != "" },
		Describe: func(at FormMatch) (string, string) {
			return "algorithm." + scalarAt(at.Node, "type") + ".on_error is removed", "algorithm.on_error, which every executor reads"
		},
		Migrate: func(m *Migration, at FormMatch) error {
			block := mappingValue(at.Node, scalarAt(at.Node, "type"))
			m.Set(at.Node, "on_error", mappingValue(block, "on_error"))
			m.Delete(block, "on_error")
			return nil
		},
	},
	{
		Code:  CodeRemovedField,
		Paths: RoutingScopes("decisions[].algorithm"),
		Match: func(node *yaml.Node, _ string) bool {
			return mappingValue(node, "on_error") != nil && slices.Contains(canonicalSelectorTypes, algorithmTypeOf(node))
		},
		What:    "algorithm.on_error of a selector is removed",
		Becomes: "nothing: a selector that fails falls back to the first candidate, and on_error is an executor setting",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Node, "on_error")
			return nil
		},
	},
	removedSelectionBlock("method", "global.router.model_selection.method is removed", "nothing: each decision's algorithm chooses its own model"),
	removedSelectionBlock("ml", "global.router.model_selection.ml, the ML selectors' trained models, is removed", "the systemone selector, which needs no trained model"),
	removedSelectionBlock("router_dc", "global.router.model_selection.router_dc is removed", "the systemone selector"),
	removedSelectionBlock("automix", "global.router.model_selection.automix is removed", "multi_factor weights on the decisions that used automix"),
	removedSelectionBlock("hybrid", "global.router.model_selection.hybrid is removed", "multi_factor weights on the decisions that used hybrid"),
}

// canonicalSelectorTypes are the built-in selectors; a decision that names no
// algorithm type uses static.
var canonicalSelectorTypes = []string{"", DecisionAlgorithmStatic, DecisionAlgorithmRandom, DecisionAlgorithmMultiFactor, DecisionAlgorithmSystemOne}

func removedAlgorithm(algorithmType, becomes string, migrate func(*Migration, FormMatch) error) ConfigForm {
	return ConfigForm{
		Code:    CodeRemovedAlgorithmType,
		Paths:   RoutingScopes("decisions[].algorithm"),
		Match:   hasType(algorithmType),
		What:    "algorithm type " + algorithmType + " is removed",
		Becomes: becomes,
		Migrate: migrate,
	}
}

func removedSelectionBlock(key, what, becomes string) ConfigForm {
	return ConfigForm{
		Code:    CodeRemovedField,
		Paths:   []string{"global.router.model_selection." + key},
		What:    what,
		Becomes: becomes,
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, at.Key)
			return nil
		},
	}
}

func algorithmTypeOf(node *yaml.Node) string {
	return scalarAt(node, "type")
}

// nestedExecutorOnError is the on_error an executor's own block still holds.
func nestedExecutorOnError(algorithm *yaml.Node) string {
	switch algorithmType := algorithmTypeOf(algorithm); algorithmType {
	case DecisionAlgorithmFusion, DecisionAlgorithmReMoM, DecisionAlgorithmRatings, DecisionAlgorithmWorkflows:
		return scalarAt(mappingValue(algorithm, algorithmType), "on_error")
	}
	return ""
}

// replaceAlgorithm makes an algorithm the given type with block as its
// settings, dropping the removed type's block and its on_error.
func replaceAlgorithm(m *Migration, algorithm *yaml.Node, removed, algorithmType string, block *yaml.Node) {
	m.Delete(algorithm, removed)
	m.Delete(algorithm, "on_error")
	m.Set(algorithm, "type", Scalar(algorithmType))
	m.Set(algorithm, algorithmType, block)
}

func floatAt(node *yaml.Node, key string) (float64, bool) {
	value, err := strconv.ParseFloat(scalarAt(node, key), 64)
	return value, err == nil
}

func intAt(node *yaml.Node, key string) (int, bool) {
	value, err := strconv.Atoi(scalarAt(node, key))
	return value, err == nil
}

// roundWeight keeps a derived weight such as 1-t free of float noise.
func roundWeight(value float64) float64 {
	return math.Round(value*1e6) / 1e6
}

// confidenceThresholdDefaults are the thresholds the confidence methods used
// when none was set; avg_logprob's accepted every scored answer.
var confidenceThresholdDefaults = map[string]float64{
	ConfidenceMethodAverageLogprob: 0,
	ConfidenceMethodMargin:         0.5,
	ConfidenceMethodHybrid:         0.5,
	ConfidenceMethodSelfVerify:     0.7,
}

func migrateConfidenceAlgorithm(m *Migration, at FormMatch) error {
	confidence := mappingValue(at.Node, "confidence")
	escalate := newMapping()
	switch order := scalarAt(confidence, "escalation_order"); order {
	case "", EscalateOrderSize, "small_to_large":
		m.Set(escalate, "order", Scalar(EscalateOrderSize))
	case EscalateOrderCost:
		m.Set(escalate, "order", Scalar(EscalateOrderCost))
	case "declared":
	case "automix":
		m.Note(at.Path, "escalation_order automix, a cost-quality ordering, is removed; the models escalate in modelRefs order: list them in the order to try, or set escalate.order: cost")
	default:
		return fmt.Errorf("confidence.escalation_order %q is not one confidence accepted", order)
	}
	accept, err := confidenceAccept(m, at, confidence)
	if err != nil {
		return err
	}
	m.Set(escalate, "accept", accept)
	onError := scalarAt(confidence, "on_error")
	replaceAlgorithm(m, at.Node, "confidence", DecisionAlgorithmEscalate, escalate)
	switch onError {
	case "", "skip":
	case EscalateOnErrorFail:
		m.Set(at.Node, "on_error", Scalar(EscalateOnErrorFail))
		m.Note(at.Path+".on_error", "on_error: fail now fails the request when no model's answer is accepted; a failed model call moves on to the next model, where confidence's fail stopped at the first failed call")
	default:
		return fmt.Errorf("confidence.on_error %q is not one confidence accepted", onError)
	}
	return nil
}

// confidenceAccept is escalate's accept for a confidence block: its method
// and threshold, or the decision model's answer_check in place of the
// automix_entailment verifier.
func confidenceAccept(m *Migration, at FormMatch, confidence *yaml.Node) (*yaml.Node, error) {
	accept := newMapping()
	method := scalarAt(confidence, "confidence_method")
	threshold, set := floatAt(confidence, "threshold")
	set = set && threshold != 0
	if method == "automix_entailment" {
		check := newMapping()
		gte := 0.7
		if set {
			gte = threshold
		}
		if err := m.SetValue(check, "gte", gte); err != nil {
			return nil, err
		}
		if seconds, ok := intAt(confidence, "verifier_timeout_seconds"); ok && seconds > 0 {
			if err := m.SetValue(check, "timeout_ms", min(seconds*1000, MaxSystemOneTimeoutMs)); err != nil {
				return nil, err
			}
		}
		m.Set(accept, "systemone", check)
		m.Note(at.Path, "the automix_entailment verifier is replaced by the decision model's answer_check question, so its server is no longer called; re-fit escalate.accept.systemone.gte, which was the verifier's threshold")
		return accept, nil
	}
	if method == "" {
		method = ConfidenceMethodAverageLogprob
	}
	gte, known := confidenceThresholdDefaults[method]
	if !known {
		return nil, fmt.Errorf("confidence.confidence_method %q is not one confidence accepted", method)
	}
	if set {
		gte = threshold
	}
	check := newMapping()
	m.Set(check, "method", Scalar(method))
	if err := m.SetValue(check, "gte", gte); err != nil {
		return nil, err
	}
	if weights := mappingValue(confidence, "hybrid_weights"); weights != nil {
		m.Set(check, "hybrid_weights", weights)
	}
	if filter := scalarAt(confidence, "token_filter"); filter != "" {
		m.Set(check, "token_filter", Scalar(filter))
	}
	m.Set(accept, "confidence", check)
	return accept, nil
}

func migrateCascadeAlgorithm(m *Migration, at FormMatch) error {
	escalate := newMapping()
	if quality := mappingValue(at.Node, "quality"); quality != nil {
		if acceptance := mappingValue(quality, "acceptance"); acceptance != nil {
			accept, err := nativeAccept(acceptance)
			if err != nil {
				return fmt.Errorf("quality.acceptance: %w", err)
			}
			m.Delete(quality, "acceptance")
			m.Set(quality, "accept", accept)
		}
		m.Set(escalate, "quality", quality)
	}
	if stages := mappingValue(at.Node, "stages"); stages != nil {
		for index, stage := range stages.Content {
			if scalarAt(stage, "kind") == EscalateStageNative {
				m.Delete(stage, "kind")
			}
			if acceptance := mappingValue(stage, "accept"); acceptance != nil {
				accept, err := nativeAccept(acceptance)
				if err != nil {
					return fmt.Errorf("stages[%d].accept: %w", index, err)
				}
				m.Set(stage, "accept", accept)
			}
		}
		m.Set(escalate, "stages", stages)
	}
	m.Delete(at.Node, "quality")
	m.Delete(at.Node, "stages")
	m.Set(at.Node, "type", Scalar(DecisionAlgorithmEscalate))
	m.Set(at.Node, "escalate", escalate)
	return nil
}

// nativeAccept writes {rules: [{..., predicate: {gte: x}}]} as escalate's
// {native: [{..., gte: x}]}: the predicate's bounds sit on the rule.
func nativeAccept(acceptance *yaml.Node) (*yaml.Node, error) {
	rules := mappingValue(acceptance, "rules")
	if rules == nil || rules.Kind != yaml.SequenceNode {
		return nil, fmt.Errorf("rules must be a list")
	}
	for _, rule := range rules.Content {
		predicate := mappingValue(rule, "predicate")
		if predicate == nil || predicate.Kind != yaml.MappingNode {
			continue
		}
		for i := 0; i+1 < len(rule.Content); i += 2 {
			if rule.Content[i].Value == "predicate" {
				rule.Content = slices.Concat(rule.Content[:i], predicate.Content, rule.Content[i+2:])
				break
			}
		}
	}
	accept := newMapping()
	accept.Style = acceptance.Style
	setNode(accept, "native", rules)
	return accept, nil
}

func migratePromptAlgorithm(m *Migration, at FormMatch) error {
	if migratedToStatic(m, at) {
		return nil
	}
	prompt := mappingValue(at.Node, "prompt")
	selector := newMapping()
	instructions := mappingValue(prompt, "instructions")
	if instructions == nil {
		instructions = Scalar(defaultSystemOneSelectorInstructions)
	}
	m.Set(selector, "instructions", instructions)
	if seconds, ok := intAt(prompt, "timeout_seconds"); ok && seconds > 0 {
		milliseconds := min(seconds*1000, MaxSystemOneTimeoutMs)
		if err := m.SetValue(selector, "timeout_ms", milliseconds); err != nil {
			return err
		}
	}
	if model := scalarAt(prompt, "model"); model != "" {
		m.Note(at.Path, fmt.Sprintf("the helper model %s no longer chooses the candidate: the decision model answers the same instructions (set algorithm.systemone.deployment to ask another System One deployment)", model))
	}
	replaceAlgorithm(m, at.Node, "prompt", DecisionAlgorithmSystemOne, selector)
	return nil
}

// migratedToStatic makes a selector static when the systemone selector, which
// chooses among two or more distinct models, has nothing to choose: a decision
// with one candidate, or one whose modelRefs all name one model, as the LoRA
// adapters of one base model do. It reports whether it did. Otherwise the
// selector keeps the first modelRef of each model, and the note names the
// ones it drops.
func migratedToStatic(m *Migration, at FormMatch) bool {
	refs := mappingValue(at.Parent, "modelRefs")
	seen := map[string]bool{}
	var kept []*yaml.Node
	var dropped []string
	for _, ref := range sequenceItems(refs) {
		model := scalarAt(ref, "model")
		if !seen[model] {
			seen[model] = true
			kept = append(kept, ref)
		} else if lora := scalarAt(ref, "lora_name"); lora != "" {
			dropped = append(dropped, model+" with lora_name "+lora)
		} else {
			dropped = append(dropped, model)
		}
	}
	if len(kept) >= 2 {
		if len(dropped) > 0 {
			refs.Content = kept
			m.Note(at.Path, "the systemone selector chooses among distinct models, so the decision keeps the first modelRef of each model and drops "+
				strings.Join(dropped, ", ")+"; to route to a dropped one, give it a decision of its own")
		}
		return false
	}
	if len(dropped) > 0 {
		m.Note(at.Path, "the decision's modelRefs repeat model "+scalarAt(kept[0], "model")+", which the systemone selector cannot choose between, so the decision is static "+
			"and its first candidate serves every request; list each model once for the decision model to choose")
	}
	m.Delete(at.Node, algorithmTypeOf(at.Node))
	m.Delete(at.Node, "on_error")
	m.Set(at.Node, "type", Scalar(DecisionAlgorithmStatic))
	return true
}

// migrateSystemOneSelector rewrites a selector that asked no question as the
// systemone selector, which chooses by the candidates' descriptions, or as
// static when it has nothing to choose.
func migrateSystemOneSelector(replaced string) func(*Migration, FormMatch) error {
	return func(m *Migration, at FormMatch) error {
		if migratedToStatic(m, at) {
			return nil
		}
		selector := newMapping()
		m.Set(selector, "instructions", Scalar(defaultSystemOneSelectorInstructions))
		replaceAlgorithm(m, at.Node, algorithmTypeOf(at.Node), DecisionAlgorithmSystemOne, selector)
		m.Note(at.Path, "the decision model chooses among the candidates by their descriptions in place of "+replaced+"; describe each candidate (its model description, or algorithm.systemone.candidates)")
		return nil
	}
}

// selectionDefault reads a setting of a removed selector: the decision's
// block, then the global model_selection block of that selector, then the
// selector's own default.
func selectionDefault(at FormMatch, selector, key string, fallback float64) float64 {
	if value, ok := floatAt(mappingValue(at.Node, selector), key); ok && value != 0 {
		return value
	}
	if value, ok := floatAt(mappingAt(at.Root, "global", "router", "model_selection", selector), key); ok && value != 0 {
		return value
	}
	return fallback
}

func migrateAutoMixAlgorithm(m *Migration, at FormMatch) error {
	tradeoff := selectionDefault(at, "automix", "cost_quality_tradeoff", 0.3)
	weights := newMapping()
	if err := m.SetValue(weights, "quality", roundWeight(1-tradeoff)); err != nil {
		return err
	}
	if err := m.SetValue(weights, "cost", tradeoff); err != nil {
		return err
	}
	block := newMapping()
	m.Set(block, "weights", weights)
	replaceAlgorithm(m, at.Node, "automix", DecisionAlgorithmMultiFactor, block)
	m.Note(at.Path, fmt.Sprintf("automix's learned POMDP policy and self-verification are not used, and multi_factor cannot keep automix's cost_quality_tradeoff %g: "+
		"automix valued a candidate at its quality less %g times its prompt price per 1M tokens divided by 10, while multi_factor rescales each factor across the candidates, "+
		"so of two candidates the larger weight decides (quality %g, cost %g); to take a cheaper model within a quality margin, set multi_factor.objective: "+
		"{strategy: lexicographic, priorities: [{factor: quality, tolerance: <margin>}, {factor: cost}]}", tradeoff, tradeoff, roundWeight(1-tradeoff), tradeoff))
	return nil
}

func migrateHybridAlgorithm(m *Migration, at FormMatch) error {
	quality := selectionDefault(at, "hybrid", "automix_weight", 0.2)
	cost := selectionDefault(at, "hybrid", "cost_weight", 0.2)
	experience := selectionDefault(at, "hybrid", "experience_weight", 0.3)
	similarity := selectionDefault(at, "hybrid", "router_dc_weight", 0.3)
	block := newMapping()
	weights := newMapping()
	if err := m.SetValue(weights, "quality", quality); err != nil {
		return err
	}
	if err := m.SetValue(weights, "cost", cost); err != nil {
		return err
	}
	m.Set(block, "weights", weights)
	replaceAlgorithm(m, at.Node, "hybrid", DecisionAlgorithmMultiFactor, block)
	m.Note(at.Path, fmt.Sprintf("hybrid's experience weight %g (feedback ratings) and router_dc weight %g (embedding similarity) are dropped; multi_factor weighs quality %g (hybrid's automix weight) and cost %g", experience, similarity, quality, cost))
	return nil
}

func migrateLatencyAwareAlgorithm(m *Migration, at FormMatch) error {
	latencyAware := mappingValue(at.Node, "latency_aware")
	tpot, _ := intAt(latencyAware, "tpot_percentile")
	ttft, _ := intAt(latencyAware, "ttft_percentile")
	block := newMapping()
	weights := newMapping()
	if err := m.SetValue(weights, "latency", 1); err != nil {
		return err
	}
	m.Set(block, "weights", weights)
	switch {
	case tpot > 0 && ttft > 0:
		if err := m.SetValue(block, "latency_percentile", tpot); err != nil {
			return err
		}
		m.Note(at.Path, fmt.Sprintf("latency_aware combined TPOT p%d and TTFT p%d; multi_factor compares one measurement, TPOT or a candidate's TTFT when it has no TPOT samples, at p%d: set multi_factor.latency_metric and latency_percentile to choose", tpot, ttft, tpot))
	case tpot > 0:
		m.Set(block, "latency_metric", Scalar("tpot"))
		if err := m.SetValue(block, "latency_percentile", tpot); err != nil {
			return err
		}
	case ttft > 0:
		m.Set(block, "latency_metric", Scalar("ttft"))
		if err := m.SetValue(block, "latency_percentile", ttft); err != nil {
			return err
		}
	}
	replaceAlgorithm(m, at.Node, "latency_aware", DecisionAlgorithmMultiFactor, block)
	return nil
}
