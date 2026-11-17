package classification

import (
	"context"
	"maps"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// ResponseConditions reads rule trees over one answer: the request-stage
// signals routing published for its request, and the response-stage rules
// (systemone rules and application rules with stage: response, such as the
// jailbreak, hallucination and refusal rules), each asked the first time a
// tree reads it. The first read of an application type asks every
// response-stage rule of the type in one call, so its rules of one preset
// share the preset's question, as they do at the request stage.
type ResponseConditions struct {
	classifier *Classifier
	ctx        context.Context
	input      ResponseInput
	engine     *decision.DecisionEngine
	signals    *decision.SignalMatches
	observed   *ResponseSignal
	asked      map[string]bool
}

// ResponseSignal is what the response-stage rules made of one answer, keyed
// as the request stage keys its signals ("<type>:<rule>"): the matched
// references of each type and those its matched-rule header lists
// (SignalResults.ReportedMatches), each rule's value, confidence or failure,
// and the spans of the span rules that matched.
type ResponseSignal struct {
	Matched     map[string][]string
	Reported    map[string][]string
	Confidences map[string]float64
	Values      map[string]float64
	Errors      map[string]string
	Spans       map[string][]modelservice.Span
}

// NewResponseConditions starts reading conditions over one answer. routed are
// the request's request-stage signals; they are copied, so the observations
// made here never change them.
func (c *Classifier) NewResponseConditions(ctx context.Context, routed *decision.SignalMatches, input ResponseInput) *ResponseConditions {
	if ctx == nil {
		ctx = context.Background()
	}
	return &ResponseConditions{
		classifier: c,
		ctx:        ctx,
		input:      input,
		engine:     decision.NewDecisionEngine(nil, c.Config.Strategy).WithRoutingScope(c.Config.RoutingScope),
		signals:    cloneSignalMatches(routed),
		observed:   newResponseSignal(),
		asked:      map[string]bool{},
	}
}

// Evaluate reads one rule tree, asking each response-stage rule it reaches
// that no tree has read yet; an AND stops at its first false condition.
func (r *ResponseConditions) Evaluate(tree config.RuleNode) decision.RuleOutcome {
	return r.engine.EvaluateRule(tree, r.signals, r.observe)
}

// Observed is what the response-stage rules the trees read made of the
// answer.
func (r *ResponseConditions) Observed() *ResponseSignal { return r.observed }

func (r *ResponseConditions) observe(leaf config.RuleNode) {
	signalType := strings.ToLower(strings.TrimSpace(leaf.Type))
	if r.classifier.Config.SignalStageOf(signalType, leaf.Name) != config.SignalStageResponse {
		return
	}
	scope := signalType
	if signalType == config.SignalTypeSystemOne {
		ruleName, _, _ := strings.Cut(leaf.Name, ":")
		scope = signalConfidenceKey(signalType, ruleName)
	}
	if r.asked[scope] {
		return
	}
	r.asked[scope] = true
	signal := r.classifier.EvaluateResponseSignals(r.ctx, r.input, func(ruleType, ruleName string) bool {
		if signalType == config.SignalTypeSystemOne {
			return ruleType == signalType && signalConfidenceKey(ruleType, ruleName) == scope
		}
		return ruleType == signalType
	})
	if signalType == config.SignalTypeHallucination && strings.TrimSpace(r.input.Context) == "" {
		metrics.RecordUnverifiedFactualResponse()
	}
	r.observed.merge(signal)
	for ruleType, matched := range signal.Matched {
		appendSignalMatches(r.signals, ruleType, matched)
	}
	maps.Copy(r.signals.SignalConfidences, signal.Confidences)
	maps.Copy(r.signals.SignalValues, signal.Values)
	maps.Copy(r.signals.SignalErrors, signal.Errors)
}

func newResponseSignal() *ResponseSignal {
	return &ResponseSignal{
		Matched:     map[string][]string{},
		Reported:    map[string][]string{},
		Confidences: map[string]float64{},
		Values:      map[string]float64{},
		Errors:      map[string]string{},
		Spans:       map[string][]modelservice.Span{},
	}
}

func (s *ResponseSignal) merge(other *ResponseSignal) {
	for signalType, matched := range other.Matched {
		s.Matched[signalType] = append(s.Matched[signalType], matched...)
	}
	for signalType, reported := range other.Reported {
		s.Reported[signalType] = append(s.Reported[signalType], reported...)
	}
	maps.Copy(s.Confidences, other.Confidences)
	maps.Copy(s.Values, other.Values)
	maps.Copy(s.Errors, other.Errors)
	maps.Copy(s.Spans, other.Spans)
}

// appendSignalMatches adds the matched references of a response-stage type
// to what the decision engine reads.
func appendSignalMatches(signals *decision.SignalMatches, signalType string, matched []string) {
	switch signalType {
	case config.SignalTypeSystemOne:
		signals.SystemOneRules = append(signals.SystemOneRules, matched...)
	case config.SignalTypeJailbreak:
		signals.JailbreakRules = append(signals.JailbreakRules, matched...)
	case config.SignalTypeSafety:
		signals.SafetyRules = append(signals.SafetyRules, matched...)
	case config.SignalTypeHallucination:
		signals.HallucinationRules = append(signals.HallucinationRules, matched...)
	case config.SignalTypeRefusal:
		signals.RefusalRules = append(signals.RefusalRules, matched...)
	}
}

// cloneSignalMatches copies what the response stage adds to, so the routed
// signals stay as routing published them.
func cloneSignalMatches(routed *decision.SignalMatches) *decision.SignalMatches {
	signals := &decision.SignalMatches{}
	if routed != nil {
		*signals = *routed
	}
	signals.SystemOneRules = slices.Clone(signals.SystemOneRules)
	signals.JailbreakRules = slices.Clone(signals.JailbreakRules)
	signals.SafetyRules = slices.Clone(signals.SafetyRules)
	signals.HallucinationRules = slices.Clone(signals.HallucinationRules)
	signals.RefusalRules = slices.Clone(signals.RefusalRules)
	signals.SignalConfidences = cloneOrEmpty(signals.SignalConfidences)
	signals.SignalValues = cloneOrEmpty(signals.SignalValues)
	signals.SignalErrors = cloneOrEmpty(signals.SignalErrors)
	return signals
}

func cloneOrEmpty[V any](values map[string]V) map[string]V {
	if values == nil {
		return map[string]V{}
	}
	return maps.Clone(values)
}
