package classification

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// ResponseConditions reads rule trees over one answer: the request-stage
// signals routing published for its request, and the response-stage rules
// (jailbreak rules with direction: response, hallucination rules, systemone
// rules with stage: response), each evaluated the first time a tree reads
// it. One detector call serves every rule of a type that reads the same
// answer, as it does at the request stage.
type ResponseConditions struct {
	classifier *Classifier
	ctx        context.Context
	input      ResponseInput
	engine     *decision.DecisionEngine
	signals    *decision.SignalMatches

	jailbreak      *ResponseJailbreakObservation
	hallucination  *ResponseHallucinationObservation
	systemOne      *ResponseSystemOneSignal
	askedSystemOne map[string]bool
}

// ResponseJailbreakObservation is how the response-direction jailbreak rules
// read one answer: the scan, when the detector produced one, and each rule's
// match, score or failure.
type ResponseJailbreakObservation struct {
	Scan   *JailbreakScan
	Signal *ResponseJailbreakSignal
}

// ResponseHallucinationObservation is how the hallucination rules read one
// answer: the detector's evidence, or why the answer could not be checked,
// and each rule's match, score or failure.
type ResponseHallucinationObservation struct {
	Evidence    *HallucinationEvidence
	FailureCode string
	Signal      *ResponseHallucinationSignal
}

// HallucinationEvidence is the hallucination detector's verdict on one answer
// and the unsupported spans it rests on; Explained carries their offsets,
// severity and explanation when the detector explains its spans.
type HallucinationEvidence struct {
	Detected   bool
	Confidence float32
	Score      HallucinationScore
	Spans      []string
	Explained  []EnhancedHallucinationSpan
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
		engine:     decision.NewDecisionEngine(c.Config.KeywordRules, c.Config.EmbeddingRules, c.Config.Categories, nil, c.Config.Strategy).WithRoutingScope(c.Config.RoutingScope),
		signals:    cloneSignalMatches(routed),
	}
}

// ResponseSignalReady reports whether the detector behind the response-stage
// rules of signalType can answer: the jailbreak model for response-direction
// jailbreak rules, the hallucination detector for hallucination rules.
// Systemone rules are answered by their deployments.
func (c *Classifier) ResponseSignalReady(signalType string) bool {
	switch signalType {
	case config.SignalTypeJailbreak:
		return c.IsJailbreakEnabled()
	case config.SignalTypeHallucination:
		return c.IsHallucinationDetectorReady()
	}
	return true
}

// Evaluate reads one rule tree, evaluating each response-stage rule it reaches
// that no tree has read yet; an AND stops at its first false condition.
func (r *ResponseConditions) Evaluate(tree config.RuleNode) decision.RuleOutcome {
	return r.engine.EvaluateRule(tree, r.signals, r.observe)
}

// Jailbreak is the response-direction jailbreak observation, or nil when no
// tree read a response-direction rule.
func (r *ResponseConditions) Jailbreak() *ResponseJailbreakObservation { return r.jailbreak }

// Hallucination is the hallucination observation, or nil when no tree read a
// hallucination rule.
func (r *ResponseConditions) Hallucination() *ResponseHallucinationObservation {
	return r.hallucination
}

// SystemOne is the response-stage systemone observation, or nil when no tree
// read a response-stage systemone rule.
func (r *ResponseConditions) SystemOne() *ResponseSystemOneSignal { return r.systemOne }

func (r *ResponseConditions) observe(leaf config.RuleNode) {
	cfg := r.classifier.Config
	signalType := strings.ToLower(strings.TrimSpace(leaf.Type))
	if cfg.SignalStageOf(signalType, leaf.Name) != config.SignalStageResponse {
		return
	}
	switch signalType {
	case config.SignalTypeJailbreak:
		if r.jailbreak == nil {
			r.observeJailbreak()
		}
	case config.SignalTypeHallucination:
		if r.hallucination == nil {
			r.observeHallucination()
		}
	case config.SignalTypeSystemOne:
		if !r.askedSystemOne[leaf.Name] {
			r.askSystemOne(leaf.Name)
		}
	}
}

// observeJailbreak scans the answer once for every response-direction rule:
// they ask the same model the same question about the same text and differ
// only in where they draw the line. An answer without text, such as one made
// of tool calls, has nothing to scan, which leaves the rules unknown rather
// than clean.
func (r *ResponseConditions) observeJailbreak() {
	c := r.classifier
	rules := c.Config.ResponseJailbreakRules()
	observation := &ResponseJailbreakObservation{}
	r.jailbreak = observation
	if c.IsJailbreakEnabled() && r.input.Answer != "" {
		start := time.Now()
		scan, err := c.ScanJailbreak(r.ctx, r.input.Answer)
		latency := time.Since(start).Seconds()
		if err != nil {
			logging.Errorf("Response jailbreak signal evaluation failed: %v", err)
		} else {
			observation.Scan = &scan
			for _, rule := range rules {
				c.recordSignalExtraction(config.SignalTypeJailbreak, rule.Name, latency)
			}
		}
	}
	observation.Signal = EvaluateResponseJailbreakSignal(rules, observation.Scan)
	if observation.Signal == nil {
		return
	}
	r.signals.JailbreakRules = append(r.signals.JailbreakRules, observation.Signal.MatchedRules...)
	maps.Copy(r.signals.SignalConfidences, observation.Signal.Confidences)
	maps.Copy(r.signals.SignalErrors, observation.Signal.Errors)
}

// observeHallucination checks the answer against its grounding context once
// for every hallucination rule. An answer with no context to check it
// against leaves the rules unknown (HallucinationSignalContextUnavailable).
func (r *ResponseConditions) observeHallucination() {
	c := r.classifier
	observation := &ResponseHallucinationObservation{}
	r.hallucination = observation
	switch {
	case !c.IsHallucinationDetectionEnabled() || r.input.Answer == "":
		observation.FailureCode = HallucinationSignalFailedCode
	case strings.TrimSpace(r.input.Context) == "":
		observation.FailureCode = HallucinationSignalContextUnavailable
		metrics.RecordUnverifiedFactualResponse()
	default:
		start := time.Now()
		evidence, err := c.hallucinationEvidence(r.ctx, r.input)
		latency := time.Since(start).Seconds()
		metrics.RecordHallucinationDetectionLatency(latency)
		if err != nil {
			logging.Errorf("Hallucination signal evaluation failed: %v", err)
			observation.FailureCode = HallucinationSignalFailedCode
			break
		}
		observation.Evidence = evidence
		for _, rule := range c.Config.HallucinationRules {
			c.recordSignalExtraction(config.SignalTypeHallucination, rule.Name, latency)
		}
	}
	detected, confidence, score := false, float32(0), HallucinationScore{}
	if evidence := observation.Evidence; evidence != nil {
		detected, confidence, score = evidence.Detected, evidence.Confidence, evidence.Score
	}
	observation.Signal = EvaluateResponseHallucinationSignal(c.Config.HallucinationRules, detected, confidence, observation.FailureCode, score)
	if observation.Signal == nil {
		return
	}
	r.signals.HallucinationRules = append(r.signals.HallucinationRules, observation.Signal.MatchedRules...)
	maps.Copy(r.signals.SignalConfidences, observation.Signal.Confidences)
	maps.Copy(r.signals.SignalValues, observation.Signal.Values)
	maps.Copy(r.signals.SignalErrors, observation.Signal.Errors)
}

// hallucinationEvidence runs the detector on the answer, with span details
// when the detector explains its spans.
func (c *Classifier) hallucinationEvidence(ctx context.Context, input ResponseInput) (*HallucinationEvidence, error) {
	if !c.Config.HallucinationMitigation.HallucinationModel.IncludeExplanation {
		result, err := c.DetectHallucination(ctx, input.Context, input.Request, input.Answer)
		if err != nil {
			return nil, err
		}
		if result == nil {
			return nil, fmt.Errorf("hallucination detector returned no result")
		}
		return &HallucinationEvidence{
			Detected: result.HallucinationDetected, Confidence: result.Confidence,
			Score: HallucinationScore{Available: result.ScoreAvailable, Kind: result.ScoreKind}, Spans: result.UnsupportedSpans,
		}, nil
	}
	result, err := c.DetectHallucinationWithExplanations(ctx, input.Context, input.Request, input.Answer)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return nil, fmt.Errorf("hallucination detector returned no result")
	}
	evidence := &HallucinationEvidence{
		Detected: result.HallucinationDetected, Confidence: result.Confidence,
		Score: HallucinationScore{Available: result.ScoreAvailable, Kind: result.ScoreKind}, Explained: result.Spans,
	}
	for _, span := range result.Spans {
		evidence.Spans = append(evidence.Spans, span.Text)
	}
	return evidence, nil
}

// askSystemOne asks one response-stage systemone rule about the answer.
func (r *ResponseConditions) askSystemOne(rule string) {
	if r.askedSystemOne == nil {
		r.askedSystemOne = map[string]bool{}
	}
	r.askedSystemOne[rule] = true
	signal := r.classifier.EvaluateResponseSystemOne(r.ctx, r.input, []string{rule})
	if r.systemOne == nil {
		r.systemOne = &ResponseSystemOneSignal{Confidences: map[string]float64{}, Values: map[string]float64{}, Errors: map[string]string{}}
	}
	r.systemOne.MatchedRules = append(r.systemOne.MatchedRules, signal.MatchedRules...)
	maps.Copy(r.systemOne.Confidences, signal.Confidences)
	maps.Copy(r.systemOne.Values, signal.Values)
	maps.Copy(r.systemOne.Errors, signal.Errors)
	r.signals.SystemOneRules = append(r.signals.SystemOneRules, signal.MatchedRules...)
	maps.Copy(r.signals.SignalConfidences, signal.Confidences)
	maps.Copy(r.signals.SignalValues, signal.Values)
	maps.Copy(r.signals.SignalErrors, signal.Errors)
}

// cloneSignalMatches copies what the response stage adds to, so the routed
// signals stay as routing published them.
func cloneSignalMatches(routed *decision.SignalMatches) *decision.SignalMatches {
	signals := &decision.SignalMatches{}
	if routed != nil {
		*signals = *routed
	}
	signals.JailbreakRules = slices.Clone(signals.JailbreakRules)
	signals.HallucinationRules = slices.Clone(signals.HallucinationRules)
	signals.SystemOneRules = slices.Clone(signals.SystemOneRules)
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
