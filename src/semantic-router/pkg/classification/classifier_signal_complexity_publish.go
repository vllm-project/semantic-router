package classification

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// complexityEvaluationFailedCode is what every complexity rule carries in
// SignalErrors when the evaluation produced no verdict at all.
const complexityEvaluationFailedCode = "complexity_evaluation_failed"

// complexitySignalSourceLocal names the prototype-scoring path in metrics
// keyed by which path ran, so a remote outage and a local one are told apart.
const complexitySignalSourceLocal = "local"

// publishComplexityValues records the numbers behind a verdict under keys that
// say what they are. The local path fuses a text margin with an image margin,
// so it publishes each component and the fused result under ":margin". A
// score.v1 backend returns one number in the model's own units - not a margin
// - so it is published under ":score" and nothing else, rather than a row of
// zero-valued margin keys that would read as "no evidence either way".
// label_distribution.v1 carries no number beyond the confidence.
func publishComplexityValues(values map[string]float64, result ComplexityRuleResult) {
	prefix := "complexity:" + result.RuleName
	switch result.SignalSource {
	case complexitySignalSourceScore:
		values[prefix+":score"] = result.FusedMargin
	case complexitySignalSourceLabels:
		return
	default:
		values[prefix+":text_hard_score"] = result.TextHardScore
		values[prefix+":text_easy_score"] = result.TextEasyScore
		values[prefix+":text_margin"] = result.TextMargin
		values[prefix+":image_hard_score"] = result.ImageHardScore
		values[prefix+":image_easy_score"] = result.ImageEasyScore
		values[prefix+":image_margin"] = result.ImageMargin
		values[prefix+":margin"] = result.FusedMargin
	}
}

// complexitySignalSource names the path that answers for this classifier,
// mirroring the dispatch in classifyComplexity.
func (c *Classifier) complexitySignalSource() string {
	switch {
	case c.complexityScoreBackend != nil:
		return complexitySignalSourceScore
	case c.complexityLabelBackend != nil:
		return complexitySignalSourceLabels
	default:
		return complexitySignalSourceLocal
	}
}

// recordComplexityFailure makes a failed evaluation visible instead of letting
// the signal silently vanish. Every rule is marked, because one evaluation
// serves all of them. The decision engine already reads SignalErrors: a rule
// node over a failed signal evaluates false unless its `on_error: match` says
// a failure counts as a match, so a decision can fail open or closed without
// a new setting here. The counter surfaces the outage in dashboards, which
// matters most for a remote scorer, where a network fault is routine.
func (c *Classifier) recordComplexityFailure(results *SignalResults, mu *sync.Mutex) {
	metrics.RecordComplexityEvaluationFailure(c.complexitySignalSource())
	mu.Lock()
	defer mu.Unlock()
	if results.SignalErrors == nil {
		results.SignalErrors = make(map[string]string)
	}
	for _, rule := range c.complexityRules() {
		results.SignalErrors[signalConfidenceKey(config.SignalTypeComplexity, rule.Name)] = complexityEvaluationFailedCode
	}
}
