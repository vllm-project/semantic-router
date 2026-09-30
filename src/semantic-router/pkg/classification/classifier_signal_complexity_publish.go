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
// serves all of them.
//
// The key must match how a decision names the condition, not how the rule is
// named: validateComplexityConditionName rejects a bare rule name at config
// load, so every complexity leaf is "<rule>:<verdict>" and the engine looks up
// "complexity:<rule>:<verdict>". A failure has no verdict, so all three are
// marked - whichever verdict a decision happens to gate on, evalLeaf finds the
// failure and, with rules.on_unknown set, the request resolves through that
// policy instead of reading the outage as a clean no-match.
func (c *Classifier) recordComplexityFailure(results *SignalResults, mu *sync.Mutex) {
	metrics.RecordComplexityEvaluationFailure(c.complexitySignalSource())
	mu.Lock()
	defer mu.Unlock()
	if results.SignalErrors == nil {
		results.SignalErrors = make(map[string]string)
	}
	for _, rule := range c.complexityRules() {
		for _, verdict := range ComplexityVerdictLabels {
			key := signalConfidenceKey(config.SignalTypeComplexity, rule.Name+":"+verdict)
			results.SignalErrors[key] = complexityEvaluationFailedCode
		}
	}
}
