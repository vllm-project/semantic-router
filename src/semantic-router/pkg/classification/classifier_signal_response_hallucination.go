package classification

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Failure codes a hallucination rule carries in SignalErrors when the answer
// could not be checked. They are distinct because the plugin acts on them
// differently: an answer with no grounding context to check against is the
// unverified-factual case it already has an action for, a detector that
// failed or was never provisioned is not.
const (
	HallucinationSignalFailedCode         = "hallucination_evaluation_failed"
	HallucinationSignalContextUnavailable = "hallucination_context_unavailable"
)

// ResponseHallucinationSignal is the response-stage hallucination observation
// for one response, in the shape the decision engine consumes request-stage
// signals in.
type ResponseHallucinationSignal struct {
	MatchedRules []string
	Confidences  map[string]float64
	Errors       map[string]string
}

// EvaluateResponseHallucinationSignal publishes one response's hallucination
// verdict under every declared rule.
//
// The detector runs once for the response, not once per rule: every rule asks
// the same detector the same question about the same answer against the same
// context, so re-running it per rule would buy nothing and cost an inference
// each time. A rule matches when the detector found an unsupported span; the
// detector's own threshold and span filters decide that, on hallucination_model.
func EvaluateResponseHallucinationSignal(rules []config.HallucinationRule, detected bool, confidence float32, failureCode string) *ResponseHallucinationSignal {
	if len(rules) == 0 {
		return nil
	}
	signal := &ResponseHallucinationSignal{
		Confidences: make(map[string]float64, len(rules)),
		Errors:      make(map[string]string),
	}
	for _, rule := range rules {
		key := signalConfidenceKey(config.SignalTypeHallucination, rule.Name)
		if failureCode != "" {
			// Unresolved, not clean: recorded where every other signal records
			// it, so the plugin applies its failure handling instead of reading
			// silence as a verified answer.
			signal.Errors[key] = failureCode
			continue
		}
		signal.Confidences[key] = float64(confidence)
		if detected {
			signal.MatchedRules = append(signal.MatchedRules, rule.Name)
		}
	}
	return signal
}
