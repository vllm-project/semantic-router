package classification

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// responseJailbreakSignalFailedCode marks a response_jailbreak rule the
// detector could not resolve. It mirrors jailbreakEvaluationFailedCode so both
// stages report an unresolved detector the same way.
const responseJailbreakSignalFailedCode = "response_jailbreak_evaluation_failed"

// ResponseJailbreakSignal is the response-stage jailbreak observation for one
// response, in the shape the decision engine consumes request-stage signals in.
type ResponseJailbreakSignal struct {
	MatchedRules []string
	Confidences  map[string]float64
	Errors       map[string]string
}

// EvaluateResponseJailbreakSignal thresholds one response's risk score against
// every declared response_jailbreak rule.
//
// The detector runs once for the response, not once per rule: every rule asks
// the same model the same question about the same text and differs only in
// where it draws the line, so re-classifying per rule would buy nothing and
// cost an inference each time. This is the same reason the request-stage
// evaluator classifies each unique content piece once and lets every rule read
// the cache.
func EvaluateResponseJailbreakSignal(rules []config.ResponseJailbreakRule, riskScore float32, resolved bool) *ResponseJailbreakSignal {
	if len(rules) == 0 {
		return nil
	}
	signal := &ResponseJailbreakSignal{
		Confidences: make(map[string]float64, len(rules)),
		Errors:      make(map[string]string),
	}
	for _, rule := range rules {
		key := signalConfidenceKey(config.SignalTypeResponseJailbreak, rule.Name)
		if !resolved {
			// Unresolved, not clean. Recorded where every other signal records
			// it, so a decision reading this rule resolves through on_unknown
			// rather than silently treating the response as verified.
			signal.Errors[key] = responseJailbreakSignalFailedCode
			continue
		}
		signal.Confidences[key] = float64(riskScore)
		if riskScore >= rule.Threshold {
			signal.MatchedRules = append(signal.MatchedRules, rule.Name)
		}
	}
	return signal
}

// ResponseJailbreakSignalKeyPrefix is the prefix every response_jailbreak
// signal key carries, for callers reading the published signal back.
const ResponseJailbreakSignalKeyPrefix = config.SignalTypeResponseJailbreak + ":"
