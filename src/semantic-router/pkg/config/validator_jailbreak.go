package config

import (
	"fmt"
	"slices"
	"strings"
)

// Recognised values for JailbreakRule.Method.
//
// The runtime dispatches on this string in
// classification.evaluateJailbreakRule, which matches "contrastive" and sends
// everything else to the model-backed path. Without validation, an
// unrecognised value is not an error but a silent downgrade: a rule written as
// `method: hybrid` with jailbreak_patterns still loads, still reports healthy,
// and never consults a single pattern, because only the contrastive path reads
// them. A guardrail that silently ignores half its configuration is worse than
// one that refuses to start.
const (
	// JailbreakMethodContrastive scores the prompt against the configured
	// jailbreak_patterns / benign_patterns.
	JailbreakMethodContrastive = "contrastive"
	// JailbreakMethodClassifier runs the configured sequence classifier.
	// "" and "model" are accepted spellings of the same path.
	JailbreakMethodClassifier = "classifier"
	JailbreakMethodModel      = "model"
)

var validJailbreakMethods = []string{
	"",
	JailbreakMethodClassifier,
	JailbreakMethodModel,
	JailbreakMethodContrastive,
}

// validateJailbreakContracts rejects jailbreak signal rules that would load
// without doing what they say.
func validateJailbreakContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	seen := make(map[string]struct{}, len(cfg.JailbreakRules))
	for _, rule := range cfg.JailbreakRules {
		name := strings.TrimSpace(rule.Name)
		if name == "" {
			return fmt.Errorf("routing.signals.jailbreak: every rule needs a name")
		}
		if _, dup := seen[name]; dup {
			return fmt.Errorf("routing.signals.jailbreak: duplicate rule name %q", name)
		}
		seen[name] = struct{}{}

		// Compared as written. The runtime dispatches on the raw value
		// (`switch rule.Method` in classification.evaluateJailbreakRule), so
		// accepting a different spelling here would let "CONTRASTIVE" load and
		// then silently take the model path, which is the failure this
		// validator exists to stop.
		method := rule.Method
		if !slices.Contains(validJailbreakMethods, method) {
			return fmt.Errorf(
				"routing.signals.jailbreak %q: unknown method %q; valid values are %q, "+
					"spelled exactly (patterns are only consulted by %q)",
				name, rule.Method,
				[]string{JailbreakMethodClassifier, JailbreakMethodModel, JailbreakMethodContrastive},
				JailbreakMethodContrastive,
			)
		}

		// Direction is compared as written for the same reason as method: the
		// runtime reads the raw value (JailbreakRule.Stage), so a misspelt
		// "Response" would silently load as a request-stage rule.
		switch rule.Direction {
		case "", SignalDirectionRequest:
		case SignalDirectionResponse:
			if method == JailbreakMethodContrastive {
				return fmt.Errorf(
					"routing.signals.jailbreak %q: direction %q scores the model's output with "+
						"the sequence classifier; method %q compares a prompt against patterns "+
						"and is request-stage only",
					name, rule.Direction, JailbreakMethodContrastive,
				)
			}
			if rule.IncludeHistory {
				return fmt.Errorf(
					"routing.signals.jailbreak %q: include_history has no meaning with direction %q; "+
						"a single response carries no conversation history",
					name, rule.Direction,
				)
			}
		default:
			return fmt.Errorf(
				"routing.signals.jailbreak %q: unknown direction %q; valid values are %q and %q",
				name, rule.Direction, SignalDirectionRequest, SignalDirectionResponse,
			)
		}

		if method != JailbreakMethodContrastive &&
			(len(rule.JailbreakPatterns) > 0 || len(rule.BenignPatterns) > 0) {
			return fmt.Errorf(
				"routing.signals.jailbreak %q: jailbreak_patterns/benign_patterns are only "+
					"evaluated by method %q, but this rule uses method %q, so the patterns "+
					"would be silently ignored",
				name, JailbreakMethodContrastive, rule.Method,
			)
		}

		// The runtime match is `riskScore >= threshold`, so a threshold of 0
		// matches every request including plainly benign ones - it does not
		// mean "use a default". Go's zero value makes an omitted threshold
		// identical to an explicit 0, so both are rejected here rather than
		// silently turning the rule into an always-match.
		switch {
		case rule.Threshold <= 0:
			return fmt.Errorf(
				"routing.signals.jailbreak %q: threshold must be greater than 0 "+
					"(the runtime matches when risk >= threshold, so 0 marks every request "+
					"as a jailbreak); set an explicit threshold",
				name,
			)
		case rule.Threshold > 1:
			return fmt.Errorf(
				"routing.signals.jailbreak %q: threshold must be <= 1, got %v",
				name, rule.Threshold,
			)
		}
	}
	return nil
}
