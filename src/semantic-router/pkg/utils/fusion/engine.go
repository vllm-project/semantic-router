package fusion

import (
	"fmt"
	"sort"
)

// Engine is the main signal fusion engine that evaluates policies
type Engine struct {
	policy *Policy
}

// NewEngine creates a new fusion engine with the given policy
func NewEngine(policy *Policy) *Engine {
	// Sort rules by priority (highest first) for efficient evaluation
	sortedRules := make([]Rule, len(policy.Rules))
	copy(sortedRules, policy.Rules)
	sort.Slice(sortedRules, func(i, j int) bool {
		return sortedRules[i].Priority > sortedRules[j].Priority
	})

	return &Engine{
		policy: &Policy{Rules: sortedRules},
	}
}

// Evaluate evaluates the policy against the given signal context
// Returns the first matching rule result (short-circuit evaluation)
func (e *Engine) Evaluate(context *SignalContext) (*EvaluationResult, error) {
	if e.policy == nil || len(e.policy.Rules) == 0 {
		return &EvaluationResult{
			Matched: false,
			Action:  ActionFallthrough,
		}, nil
	}

	evaluator := NewExpressionEvaluator(context)

	// Evaluate rules in priority order (short-circuit: first match wins)
	for _, rule := range e.policy.Rules {
		matched, err := evaluator.Evaluate(rule.Condition)
		if err != nil {
			return nil, fmt.Errorf("error evaluating rule %s: %w", rule.Name, err)
		}

		if matched {
			// First matching rule wins
			return &EvaluationResult{
				Matched:     true,
				MatchedRule: rule.Name,
				Action:      rule.Action,
				Models:      rule.Models,
				Category:    rule.Category,
				BoostWeight: rule.BoostWeight,
				Message:     rule.Message,
			}, nil
		}
	}

	// No rules matched - fallthrough
	return &EvaluationResult{
		Matched: false,
		Action:  ActionFallthrough,
	}, nil
}

// NewSignalContext creates a new empty signal context
func NewSignalContext() *SignalContext {
	return &SignalContext{
		Signals: make(map[string]Signal),
	}
}

// AddSignal adds a signal to the context
func (sc *SignalContext) AddSignal(signal Signal) {
	key := signal.Provider + "." + signal.Name
	sc.Signals[key] = signal
}

// GetSignal retrieves a signal from the context
func (sc *SignalContext) GetSignal(provider, name string) (Signal, bool) {
	key := provider + "." + name
	signal, exists := sc.Signals[key]
	return signal, exists
}
