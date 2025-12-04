/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package fusion

import (
	"fmt"
	"sort"
	"sync"
)

// FusionEngine evaluates fusion policies and returns routing decisions
type FusionEngine struct {
	policies []FusionPolicy
	cache    map[string]*ASTNode
	cacheMu  sync.RWMutex
	parser   *Parser
}

// NewFusionEngine creates a new fusion engine with the given policies
func NewFusionEngine(policies []FusionPolicy) (*FusionEngine, error) {
	if len(policies) == 0 {
		return nil, fmt.Errorf("no fusion policies provided")
	}

	engine := &FusionEngine{
		policies: policies,
		cache:    make(map[string]*ASTNode),
		parser:   NewParser(),
	}

	// Pre-compile all policy expressions
	for i := range engine.policies {
		policy := &engine.policies[i]
		if err := engine.compilePolicy(policy); err != nil {
			return nil, fmt.Errorf("failed to compile policy %s: %w", policy.Name, err)
		}
	}

	return engine, nil
}

// compilePolicy pre-compiles a policy's expression into an AST
func (e *FusionEngine) compilePolicy(policy *FusionPolicy) error {
	ast, err := e.parser.Parse(policy.Condition)
	if err != nil {
		return err
	}

	e.cacheMu.Lock()
	e.cache[policy.Condition] = ast
	e.cacheMu.Unlock()

	return nil
}

// Evaluate evaluates all policies and returns the result of the first matching policy
// Policies are evaluated in priority order (highest priority first)
// Returns error if no policy matches
func (e *FusionEngine) Evaluate(ctx *SignalContext) (*FusionResult, error) {
	if ctx == nil {
		return nil, fmt.Errorf("nil signal context")
	}

	// Sort policies by priority (descending)
	sortedPolicies := make([]FusionPolicy, len(e.policies))
	copy(sortedPolicies, e.policies)
	sort.Slice(sortedPolicies, func(i, j int) bool {
		return sortedPolicies[i].Priority > sortedPolicies[j].Priority
	})

	// Evaluate policies in priority order (SHORT-CIRCUIT on first match)
	for _, policy := range sortedPolicies {
		matched, err := e.evaluatePolicy(&policy, ctx)
		if err != nil {
			// Log error but continue to next policy
			// In production, this should use proper logging
			continue
		}

		if matched {
			// SHORT-CIRCUIT: First match wins!
			return e.createResult(&policy), nil
		}
	}

	// No policy matched
	return nil, fmt.Errorf("no fusion policy matched")
}

// evaluatePolicy evaluates a single policy against the signal context
func (e *FusionEngine) evaluatePolicy(policy *FusionPolicy, ctx *SignalContext) (bool, error) {
	// Get pre-compiled AST from cache
	e.cacheMu.RLock()
	ast, exists := e.cache[policy.Condition]
	e.cacheMu.RUnlock()

	if !exists {
		// This shouldn't happen if compilePolicy was called in constructor
		// But handle it gracefully
		var err error
		ast, err = e.parser.Parse(policy.Condition)
		if err != nil {
			return false, fmt.Errorf("failed to parse condition: %w", err)
		}

		// Cache for future use
		e.cacheMu.Lock()
		e.cache[policy.Condition] = ast
		e.cacheMu.Unlock()
	}

	// Evaluate the AST
	evaluator := NewEvaluator(ast, ctx)
	return evaluator.Evaluate()
}

// createResult creates a fusion result from a matched policy
func (e *FusionEngine) createResult(policy *FusionPolicy) *FusionResult {
	return &FusionResult{
		Action:        policy.Action,
		Models:        policy.Models,
		Category:      policy.Category,
		BoostWeight:   policy.BoostWeight,
		Message:       policy.Message,
		MatchedPolicy: policy.Name,
		Priority:      policy.Priority,
	}
}

// ValidatePolicy validates a single policy
func ValidatePolicy(policy *FusionPolicy) error {
	if policy.Name == "" {
		return fmt.Errorf("policy name cannot be empty")
	}

	if policy.Condition == "" {
		return fmt.Errorf("policy %s: condition cannot be empty", policy.Name)
	}

	// Validate that the condition can be parsed
	parser := NewParser()
	if _, err := parser.Parse(policy.Condition); err != nil {
		return fmt.Errorf("policy %s: invalid condition: %w", policy.Name, err)
	}

	// Validate action-specific fields
	switch policy.Action {
	case ActionBlock:
		if policy.Message == "" {
			return fmt.Errorf("policy %s: block action requires a message", policy.Name)
		}

	case ActionRoute:
		if len(policy.Models) == 0 {
			return fmt.Errorf("policy %s: route action requires at least one model", policy.Name)
		}

	case ActionBoost:
		if policy.Category == "" {
			return fmt.Errorf("policy %s: boost_category action requires a category", policy.Name)
		}
		if policy.BoostWeight <= 0 {
			return fmt.Errorf("policy %s: boost_category action requires a positive boost_weight", policy.Name)
		}

	case ActionFallthrough:
		// No additional validation needed

	case "":
		return fmt.Errorf("policy %s: action cannot be empty", policy.Name)

	default:
		return fmt.Errorf("policy %s: unknown action %s", policy.Name, policy.Action)
	}

	return nil
}

// ValidatePolicies validates multiple policies
func ValidatePolicies(policies []FusionPolicy) error {
	if len(policies) == 0 {
		return fmt.Errorf("at least one policy is required")
	}

	// Check for duplicate names
	names := make(map[string]bool)
	for i := range policies {
		policy := &policies[i]

		// Validate individual policy
		if err := ValidatePolicy(policy); err != nil {
			return err
		}

		// Check for duplicate names
		if names[policy.Name] {
			return fmt.Errorf("duplicate policy name: %s", policy.Name)
		}
		names[policy.Name] = true
	}

	return nil
}
