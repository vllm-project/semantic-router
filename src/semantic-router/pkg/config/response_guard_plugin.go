package config

import (
	"fmt"
	"slices"
	"strings"
)

// DecisionPluginResponseGuard blocks or annotates a decision's answers on
// the signals they are checked with.
const DecisionPluginResponseGuard = "response_guard"

// What a response_guard rule does with an answer its condition matches, and
// what on_unknown does with one it cannot decide.
const (
	ResponseGuardBlock    = "block"
	ResponseGuardAnnotate = "annotate"
	ResponseGuardNone     = "none"
)

var responseGuardActions = []string{ResponseGuardBlock, ResponseGuardAnnotate, ResponseGuardNone}

// ResponseGuardPluginConfig acts on the answers of a decision. A rule's when
// is a rule tree like a decision's rules: it reads request-stage signals as
// routing evaluated them, and response-stage ones (jailbreak, safety and
// systemone rules with stage: response, and every hallucination and refusal
// rule), each evaluated the first time a rule reads it. Rules run in order,
// and the first that blocks ends the guard.
type ResponseGuardPluginConfig struct {
	Rules []ResponseGuardRule `json:"rules" yaml:"rules"`
	// OnUnknown is what a rule does when its condition cannot be decided: a
	// signal it reads failed, timed out or, for a hallucination rule, had no
	// grounding context to check the answer against. Default annotate.
	OnUnknown string `json:"on_unknown,omitempty" yaml:"on_unknown,omitempty" jsonschema:"enum=block,enum=annotate,enum=none"`
}

// ResponseGuardRule is one condition on an answer and what happens to an
// answer that meets it: block answers with the Router's 403 error, annotate
// adds a code to x-vsr-response-warnings and none only records the outcome.
type ResponseGuardRule struct {
	When   RuleNode `json:"when" yaml:"when"`
	Action string   `json:"action" yaml:"action" jsonschema:"enum=block,enum=annotate,enum=none"`
	// Details also writes a note naming what was found at the head of an
	// annotated answer.
	Details bool `json:"details,omitempty" yaml:"details,omitempty"`
}

// GetResponseGuardConfig returns the response_guard plugin configuration with
// its rule trees normalized, or nil when the decision has none.
func (d *Decision) GetResponseGuardConfig() *ResponseGuardPluginConfig {
	guard := decodeDecisionPlugin(d, DecisionPluginResponseGuard, &ResponseGuardPluginConfig{})
	if guard == nil {
		return nil
	}
	for i := range guard.Rules {
		// Validated at load; normalizing gives an omitted operator its AND.
		_ = normalizeRuleOperator(&guard.Rules[i].When, "when", false)
	}
	return guard
}

// Reads reports whether a rule of the guard reads a signal of signalType.
func (c *ResponseGuardPluginConfig) Reads(signalType string) bool {
	for i := range c.Rules {
		if len(collectSignalNames(&c.Rules[i].When, signalType)) > 0 {
			return true
		}
	}
	return false
}

// EffectiveOnUnknown is what a rule whose condition is unknown does.
func (c *ResponseGuardPluginConfig) EffectiveOnUnknown() string {
	if c == nil || c.OnUnknown == "" {
		return ResponseGuardAnnotate
	}
	return c.OnUnknown
}

// validateResponseGuardPlugin checks a guard's shape: its rules, their
// actions and the condition trees they read. Which signals a tree may name
// depends on its routing scope (validateResponseGuardReferences).
func validateResponseGuardPlugin(at PluginAt, p *ResponseGuardPluginConfig) error {
	if len(p.Rules) == 0 {
		return at.Errorf("rules: at least one rule is required")
	}
	if p.OnUnknown != "" && !slices.Contains(responseGuardActions, p.OnUnknown) {
		return at.Errorf("on_unknown must be %s", strings.Join(responseGuardActions, ", "))
	}
	for i := range p.Rules {
		rule := &p.Rules[i]
		path := fmt.Sprintf("rules[%d]", i)
		if !slices.Contains(responseGuardActions, rule.Action) {
			return at.Errorf("%s.action must be %s", path, strings.Join(responseGuardActions, ", "))
		}
		if rule.Details && rule.Action != ResponseGuardAnnotate {
			return at.Errorf("%s.details applies only to action: annotate", path)
		}
		if err := normalizeRuleOperator(&rule.When, path+".when", false); err != nil {
			return at.Errorf("%w", err)
		}
		if err := validateResponseGuardTree(&rule.When, path+".when"); err != nil {
			return at.Errorf("%w", err)
		}
	}
	return nil
}

// validateResponseGuardTree refuses what a guard condition cannot carry: an
// unknown condition takes the guard's on_unknown, so no node sets its own
// on_error or on_unknown.
func validateResponseGuardTree(node *RuleNode, path string) error {
	if node.OnUnknown != "" || node.OnError != "" {
		return fmt.Errorf("%s: a response_guard condition takes the guard's on_unknown; remove on_error and on_unknown", path)
	}
	if !node.IsLeaf() {
		for i := range node.Conditions {
			if err := validateResponseGuardTree(&node.Conditions[i], fmt.Sprintf("%s.conditions[%d]", path, i)); err != nil {
				return err
			}
		}
		return nil
	}
	signalType := strings.ToLower(strings.TrimSpace(node.Type))
	if signalType != SignalTypeProjection && !IsSupportedSignalType(signalType) {
		return fmt.Errorf("%s: unsupported signal type %q", path, node.Type)
	}
	return nil
}

// validateResponseGuardReferences resolves every condition of a routing
// scope's guards against the signals the scope declares; a guard may name a
// response-stage rule, which no decision rule can.
func validateResponseGuardReferences(cfg *RouterConfig) error {
	declared := projectionDeclaredSignals(cfg)
	declared[SignalTypeHallucination] = collectHallucinationRuleNames(cfg.HallucinationRules)
	declared[SignalTypeProjection] = collectProjectionOutputNames(cfg.Projections.Mappings)
	for i := range cfg.Decisions {
		decision := &cfg.Decisions[i]
		guard := decision.GetResponseGuardConfig()
		if guard == nil {
			continue
		}
		for j := range guard.Rules {
			if err := validateResponseGuardLeaves(cfg, decision.Name, j, &guard.Rules[j].When, declared); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateResponseGuardLeaves(cfg *RouterConfig, decisionName string, rule int, node *RuleNode, declared map[string]map[string]struct{}) error {
	if !node.IsLeaf() {
		for i := range node.Conditions {
			if err := validateResponseGuardLeaves(cfg, decisionName, rule, &node.Conditions[i], declared); err != nil {
				return err
			}
		}
		return nil
	}
	signalType := strings.ToLower(strings.TrimSpace(node.Type))
	if !projectionInputDeclared(declared, signalType, node.Name) {
		return fmt.Errorf("decision %q response_guard rules[%d]: signal %s(%q) is not declared in this recipe", decisionName, rule, signalType, node.Name)
	}
	return validateDecisionLeafNode(cfg, decisionName, node)
}

func collectHallucinationRuleNames(rules []HallucinationRule) map[string]struct{} {
	names := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		names[rule.Name] = struct{}{}
	}
	return names
}

func collectProjectionOutputNames(mappings []ProjectionMapping) map[string]struct{} {
	names := map[string]struct{}{}
	for _, mapping := range mappings {
		for _, output := range mapping.Outputs {
			names[output.Name] = struct{}{}
		}
	}
	return names
}

// ResponseGuardRequestConditions are the request-stage conditions of a
// decision's guard: routing evaluates them with the decisions' own, so the
// guard reads them once the model has answered.
func (c *RouterConfig) ResponseGuardRequestConditions(decision *Decision) []RuleNode {
	guard := decision.GetResponseGuardConfig()
	if guard == nil {
		return nil
	}
	var leaves []RuleNode
	var collect func(node *RuleNode)
	collect = func(node *RuleNode) {
		if !node.IsLeaf() {
			for i := range node.Conditions {
				collect(&node.Conditions[i])
			}
			return
		}
		if c.SignalStageOf(node.Type, node.Name) == SignalStageRequest {
			leaves = append(leaves, *node)
		}
	}
	for i := range guard.Rules {
		collect(&guard.Rules[i].When)
	}
	return leaves
}
