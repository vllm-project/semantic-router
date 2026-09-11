package evaluationplane

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

const (
	maxRoutingTraceLineBytes = 256 * 1024
	maxRoutingTraceDepth     = 8
	maxRoutingTraceChildren  = 32
	maxRoutingTraceDecisions = 64
	maxRoutingTraceSignals   = 128
	maxRoutingTraceTokens    = 128
	maxRoutingTraceNodes     = 256
)

var routingSafeTokenPattern = regexp.MustCompile(`^[A-Za-z0-9_.:/+ -]+$`)

type routingTraceNodeEvidence struct {
	NodeType         string                     `json:"node_type"`
	SignalType       *string                    `json:"signal_type"`
	SignalName       *string                    `json:"signal_name"`
	Label            *string                    `json:"label"`
	State            *string                    `json:"state"`
	Matched          bool                       `json:"matched"`
	Confidence       *float64                   `json:"confidence"`
	HasSignalError   bool                       `json:"has_signal_error"`
	ConfidenceScored bool                       `json:"confidence_scored"`
	Children         []routingTraceNodeEvidence `json:"children"`
}

type routingDecisionTraceEvidence struct {
	DecisionName string                    `json:"decision_name"`
	State        *string                   `json:"state"`
	Matched      bool                      `json:"matched"`
	Confidence   *float64                  `json:"confidence"`
	OnUnknown    *string                   `json:"on_unknown"`
	RootTrace    *routingTraceNodeEvidence `json:"root_trace"`
}

type routingSignalEvidence struct {
	Key        string   `json:"key"`
	Confidence *float64 `json:"confidence"`
	Value      *float64 `json:"value"`
	HasError   bool     `json:"has_error"`
}

type routingDiagnosticEvidence struct {
	SchemaVersion          string                         `json:"schema_version"`
	CaseID                 string                         `json:"case_id"`
	Truncated              bool                           `json:"truncated"`
	Recipe                 *string                        `json:"recipe"`
	DecisionName           *string                        `json:"decision_name"`
	Algorithm              *string                        `json:"algorithm"`
	Plugins                []string                       `json:"plugins"`
	RecommendedModels      []string                       `json:"recommended_models"`
	SelectedModel          *string                        `json:"selected_model"`
	SelectionStatus        *string                        `json:"selection_status"`
	SelectionMethod        *string                        `json:"selection_method"`
	RoutingDecision        *string                        `json:"routing_decision"`
	Traces                 []routingDecisionTraceEvidence `json:"traces"`
	Signals                []routingSignalEvidence        `json:"signals"`
	AppliedUnknownPolicies [][]string                     `json:"applied_unknown_policies"`
}

func validateRoutingTraceArtifact(runDir string, caseIDs map[string]struct{}) error {
	if len(caseIDs) == 0 {
		return fmt.Errorf("%w: routing traces require a validated case set", ErrInvalid)
	}
	seenCases := make(map[string]struct{}, len(caseIDs))
	path := filepath.Join(runDir, "routing-traces.jsonl")
	err := scanEvidenceJSONLines(
		path,
		maxWorkerArtifactBytes,
		maxRoutingTraceLineBytes,
		len(caseIDs),
		func(line []byte, lineNumber int) error {
			var trace routingDiagnosticEvidence
			if decodeErr := decodeStrictJSONLine(line, &trace); decodeErr != nil {
				return fmt.Errorf("%w: routing-traces.jsonl line %d is invalid: %w", ErrInvalid, lineNumber, decodeErr)
			}
			if validationErr := validateRoutingDiagnostic(trace, caseIDs); validationErr != nil {
				return fmt.Errorf("%w: routing-traces.jsonl line %d: %w", ErrInvalid, lineNumber, validationErr)
			}
			if _, duplicate := seenCases[trace.CaseID]; duplicate {
				return fmt.Errorf("%w: routing-traces.jsonl contains duplicate case_id %q", ErrInvalid, trace.CaseID)
			}
			seenCases[trace.CaseID] = struct{}{}
			return nil
		},
	)
	return err
}

func validateRoutingDiagnostic(trace routingDiagnosticEvidence, caseIDs map[string]struct{}) error {
	if trace.SchemaVersion != SchemaVersion {
		return fmt.Errorf("schema_version must be %q", SchemaVersion)
	}
	if !evidenceIDPattern.MatchString(trace.CaseID) {
		return fmt.Errorf("case_id must be a portable non-empty identity")
	}
	if _, ok := caseIDs[trace.CaseID]; !ok {
		return fmt.Errorf("case_id %q is absent from the validated case set", trace.CaseID)
	}
	if trace.Plugins == nil || trace.RecommendedModels == nil || trace.Traces == nil || trace.Signals == nil ||
		trace.AppliedUnknownPolicies == nil {
		return fmt.Errorf("routing trace collections cannot be null")
	}
	if len(trace.Plugins) > maxRoutingTraceTokens || len(trace.RecommendedModels) > maxRoutingTraceTokens ||
		len(trace.Traces) > maxRoutingTraceDecisions || len(trace.Signals) > maxRoutingTraceSignals ||
		len(trace.AppliedUnknownPolicies) > maxRoutingTraceSignals {
		return fmt.Errorf("routing trace collection exceeds its cardinality limit")
	}
	for name, value := range map[string]*string{
		"recipe": trace.Recipe, "decision_name": trace.DecisionName, "algorithm": trace.Algorithm,
		"selected_model": trace.SelectedModel, "selection_status": trace.SelectionStatus,
		"selection_method": trace.SelectionMethod, "routing_decision": trace.RoutingDecision,
	} {
		limit := 160
		switch name {
		case "selected_model":
			limit = 256
		case "selection_status":
			limit = 64
		case "selection_method":
			limit = 128
		}
		if err := validateRoutingSafeToken(name, value, limit); err != nil {
			return err
		}
	}
	for index := range trace.Plugins {
		if err := validateRoutingSafeToken("plugin", &trace.Plugins[index], 160); err != nil {
			return err
		}
	}
	for index := range trace.RecommendedModels {
		if err := validateRoutingSafeToken("recommended_model", &trace.RecommendedModels[index], 256); err != nil {
			return err
		}
	}
	nodeCount := 0
	for index, decision := range trace.Traces {
		if err := validateRoutingDecisionTrace(decision, &nodeCount); err != nil {
			return fmt.Errorf("decision trace %d: %w", index+1, err)
		}
	}
	for index, signal := range trace.Signals {
		if err := validateRoutingSafeToken("signal key", &signal.Key, 160); err != nil {
			return fmt.Errorf("signal %d: %w", index+1, err)
		}
		for name, value := range map[string]*float64{"confidence": signal.Confidence, "value": signal.Value} {
			if value != nil && !finiteFloat(*value) {
				return fmt.Errorf("signal %d %s must be finite", index+1, name)
			}
		}
	}
	previousPolicyKey := ""
	for index, policy := range trace.AppliedUnknownPolicies {
		if len(policy) != 2 {
			return fmt.Errorf("applied unknown policy %d must contain a key and policy", index+1)
		}
		if err := validateRoutingSafeToken("applied unknown policy key", &policy[0], 128); err != nil {
			return fmt.Errorf("applied unknown policy %d: %w", index+1, err)
		}
		if err := validateRoutingSafeToken("applied unknown policy", &policy[1], 32); err != nil {
			return fmt.Errorf("applied unknown policy %d: %w", index+1, err)
		}
		if previousPolicyKey != "" && policy[0] <= previousPolicyKey {
			return fmt.Errorf("applied unknown policies must have unique canonical key order")
		}
		previousPolicyKey = policy[0]
	}
	return nil
}

func validateRoutingDecisionTrace(trace routingDecisionTraceEvidence, nodeCount *int) error {
	if err := validateRoutingSafeToken("decision_name", &trace.DecisionName, 128); err != nil {
		return err
	}
	if err := validateRoutingSafeToken("decision state", trace.State, 32); err != nil {
		return err
	}
	if err := validateRoutingSafeToken("on_unknown", trace.OnUnknown, 32); err != nil {
		return err
	}
	if trace.Confidence != nil && (!finiteFloat(*trace.Confidence) || *trace.Confidence < 0 || *trace.Confidence > 1) {
		return fmt.Errorf("confidence must be a finite fraction")
	}
	if trace.RootTrace != nil {
		return validateRoutingTraceNode(*trace.RootTrace, 0, nodeCount)
	}
	return nil
}

func validateRoutingTraceNode(node routingTraceNodeEvidence, depth int, nodeCount *int) error {
	if depth >= maxRoutingTraceDepth {
		return fmt.Errorf("trace tree exceeds its depth limit")
	}
	if *nodeCount >= maxRoutingTraceNodes {
		return fmt.Errorf("routing trace exceeds its global node budget")
	}
	*nodeCount++
	if err := validateRoutingSafeToken("node_type", &node.NodeType, 64); err != nil {
		return err
	}
	for name, value := range map[string]*string{
		"signal_type": node.SignalType, "signal_name": node.SignalName, "label": node.Label,
	} {
		if err := validateRoutingSafeToken(name, value, 128); err != nil {
			return err
		}
	}
	if err := validateRoutingSafeToken("node state", node.State, 32); err != nil {
		return err
	}
	if node.Confidence != nil && (!finiteFloat(*node.Confidence) || *node.Confidence < 0 || *node.Confidence > 1) {
		return fmt.Errorf("confidence must be a finite fraction")
	}
	if node.Children == nil {
		return fmt.Errorf("trace children cannot be null")
	}
	if len(node.Children) > maxRoutingTraceChildren {
		return fmt.Errorf("trace node exceeds its child limit")
	}
	for _, child := range node.Children {
		if err := validateRoutingTraceNode(child, depth+1, nodeCount); err != nil {
			return err
		}
	}
	return nil
}

func validateRoutingSafeToken(name string, value *string, limit int) error {
	if value == nil {
		return nil
	}
	trimmed := strings.TrimSpace(*value)
	if trimmed == "" || trimmed != *value || len(trimmed) > limit || strings.Contains(trimmed, "://") ||
		strings.Contains(trimmed, "@") || !routingSafeTokenPattern.MatchString(trimmed) {
		return fmt.Errorf("%s is not a bounded safe token", name)
	}
	return nil
}
