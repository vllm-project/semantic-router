package decision

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TraceNode captures the evaluation result of a single node in a decision's
// rule tree. When trace mode is enabled, every evalNode call produces a
// TraceNode, forming a tree that mirrors the boolean expression structure.
type TraceNode struct {
	NodeType    string  `json:"node_type"`             // "leaf", "AND", "OR", "NOT"
	SignalType  string  `json:"signal_type,omitempty"` // populated for leaf nodes
	SignalName  string  `json:"signal_name,omitempty"` // populated for leaf nodes
	Label       string  `json:"label,omitempty"`       // optional classifier label
	State       string  `json:"state"`
	Matched     bool    `json:"matched"`
	Confidence  float64 `json:"confidence"`
	SignalError string  `json:"signal_error,omitempty"`
	// ConfidenceScored mirrors DecisionResult.ConfidenceScored at node level:
	// whether this node's confidence came entirely from reported signal
	// scores rather than structural 1.0 constants.
	ConfidenceScored bool         `json:"confidence_scored,omitempty"`
	Children         []*TraceNode `json:"children,omitempty"`
}

// DecisionTrace captures the full trace of a decision evaluation.
type DecisionTrace struct {
	DecisionName string     `json:"decision_name"`
	State        string     `json:"state"`
	Matched      bool       `json:"matched"`
	Confidence   float64    `json:"confidence"`
	OnUnknown    string     `json:"on_unknown,omitempty"`
	RootTrace    *TraceNode `json:"root_trace"`
}

// EvaluateDecisionsWithTraceAndDiagnostics evaluates all decisions and returns
// the best match with a trace of every decision's evaluation tree.
func (e *DecisionEngine) EvaluateDecisionsWithTraceAndDiagnostics(
	signals *SignalMatches,
) (*DecisionResult, []DecisionTrace, EvaluationDiagnostics, error) {
	evaluations := e.evaluateDecisions(signals, true)
	return evaluations.result, evaluations.traces, evaluations.diagnostics, evaluations.failure
}

func (e *DecisionEngine) evalDecisionWithTrace(
	decision *config.Decision,
	signals *SignalMatches,
	policy config.UnknownPolicy,
) (nodeEvaluation, *TraceNode) {
	if decision.Rules.IsEmpty() {
		return nodeEvaluation{state: evaluationTrue, scored: true}, &TraceNode{
			NodeType:         "fallback",
			State:            evaluationTrue.String(),
			Matched:          true,
			ConfidenceScored: true,
		}
	}
	return e.evalNode(decision.Rules, signals, policy, true)
}

func newDecisionTrace(
	decision *config.Decision,
	evaluation nodeEvaluation,
	originalState evaluationState,
	policy string,
	trace *TraceNode,
) DecisionTrace {
	return DecisionTrace{
		DecisionName: decision.Name,
		State:        originalState.String(),
		Matched:      evaluation.state == evaluationTrue,
		Confidence:   evaluation.confidence,
		OnUnknown:    policy,
		RootTrace:    trace,
	}
}

// FormatTrace returns a human-readable string representation of a decision trace.
func FormatTrace(dt DecisionTrace) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Decision: %s (state=%s, matched=%v, confidence=%.3f",
		dt.DecisionName, dt.State, dt.Matched, dt.Confidence))
	if dt.OnUnknown != "" {
		sb.WriteString(fmt.Sprintf(", on_unknown=%s", dt.OnUnknown))
	}
	sb.WriteString(")\n")
	if dt.RootTrace != nil {
		formatTraceNode(&sb, dt.RootTrace, 1)
	}
	return sb.String()
}

func formatTraceNode(sb *strings.Builder, node *TraceNode, depth int) {
	indent := strings.Repeat("  ", depth)
	matchSymbol := "✗"
	if node.State == evaluationUnknown.String() {
		matchSymbol = "?"
	} else if node.Matched {
		matchSymbol = "✓"
	}

	if node.NodeType == "leaf" {
		fmt.Fprintf(sb, "%s%s %s(%q) conf=%.3f\n",
			indent, matchSymbol, node.SignalType, node.SignalName, node.Confidence)
		return
	}

	fmt.Fprintf(sb, "%s%s %s conf=%.3f\n",
		indent, matchSymbol, node.NodeType, node.Confidence)
	for _, child := range node.Children {
		formatTraceNode(sb, child, depth+1)
	}
}
