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
	NodeType   string  `json:"node_type"`             // "leaf", "AND", "OR", "NOT"
	SignalType string  `json:"signal_type,omitempty"` // populated for leaf nodes
	SignalName string  `json:"signal_name,omitempty"` // populated for leaf nodes
	Label      string  `json:"label,omitempty"`       // optional classifier label
	Matched    bool    `json:"matched"`
	Confidence float64 `json:"confidence"`
	// ConfidenceScored mirrors DecisionResult.ConfidenceScored at node level:
	// whether this node's confidence came entirely from reported signal
	// scores rather than structural 1.0 constants.
	ConfidenceScored bool         `json:"confidence_scored,omitempty"`
	Children         []*TraceNode `json:"children,omitempty"`
}

// DecisionTrace captures the full trace of a decision evaluation.
type DecisionTrace struct {
	DecisionName string     `json:"decision_name"`
	Matched      bool       `json:"matched"`
	Confidence   float64    `json:"confidence"`
	RootTrace    *TraceNode `json:"root_trace"`
}

// EvaluateDecisionsWithTrace evaluates all decisions and returns both the
// best match and a trace of every decision's evaluation tree.
func (e *DecisionEngine) EvaluateDecisionsWithTrace(
	signals *SignalMatches,
) (*DecisionResult, []DecisionTrace) {
	if len(e.decisions) == 0 {
		return nil, nil
	}

	var results []DecisionResult
	traces := make([]DecisionTrace, 0, len(e.decisions))

	for i := range e.decisions {
		decision := &e.decisions[i]
		evaluation, trace := e.evalDecisionWithTrace(decision, signals, false)
		if evaluation.state == evaluationUnknown {
			evaluation, trace = e.evalDecisionWithTrace(decision, signals, true)
		}
		matched := evaluation.state == evaluationTrue

		traces = append(traces, DecisionTrace{
			DecisionName: decision.Name,
			Matched:      matched,
			Confidence:   evaluation.confidence,
			RootTrace:    trace,
		})

		if matched {
			results = append(results, DecisionResult{
				Decision:         decision,
				Confidence:       evaluation.confidence,
				MatchedRules:     evaluation.matchedRules,
				ConfidenceScored: evaluation.scored,
				CatchAll:         isCatchAllRules(decision.Rules),
			})
		}
	}

	var best *DecisionResult
	if len(results) > 0 {
		best = e.selectBestDecision(results)
	}
	return best, traces
}

func (e *DecisionEngine) evalDecisionWithTrace(
	decision *config.Decision,
	signals *SignalMatches,
	legacy bool,
) (nodeEvaluation, *TraceNode) {
	if decision.Rules.IsEmpty() {
		return nodeEvaluation{state: evaluationTrue, scored: true}, &TraceNode{
			NodeType:         "fallback",
			Matched:          true,
			ConfidenceScored: true,
		}
	}
	return e.evalNode(decision.Rules, signals, legacy, true)
}

// FormatTrace returns a human-readable string representation of a decision trace.
func FormatTrace(dt DecisionTrace) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Decision: %s (matched=%v, confidence=%.3f)\n",
		dt.DecisionName, dt.Matched, dt.Confidence))
	if dt.RootTrace != nil {
		formatTraceNode(&sb, dt.RootTrace, 1)
	}
	return sb.String()
}

func formatTraceNode(sb *strings.Builder, node *TraceNode, depth int) {
	indent := strings.Repeat("  ", depth)
	matchSymbol := "✗"
	if node.Matched {
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
