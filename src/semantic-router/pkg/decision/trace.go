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
	NodeType   string       `json:"node_type"`             // "leaf", "AND", "OR", "NOT"
	SignalType string       `json:"signal_type,omitempty"` // populated for leaf nodes
	SignalName string       `json:"signal_name,omitempty"` // populated for leaf nodes
	Matched    bool         `json:"matched"`
	Confidence float64      `json:"confidence"`
	Children   []*TraceNode `json:"children,omitempty"`
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
		matched, confidence, matchedRules, trace := e.evalNodeWithTrace(decision.Rules, signals)

		dt := DecisionTrace{
			DecisionName: decision.Name,
			Matched:      matched,
			Confidence:   confidence,
			RootTrace:    trace,
		}
		traces = append(traces, dt)

		if matched {
			results = append(results, DecisionResult{
				Decision:     decision,
				Confidence:   confidence,
				MatchedRules: matchedRules,
			})
		}
	}

	var best *DecisionResult
	if len(results) > 0 {
		best = e.selectBestDecision(results)
	}
	return best, traces
}

// evalNodeWithTrace mirrors evalNode but also builds a TraceNode tree.
func (e *DecisionEngine) evalNodeWithTrace(
	node config.RuleNode,
	signals *SignalMatches,
) (matched bool, confidence float64, matchedRules []string, trace *TraceNode) {
	if node.IsLeaf() {
		m, c, r := e.evalLeaf(node.Type, node.Name, signals)
		return m, c, r, &TraceNode{
			NodeType:   "leaf",
			SignalType: node.Type,
			SignalName: node.Name,
			Matched:    m,
			Confidence: c,
		}
	}

	op := strings.ToUpper(node.Operator)
	switch op {
	case "AND":
		return e.evalANDWithTrace(node.Conditions, signals)
	case "NOT":
		return e.evalNOTWithTrace(node.Conditions, signals)
	default:
		return e.evalORWithTrace(node.Conditions, signals)
	}
}

func (e *DecisionEngine) evalANDWithTrace(
	children []config.RuleNode,
	signals *SignalMatches,
) (bool, float64, []string, *TraceNode) {
	trace := &TraceNode{NodeType: "AND"}

	if len(children) == 0 {
		trace.Matched = true
		trace.Confidence = 0
		return true, 0, nil, trace
	}

	totalConf := 0.0
	var matchedRules []string
	for _, child := range children {
		m, c, r, childTrace := e.evalNodeWithTrace(child, signals)
		trace.Children = append(trace.Children, childTrace)
		if !m {
			trace.Matched = false
			trace.Confidence = 0
			return false, 0, nil, trace
		}
		totalConf += c
		matchedRules = append(matchedRules, r...)
	}

	avg := totalConf / float64(len(children))
	trace.Matched = true
	trace.Confidence = avg
	return true, avg, matchedRules, trace
}

func (e *DecisionEngine) evalORWithTrace(
	children []config.RuleNode,
	signals *SignalMatches,
) (bool, float64, []string, *TraceNode) {
	trace := &TraceNode{NodeType: "OR"}

	bestConf := 0.0
	var bestRules []string
	anyMatched := false

	for _, child := range children {
		m, c, r, childTrace := e.evalNodeWithTrace(child, signals)
		trace.Children = append(trace.Children, childTrace)
		if m {
			anyMatched = true
			if c > bestConf {
				bestConf = c
				bestRules = r
			}
		}
	}

	if anyMatched {
		trace.Matched = true
		trace.Confidence = bestConf
		return true, bestConf, bestRules, trace
	}
	return false, 0, nil, trace
}

func (e *DecisionEngine) evalNOTWithTrace(
	children []config.RuleNode,
	signals *SignalMatches,
) (bool, float64, []string, *TraceNode) {
	trace := &TraceNode{NodeType: "NOT"}

	if len(children) != 1 {
		return false, 0, nil, trace
	}

	m, c, r, childTrace := e.evalNodeWithTrace(children[0], signals)
	trace.Children = append(trace.Children, childTrace)

	if !m {
		trace.Matched = true
		trace.Confidence = 1.0
		return true, 1.0, r, trace
	}
	trace.Matched = false
	trace.Confidence = c
	return false, c, r, trace
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
