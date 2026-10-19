package topiccontinuity

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

var defaultPolicy = HistoryPolicy{
	Limits:           Limits{MaxPriorTurns: 8, MaxTurnBytes: 16384, MaxInputBytes: 147456},
	IncludeAssistant: true,
}

func defaultConfig(policy HistoryPolicy) EvalConfig {
	return EvalConfig{Name: "topic_boundary", Policy: policy, Continuation: 0.35, Change: 0.08}
}

func text(value string) llmprotocol.Content {
	return llmprotocol.Content{Kind: llmprotocol.ContentText, Text: value}
}

func user(value string) llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{text(value)}}
}

func assistant(value string) llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{text(value)}}
}

func toolCall(id, name, arguments string) llmprotocol.Content {
	return llmprotocol.Content{
		Kind:     llmprotocol.ContentToolCall,
		ToolCall: &llmprotocol.ToolCall{ID: id, Name: name, Arguments: arguments},
	}
}

func assistantCalls(calls ...llmprotocol.Content) llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: calls}
}

func toolResult(callID, value string) llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
		Kind:       llmprotocol.ContentToolResult,
		ToolResult: &llmprotocol.ToolResult{CallID: callID, Content: []llmprotocol.Content{text(value)}},
	}}}
}

func image() llmprotocol.Content {
	return llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: "https://example.invalid/a.png"}
}

// exchange is one prior user/assistant turn.
func exchange(question, answer string) []llmprotocol.Message {
	return []llmprotocol.Message{user(question), assistant(answer)}
}

func conversation(parts ...interface{}) []llmprotocol.Message {
	var out []llmprotocol.Message
	for _, part := range parts {
		switch value := part.(type) {
		case llmprotocol.Message:
			out = append(out, value)
		case []llmprotocol.Message:
			out = append(out, value...)
		default:
			panic(fmt.Sprintf("unsupported part %T", part))
		}
	}
	return out
}

// unrelatedHistory returns n prior turns about a single unrelated subject.
func unrelatedHistory(n int) []llmprotocol.Message {
	var out []llmprotocol.Message
	for i := 0; i < n; i++ {
		out = append(out, exchange(
			"Refactor the routing module so plugins load lazily",
			"Done. The loader now defers plugin initialization until first use.")...)
	}
	return out
}

func evaluate(messages []llmprotocol.Message, policy HistoryPolicy) Result {
	return evaluateConfig(messages, defaultConfig(policy))
}

func evaluateConfig(messages []llmprotocol.Message, cfg EvalConfig) Result {
	prepared := prepare(context.Background(), messages, true, cfg.Policy)
	extracted := extract(context.Background(), prepared)
	return classify(cfg, prepared, extracted)
}

// assertInvariants checks the schema v1 invariants on any result.
func assertInvariants(t *testing.T, result Result) {
	t.Helper()
	if result.SchemaVersion != SchemaVersion || result.EvaluatorVersion != EvaluatorVersion {
		t.Fatalf("versions = %q/%q", result.SchemaVersion, result.EvaluatorVersion)
	}
	class, ok := reasonClass[result.Reason]
	if !ok || class != result.Class {
		t.Fatalf("reason %q does not belong to class %q", result.Reason, result.Class)
	}
	if !strings.HasPrefix(string(result.Reason), string(result.Class)+"_") {
		t.Fatalf("reason %q lacks class prefix %q", result.Reason, result.Class)
	}
	if result.Fallback != fallbackReason(result.Reason) {
		t.Fatalf("fallback %v inconsistent with reason %q", result.Fallback, result.Reason)
	}
	switch result.Coverage {
	case CoverageFull, CoverageWindow, CoveragePartial:
	default:
		t.Fatalf("undeclared coverage %q", result.Coverage)
	}
	if result.HistorySource != SourceOriginalSnapshot {
		t.Fatalf("undeclared source %q", result.HistorySource)
	}
	c := result.Confidence
	if math.IsNaN(c) || math.IsInf(c, 0) || c < 0 || c > 1 {
		t.Fatalf("confidence %v out of range", c)
	}
	switch result.Class {
	case ClassUnknown:
		if c != 0 {
			t.Fatalf("unknown with confidence %v", c)
		}
	case ClassChange:
		if result.Coverage != CoverageFull {
			t.Fatalf("change with coverage %q", result.Coverage)
		}
		if result.Reason == ReasonExplicitChange && c < 0.9 {
			t.Fatalf("explicit change confidence %v", c)
		}
		if result.Reason == ReasonDisjoint && (c < 0.3 || c > 0.6) {
			t.Fatalf("disjoint change confidence %v", c)
		}
	}
}

func near(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}
