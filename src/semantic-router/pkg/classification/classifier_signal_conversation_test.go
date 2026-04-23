package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func ptrFloat(v float64) *float64 { return &v }

func TestConversation_MultiTurnActive(t *testing.T) {
	rule := config.ConversationRule{
		Name: "multi_turn_active",
		Feature: config.ConversationFeature{
			Type:   "count",
			Source: config.ConversationSource{Type: "message", Role: "user"},
		},
		Predicate: &config.NumericPredicate{GTE: ptrFloat(2)},
	}

	facts := ConversationFacts{UserMessageCount: 3}
	val := resolveConversationValue(rule.Feature, facts)
	if val != 3.0 {
		t.Fatalf("expected 3.0, got %v", val)
	}
	if !conversationPredicateMatches(rule, val) {
		t.Fatal("expected multi_turn_active to match with 3 user messages")
	}

	factsSingle := ConversationFacts{UserMessageCount: 1}
	valSingle := resolveConversationValue(rule.Feature, factsSingle)
	if conversationPredicateMatches(rule, valSingle) {
		t.Fatal("expected multi_turn_active NOT to match with 1 user message")
	}
}

func TestConversation_DeveloperGuided(t *testing.T) {
	rule := config.ConversationRule{
		Name: "developer_guided",
		Feature: config.ConversationFeature{
			Type:   "exists",
			Source: config.ConversationSource{Type: "message", Role: "developer"},
		},
	}

	facts := ConversationFacts{HasDeveloperMessage: true}
	val := resolveConversationValue(rule.Feature, facts)
	if val != 1.0 {
		t.Fatalf("expected 1.0 for exists with developer, got %v", val)
	}
	if !conversationPredicateMatches(rule, val) {
		t.Fatal("expected developer_guided to match")
	}

	factsNo := ConversationFacts{HasDeveloperMessage: false}
	valNo := resolveConversationValue(rule.Feature, factsNo)
	if valNo != 0.0 {
		t.Fatalf("expected 0.0, got %v", valNo)
	}
	if conversationPredicateMatches(rule, valNo) {
		t.Fatal("expected developer_guided NOT to match without developer message")
	}
}

func TestConversation_ToolingAvailable(t *testing.T) {
	rule := config.ConversationRule{
		Name: "tooling_available",
		Feature: config.ConversationFeature{
			Type:   "count",
			Source: config.ConversationSource{Type: "tool_definition"},
		},
		Predicate: &config.NumericPredicate{GT: ptrFloat(0)},
	}

	facts := ConversationFacts{ToolDefinitionCount: 3}
	val := resolveConversationValue(rule.Feature, facts)
	if val != 3.0 {
		t.Fatalf("expected 3.0, got %v", val)
	}
	if !conversationPredicateMatches(rule, val) {
		t.Fatal("expected tooling_available to match with 3 tool definitions")
	}

	factsNone := ConversationFacts{ToolDefinitionCount: 0}
	valNone := resolveConversationValue(rule.Feature, factsNone)
	if conversationPredicateMatches(rule, valNone) {
		t.Fatal("expected tooling_available NOT to match with 0 definitions")
	}
}

func TestConversation_ToolGrounded(t *testing.T) {
	rule := config.ConversationRule{
		Name: "tool_grounded",
		Feature: config.ConversationFeature{
			Type:   "count",
			Source: config.ConversationSource{Type: "message", Role: "tool"},
		},
		Predicate: &config.NumericPredicate{GT: ptrFloat(0)},
	}

	facts := ConversationFacts{ToolMessageCount: 2}
	val := resolveConversationValue(rule.Feature, facts)
	if val != 2.0 {
		t.Fatalf("expected 2.0, got %v", val)
	}
	if !conversationPredicateMatches(rule, val) {
		t.Fatal("expected tool_grounded to match with 2 tool messages")
	}

	factsNone := ConversationFacts{ToolMessageCount: 0}
	valNone := resolveConversationValue(rule.Feature, factsNone)
	if conversationPredicateMatches(rule, valNone) {
		t.Fatal("expected tool_grounded NOT to match with 0 tool messages")
	}
}

func TestConversation_ToolLoopDeep(t *testing.T) {
	rule := config.ConversationRule{
		Name: "tool_loop_deep",
		Feature: config.ConversationFeature{
			Type:   "count",
			Source: config.ConversationSource{Type: "assistant_tool_cycle"},
		},
		Predicate: &config.NumericPredicate{GTE: ptrFloat(2)},
	}

	facts := ConversationFacts{CompletedToolCycles: 3}
	val := resolveConversationValue(rule.Feature, facts)
	if val != 3.0 {
		t.Fatalf("expected 3.0, got %v", val)
	}
	if !conversationPredicateMatches(rule, val) {
		t.Fatal("expected tool_loop_deep to match with 3 cycles")
	}

	factsShallow := ConversationFacts{CompletedToolCycles: 1}
	valShallow := resolveConversationValue(rule.Feature, factsShallow)
	if conversationPredicateMatches(rule, valShallow) {
		t.Fatal("expected tool_loop_deep NOT to match with 1 cycle")
	}
}

func TestConversation_NonUserRoleCount(t *testing.T) {
	facts := ConversationFacts{
		AssistantMessageCount: 2,
		SystemMessageCount:    1,
		ToolMessageCount:      3,
		HasDeveloperMessage:   true,
	}
	got := countMessagesByRole("non_user", facts)
	want := 2 + 1 + 3 + 1 // assistant + system + tool + developer
	if got != want {
		t.Fatalf("non_user count: expected %d, got %d", want, got)
	}
}

func TestConversation_EmptyRoleCountsAll(t *testing.T) {
	facts := ConversationFacts{
		UserMessageCount:      2,
		AssistantMessageCount: 1,
		SystemMessageCount:    1,
		ToolMessageCount:      1,
		HasDeveloperMessage:   true,
	}
	got := countMessagesByRole("", facts)
	want := 2 + 1 + 1 + 1 + 1 // all roles
	if got != want {
		t.Fatalf("empty role count: expected %d, got %d", want, got)
	}
}

func TestConversation_AssistantToolCallCount(t *testing.T) {
	rule := config.ConversationRule{
		Name: "heavy_tool_use",
		Feature: config.ConversationFeature{
			Type:   "count",
			Source: config.ConversationSource{Type: "assistant_tool_call"},
		},
		Predicate: &config.NumericPredicate{GTE: ptrFloat(3)},
	}

	facts := ConversationFacts{AssistantToolCallCount: 5}
	val := resolveConversationValue(rule.Feature, facts)
	if !conversationPredicateMatches(rule, val) {
		t.Fatal("expected match with 5 tool calls >= 3")
	}

	factsLow := ConversationFacts{AssistantToolCallCount: 2}
	valLow := resolveConversationValue(rule.Feature, factsLow)
	if conversationPredicateMatches(rule, valLow) {
		t.Fatal("expected no match with 2 tool calls < 3")
	}
}
