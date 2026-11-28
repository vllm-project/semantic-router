package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func ptrFloat(v float64) *float64 { return &v }

func TestConversation_InvalidToolCallCount(t *testing.T) {
	rule := config.ConversationRule{
		Name: "invalid_tool_call",
		Feature: config.ConversationFeature{
			Type:   "count",
			Source: config.ConversationSource{Type: "invalid_tool_call"},
		},
		Predicate: &config.NumericPredicate{GTE: ptrFloat(1)},
	}

	facts := ConversationFacts{InvalidToolCalls: 2}
	val := resolveConversationValue(rule.Feature, facts)
	if val != 2.0 {
		t.Fatalf("expected 2.0, got %v", val)
	}
	if !conversationPredicateMatches(rule, val) {
		t.Fatal("expected invalid_tool_call to match with 2 invalid calls")
	}

	factsClean := ConversationFacts{InvalidToolCalls: 0}
	valClean := resolveConversationValue(rule.Feature, factsClean)
	if valClean != 0.0 {
		t.Fatalf("expected 0.0, got %v", valClean)
	}
	if conversationPredicateMatches(rule, valClean) {
		t.Fatal("expected invalid_tool_call NOT to match with 0 invalid calls")
	}
}

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

	facts := ConversationFacts{ToolResultCount: 3}
	val := resolveConversationValue(rule.Feature, facts)
	if val != 3.0 {
		t.Fatalf("expected 3.0, got %v", val)
	}
	if !conversationPredicateMatches(rule, val) {
		t.Fatal("expected tool_loop_deep to match with 3 cycles")
	}

	factsShallow := ConversationFacts{ToolResultCount: 1}
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

func TestConversation_ToolChoiceFacts(t *testing.T) {
	requiredRule := config.ConversationRule{
		Name: "required",
		Feature: config.ConversationFeature{
			Type:   "exists",
			Source: config.ConversationSource{Type: "tool_choice_required"},
		},
	}
	noneRule := config.ConversationRule{
		Name: "none",
		Feature: config.ConversationFeature{
			Type:   "exists",
			Source: config.ConversationSource{Type: "tool_choice_none"},
		},
	}

	if got := resolveConversationValue(requiredRule.Feature, ConversationFacts{ToolChoiceRequired: true}); got != 1 {
		t.Fatalf("required fact value = %v, want 1", got)
	}
	if got := resolveConversationValue(requiredRule.Feature, ConversationFacts{ToolChoiceNone: true}); got != 0 {
		t.Fatalf("none fact must not satisfy required source, got %v", got)
	}
	if got := resolveConversationValue(noneRule.Feature, ConversationFacts{ToolChoiceNone: true}); got != 1 {
		t.Fatalf("none fact value = %v, want 1", got)
	}
	if got := resolveConversationValue(noneRule.Feature, ConversationFacts{ToolChoiceRequired: true}); got != 0 {
		t.Fatalf("required fact must not satisfy none source, got %v", got)
	}
}

func TestConversation_ActiveToolLoopCount(t *testing.T) {
	rule := config.ConversationRule{
		Name: "active_tool_loop",
		Feature: config.ConversationFeature{
			Type:   "exists",
			Source: config.ConversationSource{Type: "active_tool_loop"},
		},
	}

	activeCases := []ConversationFacts{
		{LastMessageToolResult: true},
		{LastMessageRole: "tool"},
		{LastUserAfterToolResult: true},
		{LastAssistantToolCall: true},
	}
	for i, facts := range activeCases {
		val := resolveConversationValue(rule.Feature, facts)
		if val != 1.0 {
			t.Fatalf("active case %d: expected 1.0, got %v", i, val)
		}
		if !conversationPredicateMatches(rule, val) {
			t.Fatalf("active case %d: expected active_tool_loop to match", i)
		}
	}

	completedPriorRun := ConversationFacts{
		LastMessageRole:         "user",
		AssistantToolCallCount:  1,
		ToolResultCount:         1,
		LastMessageToolResult:   false,
		LastUserAfterToolResult: false,
	}
	val := resolveConversationValue(rule.Feature, completedPriorRun)
	if val != 0.0 {
		t.Fatalf("completed prior run: expected 0.0, got %v", val)
	}
	if conversationPredicateMatches(rule, val) {
		t.Fatal("completed prior run should not match active_tool_loop")
	}

	historicalUnmatchedCall := ConversationFacts{
		LastMessageRole:        "user",
		AssistantToolCallCount: 2,
		ToolResultCount:        1,
	}
	val = resolveConversationValue(rule.Feature, historicalUnmatchedCall)
	if val != 0.0 || conversationPredicateMatches(rule, val) {
		t.Fatal("historical unmatched call must not make a later user turn active")
	}
}

func TestConversation_FlowToolState(t *testing.T) {
	rule := config.ConversationRule{
		Name: "flow_tool_state",
		Feature: config.ConversationFeature{
			Type:   "exists",
			Source: config.ConversationSource{Type: "flow_tool_state"},
		},
	}

	matched := resolveConversationValue(rule.Feature, ConversationFacts{LastMessageFlowToolResult: true})
	if matched != 1 || !conversationPredicateMatches(rule, matched) {
		t.Fatal("trailing Flow tool result should match flow_tool_state")
	}
	notMatched := resolveConversationValue(rule.Feature, ConversationFacts{})
	if notMatched != 0 || conversationPredicateMatches(rule, notMatched) {
		t.Fatal("ordinary history should not match flow_tool_state")
	}
}

func TestConversation_TurnIndex(t *testing.T) {
	rule := config.ConversationRule{
		Name: "late_session",
		Feature: config.ConversationFeature{
			Type:   "count",
			Source: config.ConversationSource{Type: "turn_index"},
		},
		Predicate: &config.NumericPredicate{GTE: ptrFloat(2)},
	}

	late := resolveConversationValue(rule.Feature, ConversationFacts{TurnIndex: 2})
	if late != 2.0 || !conversationPredicateMatches(rule, late) {
		t.Fatalf("turn_index 2 should match gte 2, got %v", late)
	}
	early := resolveConversationValue(rule.Feature, ConversationFacts{TurnIndex: 1})
	if conversationPredicateMatches(rule, early) {
		t.Fatal("turn_index 1 should not match gte 2")
	}
	// No visible user history falls back to 0, so single-turn requests never
	// match a late-session rule.
	if got := resolveConversationValue(rule.Feature, ConversationFacts{}); got != 0 {
		t.Fatalf("missing facts must resolve turn_index to 0, got %v", got)
	}
}

func TestConversation_SessionHistoryTokens(t *testing.T) {
	rule := config.ConversationRule{
		Name: "long_session",
		Feature: config.ConversationFeature{
			Type:   "count",
			Source: config.ConversationSource{Type: "session_history_tokens"},
		},
		Predicate: &config.NumericPredicate{GTE: ptrFloat(100)},
	}

	loaded := resolveConversationValue(rule.Feature, ConversationFacts{HistoryTokens: 120})
	if loaded != 120 || !conversationPredicateMatches(rule, loaded) {
		t.Fatalf("history tokens 120 should match gte 100, got %v", loaded)
	}
	light := resolveConversationValue(rule.Feature, ConversationFacts{HistoryTokens: 8})
	if conversationPredicateMatches(rule, light) {
		t.Fatal("history tokens 8 should not match gte 100")
	}
	if got := resolveConversationValue(rule.Feature, ConversationFacts{}); got != 0 {
		t.Fatalf("missing facts must resolve session_history_tokens to 0, got %v", got)
	}
}

// A result failed when the client flags it, and succeeded only when Claude
// Code's foreground Bash or a Codex shell header says so. Everything else,
// text a command printed or a tool fetched included, says nothing.
func TestToolResultStatusOfReadsOnlyTheFlagOrTheCodexHeader(t *testing.T) {
	failed, cleared := true, false
	anthropic, responses := llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1
	bash := &llmprotocol.ToolCall{Name: "Bash", Arguments: `{"command":"make"}`}
	background := &llmprotocol.ToolCall{Name: "Bash", Arguments: `{"command":"make","run_in_background":true}`}
	read := &llmprotocol.ToolCall{Name: "Read", Arguments: `{"file_path":"Makefile"}`}
	exec := &llmprotocol.ToolCall{Name: "exec_command"}
	shell := &llmprotocol.ToolCall{Name: "shell"}
	header := "Exit code: 1\nWall time: 0.1 seconds\nOutput:\n"
	for _, tc := range []struct {
		name    string
		format  llmprotocol.WireFormat
		call    *llmprotocol.ToolCall
		isError *bool
		text    string
		want    ToolResultStatus
	}{
		{"flag set", anthropic, read, &failed, "File does not exist.", ToolResultFailed},
		{"flag set without the call", anthropic, nil, &failed, "denied", ToolResultFailed},
		{"bash with the flag cleared", anthropic, bash, &cleared, "ok", ToolResultSucceeded},
		{"bash in the background", anthropic, background, &cleared, "Command running in background with ID: b1.", ToolResultUnknown},
		{"another tool with the flag cleared", anthropic, read, nil, "1\tall:", ToolResultUnknown},
		{"anthropic text with a codex header", anthropic, read, nil, header, ToolResultUnknown},
		{"shell nonzero exit", responses, shell, nil, "Exit code: 127\nWall time: 0.1 seconds\nOutput:\nsh: make: not found\n", ToolResultFailed},
		{"exec_command nonzero exit", responses, exec, nil, "Chunk ID: a\nWall time: 0.1000 seconds\nProcess exited with code 2\nOriginal token count: 1\nOutput:\n", ToolResultFailed},
		{"exec_command zero exit", responses, exec, nil, "Chunk ID: a\nProcess exited with code 0\nOutput:\nExit code: 1\n", ToolResultSucceeded},
		{"still running", responses, exec, nil, "Chunk ID: a\nProcess running with session ID 3\nOutput:\n", ToolResultUnknown},
		{"a line codex does not write", responses, exec, nil, "CI log for run 42\n" + header, ToolResultUnknown},
		{"no output line", responses, exec, nil, "Exit code: 1\nbuild failed", ToolResultUnknown},
		{"another tool's header", responses, &llmprotocol.ToolCall{Name: "fetch"}, nil, header, ToolResultUnknown},
		{"a namespaced tool's header", responses, &llmprotocol.ToolCall{Name: "exec_command", Namespace: "mcp__ci"}, nil, header, ToolResultUnknown},
		{"without the call", responses, nil, nil, header, ToolResultUnknown},
	} {
		if got := ToolResultStatusOf(tc.format, tc.call, tc.isError, tc.text); got != tc.want {
			t.Errorf("%s: ToolResultStatusOf() = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// Unknown results neither add a failure nor count toward the two successes
// in a row that clear the count.
func TestToolFailureCountSkipsUnknownResults(t *testing.T) {
	var count ToolFailureCount
	for _, status := range []ToolResultStatus{ToolResultFailed, ToolResultFailed, ToolResultUnknown, ToolResultSucceeded, ToolResultUnknown} {
		count.Observe(status)
	}
	if count.Failed != 2 {
		t.Fatalf("Failed = %d after one success, want 2", count.Failed)
	}
	count.Observe(ToolResultSucceeded)
	if count.Failed != 0 {
		t.Fatalf("Failed = %d after two successes, want 0", count.Failed)
	}
}
