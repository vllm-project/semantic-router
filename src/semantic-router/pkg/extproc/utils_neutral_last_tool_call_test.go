package extproc

import (
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func toolCallMessage(names ...string) llmprotocol.Message {
	message := llmprotocol.Message{Role: llmprotocol.RoleAssistant}
	for _, name := range names {
		message.Content = append(message.Content, llmprotocol.Content{
			Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call_" + name, Name: name},
		})
	}
	return message
}

func toolResultMessage(callID string) llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: callID},
	}}}
}

// Only the latest assistant message's tool calls count: an agent turn that
// read files after an earlier edit is a read turn.
func TestLastToolCallCountsOnlyTheLatestAssistantMessage(t *testing.T) {
	snapshot := extractSemanticRequestSignals(&llmprotocol.Request{Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "fix the bug"}}},
		toolCallMessage("Edit"),
		toolResultMessage("call_Edit"),
		toolCallMessage("Read", "Grep"),
		toolResultMessage("call_Read"),
		toolResultMessage("call_Grep"),
	}})
	if !slices.Equal(snapshot.LastAssistantToolNames, []string{"Read", "Grep"}) {
		t.Fatalf("LastAssistantToolNames = %v, want [Read Grep]", snapshot.LastAssistantToolNames)
	}
	history := signalConversationHistoryFromSnapshot(snapshot)
	if !slices.Equal(history.lastAssistantToolNames, []string{"Read", "Grep"}) {
		t.Fatalf("history lastAssistantToolNames = %v, want [Read Grep]", history.lastAssistantToolNames)
	}
}

// Arguments ride along with the names, one per call, so a rule can read the command.
func TestLastToolCallKeepsEachCallsArguments(t *testing.T) {
	bash := llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
		{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "a", Name: "Bash", Arguments: `{"command":"ls"}`}},
		{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "b", Name: "Read", Arguments: `{"file_path":"/a"}`}},
	}}
	snapshot := extractSemanticRequestSignals(&llmprotocol.Request{Messages: []llmprotocol.Message{
		toolCallMessage("Edit"), toolResultMessage("call_Edit"), bash, toolResultMessage("a"), toolResultMessage("b"),
	}})
	history := signalConversationHistoryFromSnapshot(snapshot)
	if !slices.Equal(history.lastAssistantToolArgs, []string{`{"command":"ls"}`, `{"file_path":"/a"}`}) {
		t.Fatalf("lastAssistantToolArgs = %v", history.lastAssistantToolArgs)
	}
	if len(history.lastAssistantToolArgs) != len(history.lastAssistantToolNames) {
		t.Fatalf("arguments %d not aligned with names %d", len(history.lastAssistantToolArgs), len(history.lastAssistantToolNames))
	}
}

// A Responses history carries each call of a turn as its own item, which the
// codec decodes into consecutive assistant messages. They are one turn: every
// call counts, in order, whatever reasoning or text sits between them.
func TestLastToolCallJoinsConsecutiveResponsesItems(t *testing.T) {
	call := func(id, cmd string) string {
		return `{"type":"function_call","call_id":"` + id + `","name":"exec_command","arguments":"{\"cmd\":\"` + cmd + `\"}"}`
	}
	output := func(id string) string {
		return `{"type":"function_call_output","call_id":"` + id + `","output":"ok"}`
	}
	reasoning := `{"type":"reasoning","summary":[{"type":"summary_text","text":"look first"}]}`
	text := `{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Checking."}]}`
	earlier := call("e", "python3 fix.py") + "," + output("e") + ","
	for _, tc := range []struct {
		name  string
		items string
		want  []string
	}{
		{"write then read", earlier + call("a", "rm x") + "," + call("b", "ls") + "," + output("a") + "," + output("b"), []string{"rm x", "ls"}},
		{"read then write", earlier + call("a", "ls") + "," + call("b", "rm x") + "," + output("a") + "," + output("b"), []string{"ls", "rm x"}},
		{"reasoning between calls", earlier + call("a", "ls") + "," + reasoning + "," + call("b", "cat y") + "," + output("a") + "," + output("b"), []string{"ls", "cat y"}},
		{"assistant text after the calls", earlier + call("a", "ls") + "," + call("b", "cat y") + "," + text, []string{"ls", "cat y"}},
	} {
		body := `{"model":"m","input":[{"type":"message","role":"user","content":"fix it"},` + tc.items + `]}`
		request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(body))
		if err != nil {
			t.Fatalf("%s: decode: %v", tc.name, err)
		}
		history := signalConversationHistoryFromSnapshot(extractSemanticRequestSignals(&request))
		var commands []string
		for _, args := range history.lastAssistantToolArgs {
			commands = append(commands, args[len(`{"cmd":"`):len(args)-len(`"}`)])
		}
		if !slices.Equal(commands, tc.want) || len(history.lastAssistantToolNames) != len(tc.want) {
			t.Errorf("%s: last turn calls = %v %v, want %v", tc.name, history.lastAssistantToolNames, commands, tc.want)
		}
	}
}

// A decoded request of each protocol routes on its latest turn. With the
// read-only pair from the docs, the turn after a read goes to "small", and a
// turn after a write, after no call or on a compacted history does not.
func TestLastToolCallRoutesDecodedRequests(t *testing.T) {
	const readOnly = "^[ \\t]*(ls|cat|head|tail|wc|file|stat|rg|grep|tree|pwd|git (status|log|show))([ \\t][^;&|<>\\n`$]*)?$"
	reads := []config.ToolCallMatch{
		{Tools: []string{"Read", "view_image"}},
		{Tools: []string{"Bash"}, Argument: "command", Pattern: readOnly},
		{Tools: []string{"exec_command"}, Argument: "cmd", Pattern: readOnly},
	}
	one := 1.0
	source := func(match string) config.ConversationFeature {
		return config.ConversationFeature{Type: "count", Source: config.ConversationSource{Type: "last_tool_call", Calls: reads, Match: match}}
	}
	cfg := &config.RouterConfig{}
	cfg.ConversationRules = []config.ConversationRule{
		{Name: "last_turn_read", Feature: source(""), Predicate: &config.NumericPredicate{GTE: &one}},
		{Name: "last_turn_other", Feature: source("no_match"), Predicate: &config.NumericPredicate{GTE: &one}},
	}
	cfg.Strategy = "priority"
	cfg.Decisions = []config.Decision{{Name: "small", Priority: 100, Rules: config.RuleCombination{Operator: "AND", Conditions: []config.RuleNode{
		{Type: config.SignalTypeConversation, Name: "last_turn_read"},
		{Operator: "NOT", Conditions: []config.RuleNode{{Type: config.SignalTypeConversation, Name: "last_turn_other"}}},
	}}}}
	router := &OpenAIRouter{Config: cfg}
	classifier := &classification.Classifier{Config: cfg}

	messages := func(turns string) string {
		return `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"fix it"}` + turns + `]}`
	}
	use := func(id, name, input string) string {
		return `{"type":"tool_use","id":"` + id + `","name":"` + name + `","input":` + input + `}`
	}
	result := func(ids ...string) string {
		parts := make([]string, 0, len(ids))
		for _, id := range ids {
			parts = append(parts, `{"type":"tool_result","tool_use_id":"`+id+`","content":"ok"}`)
		}
		return `,{"role":"user","content":[` + strings.Join(parts, ",") + `]}`
	}
	assistant := func(calls ...string) string {
		return `,{"role":"assistant","content":[` + strings.Join(calls, ",") + `]}`
	}
	chatCall := `,{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"cat a.txt\"}"}}]},{"role":"tool","tool_call_id":"a","content":"ok"}`
	responses := func(items ...string) string {
		return `{"model":"m","input":[{"type":"message","role":"user","content":"fix it"},` + strings.Join(items, ",") + `]}`
	}
	call := func(id, cmd string) string {
		return `{"type":"function_call","call_id":"` + id + `","name":"exec_command","arguments":"{\"cmd\":\"` + cmd + `\"}"}`
	}
	output := func(id string) string { return `{"type":"function_call_output","call_id":"` + id + `","output":"ok"}` }
	for _, tc := range []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
		small  bool
	}{
		{"messages read", llmprotocol.AnthropicMessagesV1, messages(assistant(use("a", "Bash", `{"command":"ls -la"}`)) + result("a")), true},
		{"messages read after an earlier edit", llmprotocol.AnthropicMessagesV1, messages(assistant(use("e", "Edit", `{"file_path":"/a"}`)) + result("e") + assistant(use("a", "Read", `{"file_path":"/a"}`)) + result("a")), true},
		{"messages read and edit", llmprotocol.AnthropicMessagesV1, messages(assistant(use("a", "Read", `{"file_path":"/a"}`), use("b", "Edit", `{"file_path":"/a"}`)) + result("a", "b")), false},
		{"messages chained read", llmprotocol.AnthropicMessagesV1, messages(assistant(use("a", "Bash", `{"command":"ls && rm -rf build"}`)) + result("a")), false},
		{"messages compacted", llmprotocol.AnthropicMessagesV1, `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"Summary of the conversation so far."}]}`, false},
		{"chat read", llmprotocol.OpenAIChatV1, `{"model":"m","messages":[{"role":"user","content":"fix it"}` + chatCall + `]}`, true},
		{"chat answer without a call", llmprotocol.OpenAIChatV1, `{"model":"m","messages":[{"role":"user","content":"fix it"},{"role":"assistant","content":"done"},{"role":"user","content":"thanks"}]}`, false},
		{"responses two reads", llmprotocol.OpenAIResponsesV1, responses(call("a", "ls"), call("b", "cat y"), output("a"), output("b")), true},
		{"responses write then read", llmprotocol.OpenAIResponsesV1, responses(call("a", "rm x"), call("b", "ls"), output("a"), output("b")), false},
	} {
		request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(tc.format, []byte(tc.body))
		if err != nil {
			t.Fatalf("%s: decode: %v", tc.name, err)
		}
		input := router.prepareSignalEvaluationInput(signalConversationHistoryFromSnapshot(extractSemanticRequestSignals(&request)))
		signals := classifier.EvaluateAllSignalsWithContext(input.evaluationText, input.allMessagesText, input.currentUserText,
			nil, nil, false, true, "", nil, input.conversationFacts, "")
		decision, err := classifier.EvaluateDecisionWithEngine(signals)
		if err != nil {
			t.Fatalf("%s: decide: %v", tc.name, err)
		}
		if small := decision != nil && decision.Decision != nil && decision.Decision.Name == "small"; small != tc.small {
			t.Errorf("%s: routed to small = %v, want %v (matched %v)", tc.name, small, tc.small, signals.MatchedConversationRules)
		}
	}
}
