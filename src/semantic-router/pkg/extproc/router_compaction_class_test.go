package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/codexreview"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

func textMessage(role llmprotocol.Role, text string) llmprotocol.Message {
	return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}}
}

var testTools = []llmprotocol.Tool{{Name: "exec_command"}}

// Codex compacts with a tool-less call whose last user message is its
// checkpoint prompt, and names its title thread; every other call is main or
// a subagent's.
func TestCodexRequestClass(t *testing.T) {
	compaction := &llmprotocol.Request{Messages: []llmprotocol.Message{
		textMessage(llmprotocol.RoleUser, "Run the tests."),
		textMessage(llmprotocol.RoleAssistant, "Done."),
		textMessage(llmprotocol.RoleUser, "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM."),
	}}
	withTools := *compaction
	withTools.Tools = testTools
	quoted := &llmprotocol.Request{Messages: []llmprotocol.Message{
		textMessage(llmprotocol.RoleUser, "You are performing a CONTEXT CHECKPOINT COMPACTION."),
		textMessage(llmprotocol.RoleAssistant, "Summary."),
	}}
	for name, test := range map[string]struct {
		ctx      *RequestContext
		subagent bool
		class    string
	}{
		"its checkpoint":             {ctx: &RequestContext{SemanticRequest: compaction}, class: config.AgentRequestClassCompaction},
		"a subagent's checkpoint":    {ctx: &RequestContext{SemanticRequest: compaction}, subagent: true, class: config.AgentRequestClassCompaction},
		"the prompt with tools":      {ctx: &RequestContext{SemanticRequest: &withTools}, class: config.AgentRequestClassMain},
		"the prompt not last":        {ctx: &RequestContext{SemanticRequest: quoted}, class: config.AgentRequestClassMain},
		"its title thread":           {ctx: &RequestContext{AgentTurn: codexreview.Turn{ThreadSource: codexThreadSourceTitle}}, class: config.AgentRequestClassAuxiliary},
		"a subagent's ordinary turn": {ctx: &RequestContext{SemanticRequest: &withTools}, subagent: true, class: config.AgentRequestClassSubagent},
	} {
		t.Run(name, func(t *testing.T) {
			require.Equal(t, test.class, codexRequestClass(test.ctx, test.subagent))
		})
	}
}

// OpenClaw compacts with a tool-less call whose system prompt is its summary
// instructions.
func TestOpenClawRequestClass(t *testing.T) {
	summary := &llmprotocol.Request{Messages: []llmprotocol.Message{
		textMessage(llmprotocol.RoleSystem, "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant."),
		textMessage(llmprotocol.RoleUser, "The conversation."),
	}}
	turn := &llmprotocol.Request{Tools: testTools, Messages: []llmprotocol.Message{
		textMessage(llmprotocol.RoleSystem, openClawTestSystem),
		textMessage(llmprotocol.RoleUser, "Run the tests."),
	}}
	require.Equal(t, config.AgentRequestClassCompaction, openClawRequestClass(summary, false))
	require.Equal(t, config.AgentRequestClassMain, openClawRequestClass(turn, false))
	require.Equal(t, config.AgentRequestClassSubagent, openClawRequestClass(turn, true))
}

// A Codex checkpoint reaches the agent signal as a compaction, and its Replay
// record and run call are marked so.
func TestCodexCompactionIsRecordedAsACompaction(t *testing.T) {
	router := memoryReplayRouter(t)
	codex := map[string]string{headers.CodexSessionID: "cx-run", headers.CodexThreadID: "cx-run", headers.UserAgent: "codex_exec/0.170.0"}
	body := `{"model":"vllm-sr/auto","max_tokens":8,"messages":[{"role":"user","content":"Run the tests."},` +
		`{"role":"assistant","content":"Done."},` +
		`{"role":"user","content":"You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task."}]}`
	record := recordReplayCall(t, router, codex, llmprotocol.OpenAIChatV1, body)

	require.Equal(t, config.AgentRequestClassCompaction, record.Agent.RequestClass)
	call := buildReplayRunCall(record)
	require.Contains(t, call.Markers, routerreplay.RunMarkerCompaction)
	require.Equal(t, config.AgentRequestClassCompaction,
		agentFacts(&RequestContext{Headers: codex, SemanticRequest: &llmprotocol.Request{Messages: []llmprotocol.Message{
			textMessage(llmprotocol.RoleUser, "You are performing a CONTEXT CHECKPOINT COMPACTION."),
		}}}, config.RouterLearningProtectionConfig{}).RequestClass)
}

func TestAgentRulesTakeTheInferredClasses(t *testing.T) {
	for _, harness := range []string{config.AgentHarnessCodex, config.AgentHarnessOpenClaw} {
		require.NoError(t, config.ValidateAgentRuleContract(config.AgentRule{
			Name: "x", Harness: harness, RequestClasses: []string{config.AgentRequestClassCompaction},
		}))
	}
}
