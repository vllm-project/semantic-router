package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routinginput"
)

func TestExtractSignalConversationHistoryMixedNeutralRoles(t *testing.T) {
	req := &llmprotocol.Request{
		Instructions: []llmprotocol.InstructionBlock{
			{Role: llmprotocol.RoleSystem, Content: textBlocks("System prompt")},
			{Role: llmprotocol.RoleDeveloper, Content: textBlocks("Developer policy")},
		},
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: textBlocks("first question")},
			{Role: llmprotocol.RoleAssistant, Content: textBlocks("first answer")},
			{Role: llmprotocol.RoleUser, Content: textBlocks("second", "question")},
		},
	}

	history := extractSignalConversationHistory(req)
	assert.Equal(t, "second question", history.currentUserMessage)
	assert.Equal(t, []string{"first question"}, history.priorUserMessages)
	assert.Equal(t, []string{"System prompt", "Developer policy", "first answer"}, history.nonUserMessages)
	assert.True(t, history.hasAssistantReply)
	assert.True(t, history.hasDeveloperMessage)
}

func TestExtractSignalConversationHistoryFlowToolStateRequiresTrailingResult(t *testing.T) {
	req := &llmprotocol.Request{Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleUser, Content: textBlocks("run the workflow")},
		neutralToolResult("flowtool_deadbeef__call_1", "done"),
	}}
	history := extractSignalConversationHistory(req)
	assert.True(t, history.lastMessageFlowToolResult)

	req.Messages = append(req.Messages, llmprotocol.Message{
		Role: llmprotocol.RoleUser, Content: textBlocks("new request"),
	})
	history = extractSignalConversationHistory(req)
	assert.False(t, history.lastMessageFlowToolResult)
}

func TestExtractSignalConversationHistoryToolChoiceFacts(t *testing.T) {
	req := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{
		Mode: llmprotocol.ToolChoiceRequired,
	}}
	history := extractSignalConversationHistory(req)
	assert.True(t, history.toolChoiceRequired)
	assert.False(t, history.toolChoiceNone)

	req.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNamed, Name: "lookup"}
	history = extractSignalConversationHistory(req)
	assert.True(t, history.toolChoiceRequired)
	assert.False(t, history.toolChoiceNone)

	req.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceNone}
	history = extractSignalConversationHistory(req)
	assert.False(t, history.toolChoiceRequired)
	assert.True(t, history.toolChoiceNone)
}

func TestExtractSignalConversationHistoryMarksUserAfterToolResult(t *testing.T) {
	req := &llmprotocol.Request{Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleUser, Content: textBlocks("Read the log.")},
		neutralAssistantToolCalls("read_log"),
		neutralToolResult("call-read_log", "TypeError: int + str"),
		{Role: llmprotocol.RoleUser, Content: textBlocks("Now give the fix.")},
	}}
	history := extractSignalConversationHistory(req)
	assert.Equal(t, "user", history.lastMessageRole)
	assert.False(t, history.lastMessageToolResult)
	assert.True(t, history.lastUserAfterToolResult)
}

func TestExtractSignalConversationHistoryBoundsPendingToolCallToTail(t *testing.T) {
	req := &llmprotocol.Request{Messages: []llmprotocol.Message{
		neutralAssistantToolCalls("lookup"),
		{Role: llmprotocol.RoleUser, Content: textBlocks("start a separate task")},
		{Role: llmprotocol.RoleAssistant, Content: textBlocks("done")},
		{Role: llmprotocol.RoleUser, Content: textBlocks("ordinary follow-up")},
	}}

	history := extractSignalConversationHistory(req)

	assert.Equal(t, 1, history.assistantToolCallCount)
	assert.Zero(t, history.toolResultCount)
	assert.False(t, history.lastAssistantToolCall)
	assert.False(t, conversationFactsIndicateActiveToolLoop(classification.ConversationFacts{
		AssistantToolCallCount: history.assistantToolCallCount,
		ToolResultCount:        history.toolResultCount,
		LastMessageRole:        history.lastMessageRole,
		LastAssistantToolCall:  history.lastAssistantToolCall,
	}))
}

func TestSignalConversationHistoryFromSnapshot(t *testing.T) {
	snapshot := &requestSignalSnapshot{
		UserContent:            "Now run the tests.",
		UserMessageCount:       2,
		ToolMessageCount:       2,
		AssistantToolCallCount: 2,
		ToolResultCount:        2,
		ContextTokenFloor:      16_384,
		ContextTextBytes:       128,
		ContextEquivalentBytes: 65_536,
		ContextHasNonText:      true,
	}
	history := signalConversationHistoryFromSnapshot(snapshot)
	assert.Equal(t, 16_384, history.contextTokenFloor)
	assert.True(t, history.contextHasNonText)
}

func textBlocks(values ...string) []llmprotocol.Content {
	blocks := make([]llmprotocol.Content, len(values))
	for index, value := range values {
		blocks[index] = llmprotocol.Content{Kind: llmprotocol.ContentText, Text: value}
	}
	return blocks
}

func neutralAssistantToolCalls(names ...string) llmprotocol.Message {
	message := llmprotocol.Message{Role: llmprotocol.RoleAssistant}
	for _, name := range names {
		message.Content = append(message.Content, llmprotocol.Content{
			Kind:     llmprotocol.ContentToolCall,
			ToolCall: &llmprotocol.ToolCall{ID: "call-" + name, Name: name, Arguments: `{}`},
		})
	}
	return message
}

func neutralToolResult(callID, output string) llmprotocol.Message {
	return llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
		Kind:       llmprotocol.ContentToolResult,
		ToolResult: &llmprotocol.ToolResult{CallID: callID, Content: textBlocks(output)},
	}}}
}

func extractSignalConversationHistory(req *llmprotocol.Request) signalConversationHistory {
	return extractSignalConversationHistoryWith(req, nil)
}

func extractSignalConversationHistoryWith(req *llmprotocol.Request, selector *routinginput.Selector) signalConversationHistory {
	return signalConversationHistoryFromRequest(req, extractSemanticRequestSignalsWith(req, selector))
}
