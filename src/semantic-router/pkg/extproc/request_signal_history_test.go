package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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

func TestExtractToolTransitionContextFromNeutralHistory(t *testing.T) {
	req := &llmprotocol.Request{Model: "entrypoint"}
	req.Messages = append(req.Messages,
		llmprotocol.Message{Role: llmprotocol.RoleUser, Content: textBlocks("Find deployment details.")},
		neutralAssistantToolCalls("search", "lookup", "summarize"),
		neutralToolResult("call-search", "search result"),
		neutralToolResult("call-lookup", "lookup result"),
		neutralToolResult("call-summarize", "summary result"),
		llmprotocol.Message{Role: llmprotocol.RoleUser, Content: textBlocks("Now write the result.")},
	)

	transition := extractToolTransitionContextFromRequest(req, 2, &RequestContext{
		VSRSelectedCategory: "coding",
		VSRSelectedDecision: &config.Decision{Name: "agent-tools"},
	})
	assert.Equal(t, []string{"lookup", "summarize"}, transition.RecentToolNames)
	assert.Equal(t, 2, transition.UserMessageCount)
	assert.Equal(t, 3, transition.ToolResultCount)
	assert.Equal(t, "agent-tools", transition.SelectedDecision)
	assert.Equal(t, "coding", transition.SelectedCategory)
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

func TestToolTransitionContextWindowAndCopy(t *testing.T) {
	history := signalConversationHistory{
		userMessageCount:   2,
		toolResultCount:    1,
		assistantToolNames: []string{"read_file", "list_dir", "run_tests"},
	}
	transition := toolTransitionContextFromConversationHistory(history, 2, &RequestContext{
		VSRSelectedDecisionName: "fallback-decision",
		VSRSelectedCategory:     "maintenance",
	})
	assert.Equal(t, []string{"list_dir", "run_tests"}, transition.RecentToolNames)
	assert.Equal(t, 2, transition.UserMessageCount)
	assert.Equal(t, 1, transition.ToolResultCount)
	assert.Equal(t, "fallback-decision", transition.SelectedDecision)
	assert.Equal(t, "maintenance", transition.SelectedCategory)
	history.assistantToolNames[1] = "mutated"
	assert.Equal(t, []string{"list_dir", "run_tests"}, transition.RecentToolNames)
}

func TestExtractToolTransitionContextNilRequest(t *testing.T) {
	transition := extractToolTransitionContextFromRequest(nil, 2, nil)
	assert.Nil(t, transition.RecentToolNames)
	assert.Zero(t, transition.UserMessageCount)
	assert.Zero(t, transition.ToolResultCount)
}

func TestSignalConversationHistoryFromSnapshotCopiesToolNames(t *testing.T) {
	snapshot := &requestSignalSnapshot{
		UserContent:            "Now run the tests.",
		UserMessageCount:       2,
		ToolMessageCount:       2,
		AssistantToolCallCount: 2,
		ToolResultCount:        2,
		AssistantToolNames:     []string{"read_file", "run_tests"},
		ContextTokenFloor:      16_384,
		ContextTextBytes:       128,
		ContextEquivalentBytes: 65_536,
		ContextHasNonText:      true,
	}
	history := signalConversationHistoryFromSnapshot(snapshot)
	transition := toolTransitionContextFromConversationHistory(history, 1, nil)
	assert.Equal(t, []string{"run_tests"}, transition.RecentToolNames)
	assert.Equal(t, 16_384, history.contextTokenFloor)
	assert.True(t, history.contextHasNonText)
	snapshot.AssistantToolNames[1] = "mutated"
	assert.Equal(t, []string{"run_tests"}, transition.RecentToolNames)
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
