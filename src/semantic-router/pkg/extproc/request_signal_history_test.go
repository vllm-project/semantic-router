package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func TestExtractSignalConversationHistory_ChatCompletionsMixedRoles(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage([]openai.ChatCompletionContentPartTextParam{
				{Text: "System prompt"},
			}),
			openai.UserMessage("first question"),
			openai.AssistantMessage("first answer"),
			openai.ToolMessage("tool output", "tool-call-id"),
			openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
				openai.TextContentPart("second"),
				openai.TextContentPart("question"),
			}),
		},
	}

	history := extractSignalConversationHistory(req)

	assert.Equal(t, "second question", history.currentUserMessage)
	assert.Equal(t, []string{"first question"}, history.priorUserMessages)
	assert.Equal(t, []string{"System prompt", "first answer"}, history.nonUserMessages)
	assert.True(t, history.hasAssistantReply)
}

func TestExtractToolTransitionContextFromRequest_ChatCompletionsNoToolCalls(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("hello"),
		},
	}

	transition := extractToolTransitionContextFromRequest(req, 2, nil)

	assert.Nil(t, transition.RecentToolNames)
	assert.Equal(t, 1, transition.UserMessageCount)
	assert.Equal(t, 0, transition.ToolResultCount)
	assert.Empty(t, transition.SelectedDecision)
	assert.Empty(t, transition.SelectedCategory)
}

func TestExtractToolTransitionContextFromRequest_ChatCompletionsToolHistory(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Find deployment details."),
			assistantToolCallMessage("search", "lookup", "summarize"),
			openai.ToolMessage("search result", "call-search"),
			openai.ToolMessage("lookup result", "call-lookup"),
			openai.ToolMessage("summary result", "call-summarize"),
			openai.UserMessage("Now write the result."),
		},
	}

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

func TestToolTransitionContextFromConversationHistoryPreservesAllToolsWhenWindowUnset(t *testing.T) {
	history := signalConversationHistory{
		userMessageCount:   2,
		toolResultCount:    1,
		assistantToolNames: []string{"read_file", "list_dir", "run_tests"},
	}

	transition := toolTransitionContextFromConversationHistory(history, 0, &RequestContext{
		VSRSelectedDecisionName: "fallback-decision",
		VSRSelectedCategory:     "maintenance",
	})

	assert.Equal(t, []string{"read_file", "list_dir", "run_tests"}, transition.RecentToolNames)
	assert.Equal(t, 2, transition.UserMessageCount)
	assert.Equal(t, 1, transition.ToolResultCount)
	assert.Equal(t, "fallback-decision", transition.SelectedDecision)
	assert.Equal(t, "maintenance", transition.SelectedCategory)
}

func TestToolTransitionContextFromConversationHistoryPreservesAllToolsWhenWindowNegative(t *testing.T) {
	history := signalConversationHistory{
		assistantToolNames: []string{"read_file", "list_dir", "run_tests"},
	}

	transition := toolTransitionContextFromConversationHistory(history, -1, nil)

	assert.Equal(t, []string{"read_file", "list_dir", "run_tests"}, transition.RecentToolNames)
}

func TestToolTransitionContextFromConversationHistoryCopiesRecentTools(t *testing.T) {
	history := signalConversationHistory{
		assistantToolNames: []string{"a", "b", "c", "d"},
	}

	transition := toolTransitionContextFromConversationHistory(history, 2, nil)

	assert.Equal(t, []string{"c", "d"}, transition.RecentToolNames)
	history.assistantToolNames[2] = "mutated"
	assert.Equal(t, []string{"c", "d"}, transition.RecentToolNames)
}

func TestExtractToolTransitionContextFromRequest_NilRequestAndContext(t *testing.T) {
	transition := extractToolTransitionContextFromRequest(nil, 2, nil)

	assert.Nil(t, transition.RecentToolNames)
	assert.Equal(t, 0, transition.UserMessageCount)
	assert.Equal(t, 0, transition.ToolResultCount)
	assert.Empty(t, transition.SelectedDecision)
	assert.Empty(t, transition.SelectedCategory)
}

func TestExtractToolTransitionContextFromRequestCountsCompletedToolResultsOnly(t *testing.T) {
	req := &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Fetch several facts."),
			assistantToolCallMessage("search", "lookup", "summarize"),
			openai.ToolMessage("search result", "call-search"),
		},
	}

	transition := extractToolTransitionContextFromRequest(req, 0, nil)

	assert.Equal(t, []string{"search", "lookup", "summarize"}, transition.RecentToolNames)
	assert.Equal(t, 1, transition.UserMessageCount)
	assert.Equal(t, 1, transition.ToolResultCount)
}

func TestSignalConversationHistoryFromFastExtract_PreservesResponseAPIUserChain(t *testing.T) {
	store := NewMockResponseStore()
	store.responses["resp_previous123"] = &responseapi.StoredResponse{
		ID:           "resp_previous123",
		Model:        "gpt-4",
		Status:       responseapi.StatusCompleted,
		Instructions: "Remember my name is Alice.",
		Input: []responseapi.InputItem{{
			Type:    responseapi.ItemTypeMessage,
			Role:    responseapi.RoleUser,
			Content: json.RawMessage(`"Hello"`),
		}},
		Output: []responseapi.OutputItem{{
			Type:   responseapi.ItemTypeMessage,
			Role:   responseapi.RoleAssistant,
			Status: responseapi.StatusCompleted,
			Content: []responseapi.ContentPart{{
				Type: responseapi.ContentTypeOutputText,
				Text: "Hi there!",
			}},
		}},
	}

	filter := NewResponseAPIFilter(store)
	requestBody := []byte(`{
		"model": "gpt-4",
		"input": "What is my name again?",
		"previous_response_id": "resp_previous123"
	}`)

	respCtx, translatedBody, err := filter.TranslateRequest(context.Background(), requestBody)
	require.NoError(t, err)
	require.NotNil(t, respCtx)

	fast, err := extractContentFast(translatedBody)
	require.NoError(t, err)

	history := signalConversationHistoryFromFastExtract(fast)

	assert.Equal(t, "What is my name again?", history.currentUserMessage)
	assert.Equal(t, []string{"Hello"}, history.priorUserMessages)
	assert.Equal(t, []string{"Remember my name is Alice.", "Hi there!"}, history.nonUserMessages)
	assert.True(t, history.hasAssistantReply)
}

func TestSignalConversationHistoryFromFastExtract_ResponseAPITranslationPreservesToolNames(t *testing.T) {
	store := NewMockResponseStore()
	store.responses["resp_with_tool"] = &responseapi.StoredResponse{
		ID:     "resp_with_tool",
		Model:  "gpt-4",
		Status: responseapi.StatusCompleted,
		Input: []responseapi.InputItem{{
			Type:    responseapi.ItemTypeMessage,
			Role:    responseapi.RoleUser,
			Content: json.RawMessage(`"Search the docs."`),
		}},
		Output: []responseapi.OutputItem{{
			Type:      responseapi.ItemTypeFunctionCall,
			CallID:    "call_search",
			Name:      "search_docs",
			Arguments: `{"query":"router"}`,
			Status:    responseapi.StatusCompleted,
		}, {
			Type:   responseapi.ItemTypeFunctionCallOutput,
			CallID: "call_search",
			Output: `{"result":"found"}`,
			Status: responseapi.StatusCompleted,
		}},
	}

	filter := NewResponseAPIFilter(store)
	requestBody := []byte(`{
		"model": "gpt-4",
		"input": "Summarize the result.",
		"previous_response_id": "resp_with_tool"
	}`)

	respCtx, translatedBody, err := filter.TranslateRequest(context.Background(), requestBody)
	require.NoError(t, err)
	require.NotNil(t, respCtx)

	fast, err := extractContentFast(translatedBody)
	require.NoError(t, err)

	history := signalConversationHistoryFromFastExtract(fast)
	transition := toolTransitionContextFromConversationHistory(history, 1, nil)

	assert.Equal(t, []string{"search_docs"}, fast.AssistantToolNames)
	assert.Equal(t, []string{"search_docs"}, history.assistantToolNames)
	assert.Equal(t, []string{"search_docs"}, transition.RecentToolNames)
	assert.Equal(t, 2, transition.UserMessageCount)
	assert.Equal(t, 1, transition.ToolResultCount)
}

func TestSignalConversationHistoryFromFastExtract_PreservesToolTransitionNames(t *testing.T) {
	fast := &FastExtractResult{
		UserContent:            "Now run the tests.",
		UserMessageCount:       2,
		ToolMessageCount:       2,
		AssistantToolCallCount: 2,
		ToolResultCount:        2,
		AssistantToolNames:     []string{"read_file", "run_tests"},
	}

	history := signalConversationHistoryFromFastExtract(fast)
	transition := toolTransitionContextFromConversationHistory(history, 1, nil)

	assert.Equal(t, []string{"run_tests"}, transition.RecentToolNames)
	assert.Equal(t, 2, transition.UserMessageCount)
	assert.Equal(t, 2, transition.ToolResultCount)

	fast.AssistantToolNames[1] = "mutated"
	assert.Equal(t, []string{"run_tests"}, transition.RecentToolNames)
}

func assistantToolCallMessage(names ...string) openai.ChatCompletionMessageParamUnion {
	toolCalls := make([]openai.ChatCompletionMessageToolCallParam, 0, len(names))
	for _, name := range names {
		toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallParam{
			ID: "call-" + name,
			Function: openai.ChatCompletionMessageToolCallFunctionParam{
				Name:      name,
				Arguments: "{}",
			},
		})
	}
	return openai.ChatCompletionMessageParamUnion{
		OfAssistant: &openai.ChatCompletionAssistantMessageParam{
			ToolCalls: toolCalls,
		},
	}
}
