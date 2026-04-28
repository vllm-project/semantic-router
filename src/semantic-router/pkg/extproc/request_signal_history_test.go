package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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
}

func TestExtractRecentAssistantToolCallNames(t *testing.T) {
	raw := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "c1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}
			]},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "c2", "type": "function", "function": {"name": "bar", "arguments": "{}"}},
				{"id": "c3", "type": "function", "function": {"name": "baz", "arguments": "{}"}}
			]}
		]
	}`)
	req, err := parseOpenAIRequest(raw)
	require.NoError(t, err)
	assert.Equal(t, []string{"foo", "bar", "baz"}, extractRecentAssistantToolCallNames(req, 10))
	assert.Equal(t, []string{"bar", "baz"}, extractRecentAssistantToolCallNames(req, 2))
}
