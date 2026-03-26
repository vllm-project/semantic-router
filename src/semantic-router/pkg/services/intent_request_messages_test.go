package services

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func mustMessageContent(t *testing.T, value interface{}) json.RawMessage {
	t.Helper()
	data, err := json.Marshal(value)
	require.NoError(t, err)
	return data
}

func TestIntentRequestResolveSignalInput_UsesMessagesConversationHistory(t *testing.T) {
	req := IntentRequest{
		Messages: []IntentMessage{
			{
				Role:    "system",
				Content: mustMessageContent(t, "You are a careful tutor."),
			},
			{
				Role:    "user",
				Content: mustMessageContent(t, "Explain inflation vs recession in plain English."),
			},
			{
				Role:    "assistant",
				Content: mustMessageContent(t, "Inflation means prices rise over time."),
			},
			{
				Role: "user",
				Content: mustMessageContent(t, []map[string]string{
					{"type": "text", "text": "That was not clear."},
					{"type": "text", "text": "Explain inflation vs recession in plain English."},
				}),
			},
		},
	}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, "That was not clear. Explain inflation vs recession in plain English.", input.evaluationText)
	assert.Equal(t, input.evaluationText, input.currentUserText)
	assert.Equal(t, []string{"Explain inflation vs recession in plain English."}, input.priorUserMessages)
	assert.Equal(t, []string{"You are a careful tutor.", "Inflation means prices rise over time."}, input.nonUserMessages)
	assert.Equal(
		t,
		"You are a careful tutor. Inflation means prices rise over time. That was not clear. Explain inflation vs recession in plain English.",
		input.contextText,
	)
}

func TestIntentRequestResolveSignalInput_FallsBackToText(t *testing.T) {
	req := IntentRequest{Text: "Fallback single-turn request"}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, "Fallback single-turn request", input.evaluationText)
	assert.Equal(t, "Fallback single-turn request", input.contextText)
	assert.Equal(t, "Fallback single-turn request", input.currentUserText)
	assert.Empty(t, input.priorUserMessages)
	assert.Empty(t, input.nonUserMessages)
}

func TestClassificationServiceClassifyIntentForEval_AcceptsMessagesWithoutText(t *testing.T) {
	service := &ClassificationService{classifier: nil}
	req := IntentRequest{
		Messages: []IntentMessage{
			{
				Role:    "user",
				Content: mustMessageContent(t, "Explain compound interest in one paragraph."),
			},
			{
				Role:    "assistant",
				Content: mustMessageContent(t, "Compound interest is interest on interest."),
			},
			{
				Role:    "user",
				Content: mustMessageContent(t, "That was not clear. Explain compound interest in one paragraph."),
			},
		},
	}

	resp, err := service.ClassifyIntentForEval(req)
	require.NoError(t, err)
	require.NotNil(t, resp)

	assert.Equal(t, "That was not clear. Explain compound interest in one paragraph.", resp.OriginalText)
	assert.NotNil(t, resp.Metrics)
}
