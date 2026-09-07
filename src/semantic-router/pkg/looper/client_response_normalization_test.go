package looper

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseNonStreamingResponseKeepsReasoningSeparateFromContent(t *testing.T) {
	tests := []struct {
		name              string
		message           map[string]interface{}
		expectedContent   string
		expectedReasoning string
	}{
		{
			name: "reasoning content",
			message: map[string]interface{}{
				"role":              "assistant",
				"content":           "",
				"reasoning_content": "reasoned answer",
			},
			expectedReasoning: "reasoned answer",
		},
		{
			name: "reasoning",
			message: map[string]interface{}{
				"role":      "assistant",
				"content":   nil,
				"reasoning": "alternate answer",
			},
			expectedReasoning: "alternate answer",
		},
		{
			name: "content remains authoritative",
			message: map[string]interface{}{
				"role":              "assistant",
				"content":           "final answer",
				"reasoning_content": "supporting analysis",
			},
			expectedContent:   "final answer",
			expectedReasoning: "supporting analysis",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body, err := json.Marshal(map[string]interface{}{
				"id":     "chatcmpl-generic",
				"object": "chat.completion",
				"model":  "model-a",
				"choices": []map[string]interface{}{{
					"index":         0,
					"message":       test.message,
					"finish_reason": "stop",
				}},
			})
			require.NoError(t, err)

			response, err := (&Client{}).parseNonStreamingResponse(body, "model-a")
			require.NoError(t, err)
			assert.Equal(t, test.expectedContent, response.Content)
			assert.Equal(t, test.expectedReasoning, response.ReasoningContent)
		})
	}
}

func TestParseStreamingResponseCollectsReasoningFields(t *testing.T) {
	body := []byte(
		"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"reasoned \"}}]}\r\n" +
			"data: {\"choices\":[{\"delta\":{\"reasoning\":\"answer\"}}]}\r\n" +
			"data: [DONE]\r\n",
	)

	response, err := (&Client{}).parseStreamingResponse(body, "model-a")
	require.NoError(t, err)
	assert.Equal(t, "reasoned answer", response.ReasoningContent)
	assert.Empty(t, response.Content)
	assert.True(t, response.IsStreaming)
}

func TestParseStreamingResponsePrefersExplicitContent(t *testing.T) {
	body := []byte(
		"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"analysis\"}}]}\n" +
			"data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n" +
			"data: [DONE]\n",
	)

	response, err := (&Client{}).parseStreamingResponse(body, "model-a")
	require.NoError(t, err)
	assert.Equal(t, "analysis", response.ReasoningContent)
	assert.Equal(t, "answer", response.Content)
}
