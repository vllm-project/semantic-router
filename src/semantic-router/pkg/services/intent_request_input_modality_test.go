package services

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

func TestIntentRequestResolveSignalInput_CollectsInputModalityFacts(t *testing.T) {
	req := IntentRequest{
		Messages: []IntentMessage{
			{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "text", "text": "what do you hear and see?"},
					{"type": "image_url", "image_url": map[string]string{"url": "https://example.com/a.png"}},
					{"type": "input_audio", "input_audio": map[string]string{"data": "aGVsbG8=", "format": "wav"}},
					{"type": "video_url", "video_url": map[string]string{"url": "https://example.com/a.mp4"}},
				}),
			},
		},
	}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, classification.InputModalityFacts{
		TextContentCount:  1,
		ImageContentCount: 1,
		AudioContentCount: 1,
		VideoContentCount: 1,
	}, input.requestFacts.InputModality)
}

func TestIntentRequestResolveSignalInput_TextOnlyFallbackSetsTextFact(t *testing.T) {
	req := IntentRequest{Text: "just plain text"}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, classification.InputModalityFacts{TextContentCount: 1}, input.requestFacts.InputModality)
}

func TestIntentRequestResolveSignalInput_PromotedSystemTextIsNotUserText(t *testing.T) {
	req := IntentRequest{
		Messages: []IntentMessage{
			{
				Role:    "system",
				Content: mustMessageContent(t, "You are helpful."),
			},
			{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "input_audio", "input_audio": map[string]string{"data": "aGVsbG8=", "format": "wav"}},
				}),
			},
		},
	}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, classification.InputModalityFacts{AudioContentCount: 1}, input.requestFacts.InputModality,
		"system text promoted into evaluation text must not count as user text input")
}

func TestIntentRequestResolveSignalInput_TopLevelTextWithImageOnlyMessagesCounts(t *testing.T) {
	req := IntentRequest{
		Text: "what is in this picture?",
		Messages: []IntentMessage{
			{
				Role: "user",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64,aGVsbG8="}},
				}),
			},
		},
	}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, classification.InputModalityFacts{
		TextContentCount:  1,
		ImageContentCount: 1,
	}, input.requestFacts.InputModality)
}

func TestIntentRequestResolveSignalInput_NonUserMediaDoesNotCount(t *testing.T) {
	req := IntentRequest{
		Messages: []IntentMessage{
			{
				Role: "assistant",
				Content: mustMessageContent(t, []map[string]interface{}{
					{"type": "image_url", "image_url": map[string]string{"url": "https://example.com/a.png"}},
				}),
			},
			{
				Role:    "user",
				Content: mustMessageContent(t, "describe the picture you sent"),
			},
		},
	}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, classification.InputModalityFacts{TextContentCount: 1}, input.requestFacts.InputModality)
}
