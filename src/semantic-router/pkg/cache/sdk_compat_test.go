package cache

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These tests verify that the SDK-based ExtractQueryFromOpenAIRequest produces
// identical results to the removed custom-type implementation across all
// content variants (string, multimodal array, multi-turn, tools, streaming).

func TestSDKCompat_StringContent_ParsesIdentically(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"What is 2+2?"}]}`)
	model, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4", model)
	assert.Equal(t, "What is 2+2?", query)
}

func TestSDKCompat_MultimodalArray_ExtractsTextOnly(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4o",
		"messages":[{
			"role":"user",
			"content":[
				{"type":"text","text":"Describe this image"},
				{"type":"image_url","image_url":{"url":"https://example.com/img.png"}}
			]
		}]
	}`)
	model, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4o", model)
	assert.Equal(t, "Describe this image", query)
}

func TestSDKCompat_MultiTurnConversation(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4",
		"messages":[
			{"role":"system","content":"You are helpful."},
			{"role":"user","content":"First question"},
			{"role":"assistant","content":"First answer"},
			{"role":"user","content":"Follow-up question"}
		]
	}`)
	_, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "Follow-up question", query, "should extract last user message")
}

func TestSDKCompat_StreamingRequest(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4",
		"messages":[{"role":"user","content":"Hello"}],
		"stream":true
	}`)
	model, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4", model)
	assert.Equal(t, "Hello", query)
}

func TestSDKCompat_WithToolsAndParameters(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4",
		"messages":[{"role":"user","content":"What is the weather?"}],
		"temperature":0.7,
		"max_tokens":100,
		"tools":[{"type":"function","function":{"name":"get_weather"}}]
	}`)
	model, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4", model)
	assert.Equal(t, "What is the weather?", query)
}

func TestSDKCompat_EmptyMessages(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[]}`)
	model, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4", model)
	assert.Empty(t, query)
}

func TestSDKCompat_NoUserMessages(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4",
		"messages":[
			{"role":"system","content":"You are helpful."},
			{"role":"assistant","content":"Ready to help."}
		]
	}`)
	_, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Empty(t, query)
}

func TestSDKCompat_ExtractUserContent_DirectSDKConstruction(t *testing.T) {
	content := openai.ChatCompletionUserMessageParamContentUnion{
		OfString: openai.String("direct SDK content"),
	}
	assert.Equal(t, "direct SDK content", extractUserContent(content))
}

func TestSDKCompat_RoundTrip_MarshalUnmarshal(t *testing.T) {
	params := openai.ChatCompletionNewParams{
		Model: "gpt-4",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("round-trip test"),
					},
				},
			},
		},
	}
	body, err := json.Marshal(params)
	require.NoError(t, err)

	model, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4", model)
	assert.Equal(t, "round-trip test", query)
}
