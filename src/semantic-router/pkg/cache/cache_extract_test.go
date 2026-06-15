package cache

import (
	"strings"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// ExtractQueryFromOpenAIRequest — cache key extraction
// ---------------------------------------------------------------------------
//
// The semantic cache embeds the *query* text returned by this function. It must
// always be the original, uncompressed user prompt. These tests lock that
// invariant and cover multimodal content, multi-turn conversations, and
// payloads large enough to trigger prompt compression upstream.
// ---------------------------------------------------------------------------

func TestExtractQuery_SimpleTextMessage(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"What is 2+2?"}]}`)
	model, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4", model)
	assert.Equal(t, "What is 2+2?", query)
}

func TestExtractQuery_MultiTurn_ExtractsLastUserMessage(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4",
		"messages":[
			{"role":"user","content":"Hello"},
			{"role":"assistant","content":"Hi there!"},
			{"role":"user","content":"What is the capital of France?"}
		]
	}`)
	_, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "What is the capital of France?", query)
}

func TestExtractQuery_MultimodalTextAndImage_ExtractsOnlyText(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4o",
		"messages":[{
			"role":"user",
			"content":[
				{"type":"text","text":"Describe this image"},
				{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="}}
			]
		}]
	}`)
	model, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4o", model)
	assert.Equal(t, "Describe this image", query, "query must be the text part only, no base64 image data")
	assert.NotContains(t, query, "base64", "image data must not leak into cache key")
}

func TestExtractQuery_MultimodalMultipleTextParts(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4o",
		"messages":[{
			"role":"user",
			"content":[
				{"type":"text","text":"First part. "},
				{"type":"image_url","image_url":{"url":"https://example.com/img.png"}},
				{"type":"text","text":"Second part."}
			]
		}]
	}`)
	_, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "First part. Second part.", query, "all text parts concatenated, no image URLs")
}

func TestExtractQuery_MultimodalOnlyImage_EmptyQuery(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4o",
		"messages":[{
			"role":"user",
			"content":[
				{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,/9j/4AAQSk..."}}
			]
		}]
	}`)
	_, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Empty(t, query, "image-only messages should produce empty query")
}

func TestExtractQuery_SystemAndAssistantMessages_Ignored(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4",
		"messages":[
			{"role":"system","content":"You are a helpful assistant."},
			{"role":"assistant","content":"How can I help?"},
			{"role":"user","content":"Tell me a joke"}
		]
	}`)
	_, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "Tell me a joke", query, "only user messages contribute to query")
}

func TestExtractQuery_NoUserMessages_EmptyQuery(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4",
		"messages":[
			{"role":"system","content":"You are a helpful assistant."}
		]
	}`)
	_, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Empty(t, query)
}

func TestExtractQuery_EmptyContent_Skipped(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4",
		"messages":[
			{"role":"user","content":""},
			{"role":"user","content":"Real question"}
		]
	}`)
	_, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "Real question", query)
}

func TestExtractQuery_LargeText_PreservesFullOriginal(t *testing.T) {
	// Build a prompt large enough that upstream prompt compression would
	// compress it (>512 tokens). The cache must still store the FULL original.
	var sb strings.Builder
	for i := range 200 {
		sb.WriteString("This is sentence number ")
		sb.WriteString(strings.Repeat("x", 10))
		sb.WriteString(". ")
		_ = i
	}
	longPrompt := sb.String()

	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"` + longPrompt + `"}]}`)
	_, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Equal(t, longPrompt, query, "cache key must be the full original prompt, not a compressed version")
	assert.Greater(t, len(query), 2000, "sanity: prompt should be large")
}

func TestExtractQuery_InvalidJSON_ReturnsError(t *testing.T) {
	_, _, err := ExtractQueryFromOpenAIRequest([]byte(`not json`))
	require.Error(t, err)
}

func TestExtractQuery_MissingMessages_EmptyQuery(t *testing.T) {
	body := []byte(`{"model":"gpt-4"}`)
	model, query, err := ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4", model)
	assert.Empty(t, query)
}

// ---------------------------------------------------------------------------
// extractUserContent — SDK-based multimodal content parsing
// ---------------------------------------------------------------------------

func TestExtractUserContent_PlainString(t *testing.T) {
	content := openai.ChatCompletionUserMessageParamContentUnion{
		OfString: openai.String("Hello world"),
	}
	assert.Equal(t, "Hello world", extractUserContent(content))
}

func TestExtractUserContent_ContentArray_TextParts(t *testing.T) {
	content := openai.ChatCompletionUserMessageParamContentUnion{
		OfArrayOfContentParts: []openai.ChatCompletionContentPartUnionParam{
			{OfText: &openai.ChatCompletionContentPartTextParam{Text: "Part A"}},
			{OfText: &openai.ChatCompletionContentPartTextParam{Text: "Part B"}},
		},
	}
	assert.Equal(t, "Part APart B", extractUserContent(content))
}

func TestExtractUserContent_ContentArray_MixedParts(t *testing.T) {
	content := openai.ChatCompletionUserMessageParamContentUnion{
		OfArrayOfContentParts: []openai.ChatCompletionContentPartUnionParam{
			{OfText: &openai.ChatCompletionContentPartTextParam{Text: "Describe: "}},
			{OfImageURL: &openai.ChatCompletionContentPartImageParam{
				ImageURL: openai.ChatCompletionContentPartImageImageURLParam{URL: "https://example.com/img.png"},
			}},
		},
	}
	assert.Equal(t, "Describe: ", extractUserContent(content), "only text parts extracted, image_url ignored")
}

func TestExtractUserContent_ContentArray_OnlyImage(t *testing.T) {
	content := openai.ChatCompletionUserMessageParamContentUnion{
		OfArrayOfContentParts: []openai.ChatCompletionContentPartUnionParam{
			{OfImageURL: &openai.ChatCompletionContentPartImageParam{
				ImageURL: openai.ChatCompletionContentPartImageImageURLParam{URL: "data:image/png;base64,AAAA"},
			}},
		},
	}
	assert.Empty(t, extractUserContent(content))
}

func TestExtractUserContent_EmptyContent(t *testing.T) {
	content := openai.ChatCompletionUserMessageParamContentUnion{}
	assert.Empty(t, extractUserContent(content))
}
