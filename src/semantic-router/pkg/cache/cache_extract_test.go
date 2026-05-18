package cache

import (
	"encoding/json"
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

// ---------------------------------------------------------------------------
// BuildContextAwareCacheQuery — context-aware multi-turn cache keys
// ---------------------------------------------------------------------------
//
// When contextWindowTurns == 0 the function must behave identically to
// ExtractQueryFromOpenAIRequest (only the last user message).
// When contextWindowTurns > 0 the returned contextQuery must encode the
// conversation history so that the same last-user message in different
// conversation contexts produces a distinct embedding key.
// ---------------------------------------------------------------------------

func TestBuildContextAwareCacheQuery_DisabledReturnsSameAsExtract(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4",
		"messages":[
			{"role":"user","content":"Hello"},
			{"role":"assistant","content":"Hi!"},
			{"role":"user","content":"What is the capital of France?"}
		]
	}`)
	model, lastUser, contextQuery, err := BuildContextAwareCacheQuery(body, 0)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4", model)
	assert.Equal(t, "What is the capital of France?", lastUser)
	// With turns=0, contextQuery == lastUserQuery (no context prepended).
	assert.Equal(t, lastUser, contextQuery)
}

func TestBuildContextAwareCacheQuery_SingleTurnNoHistory(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"What is 2+2?"}]}`)
	_, lastUser, contextQuery, err := BuildContextAwareCacheQuery(body, 3)
	require.NoError(t, err)
	assert.Equal(t, "What is 2+2?", lastUser)
	// No prior turns or system prompt – key must still contain the current message.
	assert.Contains(t, contextQuery, "What is 2+2?")
	assert.Contains(t, contextQuery, "[user]:")
}

func TestBuildContextAwareCacheQuery_WithSystemPromptIncluded(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4",
		"messages":[
			{"role":"system","content":"You are a medical assistant."},
			{"role":"user","content":"What are the symptoms of flu?"}
		]
	}`)
	_, lastUser, contextQuery, err := BuildContextAwareCacheQuery(body, 2)
	require.NoError(t, err)
	assert.Equal(t, "What are the symptoms of flu?", lastUser)
	assert.Contains(t, contextQuery, "[system]: You are a medical assistant.")
	assert.Contains(t, contextQuery, "[user]: What are the symptoms of flu?")
}

func TestBuildContextAwareCacheQuery_ContextDifferentiatesSameLastMessage(t *testing.T) {
	// Two conversations share the same last user message but differ in prior turns.
	// With context-aware keys their contextQuery values must differ, preventing
	// a false cache hit.
	bodyMedical := []byte(`{
		"model":"gpt-4",
		"messages":[
			{"role":"system","content":"You are a medical assistant."},
			{"role":"user","content":"I have chest pain"},
			{"role":"assistant","content":"Describe the pain."},
			{"role":"user","content":"How do I fix it?"}
		]
	}`)
	bodyCode := []byte(`{
		"model":"gpt-4",
		"messages":[
			{"role":"system","content":"You are a coding assistant."},
			{"role":"user","content":"My build is failing"},
			{"role":"assistant","content":"What error do you see?"},
			{"role":"user","content":"How do I fix it?"}
		]
	}`)

	_, lastMed, ctxMed, err := BuildContextAwareCacheQuery(bodyMedical, 2)
	require.NoError(t, err)
	_, lastCode, ctxCode, err := BuildContextAwareCacheQuery(bodyCode, 2)
	require.NoError(t, err)

	// Both share the same raw last user message.
	assert.Equal(t, "How do I fix it?", lastMed)
	assert.Equal(t, "How do I fix it?", lastCode)

	// Context-aware keys must differ.
	assert.NotEqual(t, ctxMed, ctxCode, "same last-user-message in different contexts must produce different cache keys")

	// Spot-check content.
	assert.Contains(t, ctxMed, "medical assistant")
	assert.Contains(t, ctxCode, "coding assistant")
}

func TestBuildContextAwareCacheQuery_TurnWindowLimitsHistory(t *testing.T) {
	// Five prior user+assistant pairs; window of 1 should include only the last one.
	body := []byte(`{
		"model":"gpt-4",
		"messages":[
			{"role":"user","content":"turn1 user"},
			{"role":"assistant","content":"turn1 assistant"},
			{"role":"user","content":"turn2 user"},
			{"role":"assistant","content":"turn2 assistant"},
			{"role":"user","content":"turn3 user"},
			{"role":"assistant","content":"turn3 assistant"},
			{"role":"user","content":"turn4 user"},
			{"role":"assistant","content":"turn4 assistant"},
			{"role":"user","content":"turn5 user"},
			{"role":"assistant","content":"turn5 assistant"},
			{"role":"user","content":"current question"}
		]
	}`)
	_, _, contextQuery, err := BuildContextAwareCacheQuery(body, 1)
	require.NoError(t, err)
	// Only the final prior pair should appear.
	assert.Contains(t, contextQuery, "turn5 user")
	assert.Contains(t, contextQuery, "turn5 assistant")
	// Earlier turns must be excluded.
	assert.NotContains(t, contextQuery, "turn1 user")
	assert.NotContains(t, contextQuery, "turn3 assistant")
}

func TestBuildContextAwareCacheQuery_LongSegmentsTruncated(t *testing.T) {
	longSystem := strings.Repeat("s", maxContextSegmentChars+50)
	body := []byte(`{"model":"gpt-4","messages":[{"role":"system","content":"` + longSystem + `"},{"role":"user","content":"short question"}]}`)
	_, _, contextQuery, err := BuildContextAwareCacheQuery(body, 1)
	require.NoError(t, err)
	// System segment must be truncated to exactly maxContextSegmentChars chars.
	assert.Contains(t, contextQuery, "[system]:")
	// The full long string must not appear verbatim in the key.
	assert.NotContains(t, contextQuery, longSystem)
}

func TestBuildContextAwareCacheQuery_NoUserMessages_EmptyResults(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"system","content":"System only"}]}`)
	_, lastUser, contextQuery, err := BuildContextAwareCacheQuery(body, 2)
	require.NoError(t, err)
	assert.Empty(t, lastUser)
	assert.Empty(t, contextQuery)
}

func TestBuildContextAwareCacheQuery_InvalidJSON_ReturnsError(t *testing.T) {
	_, _, _, err := BuildContextAwareCacheQuery([]byte(`not json`), 1)
	require.Error(t, err)
}

// ---------------------------------------------------------------------------
// buildContextWindow — internal helper
// ---------------------------------------------------------------------------

func TestBuildContextWindow_EmptyMessages(t *testing.T) {
	result := buildContextWindow(nil, 2)
	assert.Empty(t, result)
}

func TestBuildContextWindow_OnlyCurrentUserMessage(t *testing.T) {
	msgs := []ChatMessage{
		{Role: "user", Content: json.RawMessage(`"Ask me anything"`)},
	}
	result := buildContextWindow(msgs, 2)
	assert.Equal(t, "[user]: Ask me anything", result)
}

func TestBuildContextWindow_SystemPlusCurrentUser(t *testing.T) {
	msgs := []ChatMessage{
		{Role: "system", Content: json.RawMessage(`"Be concise."`)},
		{Role: "user", Content: json.RawMessage(`"Hello"`)},
	}
	result := buildContextWindow(msgs, 1)
	assert.Equal(t, "[system]: Be concise.\n[user]: Hello", result)
}

func TestBuildContextWindow_PriorTurnsIncluded(t *testing.T) {
	msgs := []ChatMessage{
		{Role: "user", Content: json.RawMessage(`"First question"`)},
		{Role: "assistant", Content: json.RawMessage(`"First answer"`)},
		{Role: "user", Content: json.RawMessage(`"Second question"`)},
	}
	result := buildContextWindow(msgs, 1)
	assert.Contains(t, result, "[user]: First question")
	assert.Contains(t, result, "[assistant]: First answer")
	assert.Contains(t, result, "[user]: Second question")
}

func TestBuildContextWindow_ZeroTurns_OnlyCurrentMessage(t *testing.T) {
	msgs := []ChatMessage{
		{Role: "user", Content: json.RawMessage(`"Old message"`)},
		{Role: "assistant", Content: json.RawMessage(`"Old reply"`)},
		{Role: "user", Content: json.RawMessage(`"New message"`)},
	}
	// contextWindowTurns=0 means maxPrior=0, so priorMsgs are dropped.
	result := buildContextWindow(msgs, 0)
	assert.NotContains(t, result, "Old message")
	assert.NotContains(t, result, "Old reply")
	assert.Contains(t, result, "[user]: New message")
}
