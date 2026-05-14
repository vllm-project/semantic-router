package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
)

// ---------- extractContentFast ----------

func TestExtractContentFast_SimpleRequest(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"Hello world"}],"stream":true}`)
	r, err := extractContentFast(body)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4", r.Model)
	assert.True(t, r.Stream)
	assert.Equal(t, "Hello world", r.UserContent)
	assert.Empty(t, r.NonUserMessages)
	assert.False(t, r.HasAssistantReply)
	assert.Empty(t, r.FirstImageURL)
}

func TestExtractContentFast_MultiRole(t *testing.T) {
	body := []byte(`{
		"model": "auto",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "Explain quantum physics."},
			{"role": "assistant", "content": "Quantum physics is..."}
		]
	}`)
	r, err := extractContentFast(body)
	require.NoError(t, err)
	assert.Equal(t, "auto", r.Model)
	assert.False(t, r.Stream)
	assert.Equal(t, "Explain quantum physics.", r.UserContent)
	assert.Equal(t, []string{"You are a helpful assistant.", "Quantum physics is..."}, r.NonUserMessages)
	assert.True(t, r.HasAssistantReply)
}

func TestExtractContentFast_ContentParts(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "What is in this image?"},
				{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}}
			]
		}]
	}`)
	r, err := extractContentFast(body)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4o", r.Model)
	assert.Equal(t, "What is in this image?", r.UserContent)
	assert.Equal(t, "data:image/png;base64,iVBOR", r.FirstImageURL)
}

func TestExtractContentFast_NoMessages(t *testing.T) {
	body := []byte(`{"model": "gpt-4"}`)
	r, err := extractContentFast(body)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4", r.Model)
	assert.Empty(t, r.UserContent)
}

func TestExtractContentFast_MissingModel(t *testing.T) {
	body := []byte(`{"messages": [{"role": "user", "content": "hi"}]}`)
	_, err := extractContentFast(body)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "model")
}

func TestExtractContentFast_InvalidJSON(t *testing.T) {
	_, err := extractContentFast([]byte(`{not json`))
	assert.Error(t, err)
}

func TestExtractContentFast_UnsafeImageURL(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "describe"},
				{"type": "image_url", "image_url": {"url": "https://evil.com/img.png"}}
			]
		}]
	}`)
	r, err := extractContentFast(body)
	require.NoError(t, err)
	assert.Empty(t, r.FirstImageURL, "HTTP URLs must be rejected")
}

func TestExtractContentFast_SystemContentParts(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4",
		"messages": [{
			"role": "system",
			"content": [{"type": "text", "text": "Part A"}, {"type": "text", "text": "Part B"}]
		}]
	}`)
	r, err := extractContentFast(body)
	require.NoError(t, err)
	assert.Equal(t, []string{"Part A Part B"}, r.NonUserMessages)
}

// ---------- extractContentFast: multimodal isolation ----------
// These tests verify that image data never leaks into text fields
// (UserContent, NonUserMessages) that feed prompt compression.

func TestExtractContentFast_ImageDataNotInUserContent(t *testing.T) {
	b64 := strings.Repeat("ABCDEFGH", 1000) // 8KB fake base64
	body := []byte(fmt.Sprintf(`{
		"model": "gpt-4o",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "Describe this photo"},
				{"type": "image_url", "image_url": {"url": "data:image/png;base64,%s"}}
			]
		}]
	}`, b64))
	r, err := extractContentFast(body)
	require.NoError(t, err)
	assert.Equal(t, "Describe this photo", r.UserContent,
		"UserContent must contain only text, not base64 image data")
	assert.NotContains(t, r.UserContent, "ABCDEFGH",
		"image bytes must not leak into UserContent")
	assert.NotContains(t, r.UserContent, "base64",
		"data URI prefix must not leak into UserContent")
}

func TestExtractContentFast_ImageDataNotInNonUserMessages(t *testing.T) {
	b64 := strings.Repeat("XYZXYZ", 500)
	body := []byte(fmt.Sprintf(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "system", "content": [
				{"type": "text", "text": "You are a helpful assistant."},
				{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,%s"}}
			]},
			{"role": "user", "content": "Hello"}
		]
	}`, b64))
	r, err := extractContentFast(body)
	require.NoError(t, err)
	assert.Equal(t, "Hello", r.UserContent)
	require.Len(t, r.NonUserMessages, 1)
	assert.Equal(t, "You are a helpful assistant.", r.NonUserMessages[0],
		"NonUserMessages must contain only text, not image data")
	assert.NotContains(t, r.NonUserMessages[0], "XYZXYZ")
}

func TestExtractContentFast_MultipleContentParts_OnlyTextExtracted(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "First question"},
				{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
				{"type": "text", "text": "Second question"},
				{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBB"}}
			]
		}]
	}`)
	r, err := extractContentFast(body)
	require.NoError(t, err)
	assert.Equal(t, "First question Second question", r.UserContent,
		"must concatenate text parts only, skipping image_url parts")
}

func TestExtractContentFast_OnlyImages_EmptyUserContent(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
				{"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB"}}
			]
		}]
	}`)
	r, err := extractContentFast(body)
	require.NoError(t, err)
	assert.Empty(t, r.UserContent,
		"content with only images and no text parts must produce empty UserContent")
	assert.NotEmpty(t, r.FirstImageURL)
}

func TestExtractContentFast_ImageURL_SeparateFromTextFields(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "What breed is this dog?"},
				{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo"}}
			]
		}]
	}`)
	r, err := extractContentFast(body)
	require.NoError(t, err)

	assert.Equal(t, "What breed is this dog?", r.UserContent)
	assert.Equal(t, "data:image/png;base64,iVBORw0KGgo", r.FirstImageURL)

	// The key invariant: UserContent (which feeds prompt compression) never
	// contains any part of the image URL or its base64 data.
	assert.NotContains(t, r.UserContent, "data:image")
	assert.NotContains(t, r.UserContent, "iVBOR")
	assert.NotContains(t, r.UserContent, "base64")
}

func TestExtractContentFast_MultiMessage_ImageIsolated(t *testing.T) {
	body := []byte(`{
		"model": "auto",
		"messages": [
			{"role": "system", "content": "You are a vet."},
			{"role": "user", "content": [
				{"type": "text", "text": "Is this cat healthy?"},
				{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"}}
			]},
			{"role": "assistant", "content": "The cat looks healthy."},
			{"role": "user", "content": "Thanks!"}
		]
	}`)
	r, err := extractContentFast(body)
	require.NoError(t, err)

	// Last user message wins for UserContent
	assert.Equal(t, "Thanks!", r.UserContent)
	assert.Equal(t, []string{"You are a vet.", "The cat looks healthy."}, r.NonUserMessages)

	// No image data in any text field
	for _, msg := range r.NonUserMessages {
		assert.NotContains(t, msg, "base64")
		assert.NotContains(t, msg, "/9j/4AAQ")
	}
	assert.NotContains(t, r.UserContent, "base64")
}

// ---------- Cache key invariant: extractContentFast ↔ cache.ExtractQueryFromOpenAIRequest ----------
//
// The semantic cache stores the *original* uncompressed prompt as the cache key.
// Prompt compression (in performDecisionEvaluation) is a local variable that
// never escapes to ctx. These tests verify that the text the cache would embed
// matches the text classification/compression would operate on, both sourced
// from the same original body.

func TestCacheKeyInvariant_SimpleText(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"What is 2+2?"}]}`)
	fast, err := extractContentFast(body)
	require.NoError(t, err)

	_, cacheQuery, cerr := cache.ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, cerr)

	assert.Equal(t, fast.UserContent, cacheQuery,
		"cache key must equal the text that classification receives (both from original body)")
}

func TestCacheKeyInvariant_MultiTurn(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4",
		"messages":[
			{"role":"user","content":"Hi"},
			{"role":"assistant","content":"Hello!"},
			{"role":"user","content":"What is the capital of France?"}
		]
	}`)
	fast, err := extractContentFast(body)
	require.NoError(t, err)

	_, cacheQuery, cerr := cache.ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, cerr)

	assert.Equal(t, fast.UserContent, cacheQuery,
		"both must return the last user message text from the original body")
}

func TestCacheKeyInvariant_Multimodal_TextOnly(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4o",
		"messages":[{
			"role":"user",
			"content":[
				{"type":"text","text":"Describe this image"},
				{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUg"}}
			]
		}]
	}`)
	fast, err := extractContentFast(body)
	require.NoError(t, err)

	_, cacheQuery, cerr := cache.ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, cerr)

	assert.Equal(t, "Describe this image", fast.UserContent)
	assert.Equal(t, "Describe this image", cacheQuery)
	assert.NotContains(t, cacheQuery, "base64",
		"cache key must not contain image data")
}

func TestCacheKeyInvariant_LargePrompt_NotCompressed(t *testing.T) {
	// Build a prompt large enough to trigger compression upstream (>512 tokens).
	// The invariant is that the cache key is the FULL original text.
	var sb strings.Builder
	for range 200 {
		sb.WriteString("This is a sentence about machine learning and neural networks. ")
	}
	longText := sb.String()

	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"` + longText + `"}]}`)

	fast, err := extractContentFast(body)
	require.NoError(t, err)

	_, cacheQuery, cerr := cache.ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, cerr)

	assert.Equal(t, longText, fast.UserContent)
	assert.Equal(t, longText, cacheQuery,
		"cache key must be the full uncompressed prompt even when text is compressible")
	assert.Greater(t, len(cacheQuery), 5000)
}

func TestCacheKeyInvariant_OriginalBodyUnmodifiedByExtraction(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"Hello world"}],"stream":false}`)
	snapshot := make([]byte, len(body))
	copy(snapshot, body)

	_, err := extractContentFast(body)
	require.NoError(t, err)

	assert.Equal(t, snapshot, body,
		"extractContentFast must not mutate the input body (cache relies on it)")

	_, _, cerr := cache.ExtractQueryFromOpenAIRequest(body)
	require.NoError(t, cerr)

	assert.Equal(t, snapshot, body,
		"ExtractQueryFromOpenAIRequest must not mutate the input body")
}

// ---------- extractContentFast: strict type validation ----------

func TestExtractContentFast_NumericModelRejected(t *testing.T) {
	body := []byte(`{"model":42,"messages":[{"role":"user","content":"hi"}]}`)
	_, err := extractContentFast(body)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "model")
	assert.Contains(t, err.Error(), "string")
}

func TestExtractContentFast_BoolModelRejected(t *testing.T) {
	body := []byte(`{"model":true,"messages":[]}`)
	_, err := extractContentFast(body)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "model")
}

func TestExtractContentFast_NullModelRejected(t *testing.T) {
	body := []byte(`{"model":null,"messages":[]}`)
	_, err := extractContentFast(body)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "model")
}

func TestExtractContentFast_ArrayModelRejected(t *testing.T) {
	body := []byte(`{"model":["gpt-4"],"messages":[]}`)
	_, err := extractContentFast(body)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "model")
}

func TestExtractContentFast_StringStreamRejected(t *testing.T) {
	body := []byte(`{"model":"gpt-4","stream":"true"}`)
	_, err := extractContentFast(body)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "stream")
	assert.Contains(t, err.Error(), "boolean")
}

func TestExtractContentFast_NumericStreamRejected(t *testing.T) {
	body := []byte(`{"model":"gpt-4","stream":1}`)
	_, err := extractContentFast(body)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "stream")
}

func TestExtractContentFast_NullStreamRejected(t *testing.T) {
	body := []byte(`{"model":"gpt-4","stream":null}`)
	_, err := extractContentFast(body)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "stream")
}

func TestExtractContentFast_MissingStreamAllowed(t *testing.T) {
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	r, err := extractContentFast(body)
	require.NoError(t, err)
	assert.False(t, r.Stream, "missing stream should default to false")
}

func TestExtractContentFast_StreamFalseAllowed(t *testing.T) {
	body := []byte(`{"model":"gpt-4","stream":false}`)
	r, err := extractContentFast(body)
	require.NoError(t, err)
	assert.False(t, r.Stream)
}

func TestExtractContentFast_StreamTrueAllowed(t *testing.T) {
	body := []byte(`{"model":"gpt-4","stream":true}`)
	r, err := extractContentFast(body)
	require.NoError(t, err)
	assert.True(t, r.Stream)
}

// ---------- extractStreamParamFast ----------

func TestExtractStreamParamFast(t *testing.T) {
	assert.True(t, extractStreamParamFast([]byte(`{"model":"x","stream":true}`)))
	assert.False(t, extractStreamParamFast([]byte(`{"model":"x","stream":false}`)))
	assert.False(t, extractStreamParamFast([]byte(`{"model":"x"}`)))
	assert.False(t, extractStreamParamFast([]byte(`{invalid`)))
}

// ---------- rewriteModelInBodyFast ----------

func TestRewriteModelInBodyFast(t *testing.T) {
	body := []byte(`{"model":"old-model","messages":[],"temperature":0.7}`)
	out, err := rewriteModelInBodyFast(body, "new-model")
	require.NoError(t, err)

	var m map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &m))
	assert.Equal(t, "new-model", m["model"])
	assert.Equal(t, 0.7, m["temperature"])
}

func TestRewriteModelInBodyFast_PreservesAllFields(t *testing.T) {
	body := []byte(`{"model":"a","messages":[{"role":"user","content":"hi"}],"max_tokens":100,"stream":true}`)
	out, err := rewriteModelInBodyFast(body, "b")
	require.NoError(t, err)

	var m map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(out, &m))
	assert.JSONEq(t, `"b"`, string(m["model"]))
	assert.JSONEq(t, `[{"role":"user","content":"hi"}]`, string(m["messages"]))
	assert.JSONEq(t, `100`, string(m["max_tokens"]))
	assert.JSONEq(t, `true`, string(m["stream"]))
}

// ---------- addStreamFieldsFast ----------

func TestAddStreamFieldsFast(t *testing.T) {
	body := []byte(`{"model":"x","messages":[]}`)
	out := addStreamFieldsFast(body)

	var m map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &m))
	assert.Equal(t, true, m["stream"])
	opts, ok := m["stream_options"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, true, opts["include_usage"])
}

// ---------- extractModelFast ----------

func TestExtractModelFast(t *testing.T) {
	assert.Equal(t, "gpt-4", extractModelFast([]byte(`{"model":"gpt-4"}`)))
	assert.Empty(t, extractModelFast([]byte(`{}`)))
}

// ---------- Consistency: fast path vs legacy path ----------

func TestExtractStreamParam_FastMatchesLegacy(t *testing.T) {
	cases := [][]byte{
		[]byte(`{"model":"x","stream":true}`),
		[]byte(`{"model":"x","stream":false}`),
		[]byte(`{"model":"x"}`),
		[]byte(`{"model":"x","stream":"not-a-bool"}`),
	}
	for _, body := range cases {
		assert.Equal(t, extractStreamParamFast(body), extractStreamParam(body),
			"mismatch for %s", string(body))
	}
}

func TestRewriteModelInBody_FastMatchesLegacy(t *testing.T) {
	body := []byte(`{"model":"old","messages":[{"role":"user","content":"hi"}],"temperature":0.5}`)

	legacyOut, err := rewriteModelInBody(body, "new")
	require.NoError(t, err)

	fastOut, err := rewriteModelInBodyFast(body, "new")
	require.NoError(t, err)

	var legacyMap, fastMap map[string]interface{}
	require.NoError(t, json.Unmarshal(legacyOut, &legacyMap))
	require.NoError(t, json.Unmarshal(fastOut, &fastMap))
	assert.Equal(t, legacyMap["model"], fastMap["model"])
	assert.Equal(t, legacyMap["temperature"], fastMap["temperature"])
}

// ---------- Benchmarks ----------

func buildLargeBody(tokenCount int) []byte {
	words := make([]string, tokenCount)
	for i := range words {
		words[i] = fmt.Sprintf("word%d", i%1000)
	}
	content := strings.Join(words, " ")
	body, _ := json.Marshal(map[string]interface{}{
		"model":  "gpt-4",
		"stream": true,
		"messages": []map[string]interface{}{
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": content},
		},
	})
	return body
}

func BenchmarkExtractStreamParam_Legacy_1K(b *testing.B) {
	body := buildLargeBody(1000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var m map[string]interface{}
		json.Unmarshal(body, &m) //nolint:errcheck
		_ = m["stream"]
	}
}

func BenchmarkExtractStreamParam_Fast_1K(b *testing.B) {
	body := buildLargeBody(1000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractStreamParamFast(body)
	}
}

func BenchmarkExtractStreamParam_Legacy_16K(b *testing.B) {
	body := buildLargeBody(16000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var m map[string]interface{}
		json.Unmarshal(body, &m) //nolint:errcheck
		_ = m["stream"]
	}
}

func BenchmarkExtractStreamParam_Fast_16K(b *testing.B) {
	body := buildLargeBody(16000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractStreamParamFast(body)
	}
}

func BenchmarkExtractContentFast_1K(b *testing.B) {
	body := buildLargeBody(1000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractContentFast(body) //nolint:errcheck
	}
}

func BenchmarkExtractContentFast_16K(b *testing.B) {
	body := buildLargeBody(16000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractContentFast(body) //nolint:errcheck
	}
}

func BenchmarkParseOpenAIRequest_1K(b *testing.B) {
	body := buildLargeBody(1000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parseOpenAIRequest(body) //nolint:errcheck
	}
}

func BenchmarkParseOpenAIRequest_16K(b *testing.B) {
	body := buildLargeBody(16000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parseOpenAIRequest(body) //nolint:errcheck
	}
}

func BenchmarkRewriteModel_Legacy_16K(b *testing.B) {
	body := buildLargeBody(16000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var m map[string]json.RawMessage
		json.Unmarshal(body, &m) //nolint:errcheck
		m["model"], _ = json.Marshal("new-model")
		json.Marshal(m) //nolint:errcheck
	}
}

func BenchmarkRewriteModel_Fast_16K(b *testing.B) {
	body := buildLargeBody(16000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rewriteModelInBodyFast(body, "new-model") //nolint:errcheck
	}
}

func TestExtractContentFast_ConversationFacts(t *testing.T) {
	body := []byte(`{
		"model": "auto",
		"messages": [
			{"role": "developer", "content": "Fix bugs methodically."},
			{"role": "user", "content": "The login endpoint returns 500."},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
				{"id": "c2", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}
			]},
			{"role": "tool", "tool_call_id": "c1", "content": "file contents"},
			{"role": "tool", "tool_call_id": "c2", "content": "dir listing"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "c3", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}
			]},
			{"role": "tool", "tool_call_id": "c3", "content": "done"},
			{"role": "user", "content": "Now run the tests."},
			{"role": "system", "content": "Session context."}
		],
		"tools": [
			{"type": "function", "function": {"name": "read_file"}},
			{"type": "function", "function": {"name": "write_file"}},
			{"type": "function", "function": {"name": "run_tests"}}
		]
	}`)
	r, err := extractContentFast(body)
	require.NoError(t, err)

	assert.True(t, r.HasDeveloperMessage, "developer message should be detected")
	assert.Equal(t, 2, r.UserMessageCount, "two user messages")
	assert.Equal(t, 2, r.AssistantMessageCount, "two assistant messages")
	assert.Equal(t, 1, r.SystemMessageCount, "one system message")
	assert.Equal(t, 3, r.ToolMessageCount, "three tool messages")
	assert.Equal(t, 3, r.ToolDefinitionCount, "three tool definitions")
	assert.Equal(t, 3, r.AssistantToolCallCount, "three tool_calls total (2+1)")
	assert.Equal(t, []string{"read_file", "list_dir", "write_file"}, r.AssistantToolNames, "tool call names should preserve conversation order")
	assert.Equal(t, 3, r.ToolResultCount, "three tool results")
}
