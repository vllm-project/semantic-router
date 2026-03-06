package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
