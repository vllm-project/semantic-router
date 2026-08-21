package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
)

const anthropicAuthErrorEnvelope = `{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"},"request_id":"req_011CSHoEeqs5DZitY69wyoEE"}`

// An Anthropic error envelope must survive the Anthropic→OpenAI response
// translation as an OpenAI-shape error body, not be flattened into an
// empty success-shaped chat.completion.
func TestToOpenAIResponseBody_PreservesErrorEnvelope(t *testing.T) {
	out, err := ToOpenAIResponseBody([]byte(anthropicAuthErrorEnvelope), "claude-test")
	require.NoError(t, err)

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &body))

	errObj, ok := body["error"].(map[string]interface{})
	require.True(t, ok, "translated body must carry a top-level error object")
	assert.Equal(t, "authentication_error", errObj["type"])
	assert.Equal(t, "invalid x-api-key", errObj["message"])

	_, hasChoices := body["choices"]
	assert.False(t, hasChoices, "error body must not be shaped like a completion")

	_, hasRequestID := body["request_id"]
	assert.False(t, hasRequestID, "OpenAI error bodies stay spec-pure; request_id rides the sidecar")
}

// The ext-aware entrypoint takes the same guard path, and the early return
// must leave the sidecar untouched — an error body carries no usage or
// stop reason to capture.
func TestToOpenAIResponseBodyWithExt_PreservesErrorEnvelope(t *testing.T) {
	ext := &ir.IRExtensions{}
	out, err := ToOpenAIResponseBodyWithExt([]byte(anthropicAuthErrorEnvelope), "claude-test", ext)
	require.NoError(t, err)
	assert.Contains(t, string(out), `"authentication_error"`)

	assert.Empty(t, ext.AnthropicStopReason)
	assert.Zero(t, ext.CacheReadInputTokens)
	assert.Zero(t, ext.CacheCreationInputTokens)
	assert.Equal(t, "req_011CSHoEeqs5DZitY69wyoEE", ext.AnthropicErrorRequestID, "request_id captured onto the sidecar")
}

// A success-shaped message must never be converted into an error body —
// the decision rests on the envelope's type field alone, not on
// error-looking text anywhere in the content.
func TestAnthropicErrorToOpenAIBody_IgnoresSuccessBodies(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "message whose text content looks like an error",
			body: `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"error: something"}],"usage":{"input_tokens":1,"output_tokens":1}}`,
		},
		{
			name: "minimal message",
			body: `{"type":"message","content":[]}`,
		},
		{
			name: "empty object",
			body: `{}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, ok := anthropicErrorToOpenAIBody([]byte(tc.body), nil)
			assert.False(t, ok)
		})
	}
}

// An OpenAI-shape error body must be re-emitted to Anthropic clients as the
// Anthropic error envelope, not as an empty success-shaped message.
func TestEmitAnthropicResponse_ReEmitsErrorEnvelope(t *testing.T) {
	openAIError := []byte(`{"error":{"message":"invalid x-api-key","type":"authentication_error"}}`)

	out, err := EmitAnthropicResponse(openAIError, &ir.IRExtensions{}, "claude-test")
	require.NoError(t, err)

	var envelope map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &envelope))
	assert.Equal(t, "error", envelope["type"])

	errObj, ok := envelope["error"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "authentication_error", errObj["type"])
	assert.Equal(t, "invalid x-api-key", errObj["message"])
}

// Errors from OpenAI-format backends reach Anthropic clients with their
// message intact; an absent type is coerced to api_error by EmitAnthropicError.
func TestEmitAnthropicResponse_OpenAIBackendErrorWithoutType(t *testing.T) {
	openAIError := []byte(`{"error":{"message":"Rate limit reached","code":"rate_limit_exceeded"}}`)

	out, err := EmitAnthropicResponse(openAIError, nil, "claude-test")
	require.NoError(t, err)

	var envelope map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &envelope))
	errObj, ok := envelope["error"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "api_error", errObj["type"])
	assert.Equal(t, "Rate limit reached", errObj["message"])

	_, hasRequestID := envelope["request_id"]
	assert.False(t, hasRequestID, "absent request_id must be omitted, not empty")
}

// The emit path applies the error-type mapping, not just the helper in
// isolation: an OpenAI-only token arriving at EmitAnthropicResponse comes
// out as its Anthropic equivalent.
func TestEmitAnthropicResponse_MapsOpenAIErrorType(t *testing.T) {
	openAIError := []byte(`{"error":{"message":"Rate limit reached","type":"rate_limit_exceeded"}}`)

	out, err := EmitAnthropicResponse(openAIError, nil, "claude-test")
	require.NoError(t, err)

	var envelope map[string]interface{}
	require.NoError(t, json.Unmarshal(out, &envelope))
	errObj, ok := envelope["error"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "rate_limit_error", errObj["type"])
	assert.Equal(t, "Rate limit reached", errObj["message"])
}

// Full double-Anthropic round trip: envelope → OpenAI error body → envelope,
// with type and message preserved verbatim.
func TestAnthropicErrorEnvelopeRoundTrip(t *testing.T) {
	ext := &ir.IRExtensions{}

	openAIBody, err := ToOpenAIResponseBodyWithExt([]byte(anthropicAuthErrorEnvelope), "claude-test", ext)
	require.NoError(t, err)

	anthropicBody, err := EmitAnthropicResponse(openAIBody, ext, "claude-test")
	require.NoError(t, err)

	var envelope map[string]interface{}
	require.NoError(t, json.Unmarshal(anthropicBody, &envelope))
	assert.Equal(t, "error", envelope["type"])
	errObj, ok := envelope["error"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "authentication_error", errObj["type"])
	assert.Equal(t, "invalid x-api-key", errObj["message"])
	assert.Equal(t, "req_011CSHoEeqs5DZitY69wyoEE", envelope["request_id"], "request_id survives the round trip")
}

// OpenAI-only error-type tokens are mapped to Anthropic's defined error
// types; Anthropic's own tokens pass through unchanged.
func TestAnthropicErrorType_CoercesOpenAITokens(t *testing.T) {
	cases := []struct {
		name  string
		token string
		want  string
	}{
		{name: "anthropic token passes verbatim", token: "authentication_error", want: "authentication_error"},
		{name: "anthropic overloaded passes verbatim", token: "overloaded_error", want: "overloaded_error"},
		{name: "openai rate limit maps to anthropic equivalent", token: "rate_limit_exceeded", want: "rate_limit_error"},
		{name: "openai quota exhaustion maps to billing", token: "insufficient_quota", want: "billing_error"},
		{name: "openai server error maps to catch-all", token: "server_error", want: "api_error"},
		{name: "unknown token maps to catch-all", token: "something_novel", want: "api_error"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, anthropicErrorType(tc.token))
		})
	}
}

// Bodies where "error" is null or a non-object are not error bodies and
// must be left to the normal translation path.
func TestOpenAIErrorBody_IgnoresNullAndNonObjectError(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{name: "error is null", body: `{"id":"x","choices":[],"error":null}`},
		{name: "error is a string", body: `{"id":"x","choices":[],"error":"nope"}`},
		{name: "error absent", body: `{"id":"x","choices":[]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, ok := openAIErrorBody([]byte(tc.body))
			assert.False(t, ok)
		})
	}
}
