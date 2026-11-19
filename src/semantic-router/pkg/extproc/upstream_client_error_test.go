package extproc

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// vLLM's answers to an output limit beyond the model's window, before 0.12
// and since, and to max_tokens 0, each with its integer code.
const (
	vllmOutputLimitMessageLegacy = "'max_tokens' or 'max_completion_tokens' is too large: 5000000. This model's maximum context length is 262144 tokens and your request has 12 input tokens (5000000 > 262144 - 12). None"
	vllmOutputLimitMessage       = "max_completion_tokens=5000000 cannot be greater than max_model_len=262144. Please request fewer output tokens. (parameter=max_completion_tokens, value=5000000)"
	vllmZeroOutputMessage        = "max_tokens must be at least 1, got 0. (parameter=max_tokens, value=0)"
	vllmPromptLengthMessage      = "This model's maximum context length is 1048576 tokens. However, your prompt contains at least 1048577 input tokens, which exceeds this limit even without any output tokens. Please reduce the length of the input prompt. (parameter=input_tokens, value=1048577)"
)

func clientErrorFor(t *testing.T, format llmprotocol.WireFormat, outputField string, status int, backend llmprotocol.ProtocolError) llmprotocol.ProtocolError {
	t.Helper()
	ctx := &RequestContext{SourceFormat: format}
	ctx.ProtocolEnvelope.OutputLimitField = outputField
	transportError := llmprotocol.TransportError{Error: &backend}
	require.NoError(t, normalizeUpstreamClientError(ctx, status)(&transportError))
	return *transportError.Error
}

func vllmBadRequest(message, param string) llmprotocol.ProtocolError {
	return llmprotocol.ProtocolError{Category: llmprotocol.ErrorInvalidRequest, Code: "400", Message: message, Parameter: param}
}

// An output limit beyond the window is context_length_exceeded, under the
// name the client gave the limit, with the numbers in plain words: the
// client never sees the max_completion_tokens the Router sent a Chat backend,
// nor vLLM's max_model_len.
func TestBackendOutputLimitRefusalNamesTheClientsField(t *testing.T) {
	const fewer = ". Please request fewer output tokens."
	for _, test := range []struct {
		name          string
		format        llmprotocol.WireFormat
		field         string
		backend       llmprotocol.ProtocolError
		param, wanted string
	}{
		{
			"legacy vLLM to a Chat max_tokens", llmprotocol.OpenAIChatV1, "max_tokens", vllmBadRequest(vllmOutputLimitMessageLegacy, ""),
			"max_tokens", "max_tokens is too large: 5000000. This model's maximum context length is 262144 tokens, and your request has 12 input tokens" + fewer,
		},
		{
			"vLLM to a Chat max_tokens", llmprotocol.OpenAIChatV1, "max_tokens", vllmBadRequest(vllmOutputLimitMessage, "max_completion_tokens"),
			"max_tokens", "max_tokens is too large: 5000000. This model's maximum context length is 262144 tokens" + fewer,
		},
		{
			"vLLM to a Chat max_completion_tokens", llmprotocol.OpenAIChatV1, "max_completion_tokens", vllmBadRequest(vllmOutputLimitMessage, "max_completion_tokens"),
			"max_completion_tokens", "max_completion_tokens is too large: 5000000. This model's maximum context length is 262144 tokens" + fewer,
		},
		{
			"vLLM to a Responses client", llmprotocol.OpenAIResponsesV1, "", vllmBadRequest(vllmOutputLimitMessage, "max_completion_tokens"),
			"max_output_tokens", "max_output_tokens is too large: 5000000. This model's maximum context length is 262144 tokens" + fewer,
		},
		{
			"legacy vLLM to a Messages client", llmprotocol.AnthropicMessagesV1, "", vllmBadRequest(vllmOutputLimitMessageLegacy, ""),
			"max_tokens", "max_tokens is too large: 5000000. This model's maximum context length is 262144 tokens, and your request has 12 input tokens" + fewer,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			got := clientErrorFor(t, test.format, test.field, http.StatusBadRequest, test.backend)
			require.Equal(t, llmprotocol.ProtocolError{
				Category: llmprotocol.ErrorInvalidRequest, Code: "context_length_exceeded", Message: test.wanted, Parameter: test.param,
			}, got)
		})
	}
}

// Any other backend 400 is invalid_request. A field vLLM names by its own
// name for the client's limit reads as the client's, and vLLM's parameter
// suffix is dropped.
func TestBackendBadRequestIsInvalidRequestInTheClientsTerms(t *testing.T) {
	speculative := "The min_p and logit_bias sampling parameters are not yet supported with speculative decoding."
	for _, test := range []struct {
		name     string
		field    string
		backend  llmprotocol.ProtocolError
		expected llmprotocol.ProtocolError
	}{
		{
			"max_tokens 0", "max_tokens", vllmBadRequest(vllmZeroOutputMessage, "max_tokens"),
			llmprotocol.ProtocolError{Code: "invalid_request", Parameter: "max_tokens", Message: "max_tokens must be at least 1, got 0."},
		},
		{
			"max_completion_tokens 0", "max_completion_tokens", vllmBadRequest(vllmZeroOutputMessage, "max_tokens"),
			llmprotocol.ProtocolError{Code: "invalid_request", Parameter: "max_completion_tokens", Message: "max_completion_tokens must be at least 1, got 0."},
		},
		{
			"an unsupported control", "", vllmBadRequest(speculative, ""),
			llmprotocol.ProtocolError{Code: "invalid_request", Message: speculative},
		},
		{
			"vLLM's own window setting", "", vllmBadRequest("truncate_prompt_tokens=300000 cannot be greater than max_total_tokens = 262144. Please request a smaller truncation size.", "truncate_prompt_tokens"),
			llmprotocol.ProtocolError{
				Code: "invalid_request", Parameter: "truncate_prompt_tokens",
				Message: "truncate_prompt_tokens=300000 cannot be greater than the model's maximum context length of 262144 tokens. Please request a smaller truncation size.",
			},
		},
		{
			"an OpenAI code", "max_tokens",
			llmprotocol.ProtocolError{
				Category: llmprotocol.ErrorInvalidRequest, Code: "unsupported_parameter", Parameter: "max_tokens",
				Message: "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
			},
			llmprotocol.ProtocolError{
				Code: "invalid_request", Parameter: "max_tokens",
				Message: "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			got := clientErrorFor(t, llmprotocol.OpenAIChatV1, test.field, http.StatusBadRequest, test.backend)
			test.expected.Category = llmprotocol.ErrorInvalidRequest
			require.Equal(t, test.expected, got)
		})
	}
}

// A prompt beyond the window is context_length_exceeded with the client's
// name for the conversation, and the backend's message without vLLM's suffix.
func TestBackendPromptLengthRefusalNamesTheConversation(t *testing.T) {
	want := "This model's maximum context length is 1048576 tokens. However, your prompt contains at least 1048577 input tokens, which exceeds this limit even without any output tokens. Please reduce the length of the input prompt."
	for format, param := range map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1: "messages", llmprotocol.OpenAIResponsesV1: "input", llmprotocol.AnthropicMessagesV1: "messages",
	} {
		got := clientErrorFor(t, format, "", http.StatusBadRequest, vllmBadRequest(vllmPromptLengthMessage, "input_tokens"))
		require.Equal(t, llmprotocol.ProtocolError{
			Category: llmprotocol.ErrorInvalidRequest, Code: "context_length_exceeded", Message: want, Parameter: param,
		}, got, "%s", format)
	}
}

// Another 4xx keeps the backend's status and code, unless the code is vLLM's
// bare HTTP status, and a 5xx is left as it is.
func TestOtherBackendErrorsKeepTheirCode(t *testing.T) {
	notFound := clientErrorFor(t, llmprotocol.OpenAIChatV1, "", http.StatusNotFound,
		llmprotocol.ProtocolError{Category: llmprotocol.ErrorUpstreamUnavailable, Code: "404", Message: "The model `m` does not exist."})
	require.Equal(t, llmprotocol.ProtocolError{Category: llmprotocol.ErrorNotFound, Code: "not_found", Message: "The model `m` does not exist."}, notFound)

	unauthorized := llmprotocol.ProtocolError{Category: llmprotocol.ErrorAuthentication, Code: "invalid_api_key", Message: "Incorrect API key provided."}
	require.Equal(t, unauthorized, clientErrorFor(t, llmprotocol.OpenAIChatV1, "", http.StatusUnauthorized, unauthorized))

	failed := llmprotocol.ProtocolError{Category: llmprotocol.ErrorUpstreamUnavailable, Code: "500", Message: "Internal server error"}
	require.Equal(t, failed, clientErrorFor(t, llmprotocol.OpenAIChatV1, "", http.StatusInternalServerError, failed))
}
