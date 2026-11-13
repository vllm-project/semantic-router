package extproc

import (
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Phrases with which backends refuse a prompt longer than their window:
// vLLM and OpenAI ("maximum context length"), SGLang ("longer than the
// model's context length"), llama.cpp ("the available context size"),
// Anthropic ("prompt is too long") and others ("context window").
var upstreamContextLengthPhrases = []string{
	"context length",
	"context window",
	"context size",
	"prompt is too long",
}

// normalizeUpstreamContextLength gives a backend's refusal of a prompt longer
// than its window OpenAI's code for it, context_length_exceeded, so a client
// handles every such refusal alike. The backend's message, which names its
// window and the prompt's count, stays.
func normalizeUpstreamContextLength(status int) protocolcodec.TransportErrorMutation {
	return func(transportError *llmprotocol.TransportError) error {
		if transportError == nil || !upstreamContextLengthError(status, transportError.Error) {
			return nil
		}
		transportError.Error.Category = llmprotocol.ErrorInvalidRequest
		transportError.Error.Code = routingFailureContextLength.code
		return nil
	}
}

func upstreamContextLengthError(status int, protocolError *llmprotocol.ProtocolError) bool {
	if protocolError == nil || (status != http.StatusBadRequest && status != http.StatusRequestEntityTooLarge) {
		return false
	}
	if protocolError.Code == routingFailureContextLength.code || protocolError.Parameter == "input_tokens" {
		return true
	}
	message := strings.ToLower(protocolError.Message)
	for _, phrase := range upstreamContextLengthPhrases {
		if strings.Contains(message, phrase) {
			return true
		}
	}
	return false
}
