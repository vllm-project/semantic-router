package extproc

import (
	"fmt"
	"net/http"
	"regexp"
	"slices"
	"strconv"
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

// The members that set the output limit, in some API or other. A backend
// names the one the Router sent it, or its own.
var upstreamOutputLimitFields = []string{"max_completion_tokens", "max_output_tokens", "max_tokens"}

// The members a backend names for the prompt: vLLM's for a prompt over its
// window, and the conversation in each API.
var upstreamInputFields = []string{"input_tokens", "input_text", "messages", "input"}

var (
	// The numbers vLLM's refusal of an output limit over its window carries:
	// "'max_tokens' or 'max_completion_tokens' is too large: 5000000. This
	// model's maximum context length is 262144 tokens and your request has 12
	// input tokens" before 0.12, "max_completion_tokens=5000000 cannot be
	// greater than max_model_len=262144" since.
	upstreamRequestedOutput = regexp.MustCompile(`(?:is too large: |\b(?:max_completion_tokens|max_output_tokens|max_tokens) ?= ?)(\d+)`)
	upstreamWindowSize      = regexp.MustCompile(`(?:maximum context length is |\b(?:max_model_len|max_total_tokens) ?= ?)(\d+)`)
	upstreamInputTokens     = regexp.MustCompile(`request has (\d+) input tokens`)
	// vLLM names its own settings by their identifiers, which mean nothing
	// to a client.
	upstreamWindowSetting = regexp.MustCompile(`\b(?:max_model_len|max_total_tokens) ?= ?(\d+)`)
	// vLLM's validation errors end with their parameter and value, and older
	// releases' with " None" after the last sentence.
	upstreamValidationSuffix = regexp.MustCompile(`\s*\((?:parameter|value)=[^)]*\)\s*$`)
	upstreamNoneSuffix       = regexp.MustCompile(`\.\s+None\s*$`)
)

// normalizeUpstreamClientError gives a backend's 4xx error the Router's
// documented code and the client's own terms. A refusal of a prompt or an
// output limit beyond the window is context_length_exceeded, as OpenAI calls
// it; any other 400 is invalid_request, and another 4xx keeps the backend's
// code unless it is vLLM's bare HTTP status. The param and the message name
// the member the client sent, not the one the Router or the backend uses for
// it, and the message drops the backend's own setting names but keeps its
// numbers. The HTTP status stays the backend's.
func normalizeUpstreamClientError(ctx *RequestContext, status int) protocolcodec.TransportErrorMutation {
	return func(transportError *llmprotocol.TransportError) error {
		if transportError == nil || transportError.Error == nil || status < 400 || status > 499 {
			return nil
		}
		clientUpstreamError(transportError.Error, status, clientFieldsOf(ctx))
		return nil
	}
}

// clientFields are the client's names for the output limit and the prompt.
type clientFields struct{ output, input string }

func clientFieldsOf(ctx *RequestContext) clientFields {
	format := llmprotocol.OpenAIChatV1
	if ctx != nil && ctx.SourceFormat != "" {
		format = ctx.SourceFormat
	}
	switch format {
	case llmprotocol.OpenAIResponsesV1:
		return clientFields{output: "max_output_tokens", input: "input"}
	case llmprotocol.AnthropicMessagesV1:
		return clientFields{output: "max_tokens", input: "messages"}
	}
	fields := clientFields{output: "max_completion_tokens", input: "messages"}
	if ctx != nil && ctx.ProtocolEnvelope.OutputLimitField != "" {
		fields.output = ctx.ProtocolEnvelope.OutputLimitField
	}
	return fields
}

func clientUpstreamError(protocolError *llmprotocol.ProtocolError, status int, fields clientFields) {
	backendParam := protocolError.Parameter
	message := upstreamNoneSuffix.ReplaceAllString(upstreamValidationSuffix.ReplaceAllString(protocolError.Message, ""), ".")
	outputLimit := upstreamOutputLimitError(status, protocolError)
	switch {
	case outputLimit || upstreamContextLengthError(status, protocolError):
		protocolError.Category = llmprotocol.ErrorInvalidRequest
		protocolError.Code = routingFailureContextLength.code
	case status == http.StatusBadRequest:
		protocolError.Category = llmprotocol.ErrorInvalidRequest
		protocolError.Code = "invalid_request"
	default:
		generic := immediateProtocolError(status)
		if _, numeric := strconv.Atoi(protocolError.Code); numeric == nil {
			protocolError.Code = generic.Code
		}
		if protocolError.Category == llmprotocol.ErrorUpstreamUnavailable || protocolError.Category == llmprotocol.ErrorInternal {
			protocolError.Category = generic.Category
		}
	}
	switch {
	case outputLimit || slices.Contains(upstreamOutputLimitFields, backendParam):
		protocolError.Parameter = fields.output
		if backendParam != "" && backendParam != fields.output {
			message = regexp.MustCompile(`\b`+regexp.QuoteMeta(backendParam)+`\b`).ReplaceAllString(message, fields.output)
		}
	case slices.Contains(upstreamInputFields, backendParam) ||
		backendParam == "" && protocolError.Code == routingFailureContextLength.code:
		protocolError.Parameter = fields.input
	}
	if outputLimit {
		if text, ok := outputLimitMessage(message, fields.output); ok {
			message = text
		}
	}
	protocolError.Message = upstreamWindowSetting.ReplaceAllString(message, "the model's maximum context length of $1 tokens")
}

// upstreamOutputLimitError reports a backend's refusal of an output limit
// its window cannot hold: an error about the output limit that names the
// window.
func upstreamOutputLimitError(status int, protocolError *llmprotocol.ProtocolError) bool {
	if protocolError == nil || status != http.StatusBadRequest {
		return false
	}
	message := strings.ToLower(protocolError.Message)
	if !slices.Contains(upstreamOutputLimitFields, protocolError.Parameter) &&
		!slices.ContainsFunc(upstreamOutputLimitFields, func(field string) bool { return strings.Contains(message, field) }) {
		return false
	}
	return upstreamWindowSetting.MatchString(message) || slices.ContainsFunc(upstreamContextLengthPhrases, func(phrase string) bool {
		return strings.Contains(message, phrase)
	})
}

// outputLimitMessage says what an output limit refusal means with its
// numbers, under the client's name for the limit.
func outputLimitMessage(message, field string) (string, bool) {
	requested, window := upstreamRequestedOutput.FindStringSubmatch(message), upstreamWindowSize.FindStringSubmatch(message)
	if requested == nil || window == nil {
		return "", false
	}
	text := fmt.Sprintf("%s is too large: %s. This model's maximum context length is %s tokens", field, requested[1], window[1])
	if input := upstreamInputTokens.FindStringSubmatch(message); input != nil {
		text += fmt.Sprintf(", and your request has %s input tokens", input[1])
	}
	return text + ". Please request fewer output tokens.", true
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
