// Package anthropic provides transformation functions between OpenAI and Anthropic API formats.
// Used for Envoy-routed requests where the router transforms request/response bodies.
package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// DefaultMaxTokens is the default max tokens if not specified in request
const DefaultMaxTokens int64 = 4096

// AnthropicAPIVersion is the version header required for Anthropic API
const AnthropicAPIVersion = "2023-06-01"

// AnthropicMessagesPath is the path for Anthropic messages API
const AnthropicMessagesPath = "/v1/messages"

// HeaderKeyValue represents a header key-value pair
type HeaderKeyValue struct {
	Key   string
	Value string
}

// BuildRequestHeaders returns the headers needed for an Anthropic API request.
// messagesPath should include any base_url prefix (e.g. /Anthropic/v1/messages for AMD).
// When empty, AnthropicMessagesPath (/v1/messages) is used. Equivalent to
// BuildRequestHeadersWithPassthrough with a nil passthrough.
func BuildRequestHeaders(apiKey string, bodyLength int, messagesPath string) []HeaderKeyValue {
	return buildRequestHeaders(apiKey, bodyLength, messagesPath, false, nil)
}

// BuildRequestHeadersWithPassthrough is the passthrough-aware form of
// BuildRequestHeaders. When pt.AnthropicVersion is non-empty it overrides the
// package default; when pt.AnthropicBeta is non-empty an anthropic-beta header
// is appended. Passing a nil passthrough is byte-identical to
// BuildRequestHeaders.
//
// Header precedence (lowest to highest) is documented for callers that
// combine this output with profile-supplied headers: package default <
// passthrough < profile pin. Callers that append profile headers AFTER this
// function's output (the convention in pkg/extproc) get this precedence for
// free: a later HeaderValueOption with the same key wins on the Envoy side.
func BuildRequestHeadersWithPassthrough(apiKey string, bodyLength int, messagesPath string, pt *AnthropicPassthrough) []HeaderKeyValue {
	return buildRequestHeaders(apiKey, bodyLength, messagesPath, false, pt)
}

func buildRequestHeaders(apiKey string, bodyLength int, messagesPath string, streaming bool, pt *AnthropicPassthrough) []HeaderKeyValue {
	if strings.TrimSpace(messagesPath) == "" {
		messagesPath = AnthropicMessagesPath
	}
	version := AnthropicAPIVersion
	if pt != nil && pt.AnthropicVersion != "" {
		version = pt.AnthropicVersion
	}
	headers := []HeaderKeyValue{
		{Key: "anthropic-version", Value: version},
		{Key: "content-type", Value: "application/json"},
		{Key: "accept-encoding", Value: "identity"},
		{Key: ":path", Value: messagesPath},
		{Key: "content-length", Value: fmt.Sprintf("%d", bodyLength)},
	}
	if streaming {
		headers = append(headers, HeaderKeyValue{Key: "accept", Value: "text/event-stream"})
	}
	if pt != nil && pt.AnthropicBeta != "" {
		headers = append(headers, HeaderKeyValue{Key: "anthropic-beta", Value: pt.AnthropicBeta})
	}
	if strings.TrimSpace(apiKey) != "" {
		headers = append([]HeaderKeyValue{{Key: "x-api-key", Value: apiKey}}, headers...)
	}
	return headers
}

// HeadersToRemove returns headers that should be removed when routing to Anthropic.
func HeadersToRemove() []string {
	return []string{"authorization", "content-length"}
}

// ToAnthropicRequestBody transforms an OpenAI-format request to Anthropic API format (JSON).
// This is used for Envoy-routed requests where the router transforms the body
// before forwarding to Anthropic via Envoy. Equivalent to calling
// ToAnthropicRequestBodyWithPassthrough with a nil passthrough.
func ToAnthropicRequestBody(openAIRequest *openai.ChatCompletionNewParams) ([]byte, error) {
	return ToAnthropicRequestBodyWithPassthrough(openAIRequest, nil)
}

// ToAnthropicRequestBodyWithPassthrough transforms an OpenAI-format request to
// Anthropic API format and replays Anthropic-only fields carried in pt.
//
// Passing a nil passthrough is byte-identical to ToAnthropicRequestBody —
// existing callers and behaviour are preserved. When pt is non-nil, fields
// without an OpenAI representation (cache_control markers, top_k,
// metadata.user_id, multi-block system prompts, image blocks, tool_result
// error/array content) are emitted on the outbound body.
func ToAnthropicRequestBodyWithPassthrough(openAIRequest *openai.ChatCompletionNewParams, pt *AnthropicPassthrough) ([]byte, error) {
	systemPrompt, messages, err := buildAnthropicMessages(openAIRequest.Messages)
	if err != nil {
		return nil, err
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(openAIRequest.Model),
		Messages:  messages,
		MaxTokens: resolveMaxTokens(openAIRequest),
		System:    buildSystem(systemPrompt, pt),
	}

	if err := applyOpenAIToolsToAnthropicParams(&params, openAIRequest); err != nil {
		return nil, err
	}
	applyToolsCacheControl(&params, pt)
	applyMessagesCacheControl(params.Messages, pt)
	applyUserMessageImageBlocks(params.Messages, pt)
	applyToolResultPassthrough(params.Messages, pt)
	applySampling(&params, openAIRequest, pt)
	applyMetadata(&params, openAIRequest, pt)

	return json.Marshal(params)
}

// ToOpenAIResponseBody transforms an Anthropic API response to OpenAI format.
// This is used for Envoy-routed requests where the router transforms the response
// after receiving it from Anthropic via Envoy.
//
// Behavior contract for the legacy inverse cell (OpenAI client, Anthropic
// backend) is preserved: thinking, citations, and server_tool_use blocks
// are silently dropped because they have no OpenAI representation. The
// internal helpers accept an optional *ir.IRExtensions so the symmetric
// outbound emitter (EmitAnthropicResponse) can capture the dropped state
// for replay on the Anthropic→Anthropic round-trip cell.
func ToOpenAIResponseBody(anthropicResponse []byte, model string) ([]byte, error) {
	return toOpenAIResponseBody(anthropicResponse, model, nil)
}

// ToOpenAIResponseBodyWithExt is the IRExtensions-aware form of
// ToOpenAIResponseBody. When ext is non-nil the parser stashes the
// Anthropic-only stop_reason, cache usage counters, server-tool counts,
// and thinking-block signatures onto it so the symmetric outbound
// emitter (EmitAnthropicResponse) can replay them on the response body.
// Pass nil ext for byte-identical behavior with the original entrypoint.
func ToOpenAIResponseBodyWithExt(anthropicResponse []byte, model string, ext *ir.IRExtensions) ([]byte, error) {
	return toOpenAIResponseBody(anthropicResponse, model, ext)
}

// toOpenAIResponseBody is the internal form that exposes the
// IRExtensions side channel. It is called by ToOpenAIResponseBody with a
// nil ext to preserve the existing inverse cell byte-for-byte; the
// Anthropic→Anthropic outbound cell calls it with a non-nil ext so the
// stop_reason, cache usage, and server-tool counts survive the
// flattening that the OpenAI envelope cannot represent.
func toOpenAIResponseBody(anthropicResponse []byte, model string, ext *ir.IRExtensions) ([]byte, error) {
	logging.Debugf("Raw Anthropic response (%d bytes): %s", len(anthropicResponse), string(anthropicResponse))

	var resp anthropic.Message
	if err := json.Unmarshal(anthropicResponse, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse Anthropic response: %w", err)
	}

	logging.Debugf("Parsed Anthropic response - ID: %s, Content blocks: %d, StopReason: %s", resp.ID, len(resp.Content), resp.StopReason)

	toolCalls, content := openAIToolCallsFromAnthropicContent(resp.Content)
	extractThinkingBlocks(resp.Content, ext)
	captureAnthropicStopReasonIntoExt(resp.StopReason, resp.StopSequence, ext)
	extractAnthropicUsageIntoExt(resp.Usage, ext)

	finishReason := openAIFinishReasonFromAnthropic(resp.StopReason, len(toolCalls))

	openAIResp := &openai.ChatCompletion{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []openai.ChatCompletionChoice{{
			Index: 0,
			Message: openai.ChatCompletionMessage{
				Role:      "assistant",
				Content:   content,
				ToolCalls: toolCalls,
			},
			FinishReason: finishReason,
		}},
		Usage: openai.CompletionUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}

	return json.Marshal(openAIResp)
}

// openAIFinishReasonFromAnthropic maps Anthropic's StopReason alphabet
// onto the smaller OpenAI finish_reason set. Anthropic-only values
// ("pause_turn", "refusal") collapse to "stop" because the inverse
// cell cannot surface them in the OpenAI envelope; the lossy capture
// happens in captureAnthropicStopReasonIntoExt for callers that wired
// IRExtensions.
func openAIFinishReasonFromAnthropic(stopReason anthropic.StopReason, toolCallCount int) string {
	finishReason := "stop"
	switch stopReason {
	case anthropic.StopReasonMaxTokens:
		finishReason = "length"
	case anthropic.StopReasonToolUse:
		finishReason = "tool_calls"
	}
	if toolCallCount > 0 && finishReason == "stop" {
		finishReason = "tool_calls"
	}
	return finishReason
}

// captureAnthropicStopReasonIntoExt records Anthropic-only stop reasons
// ("pause_turn", "refusal", "stop_sequence") on the IRExtensions side
// channel so the symmetric outbound emitter can override the
// OpenAI-derived finish_reason mapping. The OpenAI finish_reason
// alphabet collapses all three to "stop", losing the distinction; the
// emitter restores it from this side channel. StopSequence is recorded
// only when the upstream stop reason was "stop_sequence" (the only case
// it is meaningful). Nil-safe.
func captureAnthropicStopReasonIntoExt(
	stopReason anthropic.StopReason,
	stopSequence string,
	ext *ir.IRExtensions,
) {
	if ext == nil {
		return
	}
	switch stopReason {
	case anthropic.StopReasonPauseTurn, anthropic.StopReasonRefusal, anthropic.StopReasonStopSequence:
		ext.AnthropicStopReason = string(stopReason)
	}
	if stopReason == anthropic.StopReasonStopSequence && stopSequence != "" {
		ext.AnthropicStopSequence = stopSequence
	}
}

// extractThinkingBlocks records signatures from any "thinking" content
// blocks into IRExtensions so the Anthropic outbound emitter can
// reconstruct the multi-turn extended-thinking continuity that the
// OpenAI envelope cannot represent. Nil-safe; legacy callers see no
// behavior change because the OpenAI envelope itself never carried
// these blocks.
func extractThinkingBlocks(blocks []anthropic.ContentBlockUnion, ext *ir.IRExtensions) {
	if ext == nil {
		return
	}
	thinkingIndex := 0
	for _, block := range blocks {
		if block.Type != "thinking" {
			continue
		}
		blockID := fmt.Sprintf("content[%d]", thinkingIndex)
		thinkingIndex++
		if block.Signature != "" {
			ext.SetThinkingSignature(blockID, block.Signature)
		}
	}
}

// extractAnthropicUsageIntoExt copies the Anthropic-native usage
// counters (cache_*, server_tool_use) onto IRExtensions so the
// symmetric outbound emitter can replay them on the response body.
// OpenAI's CompletionUsage envelope has no slot for cache or
// server-tool counts, so without this side channel those values are
// dropped on the floor. Nil-safe.
func extractAnthropicUsageIntoExt(u anthropic.Usage, ext *ir.IRExtensions) {
	if ext == nil {
		return
	}
	ext.CacheReadInputTokens = u.CacheReadInputTokens
	ext.CacheCreationInputTokens = u.CacheCreationInputTokens
	ext.Ephemeral5mInputTokens = u.CacheCreation.Ephemeral5mInputTokens
	ext.Ephemeral1hInputTokens = u.CacheCreation.Ephemeral1hInputTokens
	if u.ServerToolUse.WebSearchRequests > 0 {
		ext.SetServerToolUseCount("web_search", u.ServerToolUse.WebSearchRequests)
	}
}

func openAIToolCallsFromAnthropicContent(
	blocks []anthropic.ContentBlockUnion,
) ([]openai.ChatCompletionMessageToolCall, string) {
	var contentParts []string
	var toolCalls []openai.ChatCompletionMessageToolCall

	for _, block := range blocks {
		switch block.Type {
		case "text":
			if block.Text != "" {
				contentParts = append(contentParts, block.Text)
			}
		case "tool_use":
			arguments := string(block.Input)
			if len(block.Input) == 0 {
				arguments = "{}"
			}
			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCall{
				ID:   block.ID,
				Type: "function",
				Function: openai.ChatCompletionMessageToolCallFunction{
					Name:      block.Name,
					Arguments: arguments,
				},
			})
		default:
			logging.Debugf("Skipping unsupported Anthropic content block type %q in OpenAI response conversion", block.Type)
		}
	}

	return toolCalls, strings.Join(contentParts, "")
}
