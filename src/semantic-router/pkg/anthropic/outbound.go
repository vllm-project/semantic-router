// Anthropic outbound emitter: non-streaming response shape.
//
// EmitAnthropicResponse is the symmetric counterpart to
// ToOpenAIResponseBody (in client.go). It rewrites the OpenAI
// ChatCompletion envelope that vsr's pipeline normalizes to back into
// a wire-shaped anthropic.Message JSON body so that an Anthropic-SDK
// client (anthropic-sdk-go, anthropic-python, Claude Code SDK) can
// unmarshal it without translation.
//
// # cache_control vs cache usage
//
// The Anthropic spec defines cache_control as a request-side field
// only: it tells the backend which prompt segments to cache. The
// response carries usage counters (cache_creation_input_tokens,
// cache_read_input_tokens, cache_creation.ephemeral_5m_input_tokens,
// cache_creation.ephemeral_1h_input_tokens) that report what the
// cache actually did — not cache_control markers. This emitter
// therefore never re-attaches cache_control markers to response
// content blocks. The request-side markers captured in
// IRExtensions.CacheControl are preserved for plugin replay (e.g.
// router replay re-rendering the request) but have no echo on the
// response. The cache *usage* counters round-trip through
// IRExtensions.CacheRead/CreationInputTokens etc., populated by the
// inverse helpers in client.go when the upstream was Anthropic.
//
// # Thinking blocks
//
// PR #1718 upstream is in flight to add reasoning_content extraction
// from the OpenAI envelope so that on the Anthropic→Anthropic cell the
// emitter can surface the actual model-produced thinking text. Until
// that lands, the emitter uses a placeholder when
// IRExtensions.ThinkingSignatures contains entries but the OpenAI body
// has no reconstructable text. The structural shape (thinking blocks
// precede text/tool_use blocks; each carries its opaque signature) is
// correct; only the user-visible text is empty.
// TODO(upstream-1718): replace the placeholder with reasoning_content
// extraction once https://github.com/vllm-project/semantic-router/pull/1718
// lands.
package anthropic

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
)

// thinkingTextPlaceholder is emitted when IRExtensions carries a
// thinking signature but no reasoning_content has been surfaced into
// the OpenAI envelope by an upstream extractor (PR #1718 territory).
// Empty string is invalid per the Anthropic spec (the field is
// required), so a short placeholder keeps the response structurally
// valid while signalling to the client that the structured thinking
// text was not preserved through the OpenAI normalization step.
const thinkingTextPlaceholder = "[thinking content not preserved through OpenAI normalization]"

// anthropicErrorTypeAPI is the catch-all error_type token used when no
// more specific Anthropic-defined type fits. See
// https://platform.claude.com/docs/en/api/messages#errors.
const anthropicErrorTypeAPI = "api_error"

// anthropicMessageResponse mirrors the wire shape of anthropic.Message
// for marshalling. Using a dedicated struct keeps the emitter's wire
// output independent of the SDK type's required-field strictness (the
// SDK type has required fields that fail to marshal cleanly when zero
// values would be valid on the wire, e.g. ServerToolUse.ServiceTier).
type anthropicMessageResponse struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Model        string                  `json:"model"`
	Content      []anthropicContentBlock `json:"content"`
	StopReason   anthropic.StopReason    `json:"stop_reason"`
	StopSequence *string                 `json:"stop_sequence"`
	Usage        anthropicUsageResponse  `json:"usage"`
}

// anthropicContentBlock is a single union shape covering all
// emitter-supported content-block types. Only the fields relevant to
// the block's Type are populated; the rest are omitted via
// omitempty/omitzero so the wire output matches the Anthropic spec.
type anthropicContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	Signature string          `json:"signature,omitempty"`
}

// anthropicUsageResponse mirrors the wire shape of anthropic.Usage
// with the cache_creation and server_tool_use sub-objects elided when
// they are entirely zero. Per the Anthropic spec these objects are
// optional on the wire; including them when empty inflates response
// size and confuses clients that test object presence.
type anthropicUsageResponse struct {
	InputTokens              int64                          `json:"input_tokens"`
	OutputTokens             int64                          `json:"output_tokens"`
	CacheCreationInputTokens int64                          `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64                          `json:"cache_read_input_tokens"`
	CacheCreation            *anthropicCacheCreationUsage   `json:"cache_creation,omitempty"`
	ServerToolUse            *anthropicServerToolUsageBlock `json:"server_tool_use,omitempty"`
}

type anthropicCacheCreationUsage struct {
	Ephemeral5mInputTokens int64 `json:"ephemeral_5m_input_tokens"`
	Ephemeral1hInputTokens int64 `json:"ephemeral_1h_input_tokens"`
}

type anthropicServerToolUsageBlock struct {
	WebSearchRequests int64 `json:"web_search_requests"`
}

// anthropicErrorEnvelope is the wire shape Anthropic uses for all
// error responses on POST /v1/messages.
type anthropicErrorEnvelope struct {
	Type  string               `json:"type"`
	Error anthropicErrorDetail `json:"error"`
}

type anthropicErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// EmitAnthropicResponse rewrites an OpenAI ChatCompletion response
// body into the Anthropic Message wire shape so an Anthropic-SDK
// client can consume it directly.
//
// Contract:
//   - Returns (nil, err) only when responseBody is not valid OpenAI
//     ChatCompletion JSON.
//   - Tolerates a nil ext (zero usage; no warnings appended) so that
//     defensive callers do not need to guard.
//   - Always emits a non-null content array (possibly empty when the
//     model produced no text and no tool calls).
//   - When IRExtensions.ThinkingSignatures contains entries, the
//     corresponding thinking blocks precede text/tool_use blocks per
//     the Anthropic spec.
//   - When IRExtensions.AnthropicStopReason is non-empty (Anthropic-only
//     stop reasons "pause_turn" or "refusal"), it overrides the
//     OpenAI-derived mapping.
func EmitAnthropicResponse(responseBody []byte, ext *ir.IRExtensions, model string) ([]byte, error) {
	var oa openai.ChatCompletion
	if err := json.Unmarshal(responseBody, &oa); err != nil {
		return nil, fmt.Errorf("anthropic outbound: parse OpenAI response: %w", err)
	}

	choice := openai.ChatCompletionChoice{}
	if len(oa.Choices) > 0 {
		choice = oa.Choices[0]
	}

	stopReason, stopSequence := mapOpenAIFinishReasonToAnthropic(choice.FinishReason, ext)
	content := buildAnthropicContent(choice, ext)
	usage := buildAnthropicUsage(oa.Usage, ext)

	resp := anthropicMessageResponse{
		ID:           oa.ID,
		Type:         "message",
		Role:         "assistant",
		Model:        model,
		Content:      content,
		StopReason:   stopReason,
		StopSequence: stopSequence,
		Usage:        usage,
	}

	out, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("anthropic outbound: marshal response: %w", err)
	}
	return out, nil
}

// EmitAnthropicError builds an Anthropic error-envelope body. Pure
// helper — marshalling cannot fail for this fixed shape, so no error
// return.
//
// errorType should be one of the Anthropic-defined tokens
// ("invalid_request_error", "authentication_error", "permission_error",
// "not_found_error", "request_too_large", "rate_limit_error",
// "api_error", "timeout_error", "overloaded_error"). An empty or
// unknown errorType is coerced to "api_error" to keep the response
// structurally valid.
func EmitAnthropicError(errorType, message string) []byte {
	if errorType == "" {
		errorType = anthropicErrorTypeAPI
	}
	envelope := anthropicErrorEnvelope{
		Type: "error",
		Error: anthropicErrorDetail{
			Type:    errorType,
			Message: message,
		},
	}
	// Marshal cannot fail for this fixed struct shape.
	out, _ := json.Marshal(envelope)
	return out
}

// buildAnthropicContent walks the OpenAI choice's Message.Content,
// Message.Refusal, Message.ToolCalls and any thinking signatures from
// ext to produce the Anthropic content[] array in document order:
// thinking blocks first (spec requirement), then text, then tool_use
// blocks. Refusal text is surfaced as a text block; the refusal
// stop_reason is set in mapOpenAIFinishReasonToAnthropic.
func buildAnthropicContent(choice openai.ChatCompletionChoice, ext *ir.IRExtensions) []anthropicContentBlock {
	blocks := make([]anthropicContentBlock, 0)

	// Thinking blocks precede text per Anthropic spec.
	if ext != nil {
		for _, blockID := range orderedThinkingBlockIDs(ext) {
			signature := ext.ThinkingSignatures[blockID]
			blocks = append(blocks, anthropicContentBlock{
				Type:      "thinking",
				Thinking:  thinkingTextPlaceholder,
				Signature: signature,
			})
		}
	}

	// Text block from either Message.Content or Message.Refusal.
	// Refusal takes precedence: if both are populated the model
	// produced a structured refusal and the content field is just an
	// echo (vsr-internal jailbreak/hallucination synthesis surfaces
	// the refusal text in Message.Refusal, not Message.Content).
	textPayload := choice.Message.Content
	if choice.Message.Refusal != "" {
		textPayload = choice.Message.Refusal
	}
	if textPayload != "" {
		blocks = append(blocks, anthropicContentBlock{
			Type: "text",
			Text: textPayload,
		})
	}

	// Tool calls.
	for _, tc := range choice.Message.ToolCalls {
		input := json.RawMessage(tc.Function.Arguments)
		if len(input) == 0 {
			input = json.RawMessage("{}")
		}
		blocks = append(blocks, anthropicContentBlock{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: input,
		})
	}

	return blocks
}

// orderedThinkingBlockIDs returns thinking block IDs from ext in a
// deterministic order (sorted lexicographically). Map iteration is
// non-deterministic in Go; stability matters because the wire output
// must be reproducible for round-trip tests and so clients keying on
// block index see a consistent layout.
func orderedThinkingBlockIDs(ext *ir.IRExtensions) []string {
	if ext == nil || len(ext.ThinkingSignatures) == 0 {
		return nil
	}
	ids := make([]string, 0, len(ext.ThinkingSignatures))
	for id := range ext.ThinkingSignatures {
		ids = append(ids, id)
	}
	// Lexicographic sort matches the "content[0]", "content[1]"
	// emission order used by the inverse extractor in client.go.
	sort.Strings(ids)
	return ids
}

// buildAnthropicUsage maps OpenAI completion usage onto the Anthropic
// usage envelope, layering in the cache and server-tool counters from
// ext when they were captured by the inverse helpers. When the
// upstream was an OpenAI backend (no cache, no server tools), ext's
// counters are all zero and the cache_creation / server_tool_use
// sub-objects are omitted from the wire output. A
// ReasonCacheFieldsAbsent info warning is appended when an Anthropic-
// source response has all-zero cache counters, so observability can
// distinguish a legitimately uncached response from a translation gap.
func buildAnthropicUsage(usage openai.CompletionUsage, ext *ir.IRExtensions) anthropicUsageResponse {
	out := anthropicUsageResponse{
		InputTokens:  usage.PromptTokens,
		OutputTokens: usage.CompletionTokens,
	}
	if ext == nil {
		return out
	}

	out.CacheReadInputTokens = ext.CacheReadInputTokens
	out.CacheCreationInputTokens = ext.CacheCreationInputTokens
	if ext.Ephemeral5mInputTokens > 0 || ext.Ephemeral1hInputTokens > 0 {
		out.CacheCreation = &anthropicCacheCreationUsage{
			Ephemeral5mInputTokens: ext.Ephemeral5mInputTokens,
			Ephemeral1hInputTokens: ext.Ephemeral1hInputTokens,
		}
	}
	if count, ok := ext.ServerToolUseCounts["web_search"]; ok && count > 0 {
		out.ServerToolUse = &anthropicServerToolUsageBlock{
			WebSearchRequests: count,
		}
	}

	if ext.SourceProtocol == SourceProtocolAnthropic &&
		ext.CacheReadInputTokens == 0 &&
		ext.CacheCreationInputTokens == 0 &&
		ext.Ephemeral5mInputTokens == 0 &&
		ext.Ephemeral1hInputTokens == 0 {
		ext.AppendWarning(ir.Warning{
			Field:    "usage.cache",
			Reason:   ir.ReasonCacheFieldsAbsent,
			Severity: ir.WarningSeverityInfo,
			Detail:   "Anthropic client received response with no cache fields populated",
		})
	}

	return out
}

// mapOpenAIFinishReasonToAnthropic maps the OpenAI finish_reason
// alphabet onto the Anthropic stop_reason vocabulary. When ext carries
// an Anthropic-only stop reason ("pause_turn", "refusal") captured on
// the round-trip, that override wins.
//
// The returned stop_sequence is non-nil only for stop_reason ==
// "stop_sequence". OpenAI does not surface which sequence matched, so
// the value comes from ext.AnthropicStopSequence when available
// (round-trip from an Anthropic upstream) or is nil otherwise.
func mapOpenAIFinishReasonToAnthropic(finish string, ext *ir.IRExtensions) (anthropic.StopReason, *string) {
	if ext != nil && ext.AnthropicStopReason != "" {
		stopReason := anthropic.StopReason(ext.AnthropicStopReason)
		if stopReason == anthropic.StopReasonStopSequence && ext.AnthropicStopSequence != "" {
			seq := ext.AnthropicStopSequence
			return stopReason, &seq
		}
		return stopReason, nil
	}

	switch finish {
	case "stop":
		return anthropic.StopReasonEndTurn, nil
	case "length":
		return anthropic.StopReasonMaxTokens, nil
	case "tool_calls":
		return anthropic.StopReasonToolUse, nil
	case "content_filter":
		return anthropic.StopReasonRefusal, nil
	case "":
		// Empty finish_reason can appear on synthesized error
		// responses or streaming-aborted bodies; default to
		// end_turn but flag for observability so unexpected
		// upstream behavior is detectable.
		ext.AppendWarning(ir.Warning{
			Field:    "finish_reason",
			Reason:   ir.ReasonAnthropicStopReasonCoerced,
			Severity: ir.WarningSeverityInfo,
			Detail:   "empty OpenAI finish_reason coerced to end_turn",
		})
		return anthropic.StopReasonEndTurn, nil
	default:
		// Unknown finish_reason — coerce to end_turn and warn.
		ext.AppendWarning(ir.Warning{
			Field:    "finish_reason",
			Reason:   ir.ReasonAnthropicStopReasonCoerced,
			Severity: ir.WarningSeverityInfo,
			Detail:   fmt.Sprintf("unknown OpenAI finish_reason %q coerced to end_turn", finish),
		})
		return anthropic.StopReasonEndTurn, nil
	}
}
