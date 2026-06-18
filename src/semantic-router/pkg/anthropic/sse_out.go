// Anthropic outbound emitter: streaming response shape.
//
// EmitAnthropicSSEChunk is the symmetric counterpart to
// TransformSSEChunkToOpenAI in client_stream.go. It walks the OpenAI
// chat.completion.chunk SSE stream that vsr's response pipeline
// produces and re-emits each delta as the corresponding Anthropic SSE
// event sequence — message_start, content_block_start,
// content_block_delta (text_delta / input_json_delta / thinking_delta /
// signature_delta), content_block_stop, message_delta, message_stop —
// so an Anthropic-SDK client can consume the byte stream without
// translation.
//
// # State machine
//
// The emitter is driven by the StreamState struct in client_stream.go.
// Outbound bookkeeping tracks which Anthropic content_block index is
// currently open (OpenTextBlockIndex / OpenThinkingBlockIdx /
// ToolIdxToBlockIndex), whether message_start / message_stop have been
// emitted, and when the last chunk arrived (for the ping ticker
// elsewhere). Anthropic strictness requires that every
// content_block_start is paired with a content_block_stop before
// message_delta, and that two distinct block types never share the same
// block index — the open-block pointers enforce both invariants.
//
// # Tool_use accumulation
//
// OpenAI tool-call deltas arrive as repeated entries for the same
// tool_calls[i] index, with id/name on the first delta and
// function.arguments fragments on subsequent deltas. The first delta
// for index i opens a new Anthropic content_block_start with type
// tool_use, id, name, and input: {}. Subsequent argument fragments
// stream as content_block_delta events of type input_json_delta. The
// Anthropic SDK accumulator concatenates partial_json fragments and
// overwrites the {} placeholder at content_block_stop.
//
// # Thinking blocks
//
// Thinking content reaches this emitter as delta.reasoning_content on
// the OpenAI envelope (PR #1718 convention; the streaming bug-fix
// commit in this stack populates that field from upstream
// thinking_delta events). Signatures are stashed in
// IRExtensions.ThinkingSignatures by the inbound translator and
// replayed here as content_block_delta of type signature_delta keyed
// by Anthropic block index.
package anthropic

import (
	"bytes"
	"encoding/json"
	"sort"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
)

// EmitAnthropicSSEChunk converts one OpenAI chat.completion.chunk SSE
// payload (the bytes Envoy hands to the response-body callback) into
// the corresponding Anthropic SSE event sequence.
//
// Contract:
//   - state must be the same per-request *StreamState across calls.
//   - Returns (sseBytes, streamDone, err). streamDone is true after the
//     terminal message_stop event has been emitted.
//   - Tolerates a nil ext (zero cache/server-tool counters; no
//     Anthropic-only stop reasons; no thinking signatures).
//   - The OpenAI "data: [DONE]" marker is consumed and never echoed
//     onto the wire — message_stop is its Anthropic equivalent and is
//     emitted in response to the prior chunk's finish_reason.
//   - Never emits a content_block_start without a paired
//     content_block_stop before the terminal message_delta.
func EmitAnthropicSSEChunk(
	openAIChunk []byte,
	state *StreamState,
	ext *ir.IRExtensions,
	model string,
) ([]byte, bool, error) {
	if state == nil {
		state = NewStreamState()
	}

	var out bytes.Buffer
	streamDone := false
	for _, line := range extractSSEDataLines(openAIChunk) {
		// Inbound error event surfaces here too because the OpenAI
		// envelope inherits the upstream error flag from
		// extractSSEDataLines. Emit a spec-compliant Anthropic error
		// event followed by a terminal message_stop so the client
		// observes a clean stream end. Guard on MessageStopSent so a
		// second error chunk (or an error chunk after a normal terminal)
		// does not produce a duplicate message_stop.
		if line.IsError {
			if state.MessageStopSent {
				streamDone = true
				continue
			}
			out.Write(emitAnthropicError("api_error", string(line.Data)))
			out.Write(emitMessageStop())
			state.MessageStopSent = true
			streamDone = true
			continue
		}

		var chunk openai.ChatCompletionChunk
		if err := json.Unmarshal(line.Data, &chunk); err != nil {
			// Skip malformed chunks rather than aborting the stream;
			// the SSE format already allows non-data lines.
			continue
		}

		events, done, err := emitChunkEvents(chunk, state, ext, model)
		if err != nil {
			return nil, false, err
		}
		out.Write(events)
		if done {
			streamDone = true
		}
	}
	return out.Bytes(), streamDone, nil
}

// emitChunkEvents handles one parsed OpenAI chunk by driving the
// outbound state machine described at the top of this file.
func emitChunkEvents(
	chunk openai.ChatCompletionChunk,
	state *StreamState,
	ext *ir.IRExtensions,
	model string,
) ([]byte, bool, error) {
	// message_stop must fire exactly once. If a backend legally splits
	// finish_reason and usage across two chunks, the second chunk also
	// satisfies isTerminalChunk and would call emitTerminalEvents again.
	// Return early so the caller sees (nil, true, nil) rather than a
	// second message_stop on the wire.
	if state.MessageStopSent {
		return nil, true, nil
	}

	var out bytes.Buffer

	if !state.MessageStartSent {
		// Capture the upstream message ID if present so the emitted
		// message_start carries an identifier the client can use for
		// correlation.
		if state.MessageID == "" && chunk.ID != "" {
			state.MessageID = chunk.ID
		}
		out.Write(emitMessageStart(state, model))
		state.MessageStartSent = true
	}

	if len(chunk.Choices) == 0 {
		// Usage-only chunk (no choices). Whether it is terminal depends on
		// its position in the stream:
		//
		//   - AFTER content (state.contentStarted()): this is the stream's
		//     final usage summary. Some OpenAI backends end the stream with
		//     a usage-only chunk instead of a finish_reason chunk, so treat
		//     it as terminal — otherwise an Anthropic client hangs waiting
		//     for an end event. With no choices there is nothing else to emit.
		//
		//   - BEFORE any content (issue #2215): this is a gateway "preview"
		//     usage chunk (e.g. {"choices":[],"usage":{"completion_tokens":4}})
		//     sent at stream start. The real content and the real terminal
		//     chunk (carrying the true usage, e.g. completion_tokens:252)
		//     follow. Treating the preview as terminal would emit message_stop
		//     after the first chunk, truncate the response, and record the
		//     preview's token counts. Skip it and wait for the real terminal
		//     signal.
		//
		// The common split-finish/usage path (finish_reason chunk first, then
		// a trailing usage-only chunk) is handled by the MessageStopSent guard
		// above, not here.
		if chunkHasUsage(chunk) && state.contentStarted() {
			out.Write(emitTerminalEvents(state, "", chunk.Usage, ext))
			return out.Bytes(), true, nil
		}
		return out.Bytes(), false, nil
	}
	choice := chunk.Choices[0]
	delta := choice.Delta

	// Reasoning content opens or extends a thinking block. Always
	// processed first so it precedes text/tool blocks at the same
	// chunk's wire position (Anthropic spec orders thinking before
	// text/tool_use).
	if reasoning := openAIDeltaReasoning(delta); reasoning != "" {
		out.Write(emitThinkingDelta(state, reasoning))
	}

	if delta.Content != "" {
		out.Write(emitTextDelta(state, delta.Content))
	}

	for _, tc := range delta.ToolCalls {
		out.Write(emitToolCallDelta(state, tc))
	}

	out.Write(replayPendingSignature(state, ext))

	if isTerminalChunk(choice.FinishReason, chunk) {
		out.Write(emitTerminalEvents(state, choice.FinishReason, chunk.Usage, ext))
		return out.Bytes(), true, nil
	}

	return out.Bytes(), false, nil
}

// replayPendingSignature emits a signature_delta if a signature for
// the currently-open thinking block was captured by the inbound
// translator. Reuses the same thinkingBlockKey helper as the inbound
// side so a streaming round-trip preserves the per-block signature
// without double-storage. The signature is consumed (removed from ext)
// after replay so subsequent chunks for the same block do not re-emit.
func replayPendingSignature(state *StreamState, ext *ir.IRExtensions) []byte {
	if ext == nil || state.OpenThinkingBlockIdx == nil {
		return nil
	}
	key := thinkingBlockKey(*state.OpenThinkingBlockIdx)
	sig, ok := ext.ThinkingSignatures[key]
	if !ok || sig == "" {
		return nil
	}
	delete(ext.ThinkingSignatures, key)
	return emitSignatureDelta(state, sig)
}

// isTerminalChunk reports whether an OpenAI chunk should trigger the
// terminal Anthropic event sequence (close blocks, message_delta,
// message_stop). OpenAI providers may surface finish_reason and usage
// on the same chunk or on separate chunks; either signal terminates.
func isTerminalChunk(finishReason string, chunk openai.ChatCompletionChunk) bool {
	return finishReason != "" || chunkHasUsage(chunk)
}

// emitTerminalEvents closes any open content blocks then emits the
// terminal message_delta + message_stop pair. Marks the state as
// done so callers can stop the ping ticker.
func emitTerminalEvents(state *StreamState, finishReason string, usage openai.CompletionUsage, ext *ir.IRExtensions) []byte {
	var buf bytes.Buffer
	buf.Write(closeAllOpenBlocks(state))
	buf.Write(emitMessageDelta(finishReason, usage, ext))
	buf.Write(emitMessageStop())
	state.MessageStopSent = true
	return buf.Bytes()
}

// emitMessageStart returns the bytes for an Anthropic message_start
// event carrying an empty content array and a placeholder usage shape.
// The cumulative message_delta.usage event at end-of-stream supplies
// the real counts.
func emitMessageStart(state *StreamState, model string) []byte {
	usage := state.InitialUsage
	// Per Anthropic spec the initial usage carries input_tokens (known
	// up front) plus output_tokens: 1 as a placeholder so clients can
	// initialize accumulators.
	if usage.OutputTokens == 0 {
		usage.OutputTokens = 1
	}
	payload := map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            state.MessageID,
			"type":          "message",
			"role":          "assistant",
			"model":         model,
			"content":       []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage":         usage,
		},
	}
	data, _ := json.Marshal(payload)
	return formatAnthropicSSEEvent("message_start", data)
}

// emitTextDelta opens a text block if none is currently open and emits
// a content_block_delta carrying the text. Any open thinking or tool
// block is closed first to maintain Anthropic's invariant that one
// block type owns each block index.
func emitTextDelta(state *StreamState, text string) []byte {
	var buf bytes.Buffer
	if state.OpenTextBlockIndex == nil {
		buf.Write(closeNonTextBlocks(state))
		idx := state.NextBlockIndex
		state.NextBlockIndex++
		state.OpenTextBlockIndex = &idx
		buf.Write(emitContentBlockStartText(idx))
	}
	buf.Write(emitContentBlockDeltaText(*state.OpenTextBlockIndex, text))
	return buf.Bytes()
}

// emitThinkingDelta opens a thinking block if none is currently open
// and emits a content_block_delta carrying the reasoning text. Any
// open text or tool block is closed first.
func emitThinkingDelta(state *StreamState, reasoning string) []byte {
	var buf bytes.Buffer
	if state.OpenThinkingBlockIdx == nil {
		buf.Write(closeNonThinkingBlocks(state))
		idx := state.NextBlockIndex
		state.NextBlockIndex++
		state.OpenThinkingBlockIdx = &idx
		buf.Write(emitContentBlockStartThinking(idx))
	}
	buf.Write(emitContentBlockDeltaThinking(*state.OpenThinkingBlockIdx, reasoning))
	return buf.Bytes()
}

// emitSignatureDelta emits a signature_delta for the currently-open
// thinking block, then closes that block. Anthropic clients expect the
// signature as the final delta of a thinking block.
func emitSignatureDelta(state *StreamState, signature string) []byte {
	if state.OpenThinkingBlockIdx == nil {
		return nil
	}
	var buf bytes.Buffer
	buf.Write(emitContentBlockDeltaSignature(*state.OpenThinkingBlockIdx, signature))
	buf.Write(emitContentBlockStop(*state.OpenThinkingBlockIdx))
	state.OpenThinkingBlockIdx = nil
	return buf.Bytes()
}

// emitToolCallDelta opens a tool_use block on the first delta for a
// given tool index and emits input_json_delta for any argument
// fragment. The block stays open until closeAllOpenBlocks is called at
// stream termination.
func emitToolCallDelta(state *StreamState, tc openai.ChatCompletionChunkChoiceDeltaToolCall) []byte {
	toolIdx := int(tc.Index)
	var buf bytes.Buffer
	blockIdx, seen := state.ToolIdxToBlockIndex[toolIdx]
	if !seen {
		buf.Write(closeNonToolBlocks(state))
		blockIdx = state.NextBlockIndex
		state.NextBlockIndex++
		state.ToolIdxToBlockIndex[toolIdx] = blockIdx
		buf.Write(emitContentBlockStartToolUse(blockIdx, tc.ID, tc.Function.Name))
	}
	if tc.Function.Arguments != "" {
		buf.Write(emitContentBlockDeltaInputJSON(blockIdx, tc.Function.Arguments))
	}
	return buf.Bytes()
}

// closeAllOpenBlocks emits content_block_stop for every block currently
// tracked as open and clears the bookkeeping. Called once at stream
// termination before message_delta / message_stop.
//
// Tool blocks are closed in ascending block-index order to produce a
// deterministic event sequence for streams with multiple concurrent
// tool calls. Go map iteration order is non-deterministic; sorting by
// block index (not tool index) matches Anthropic's content[] position.
func closeAllOpenBlocks(state *StreamState) []byte {
	var buf bytes.Buffer
	if state.OpenTextBlockIndex != nil {
		buf.Write(emitContentBlockStop(*state.OpenTextBlockIndex))
		state.OpenTextBlockIndex = nil
	}
	if state.OpenThinkingBlockIdx != nil {
		buf.Write(emitContentBlockStop(*state.OpenThinkingBlockIdx))
		state.OpenThinkingBlockIdx = nil
	}
	// Sort tool indices by their assigned block index so the wire order
	// is stable regardless of Go's map iteration randomness.
	toolIdxs := make([]int, 0, len(state.ToolIdxToBlockIndex))
	for toolIdx := range state.ToolIdxToBlockIndex {
		toolIdxs = append(toolIdxs, toolIdx)
	}
	sort.Slice(toolIdxs, func(i, j int) bool {
		return state.ToolIdxToBlockIndex[toolIdxs[i]] < state.ToolIdxToBlockIndex[toolIdxs[j]]
	})
	for _, toolIdx := range toolIdxs {
		buf.Write(emitContentBlockStop(state.ToolIdxToBlockIndex[toolIdx]))
		delete(state.ToolIdxToBlockIndex, toolIdx)
	}
	return buf.Bytes()
}

// closeNonTextBlocks closes any open thinking or tool blocks before
// opening a new text block.
func closeNonTextBlocks(state *StreamState) []byte {
	var buf bytes.Buffer
	if state.OpenThinkingBlockIdx != nil {
		buf.Write(emitContentBlockStop(*state.OpenThinkingBlockIdx))
		state.OpenThinkingBlockIdx = nil
	}
	for toolIdx, blockIdx := range state.ToolIdxToBlockIndex {
		buf.Write(emitContentBlockStop(blockIdx))
		delete(state.ToolIdxToBlockIndex, toolIdx)
	}
	return buf.Bytes()
}

// closeNonThinkingBlocks closes any open text or tool blocks before
// opening a new thinking block.
func closeNonThinkingBlocks(state *StreamState) []byte {
	var buf bytes.Buffer
	if state.OpenTextBlockIndex != nil {
		buf.Write(emitContentBlockStop(*state.OpenTextBlockIndex))
		state.OpenTextBlockIndex = nil
	}
	for toolIdx, blockIdx := range state.ToolIdxToBlockIndex {
		buf.Write(emitContentBlockStop(blockIdx))
		delete(state.ToolIdxToBlockIndex, toolIdx)
	}
	return buf.Bytes()
}

// closeNonToolBlocks closes any open text or thinking block before
// opening a new tool block.
func closeNonToolBlocks(state *StreamState) []byte {
	var buf bytes.Buffer
	if state.OpenTextBlockIndex != nil {
		buf.Write(emitContentBlockStop(*state.OpenTextBlockIndex))
		state.OpenTextBlockIndex = nil
	}
	if state.OpenThinkingBlockIdx != nil {
		buf.Write(emitContentBlockStop(*state.OpenThinkingBlockIdx))
		state.OpenThinkingBlockIdx = nil
	}
	return buf.Bytes()
}

// emitContentBlockStartText returns the bytes for a content_block_start
// of type text at the given index.
func emitContentBlockStartText(idx int64) []byte {
	payload := map[string]any{
		"type":  "content_block_start",
		"index": idx,
		"content_block": map[string]any{
			"type": "text",
			"text": "",
		},
	}
	data, _ := json.Marshal(payload)
	return formatAnthropicSSEEvent("content_block_start", data)
}

// emitContentBlockStartToolUse returns the bytes for a
// content_block_start of type tool_use. Anthropic spec requires the
// input field to be present; an empty object {} is the canonical
// placeholder for "real value arrives via input_json_delta events".
func emitContentBlockStartToolUse(idx int64, id, name string) []byte {
	payload := map[string]any{
		"type":  "content_block_start",
		"index": idx,
		"content_block": map[string]any{
			"type":  "tool_use",
			"id":    id,
			"name":  name,
			"input": map[string]any{},
		},
	}
	data, _ := json.Marshal(payload)
	return formatAnthropicSSEEvent("content_block_start", data)
}

// emitContentBlockStartThinking returns the bytes for a
// content_block_start of type thinking with an empty thinking string.
func emitContentBlockStartThinking(idx int64) []byte {
	payload := map[string]any{
		"type":  "content_block_start",
		"index": idx,
		"content_block": map[string]any{
			"type":     "thinking",
			"thinking": "",
		},
	}
	data, _ := json.Marshal(payload)
	return formatAnthropicSSEEvent("content_block_start", data)
}

// emitContentBlockDeltaText returns a content_block_delta carrying a
// text_delta.
func emitContentBlockDeltaText(idx int64, text string) []byte {
	payload := map[string]any{
		"type":  "content_block_delta",
		"index": idx,
		"delta": map[string]any{
			"type": "text_delta",
			"text": text,
		},
	}
	data, _ := json.Marshal(payload)
	return formatAnthropicSSEEvent("content_block_delta", data)
}

// emitContentBlockDeltaThinking returns a content_block_delta carrying
// a thinking_delta.
func emitContentBlockDeltaThinking(idx int64, thinking string) []byte {
	payload := map[string]any{
		"type":  "content_block_delta",
		"index": idx,
		"delta": map[string]any{
			"type":     "thinking_delta",
			"thinking": thinking,
		},
	}
	data, _ := json.Marshal(payload)
	return formatAnthropicSSEEvent("content_block_delta", data)
}

// emitContentBlockDeltaSignature returns a content_block_delta carrying
// a signature_delta.
func emitContentBlockDeltaSignature(idx int64, signature string) []byte {
	payload := map[string]any{
		"type":  "content_block_delta",
		"index": idx,
		"delta": map[string]any{
			"type":      "signature_delta",
			"signature": signature,
		},
	}
	data, _ := json.Marshal(payload)
	return formatAnthropicSSEEvent("content_block_delta", data)
}

// emitContentBlockDeltaInputJSON returns a content_block_delta carrying
// an input_json_delta fragment for a tool_use block.
func emitContentBlockDeltaInputJSON(idx int64, partialJSON string) []byte {
	payload := map[string]any{
		"type":  "content_block_delta",
		"index": idx,
		"delta": map[string]any{
			"type":         "input_json_delta",
			"partial_json": partialJSON,
		},
	}
	data, _ := json.Marshal(payload)
	return formatAnthropicSSEEvent("content_block_delta", data)
}

// emitContentBlockStop returns a content_block_stop at the given index.
func emitContentBlockStop(idx int64) []byte {
	payload := map[string]any{
		"type":  "content_block_stop",
		"index": idx,
	}
	data, _ := json.Marshal(payload)
	return formatAnthropicSSEEvent("content_block_stop", data)
}

// emitMessageDelta returns the terminal message_delta carrying the
// final stop_reason / stop_sequence / cumulative usage. Reuses PR4's
// helpers so streaming and non-streaming output are byte-identical for
// the same upstream signal.
func emitMessageDelta(finishReason string, usage openai.CompletionUsage, ext *ir.IRExtensions) []byte {
	stopReason, stopSequence := mapOpenAIFinishReasonToAnthropic(finishReason, ext)
	anthropicUsage := buildAnthropicUsage(usage, ext)

	payload := map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   stopReason,
			"stop_sequence": stopSequence,
		},
		"usage": anthropicUsage,
	}
	data, _ := json.Marshal(payload)
	return formatAnthropicSSEEvent("message_delta", data)
}

// emitMessageStop returns the terminal message_stop event.
func emitMessageStop() []byte {
	payload := map[string]any{
		"type": "message_stop",
	}
	data, _ := json.Marshal(payload)
	return formatAnthropicSSEEvent("message_stop", data)
}

// emitAnthropicError emits an error event in Anthropic SSE shape.
// Reuses the same envelope used by the non-streaming emitter so error
// payloads are structurally identical across the two cells.
func emitAnthropicError(errorType, message string) []byte {
	envelope := anthropicErrorEnvelope{
		Type: "error",
		Error: anthropicErrorDetail{
			Type:    errorType,
			Message: message,
		},
	}
	data, _ := json.Marshal(envelope)
	return formatAnthropicSSEEvent("error", data)
}

// EmitAnthropicPingEvent returns the bytes for a single ping event.
// Consumed by the keepalive ticker in the extproc dispatcher.
func EmitAnthropicPingEvent() []byte {
	payload := map[string]any{
		"type": "ping",
	}
	data, _ := json.Marshal(payload)
	return formatAnthropicSSEEvent("ping", data)
}

// formatAnthropicSSEEvent renders one named SSE event in the wire shape
// Anthropic uses: "event: <name>\ndata: <json>\n\n". The named event
// header is what distinguishes Anthropic SSE from the OpenAI shape
// (which uses bare data: lines).
func formatAnthropicSSEEvent(eventName string, payload []byte) []byte {
	var buf bytes.Buffer
	buf.WriteString("event: ")
	buf.WriteString(eventName)
	buf.WriteByte('\n')
	buf.WriteString("data: ")
	buf.Write(payload)
	buf.WriteString("\n\n")
	return buf.Bytes()
}

// contentStarted reports whether the outbound emitter has opened at
// least one Anthropic content block (text, thinking, or tool_use) for
// this stream. NextBlockIndex is allocated monotonically as blocks open
// and is never reset, so a non-zero value means real content has already
// been emitted. Used to tell a stream-ending usage-only summary chunk
// (after content) from a gateway "preview" usage chunk (before any
// content; issue #2215).
func (s *StreamState) contentStarted() bool {
	return s.NextBlockIndex > 0
}

// chunkHasUsage returns true when an OpenAI chunk carries a non-zero
// usage object. OpenAI streams may put usage on a chunk separate from
// the one carrying finish_reason, so the emitter treats either as
// "this is the terminal signal".
func chunkHasUsage(chunk openai.ChatCompletionChunk) bool {
	u := chunk.Usage
	return u.PromptTokens != 0 || u.CompletionTokens != 0 || u.TotalTokens != 0
}

// openAIDeltaReasoning extracts the non-standard reasoning_content
// field from an OpenAI chunk delta. The OpenAI Go SDK preserves
// unknown fields in delta.JSON.ExtraFields; vsr's convention (matching
// PR #1718 on the non-streaming side) reads it from there. Returns
// the empty string when the delta carries no reasoning_content.
func openAIDeltaReasoning(delta openai.ChatCompletionChunkChoiceDelta) string {
	extra, ok := delta.JSON.ExtraFields["reasoning_content"]
	if !ok {
		return ""
	}
	var reasoning string
	if err := json.Unmarshal([]byte(extra.Raw()), &reasoning); err != nil {
		return ""
	}
	return reasoning
}
