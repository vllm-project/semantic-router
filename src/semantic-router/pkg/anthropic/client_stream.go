package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go"
	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
)

// StreamState tracks per-request streaming translation state. The struct
// is shared by both directions:
//
//   - Inbound (Anthropic SSE → OpenAI SSE), driven by TransformSSEChunkToOpenAI:
//     uses MessageID, Created, RoleSent, BlockIndexToToolIdx, NextToolIdx,
//     BlockIndexToThinkingActive.
//
//   - Outbound (OpenAI SSE → Anthropic SSE), driven by the PR5 emitter:
//     uses MessageStartSent, MessageStopSent, NextBlockIndex, the Open*
//     pointers tracking which block index is currently open,
//     ToolIdxToBlockIndex (the reverse of the inbound pair),
//     LastChunkAt for the ping keepalive ticker, and InitialUsage for the
//     message_start usage placeholder.
//
// One ClientProtocol per request means only one direction is active at a
// time, so the two sub-sets never collide on the same instance.
type StreamState struct {
	// Inbound direction (Anthropic → OpenAI).
	MessageID                  string
	Created                    int64
	RoleSent                   bool
	BlockIndexToToolIdx        map[int64]int
	NextToolIdx                int
	BlockIndexToThinkingActive map[int64]bool

	// Outbound direction (OpenAI → Anthropic).
	MessageStartSent     bool
	MessageStopSent      bool
	NextBlockIndex       int64
	OpenTextBlockIndex   *int64
	OpenThinkingBlockIdx *int64
	ToolIdxToBlockIndex  map[int]int64
	LastChunkAt          time.Time
	InitialUsage         anthropic.Usage
}

// NewStreamState returns initialized stream translation state with maps
// allocated for both directions.
func NewStreamState() *StreamState {
	return &StreamState{
		BlockIndexToToolIdx:        make(map[int64]int),
		BlockIndexToThinkingActive: make(map[int64]bool),
		ToolIdxToBlockIndex:        make(map[int]int64),
	}
}

// extractedSSELine carries one parsed `data:` payload along with the most
// recent `event:` header above it. The header lets us route Anthropic
// `event: error` lines into the error arm of transformStreamEvent — those
// lines unmarshal as `{"type":"error",...}` which the switch otherwise
// drops silently.
type extractedSSELine struct {
	Data    []byte
	IsError bool
}

// TransformSSEChunkToOpenAI converts Anthropic SSE bytes from Envoy into OpenAI SSE.
// streamDone is true when message_stop is observed.
//
// ext is optional; when non-nil, signature_delta events are captured into
// ext.ThinkingSignatures so the symmetric outbound emitter can replay them
// on multi-turn requests. Pass nil for the existing OpenAI-client cell
// where signatures have no downstream consumer.
func TransformSSEChunkToOpenAI(
	anthropicChunk []byte,
	state *StreamState,
	model string,
	ext *ir.IRExtensions,
) ([]byte, bool, error) {
	if state == nil {
		state = NewStreamState()
	}

	var out bytes.Buffer
	streamDone := false
	for _, line := range extractSSEDataLines(anthropicChunk) {
		if line.IsError {
			chunks, err := handleErrorEvent(line.Data, state, model)
			if err != nil {
				return nil, false, err
			}
			for _, chunk := range chunks {
				out.Write(formatOpenAISSELine(chunk))
			}
			continue
		}

		var event anthropic.MessageStreamEventUnion
		if err := json.Unmarshal(line.Data, &event); err != nil {
			continue
		}
		chunks, done, err := transformStreamEvent(event, state, model, ext)
		if err != nil {
			return nil, false, err
		}
		if done {
			streamDone = true
		}
		for _, chunk := range chunks {
			out.Write(formatOpenAISSELine(chunk))
		}
	}

	if streamDone {
		out.WriteString("data: [DONE]\n\n")
	}

	return out.Bytes(), streamDone, nil
}

// extractSSEDataLines walks the Anthropic SSE framing in chunk and yields
// each `data:` payload along with whether the preceding `event:` header
// flagged it as an error event. Lines that begin with `{` (some
// transports omit the `data:` prefix) are surfaced as ordinary payloads.
func extractSSEDataLines(chunk []byte) []extractedSSELine {
	var lines []extractedSSELine
	currentEvent := ""
	for _, raw := range bytes.Split(chunk, []byte("\n")) {
		line := bytes.TrimSpace(raw)
		if len(line) == 0 {
			// Blank line terminates the current SSE event; reset the
			// per-event header tracker so the next data: line is not
			// mis-classified as belonging to a stale event header.
			currentEvent = ""
			continue
		}
		if bytes.HasPrefix(line, []byte("event:")) {
			currentEvent = string(bytes.TrimSpace(bytes.TrimPrefix(line, []byte("event:"))))
			continue
		}
		if bytes.HasPrefix(line, []byte("data:")) {
			payload := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
			if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
				continue
			}
			lines = append(lines, extractedSSELine{
				Data:    payload,
				IsError: currentEvent == "error",
			})
			continue
		}
		if line[0] == '{' {
			lines = append(lines, extractedSSELine{
				Data:    line,
				IsError: currentEvent == "error",
			})
		}
	}
	return lines
}

func transformStreamEvent(
	event anthropic.MessageStreamEventUnion,
	state *StreamState,
	model string,
	ext *ir.IRExtensions,
) ([][]byte, bool, error) {
	switch event.Type {
	case "message_start":
		return handleMessageStart(event, state, model)
	case "content_block_start":
		return handleContentBlockStart(event, state, model)
	case "content_block_delta":
		return handleContentBlockDelta(event, state, model, ext)
	case "message_delta":
		return handleMessageDelta(event, state, model, ext)
	case "message_stop":
		return nil, true, nil
	case "content_block_stop", "ping", "vertex_event":
		return nil, false, nil
	case "error":
		// Some transports surface error events as ordinary
		// MessageStreamEventUnion entries (no `event: error` header).
		// Route them through the same error pipeline.
		chunks, err := handleErrorEvent([]byte(event.RawJSON()), state, model)
		return chunks, false, err
	default:
		return nil, false, nil
	}
}

func handleMessageStart(
	event anthropic.MessageStreamEventUnion,
	state *StreamState,
	model string,
) ([][]byte, bool, error) {
	start := event.AsMessageStart()
	if start.Message.ID != "" {
		state.MessageID = start.Message.ID
	}
	if state.Created == 0 {
		state.Created = time.Now().Unix()
	}
	// Capture the upstream's initial usage so the symmetric outbound
	// emitter (sse_out.go) can echo it onto its synthesized
	// message_start event, preserving the per-request input token
	// count clients use to seed their usage accumulators.
	state.InitialUsage = start.Message.Usage
	if state.RoleSent {
		return nil, false, nil
	}
	state.RoleSent = true
	chunk, err := buildOpenAIStreamChunk(state, model, openai.ChatCompletionChunkChoiceDelta{
		Role: "assistant",
	})
	return [][]byte{chunk}, false, err
}

func handleContentBlockStart(
	event anthropic.MessageStreamEventUnion,
	state *StreamState,
	model string,
) ([][]byte, bool, error) {
	start := event.AsContentBlockStart()
	block := start.ContentBlock

	switch block.Type {
	case "text":
		if strings.TrimSpace(block.Text) == "" {
			return nil, false, nil
		}
		chunk, err := buildOpenAIStreamChunk(state, model, openai.ChatCompletionChunkChoiceDelta{
			Content: block.Text,
		})
		return [][]byte{chunk}, false, err
	case "tool_use", "server_tool_use":
		toolIdx := state.NextToolIdx
		state.NextToolIdx++
		state.BlockIndexToToolIdx[start.Index] = toolIdx
		chunk, err := buildOpenAIStreamToolChunk(state, model, toolIdx, block.ID, block.Name, "")
		return [][]byte{chunk}, false, err
	case "thinking":
		// Mark the block index as a thinking block so subsequent
		// thinking_delta and signature_delta events can be routed back to
		// it. Emit a bootstrap chunk carrying an empty reasoning_content
		// string so downstream consumers see the field appear (mirroring
		// the text case's bootstrap content).
		state.BlockIndexToThinkingActive[start.Index] = true
		chunk, err := buildOpenAIReasoningChunk(state, model, "")
		return [][]byte{chunk}, false, err
	default:
		return nil, false, nil
	}
}

func handleContentBlockDelta(
	event anthropic.MessageStreamEventUnion,
	state *StreamState,
	model string,
	ext *ir.IRExtensions,
) ([][]byte, bool, error) {
	deltaEvent := event.AsContentBlockDelta()
	delta := deltaEvent.Delta

	switch delta.Type {
	case "text_delta":
		textDelta := delta.AsTextDelta()
		if textDelta.Text == "" {
			return nil, false, nil
		}
		chunk, err := buildOpenAIStreamChunk(state, model, openai.ChatCompletionChunkChoiceDelta{
			Content: textDelta.Text,
		})
		return [][]byte{chunk}, false, err
	case "input_json_delta":
		toolIdx, ok := state.BlockIndexToToolIdx[deltaEvent.Index]
		if !ok {
			return nil, false, nil
		}
		jsonDelta := delta.AsInputJSONDelta()
		chunk, err := buildOpenAIStreamToolChunk(state, model, toolIdx, "", "", jsonDelta.PartialJSON)
		return [][]byte{chunk}, false, err
	case "thinking_delta":
		return handleThinkingDelta(deltaEvent, state, model)
	case "signature_delta":
		return handleSignatureDelta(deltaEvent, state, ext)
	default:
		return nil, false, nil
	}
}

// handleThinkingDelta translates an Anthropic thinking_delta into an OpenAI
// chunk carrying delta.reasoning_content. Returns nil for deltas that arrive
// without a matching content_block_start (orphan deltas cannot be routed).
func handleThinkingDelta(
	deltaEvent anthropic.ContentBlockDeltaEvent,
	state *StreamState,
	model string,
) ([][]byte, bool, error) {
	if !state.BlockIndexToThinkingActive[deltaEvent.Index] {
		return nil, false, nil
	}
	thinkingDelta := deltaEvent.Delta.AsThinkingDelta()
	if thinkingDelta.Thinking == "" {
		return nil, false, nil
	}
	chunk, err := buildOpenAIReasoningChunk(state, model, thinkingDelta.Thinking)
	return [][]byte{chunk}, false, err
}

// handleSignatureDelta captures the per-block signature into IRExtensions
// so the symmetric outbound emitter can replay it on multi-turn requests.
// Signatures have no representation on the OpenAI envelope; with nil ext
// (the existing OpenAI-client cell), the signature is dropped — no consumer.
func handleSignatureDelta(
	deltaEvent anthropic.ContentBlockDeltaEvent,
	state *StreamState,
	ext *ir.IRExtensions,
) ([][]byte, bool, error) {
	if ext == nil {
		return nil, false, nil
	}
	if !state.BlockIndexToThinkingActive[deltaEvent.Index] {
		return nil, false, nil
	}
	sigDelta := deltaEvent.Delta.AsSignatureDelta()
	if sigDelta.Signature == "" {
		return nil, false, nil
	}
	ext.SetThinkingSignature(thinkingBlockKey(deltaEvent.Index), sigDelta.Signature)
	return nil, false, nil
}

// thinkingBlockKey is the stable IRExtensions.ThinkingSignatures key for
// a thinking block at Anthropic content_block index. Mirrors the
// non-streaming inverse helper in client.go so a request that becomes
// streaming mid-way does not double-store signatures.
func thinkingBlockKey(blockIdx int64) string {
	return fmt.Sprintf("content[%d]", blockIdx)
}

func handleMessageDelta(
	event anthropic.MessageStreamEventUnion,
	state *StreamState,
	model string,
	ext *ir.IRExtensions,
) ([][]byte, bool, error) {
	deltaEvent := event.AsMessageDelta()
	finishReason := mapAnthropicStopReasonToOpenAI(deltaEvent.Delta.StopReason)

	// Capture Anthropic-only stop reasons ("pause_turn", "refusal")
	// onto the IR sidecar so the symmetric outbound emitter can
	// surface them verbatim on the client's message_delta. OpenAI's
	// finish_reason alphabet has no equivalent, so without this
	// round-trip the value would be lossily mapped to end_turn.
	if ext != nil && isAnthropicOnlyStopReason(deltaEvent.Delta.StopReason) {
		ext.AnthropicStopReason = string(deltaEvent.Delta.StopReason)
		if deltaEvent.Delta.StopSequence != "" {
			ext.AnthropicStopSequence = deltaEvent.Delta.StopSequence
		}
	}

	chunk, err := buildOpenAIStreamFinishChunk(state, model, finishReason, deltaEvent.Usage)
	if err != nil {
		return nil, false, err
	}
	if chunk == nil {
		return nil, false, nil
	}
	return [][]byte{chunk}, false, nil
}

// isAnthropicOnlyStopReason returns true for stop_reason values that
// have no equivalent in the OpenAI finish_reason alphabet and must
// round-trip via IRExtensions to survive the OpenAI normalization step.
//
// The full set: pause_turn, refusal, and stop_sequence all map onto
// "stop" in mapAnthropicStopReasonToOpenAI (refusal via the default arm,
// stop_sequence explicitly). The outbound emitter restores the original
// value from IRExtensions so an Anthropic client in a double-Anthropic
// cell sees the correct stop_reason.
func isAnthropicOnlyStopReason(reason anthropic.StopReason) bool {
	switch reason {
	case anthropic.StopReasonPauseTurn, anthropic.StopReasonRefusal, anthropic.StopReasonStopSequence:
		return true
	default:
		return false
	}
}

// handleErrorEvent maps an Anthropic `event: error` data payload onto an
// OpenAI terminal chunk. Per the plan's Q4 resolution: emit
// finish_reason="error" plus a synthetic content delta carrying the error
// message, so clients that ignore the non-standard finish_reason still
// see the human-readable failure in the assistant text stream.
func handleErrorEvent(data []byte, state *StreamState, model string) ([][]byte, error) {
	var envelope anthropicErrorEnvelope
	if err := json.Unmarshal(data, &envelope); err != nil {
		// Malformed error payload; surface a generic message so the
		// client at least sees that the stream aborted abnormally.
		envelope = anthropicErrorEnvelope{
			Type: "error",
			Error: anthropicErrorDetail{
				Type:    anthropicErrorTypeAPI,
				Message: "unparsable Anthropic error event",
			},
		}
	}
	message := envelope.Error.Message
	if message == "" {
		message = "Anthropic upstream error"
	}

	contentChunk, err := buildOpenAIStreamChunk(state, model, openai.ChatCompletionChunkChoiceDelta{
		Content: fmt.Sprintf("[error: %s] %s", envelope.Error.Type, message),
	})
	if err != nil {
		return nil, err
	}

	finishChunk, err := buildOpenAIStreamFinishChunk(state, model, "error", anthropic.MessageDeltaUsage{})
	if err != nil {
		return nil, err
	}
	if finishChunk == nil {
		return [][]byte{contentChunk}, nil
	}
	return [][]byte{contentChunk, finishChunk}, nil
}

func buildOpenAIStreamChunk(
	state *StreamState,
	model string,
	delta openai.ChatCompletionChunkChoiceDelta,
) ([]byte, error) {
	chunk := openai.ChatCompletionChunk{
		ID:      state.MessageID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   model,
		Choices: []openai.ChatCompletionChunkChoice{{
			Index: 0,
			Delta: delta,
		}},
	}
	return json.Marshal(chunk)
}

// buildOpenAIReasoningChunk emits an OpenAI chunk carrying
// `reasoning_content` on the delta. The OpenAI Go SDK's delta struct has
// no first-class reasoning_content field, so the value is injected via
// sjson after marshal — keeping interop with vsr's existing convention
// (the non-streaming path reads reasoning_content the same way).
func buildOpenAIReasoningChunk(state *StreamState, model string, reasoning string) ([]byte, error) {
	base, err := buildOpenAIStreamChunk(state, model, openai.ChatCompletionChunkChoiceDelta{})
	if err != nil {
		return nil, err
	}
	withReasoning, err := sjson.SetBytes(base, "choices.0.delta.reasoning_content", reasoning)
	if err != nil {
		return nil, fmt.Errorf("inject reasoning_content: %w", err)
	}
	return withReasoning, nil
}

func buildOpenAIStreamToolChunk(
	state *StreamState,
	model string,
	toolIdx int,
	id string,
	name string,
	arguments string,
) ([]byte, error) {
	toolCall := openai.ChatCompletionChunkChoiceDeltaToolCall{
		Index: int64(toolIdx),
	}
	if id != "" {
		toolCall.ID = id
	}
	if name != "" {
		toolCall.Type = "function"
		toolCall.Function.Name = name
	}
	if arguments != "" {
		if toolCall.Type == "" {
			toolCall.Type = "function"
		}
		toolCall.Function.Arguments = arguments
	}

	chunk := openai.ChatCompletionChunk{
		ID:      state.MessageID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   model,
		Choices: []openai.ChatCompletionChunkChoice{{
			Index: 0,
			Delta: openai.ChatCompletionChunkChoiceDelta{
				ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{toolCall},
			},
		}},
	}
	return json.Marshal(chunk)
}

func buildOpenAIStreamFinishChunk(
	state *StreamState,
	model string,
	finishReason string,
	usage anthropic.MessageDeltaUsage,
) ([]byte, error) {
	if finishReason == "" && usage.InputTokens == 0 && usage.OutputTokens == 0 {
		return nil, nil
	}

	choice := openai.ChatCompletionChunkChoice{
		Index: 0,
		Delta: openai.ChatCompletionChunkChoiceDelta{},
	}
	if finishReason != "" {
		choice.FinishReason = finishReason
	}

	chunk := openai.ChatCompletionChunk{
		ID:      state.MessageID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   model,
		Choices: []openai.ChatCompletionChunkChoice{choice},
	}
	if usage.InputTokens > 0 || usage.OutputTokens > 0 {
		chunk.Usage = openai.CompletionUsage{
			PromptTokens:     usage.InputTokens,
			CompletionTokens: usage.OutputTokens,
			TotalTokens:      usage.InputTokens + usage.OutputTokens,
		}
	}
	return json.Marshal(chunk)
}

func mapAnthropicStopReasonToOpenAI(reason anthropic.StopReason) string {
	switch reason {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence:
		return "stop"
	case anthropic.StopReasonMaxTokens:
		return "length"
	case anthropic.StopReasonToolUse:
		return "tool_calls"
	default:
		if reason == "" {
			return ""
		}
		return "stop"
	}
}

func formatOpenAISSELine(chunkJSON []byte) []byte {
	var buf bytes.Buffer
	buf.WriteString("data: ")
	buf.Write(chunkJSON)
	buf.WriteString("\n\n")
	return buf.Bytes()
}

// WithStreamingRequestBody adds stream=true to an Anthropic request JSON body.
func WithStreamingRequestBody(body []byte) ([]byte, error) {
	return withStreamFlag(body)
}

// BuildStreamingRequestHeaders returns headers for a streaming Anthropic request.
// Equivalent to BuildStreamingRequestHeadersWithPassthrough with a nil passthrough.
func BuildStreamingRequestHeaders(apiKey string, bodyLength int, messagesPath string) []HeaderKeyValue {
	return buildRequestHeaders(apiKey, bodyLength, messagesPath, true, nil)
}

// BuildStreamingRequestHeadersWithPassthrough is the passthrough-aware form of
// BuildStreamingRequestHeaders. See BuildRequestHeadersWithPassthrough for the
// header precedence contract.
func BuildStreamingRequestHeadersWithPassthrough(apiKey string, bodyLength int, messagesPath string, pt *AnthropicPassthrough) []HeaderKeyValue {
	return buildRequestHeaders(apiKey, bodyLength, messagesPath, true, pt)
}

// withStreamFlag adds stream=true to an Anthropic request JSON body.
func withStreamFlag(body []byte) ([]byte, error) {
	out, err := sjson.SetBytes(body, "stream", true)
	if err != nil {
		return nil, fmt.Errorf("set stream flag: %w", err)
	}
	return out, nil
}
