package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type sseFrame struct {
	Event string
	Data  []byte
}

// sseFramer turns arbitrary transport chunks into complete SSE events. It is
// request-scoped, retains at most one bounded unfinished event, and accepts LF,
// CRLF, or CR line endings without assuming network read boundaries.
type sseFramer struct {
	buffer []byte
	limit  int
}

func newSSEFramer(limit int) sseFramer { return sseFramer{limit: limit} }

func (framer *sseFramer) Push(chunk []byte) ([][]byte, error) {
	if len(chunk) == 0 {
		return nil, nil
	}
	framer.buffer = append(framer.buffer, chunk...)
	frames := make([][]byte, 0, 1)
	for {
		end, complete := completeSSEFrame(framer.buffer)
		if !complete {
			break
		}
		if end > framer.limit {
			return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit", "SSE frame is too large", nil)
		}
		frames = append(frames, append([]byte(nil), framer.buffer[:end]...))
		framer.buffer = append(framer.buffer[:0], framer.buffer[end:]...)
	}
	if len(framer.buffer) > framer.limit {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit", "unfinished SSE frame is too large", nil)
	}
	return frames, nil
}

func (framer *sseFramer) Finalize() ([][]byte, error) {
	if len(bytes.TrimSpace(framer.buffer)) == 0 {
		framer.buffer = nil
		return nil, nil
	}
	if len(framer.buffer) > framer.limit {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit", "unfinished SSE frame is too large", nil)
	}
	frame := append([]byte(nil), framer.buffer...)
	framer.buffer = nil
	return [][]byte{frame}, nil
}

func completeSSEFrame(payload []byte) (int, bool) {
	lineStart := 0
	for index := 0; index < len(payload); {
		lineEnd := index
		terminator := 0
		switch payload[index] {
		case '\n':
			terminator = 1
		case '\r':
			terminator = 1
			if index+1 < len(payload) && payload[index+1] == '\n' {
				terminator = 2
			}
		default:
			index++
			continue
		}
		if lineEnd == lineStart {
			return index + terminator, true
		}
		index += terminator
		lineStart = index
	}
	return 0, false
}

type (
	decoderFrameFinalizer func() ([][]byte, error)
	decoderFrameProcessor func([]byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error)
)

func finalizeDecoderFrames(
	finalize decoderFrameFinalizer,
	process decoderFrameProcessor,
	diagnosticLimit int,
) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	frames, err := finalize()
	if err != nil {
		return nil, nil, err
	}
	var events []llmprotocol.Event
	var diagnostics llmprotocol.Diagnostics
	for _, frame := range frames {
		decoded, frameDiagnostics, decodeErr := process(frame)
		events = append(events, decoded...)
		diagnostics = appendDiagnostics(diagnostics, frameDiagnostics, diagnosticLimit)
		if decodeErr != nil {
			return events, diagnostics, decodeErr
		}
	}
	return events, diagnostics, nil
}

func parseSSEFrame(frame []byte, limit int) (sseFrame, error) {
	if len(frame) == 0 || len(frame) > limit {
		return sseFrame{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit", "SSE frame is empty or too large", nil)
	}
	var result sseFrame
	normalized := bytes.ReplaceAll(frame, []byte("\r\n"), []byte("\n"))
	normalized = bytes.ReplaceAll(normalized, []byte("\r"), []byte("\n"))
	for _, line := range bytes.Split(normalized, []byte("\n")) {
		if len(line) == 0 || line[0] == ':' {
			continue
		}
		name, value, found := bytes.Cut(line, []byte{':'})
		if !found {
			name, value = line, nil
		}
		value = bytes.TrimPrefix(value, []byte{' '})
		switch string(name) {
		case "event":
			result.Event = string(value)
		case "data":
			if len(result.Data) > 0 {
				result.Data = append(result.Data, '\n')
			}
			result.Data = append(result.Data, value...)
		}
	}
	if len(result.Data) == 0 {
		return sseFrame{}, nil
	}
	return result, nil
}

func encodeSSE(event string, data any) ([]byte, error) {
	body, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}
	var buffer bytes.Buffer
	if event != "" {
		buffer.WriteString("event: ")
		buffer.WriteString(event)
		buffer.WriteByte('\n')
	}
	buffer.WriteString("data: ")
	buffer.Write(body)
	buffer.WriteString("\n\n")
	return buffer.Bytes(), nil
}

type streamState struct {
	context        llmprotocol.StreamContext
	policy         llmprotocol.Policy
	sequence       uint64
	events         int
	terminal       bool
	started        bool
	usage          llmprotocol.Usage
	stop           llmprotocol.StopReason
	items          map[int]bool
	completedItems map[int]bool
	itemKinds      map[int]llmprotocol.ContentKind
	itemIDs        map[int]string
	itemTextBytes  map[int]int
	itemTextRunes  map[int]int64
	itemCitations  map[int]int
	toolCalls      map[int]llmprotocol.ToolCall
	toolArguments  map[int][]byte
}

func (state *streamState) next(event llmprotocol.Event) (llmprotocol.Event, error) {
	if state.terminal {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already terminal", nil)
	}
	switch event.Type {
	case llmprotocol.EventResponseStarted, llmprotocol.EventOutputItemStarted,
		llmprotocol.EventOutputTextDelta, llmprotocol.EventReasoningDelta,
		llmprotocol.EventToolCallDelta, llmprotocol.EventOutputItemCompleted,
		llmprotocol.EventUsageUpdated, llmprotocol.EventResponseCompleted,
		llmprotocol.EventResponseFailed, llmprotocol.EventProviderOpaque:
	default:
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "unknown_stream_event", "upstream stream event type is invalid", nil)
	}
	state.events++
	if state.policy.Limits.Events > 0 && state.events > state.policy.Limits.Events {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_event_limit", "stream event limit exceeded", nil)
	}
	state.sequence++
	event.Sequence = state.sequence
	if event.ResponseID == "" {
		event.ResponseID = state.context.ResponseID
	}
	if event.Model == "" {
		event.Model = state.context.PublicModel
	}
	if state.items == nil {
		state.items = make(map[int]bool)
		state.completedItems = make(map[int]bool)
		state.itemKinds = make(map[int]llmprotocol.ContentKind)
		state.itemIDs = make(map[int]string)
		state.itemTextBytes = make(map[int]int)
		state.itemTextRunes = make(map[int]int64)
		state.itemCitations = make(map[int]int)
		state.toolCalls = make(map[int]llmprotocol.ToolCall)
		state.toolArguments = make(map[int][]byte)
		if state.usage.State == "" {
			state.usage.State = llmprotocol.UsageUnavailable
		}
	}
	if event.Type == llmprotocol.EventResponseStarted {
		if state.started {
			return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "duplicate_stream_start", "upstream stream started more than once", nil)
		}
		if event.ResponseID == "" && state.policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
			event.ResponseID = llmprotocol.StableID("stream-response", state.context.PublicModel, state.context.ProviderModel)
		}
		if event.ResponseID == "" {
			return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_response_id", "upstream stream response ID is missing", nil)
		}
		state.context.ResponseID = event.ResponseID
		state.started = true
	} else if event.Type != llmprotocol.EventProviderOpaque && event.Type != llmprotocol.EventResponseFailed && !state.started {
		return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_start_missing", "upstream stream emitted output before response start", nil)
	}
	switch event.Type {
	case llmprotocol.EventOutputItemStarted:
		if event.ItemIndex < 0 {
			return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_item_index", "upstream output item index is invalid", nil)
		}
		if state.items[event.ItemIndex] {
			return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "duplicate_item_start", "upstream output item started more than once", nil)
		}
		state.items[event.ItemIndex] = true
		if event.ItemID == "" && state.policy.MissingStableIDs == llmprotocol.MissingIDGenerateStable {
			event.ItemID = llmprotocol.StableID(event.ResponseID, string(event.Type), fmt.Sprint(event.ItemIndex))
		}
		if event.ItemID == "" {
			return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_id", "upstream output item ID is missing", nil)
		}
		state.itemIDs[event.ItemIndex] = event.ItemID
		if event.ToolCall != nil {
			state.itemKinds[event.ItemIndex] = llmprotocol.ContentToolCall
			state.toolCalls[event.ItemIndex] = *event.ToolCall
			state.toolArguments[event.ItemIndex] = append([]byte(nil), event.ToolCall.Arguments...)
		} else if event.Content != nil {
			state.itemKinds[event.ItemIndex] = event.Content.Kind
		}
	case llmprotocol.EventOutputTextDelta, llmprotocol.EventReasoningDelta, llmprotocol.EventToolCallDelta:
		if !state.items[event.ItemIndex] || state.completedItems[event.ItemIndex] {
			return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_item_lifecycle", "upstream delta does not reference an active output item", nil)
		}
		if event.Content != nil && len(event.Content.Citations) > 0 &&
			(event.Type != llmprotocol.EventOutputTextDelta || event.Content.Kind != llmprotocol.ContentText) {
			return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_citation", "upstream citations require a text delta", nil)
		}
		if event.Type == llmprotocol.EventOutputTextDelta {
			textBytes := state.itemTextBytes[event.ItemIndex] + len(event.Delta)
			if state.policy.Limits.TextBytes > 0 && textBytes > state.policy.Limits.TextBytes {
				return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "text_limit", "content text exceeds the configured limit", nil)
			}
			textRunes := state.itemTextRunes[event.ItemIndex] + int64(utf8.RuneCountInString(event.Delta))
			citationCount := state.itemCitations[event.ItemIndex]
			var citationBatch []llmprotocol.Citation
			if event.Content != nil && len(event.Content.Citations) > 0 {
				citationBatch = event.Content.Citations
				citationCount += len(citationBatch)
			}
			if err := llmprotocol.ValidateCitationBatch(
				textRunes, citationCount, citationBatch, state.policy.Limits,
			); err != nil {
				return llmprotocol.Event{}, err
			}
			state.itemTextBytes[event.ItemIndex] = textBytes
			state.itemTextRunes[event.ItemIndex] = textRunes
			state.itemCitations[event.ItemIndex] = citationCount
		}
		if event.Type == llmprotocol.EventToolCallDelta {
			if event.ToolCall == nil {
				return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "tool_delta_missing", "upstream tool delta is missing", nil)
			}
			state.itemKinds[event.ItemIndex] = llmprotocol.ContentToolCall
			call := state.toolCalls[event.ItemIndex]
			if event.ToolCall.ID != "" {
				call.ID = event.ToolCall.ID
			}
			if event.ToolCall.Name != "" {
				call.Name = event.ToolCall.Name
			}
			state.toolCalls[event.ItemIndex] = call
			event.ToolCall.ID = call.ID
			event.ToolCall.Name = call.Name
			current := state.toolArguments[event.ItemIndex]
			if bytes.Equal(bytes.TrimSpace(current), []byte("{}")) && event.ToolCall.Arguments != "" {
				current = nil
			}
			if state.policy.Limits.UnfinishedArguments > 0 && len(event.ToolCall.Arguments) > state.policy.Limits.UnfinishedArguments-len(current) {
				return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "tool_arguments_limit", "unfinished tool arguments exceed the configured limit", nil)
			}
			state.toolArguments[event.ItemIndex] = append(current, event.ToolCall.Arguments...)
		} else if event.Content != nil {
			state.itemKinds[event.ItemIndex] = event.Content.Kind
		} else if event.Type == llmprotocol.EventReasoningDelta {
			state.itemKinds[event.ItemIndex] = llmprotocol.ContentReasoning
		} else if state.itemKinds[event.ItemIndex] == "" {
			state.itemKinds[event.ItemIndex] = llmprotocol.ContentText
		}
	case llmprotocol.EventOutputItemCompleted:
		if !state.items[event.ItemIndex] || state.completedItems[event.ItemIndex] {
			return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_item_lifecycle", "upstream completed an inactive output item", nil)
		}
		if state.itemKinds[event.ItemIndex] == llmprotocol.ContentToolCall {
			arguments := state.toolArguments[event.ItemIndex]
			if event.ToolCall != nil && event.ToolCall.Arguments != "" {
				if len(arguments) > 0 && string(arguments) != event.ToolCall.Arguments {
					return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_tool_arguments_mismatch", "upstream final tool arguments do not match streamed arguments", nil)
				}
				arguments = []byte(event.ToolCall.Arguments)
			}
			if !isJSONObject(arguments, state.policy.Limits.JSONDepth) {
				return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_tool_arguments", "upstream streamed tool arguments are not a JSON object", nil)
			}
			call := state.toolCalls[event.ItemIndex]
			if event.ToolCall != nil {
				if event.ToolCall.ID != "" {
					call.ID = event.ToolCall.ID
				}
				if event.ToolCall.Name != "" {
					call.Name = event.ToolCall.Name
				}
			}
			call.Arguments = string(arguments)
			event.ToolCall = &call
		} else if event.Content == nil && state.itemKinds[event.ItemIndex] != "" {
			event.Content = &llmprotocol.Content{Kind: state.itemKinds[event.ItemIndex]}
		}
		state.completedItems[event.ItemIndex] = true
		delete(state.itemTextBytes, event.ItemIndex)
		delete(state.itemTextRunes, event.ItemIndex)
		delete(state.itemCitations, event.ItemIndex)
		delete(state.toolArguments, event.ItemIndex)
	}
	if event.Usage != nil {
		merged, err := mergeMonotonicUsage(state.usage, *event.Usage)
		if err != nil {
			return llmprotocol.Event{}, err
		}
		state.usage = merged
		usage := state.usage
		event.Usage = &usage
	}
	if event.StopReason != "" {
		state.stop = event.StopReason
	}
	if event.Type == llmprotocol.EventResponseCompleted || event.Type == llmprotocol.EventResponseFailed {
		if event.Type == llmprotocol.EventResponseCompleted {
			if !state.started {
				return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_start_missing", "upstream stream completed before response start", nil)
			}
			if len(state.items) == 0 {
				return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_output_missing", "upstream stream completed without output", nil)
			}
			for itemIndex := range state.items {
				if !state.completedItems[itemIndex] {
					return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_incomplete", "upstream stream completed with an active output item", nil)
				}
			}
			if len(state.toolArguments) != 0 {
				return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_tool_arguments_incomplete", "upstream stream completed with unfinished tool arguments", nil)
			}
			if event.Error != nil {
				return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_terminal_shape", "completed stream cannot contain an error", nil)
			}
			if event.StopReason == "" {
				event.StopReason = llmprotocol.StopUnknown
			}
			if event.Usage == nil {
				usage := state.usage
				event.Usage = &usage
			}
			if err := llmprotocol.ValidateUsage(*event.Usage); err != nil {
				return llmprotocol.Event{}, err
			}
		} else if event.Error == nil || event.Usage != nil {
			return llmprotocol.Event{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_failure_shape", "failed stream requires an error and cannot contain successful usage", nil)
		}
		state.terminal = true
	}
	return event, nil
}

func mergeMonotonicUsage(current, update llmprotocol.Usage) (llmprotocol.Usage, error) {
	if err := llmprotocol.ValidateUsage(update); err != nil {
		return llmprotocol.Usage{}, err
	}
	if current.State == llmprotocol.UsageAvailable && update.State == llmprotocol.UsageUnavailable {
		return llmprotocol.Usage{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "usage_evidence_decreased", "upstream streaming usage evidence became unavailable", nil)
	}
	merge := func(existing, incoming llmprotocol.TokenCount) (llmprotocol.TokenCount, error) {
		if incoming.Value == nil {
			return existing, nil
		}
		if existing.Value != nil && *existing.Value > *incoming.Value {
			return llmprotocol.TokenCount{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "usage_decreased", "upstream streaming usage counter decreased", nil)
		}
		if existing.Value != nil && usageEvidenceRank(incoming.Provenance) < usageEvidenceRank(existing.Provenance) {
			return llmprotocol.TokenCount{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "usage_evidence_decreased", "upstream streaming usage evidence quality decreased", nil)
		}
		return incoming, nil
	}
	state := update.State
	if state == "" {
		state = current.State
	}
	if state == "" {
		state = llmprotocol.UsageUnavailable
	}
	currentCounts := []llmprotocol.TokenCount{current.InputUncached, current.InputCacheRead, current.InputCacheWrite, current.OutputReasoning, current.OutputOther, current.InputTotal, current.OutputTotal, current.Total}
	updateCounts := []llmprotocol.TokenCount{update.InputUncached, update.InputCacheRead, update.InputCacheWrite, update.OutputReasoning, update.OutputOther, update.InputTotal, update.OutputTotal, update.Total}
	merged := make([]llmprotocol.TokenCount, len(currentCounts))
	for index := range merged {
		value, err := merge(currentCounts[index], updateCounts[index])
		if err != nil {
			return llmprotocol.Usage{}, err
		}
		merged[index] = value
	}
	result := llmprotocol.Usage{State: state, InputUncached: merged[0], InputCacheRead: merged[1], InputCacheWrite: merged[2], OutputReasoning: merged[3], OutputOther: merged[4], InputTotal: merged[5], OutputTotal: merged[6], Total: merged[7]}
	if result.InputTotal.Value != nil && result.OutputTotal.Value != nil &&
		(result.Total.Value == nil || result.Total.Provenance == llmprotocol.UsageDerived) {
		if *result.OutputTotal.Value > math.MaxInt64-*result.InputTotal.Value {
			return llmprotocol.Usage{}, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "usage_overflow", "upstream streaming usage total overflowed", nil)
		}
		value := *result.InputTotal.Value + *result.OutputTotal.Value
		result.Total = llmprotocol.TokenCount{Value: llmprotocol.Int64(value), Provenance: llmprotocol.UsageDerived}
	}
	if err := llmprotocol.ValidateUsage(result); err != nil {
		return llmprotocol.Usage{}, err
	}
	return result, nil
}

func usageEvidenceRank(provenance llmprotocol.UsageProvenance) int {
	switch provenance {
	case llmprotocol.UsageAuthoritative:
		return 3
	case llmprotocol.UsageDerived:
		return 2
	case llmprotocol.UsageEstimated:
		return 1
	default:
		return 0
	}
}

func (state *streamState) finalize(reason error) ([]llmprotocol.Event, error) {
	if state.terminal {
		return nil, nil
	}
	if reason == nil {
		reason = errors.New("upstream stream ended without a terminal event")
	}
	protocolError := llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_incomplete", "upstream stream ended before completion", reason)
	if errors.Is(reason, context.Canceled) {
		protocolError = llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_canceled", "stream was canceled", reason)
	}
	event, err := state.next(llmprotocol.Event{Type: llmprotocol.EventResponseFailed, Error: protocolError, StopReason: llmprotocol.StopError})
	if err != nil {
		return nil, err
	}
	return []llmprotocol.Event{event}, nil
}

type chatStreamDecoder struct {
	streamState
	framer sseFramer
}
type chatStreamEncoder struct {
	streamState
	itemStarted bool
	toolIndexes map[string]int
}

func (OpenAIChatCodec) NewDecoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamDecoder {
	return &chatStreamDecoder{streamState: streamState{context: context, policy: policy}, framer: newSSEFramer(policy.Limits.SSEFrameBytes)}
}

func (OpenAIChatCodec) NewEncoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamEncoder {
	return &chatStreamEncoder{streamState: streamState{context: context, policy: policy}, toolIndexes: make(map[string]int)}
}

type chatChunkWire struct {
	ID      string                `json:"id"`
	Object  string                `json:"object,omitempty"`
	Created int64                 `json:"created,omitempty"`
	Model   string                `json:"model,omitempty"`
	Choices []chatChunkChoiceWire `json:"choices,omitempty"`
	Usage   *chatUsageWire        `json:"usage,omitempty"`
	Error   *chatErrorWire        `json:"error,omitempty"`
}

type chatChunkChoiceWire struct {
	Index        int                `json:"index"`
	Delta        chatChunkDeltaWire `json:"delta"`
	FinishReason *string            `json:"finish_reason"`
}

type chatChunkDeltaWire struct {
	Role               string                  `json:"role,omitempty"`
	Content            *string                 `json:"content,omitempty"`
	Reasoning          *string                 `json:"reasoning_content,omitempty"`
	AlternateReasoning *string                 `json:"reasoning,omitempty"`
	Refusal            *string                 `json:"refusal,omitempty"`
	ToolCalls          []chatChunkToolCallWire `json:"tool_calls,omitempty"`
	Annotations        []chatAnnotationWire    `json:"annotations,omitempty"`
}

type chatChunkToolCallWire struct {
	Index    int              `json:"index"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function chatFunctionWire `json:"function"`
}

func (decoder *chatStreamDecoder) Push(chunk []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	frames, err := decoder.framer.Push(chunk)
	if err != nil {
		return nil, nil, err
	}
	var events []llmprotocol.Event
	var diagnostics llmprotocol.Diagnostics
	for _, frame := range frames {
		decoded, frameDiagnostics, decodeErr := decoder.pushFrame(frame)
		events = append(events, decoded...)
		diagnostics = appendDiagnostics(diagnostics, frameDiagnostics, decoder.policy.Limits.Diagnostics)
		if decodeErr != nil {
			return events, diagnostics, decodeErr
		}
	}
	return events, diagnostics, nil
}

func (decoder *chatStreamDecoder) pushFrame(frame []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	parsed, err := parseSSEFrame(frame, decoder.policy.Limits.SSEFrameBytes)
	if err != nil || len(parsed.Data) == 0 {
		return nil, nil, err
	}
	if bytes.Equal(bytes.TrimSpace(parsed.Data), []byte("[DONE]")) {
		event, err := decoder.next(llmprotocol.Event{Type: llmprotocol.EventResponseCompleted, StopReason: decoder.stop, Usage: &decoder.usage})
		return []llmprotocol.Event{event}, nil, err
	}
	var chunk chatChunkWire
	if err := decodeProviderWire(parsed.Data, &chunk, decoder.policy); err != nil {
		return nil, nil, err
	}
	if chunk.Error != nil {
		event, err := decoder.next(llmprotocol.Event{Type: llmprotocol.EventResponseFailed, Error: &llmprotocol.ProtocolError{Category: llmprotocol.ErrorUpstreamUnavailable, Code: chunk.Error.Code, Message: chunk.Error.Message}, StopReason: llmprotocol.StopError})
		return []llmprotocol.Event{event}, nil, err
	}
	events := make([]llmprotocol.Event, 0, 8)
	if !decoder.started {
		event, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventResponseStarted, ResponseID: chunk.ID, Model: chunk.Model})
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, event)
	}
	for _, choice := range chunk.Choices {
		if choice.Index != 0 {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "stream_multiple_choices", "streaming multiple choices is unsupported", nil)
		}
		needsItem := choice.Delta.Role != "" || choice.Delta.Content != nil || len(choice.Delta.Annotations) > 0 ||
			choice.Delta.Reasoning != nil || choice.Delta.AlternateReasoning != nil || choice.Delta.Refusal != nil
		if needsItem && !decoder.items[choice.Index] {
			event, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted, ItemIndex: choice.Index, Role: llmprotocol.RoleAssistant})
			if nextErr != nil {
				return nil, nil, nextErr
			}
			events = append(events, event)
		}
		if choice.Delta.Content != nil {
			event, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventOutputTextDelta, ItemIndex: choice.Index, Delta: *choice.Delta.Content})
			if nextErr != nil {
				return nil, nil, nextErr
			}
			events = append(events, event)
		}
		if len(choice.Delta.Annotations) > 0 {
			citations, decodeErr := decodeChatAnnotations(choice.Delta.Annotations)
			if decodeErr != nil {
				return nil, nil, decodeErr
			}
			content := llmprotocol.Content{Kind: llmprotocol.ContentText, Citations: citations}
			event, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventOutputTextDelta, ItemIndex: choice.Index, Content: &content})
			if nextErr != nil {
				return nil, nil, nextErr
			}
			events = append(events, event)
		}
		reasoning := choice.Delta.Reasoning
		if reasoning == nil {
			reasoning = choice.Delta.AlternateReasoning
		}
		if reasoning != nil {
			event, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventReasoningDelta, ItemIndex: choice.Index, Delta: *reasoning})
			if nextErr != nil {
				return nil, nil, nextErr
			}
			events = append(events, event)
		}
		if choice.Delta.Refusal != nil {
			content := llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: *choice.Delta.Refusal}
			event, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventOutputTextDelta, ItemIndex: choice.Index, Delta: *choice.Delta.Refusal, Content: &content})
			if nextErr != nil {
				return nil, nil, nextErr
			}
			events = append(events, event)
		}
		for _, call := range choice.Delta.ToolCalls {
			itemIndex := call.Index + 1
			if !decoder.items[itemIndex] {
				started, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted, ItemIndex: itemIndex, Role: llmprotocol.RoleAssistant, ToolCall: &llmprotocol.ToolCall{ID: call.ID, Name: call.Function.Name}})
				if nextErr != nil {
					return nil, nil, nextErr
				}
				events = append(events, started)
			}
			event, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventToolCallDelta, ItemIndex: itemIndex, ToolCall: &llmprotocol.ToolCall{ID: call.ID, Name: call.Function.Name, Arguments: call.Function.Arguments}})
			if nextErr != nil {
				return nil, nil, nextErr
			}
			events = append(events, event)
		}
		if choice.FinishReason != nil {
			decoder.stop = decodeChatStop(*choice.FinishReason)
			active := make([]int, 0, len(decoder.items))
			for itemIndex := range decoder.items {
				if !decoder.completedItems[itemIndex] {
					active = append(active, itemIndex)
				}
			}
			sort.Ints(active)
			for _, itemIndex := range active {
				event, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: itemIndex, StopReason: decoder.stop})
				if nextErr != nil {
					return nil, nil, nextErr
				}
				events = append(events, event)
			}
		}
	}
	if chunk.Usage != nil {
		usage := decodeChatUsage(*chunk.Usage)
		event, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventUsageUpdated, Usage: &usage})
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, event)
	}
	return events, nil, nil
}

func (decoder *chatStreamDecoder) Finalize(reason error) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	events, diagnostics, frameErr := finalizeDecoderFrames(decoder.framer.Finalize, decoder.pushFrame, decoder.policy.Limits.Diagnostics)
	if frameErr != nil {
		return events, diagnostics, frameErr
	}
	terminalEvents, err := decoder.finalize(reason)
	events = append(events, terminalEvents...)
	return events, diagnostics, err
}

func (encoder *chatStreamEncoder) Push(event llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already terminal", nil)
	}
	normalized, err := encoder.next(event)
	if err != nil {
		return nil, nil, err
	}
	event = normalized
	chunk := chatChunkWire{ID: event.ResponseID, Object: "chat.completion.chunk", Model: event.Model}
	choice := chatChunkChoiceWire{Index: 0}
	emitChunk := true
	switch event.Type {
	case llmprotocol.EventResponseStarted:
		choice.Delta.Role = "assistant"
		encoder.itemStarted = true
	case llmprotocol.EventOutputItemStarted:
		if encoder.itemStarted {
			return nil, nil, nil
		}
		choice.Delta.Role = "assistant"
		encoder.itemStarted = true
	case llmprotocol.EventOutputTextDelta:
		if event.Content != nil && event.Content.Kind == llmprotocol.ContentRefusal {
			choice.Delta.Refusal = &event.Delta
		} else {
			choice.Delta.Content = &event.Delta
		}
		if event.Content != nil && len(event.Content.Citations) > 0 {
			choice.Delta.Annotations = encodeChatAnnotations(event.Content.Citations)
		}
	case llmprotocol.EventReasoningDelta:
		if event.Content != nil && event.Content.Signature != "" {
			var diagnostics llmprotocol.Diagnostics
			if lossyErr := appendLossy(&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target, "reasoning.signature", "Chat Completions cannot represent a signed reasoning delta"); lossyErr != nil {
				return nil, diagnostics, lossyErr
			}
			choice.Delta.Reasoning = &event.Delta
			frame, encodeErr := encodeSSE("", chatChunkWire{ID: event.ResponseID, Object: "chat.completion.chunk", Model: event.Model, Choices: []chatChunkChoiceWire{choice}})
			return [][]byte{frame}, diagnostics, encodeErr
		}
		choice.Delta.Reasoning = &event.Delta
	case llmprotocol.EventToolCallDelta:
		if event.ToolCall == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_event_invalid", "tool event is invalid", nil)
		}
		index, found := encoder.toolIndexes[event.ToolCall.ID]
		if !found {
			index = len(encoder.toolIndexes)
			encoder.toolIndexes[event.ToolCall.ID] = index
		}
		choice.Delta.ToolCalls = []chatChunkToolCallWire{{Index: index, ID: event.ToolCall.ID, Type: "function", Function: chatFunctionWire{Name: event.ToolCall.Name, Arguments: event.ToolCall.Arguments}}}
	case llmprotocol.EventOutputItemCompleted:
		// Chat exposes one choice lifecycle even when the neutral response has
		// several ordered content items (for example reasoning followed by
		// text). Emit the choice finish only at the semantic response terminal;
		// completing an intermediate item must not terminate the public stream.
		return nil, nil, nil
	case llmprotocol.EventUsageUpdated:
		if event.Usage == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "usage event is invalid", nil)
		}
		chunk.Usage = encodeChatUsage(*event.Usage)
		emitChunk = true
	case llmprotocol.EventResponseCompleted:
		encoder.terminal = true
		reason := encodeChatStop(event.StopReason)
		finishFrame, encodeErr := encodeSSE("", chatChunkWire{
			ID: event.ResponseID, Object: "chat.completion.chunk", Model: event.Model,
			Choices: []chatChunkChoiceWire{{
				Index: 0, Delta: chatChunkDeltaWire{}, FinishReason: &reason,
			}},
		})
		if encodeErr != nil {
			return nil, nil, encodeErr
		}
		if event.Usage != nil && event.Usage.State == llmprotocol.UsageAvailable {
			usageChunk := chatChunkWire{ID: event.ResponseID, Object: "chat.completion.chunk", Model: event.Model, Usage: encodeChatUsage(*event.Usage)}
			usageFrame, encodeErr := encodeSSE("", usageChunk)
			if encodeErr != nil {
				return nil, nil, encodeErr
			}
			return [][]byte{finishFrame, usageFrame, []byte("data: [DONE]\n\n")}, nil, nil
		}
		return [][]byte{finishFrame, []byte("data: [DONE]\n\n")}, nil, nil
	case llmprotocol.EventResponseFailed:
		encoder.terminal = true
		if event.Error == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "error_event_invalid", "error event is invalid", nil)
		}
		chunk.Error = &chatErrorWire{Message: event.Error.Message, Type: string(event.Error.Category), Code: event.Error.Code}
	case llmprotocol.EventProviderOpaque:
		if encoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || encoder.context.Source != encoder.context.Target {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "opaque_event", "opaque provider event cannot cross formats", nil)
		}
		return [][]byte{append([]byte(nil), event.Opaque...)}, nil, nil
	default:
		emitChunk = false
	}
	if !emitChunk {
		return nil, nil, nil
	}
	if event.Type != llmprotocol.EventUsageUpdated && event.Type != llmprotocol.EventResponseFailed {
		chunk.Choices = []chatChunkChoiceWire{choice}
	}
	frame, err := encodeSSE("", chunk)
	return [][]byte{frame}, nil, err
}

func (encoder *chatStreamEncoder) Finalize(reason error) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, nil
	}
	encoder.terminal = true
	if reason != nil {
		body := OpenAIChatCodec{}.EncodeError(llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_incomplete", "stream ended before completion", reason))
		return [][]byte{append([]byte("data: "), append(body, []byte("\n\n")...)...)}, nil, nil
	}
	return [][]byte{[]byte("data: [DONE]\n\n")}, nil, nil
}

type responsesStreamDecoder struct {
	streamState
	framer                sseFramer
	nextAnnotationIndexes map[int]int
}
type responsesStreamEncoder struct {
	streamState
	startedItems     map[int]bool
	contentStarted   map[int]bool
	contentKinds     map[int]llmprotocol.ContentKind
	contentText      map[int]*strings.Builder
	contentCitations map[int][]llmprotocol.Citation
	wireSequence     uint64
}

func (OpenAIResponsesCodec) NewDecoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamDecoder {
	return &responsesStreamDecoder{
		streamState:           streamState{context: context, policy: policy},
		framer:                newSSEFramer(policy.Limits.SSEFrameBytes),
		nextAnnotationIndexes: make(map[int]int),
	}
}

func (OpenAIResponsesCodec) NewEncoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamEncoder {
	return &responsesStreamEncoder{
		streamState:      streamState{context: context, policy: policy},
		startedItems:     make(map[int]bool),
		contentStarted:   make(map[int]bool),
		contentKinds:     make(map[int]llmprotocol.ContentKind),
		contentText:      make(map[int]*strings.Builder),
		contentCitations: make(map[int][]llmprotocol.Citation),
	}
}

type responsesEventWire struct {
	Type            string                   `json:"type"`
	Sequence        uint64                   `json:"sequence_number,omitempty"`
	Response        *responsesResponseWire   `json:"response,omitempty"`
	Item            *responsesItemWire       `json:"item,omitempty"`
	ItemID          string                   `json:"item_id,omitempty"`
	OutputIndex     int                      `json:"output_index,omitempty"`
	ContentIndex    *int                     `json:"content_index,omitempty"`
	AnnotationIndex *int                     `json:"annotation_index,omitempty"`
	Delta           string                   `json:"delta,omitempty"`
	Text            string                   `json:"text,omitempty"`
	Part            *responsesContentWire    `json:"part,omitempty"`
	Annotation      *responsesAnnotationWire `json:"annotation,omitempty"`
	Name            string                   `json:"name,omitempty"`
	Error           *responsesErrorWire      `json:"error,omitempty"`
}

func (decoder *responsesStreamDecoder) Push(chunk []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	frames, err := decoder.framer.Push(chunk)
	if err != nil {
		return nil, nil, err
	}
	var events []llmprotocol.Event
	var diagnostics llmprotocol.Diagnostics
	for _, frame := range frames {
		decoded, frameDiagnostics, decodeErr := decoder.pushFrame(frame)
		events = append(events, decoded...)
		diagnostics = appendDiagnostics(diagnostics, frameDiagnostics, decoder.policy.Limits.Diagnostics)
		if decodeErr != nil {
			return events, diagnostics, decodeErr
		}
	}
	return events, diagnostics, nil
}

func (decoder *responsesStreamDecoder) pushFrame(frame []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	parsed, err := parseSSEFrame(frame, decoder.policy.Limits.SSEFrameBytes)
	if err != nil || len(parsed.Data) == 0 {
		return nil, nil, err
	}
	if bytes.Equal(bytes.TrimSpace(parsed.Data), []byte("[DONE]")) {
		// A Responses terminal event is authoritative. Some compatible
		// transports append this sentinel for framing compatibility; it is not
		// a second semantic completion.
		if decoder.terminal {
			return nil, nil, nil
		}
		event, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventResponseCompleted, StopReason: decoder.stop, Usage: &decoder.usage})
		return []llmprotocol.Event{event}, nil, nextErr
	}
	var wire responsesEventWire
	if err := decodeProviderWire(parsed.Data, &wire, decoder.policy); err != nil {
		return nil, nil, err
	}
	if wire.Type == "" {
		wire.Type = parsed.Event
	}
	event := llmprotocol.Event{ResponseID: decoder.context.ResponseID, Model: decoder.context.PublicModel, ItemIndex: wire.OutputIndex, ItemID: wire.ItemID, Delta: wire.Delta}
	switch wire.Type {
	case "response.created", "response.in_progress":
		if decoder.started {
			return nil, nil, nil
		}
		event.Type = llmprotocol.EventResponseStarted
		if wire.Response != nil {
			event.ResponseID, event.Model = wire.Response.ID, wire.Response.Model
		}
	case "response.output_item.added":
		event.Type = llmprotocol.EventOutputItemStarted
		if wire.Item != nil {
			event.ItemID = wire.Item.ID
			event.Role = llmprotocol.RoleAssistant
			if wire.Item.Type == "function_call" {
				event.ToolCall = &llmprotocol.ToolCall{ID: wire.Item.CallID, Name: wire.Item.Name, Arguments: wire.Item.Arguments}
			}
		}
	case "response.output_text.delta":
		event.Type = llmprotocol.EventOutputTextDelta
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentText, Text: wire.Delta}
	case "response.output_text.annotation.added":
		if wire.Annotation == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_missing", "Responses citation event is missing its annotation", nil)
		}
		itemID, itemFound := decoder.itemIDs[wire.OutputIndex]
		if !itemFound || wire.ItemID == "" || wire.ItemID != itemID {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_item", "Responses citation event does not match its active output item", nil)
		}
		if wire.ContentIndex == nil || *wire.ContentIndex != 0 {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_content_index", "Responses citation event has an unsupported content index", nil)
		}
		expectedAnnotationIndex := decoder.nextAnnotationIndexes[wire.OutputIndex]
		if wire.AnnotationIndex == nil || *wire.AnnotationIndex != expectedAnnotationIndex {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_annotation_index", "Responses citation indexes must be monotonic and contiguous", nil)
		}
		citations, decodeErr := decodeResponsesAnnotations([]responsesAnnotationWire{*wire.Annotation})
		if decodeErr != nil {
			return nil, nil, decodeErr
		}
		event.Type = llmprotocol.EventOutputTextDelta
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentText, Citations: citations}
		decoder.nextAnnotationIndexes[wire.OutputIndex] = expectedAnnotationIndex + 1
	case "response.refusal.delta":
		event.Type = llmprotocol.EventOutputTextDelta
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: wire.Delta}
	case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
		event.Type = llmprotocol.EventReasoningDelta
		event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: wire.Delta}
	case "response.function_call_arguments.delta":
		event.Type = llmprotocol.EventToolCallDelta
		call := decoder.toolCalls[wire.OutputIndex]
		if wire.Item != nil && wire.Item.CallID != "" {
			call.ID = wire.Item.CallID
		}
		if wire.Name != "" {
			call.Name = wire.Name
		}
		call.Arguments = wire.Delta
		event.ToolCall = &call
	case "response.output_item.done":
		event.Type = llmprotocol.EventOutputItemCompleted
		if wire.Item != nil {
			event.ItemID = wire.Item.ID
			switch wire.Item.Type {
			case "function_call":
				event.ToolCall = &llmprotocol.ToolCall{ID: wire.Item.CallID, Name: wire.Item.Name, Arguments: wire.Item.Arguments}
			case "message":
				if decoder.itemKinds[wire.OutputIndex] == llmprotocol.ContentToolCall {
					return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_item_kind_mismatch", "upstream completed a tool item as a message", nil)
				}
			case "reasoning":
				event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentReasoning}
			default:
				return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_output_item", "Responses completed an unsupported output item", nil)
			}
		}
	case "response.completed", "response.incomplete":
		event.Type = llmprotocol.EventResponseCompleted
		event.StopReason = llmprotocol.StopEndTurn
		if wire.Type == "response.incomplete" {
			event.StopReason = llmprotocol.StopMaxTokens
		}
		if wire.Response != nil && wire.Response.Usage != nil {
			usage := decodeResponsesUsage(*wire.Response.Usage)
			event.Usage = &usage
		}
	case "response.failed", "error":
		event.Type = llmprotocol.EventResponseFailed
		event.StopReason = llmprotocol.StopError
		event.Error = llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_stream_error", "upstream stream failed", nil)
		upstreamError := wire.Error
		if upstreamError == nil && wire.Response != nil {
			upstreamError = wire.Response.Error
		}
		if upstreamError != nil {
			event.Error.Code, event.Error.Message = upstreamError.Code, upstreamError.Message
		}
	case "response.content_part.added", "response.content_part.done", "response.output_text.done", "response.refusal.done", "response.function_call_arguments.done":
		return nil, nil, nil
	default:
		if decoder.policy.UnknownFields == llmprotocol.UnknownPreserveSameFormat && decoder.context.Source == decoder.context.Target {
			event.Type, event.Opaque = llmprotocol.EventProviderOpaque, append([]byte(nil), frame...)
		} else {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unknown_stream_event", "Responses stream event is unsupported", nil)
		}
	}
	normalized, nextErr := decoder.next(event)
	return []llmprotocol.Event{normalized}, nil, nextErr
}

func (decoder *responsesStreamDecoder) Finalize(reason error) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	events, diagnostics, frameErr := finalizeDecoderFrames(decoder.framer.Finalize, decoder.pushFrame, decoder.policy.Limits.Diagnostics)
	if frameErr != nil {
		return events, diagnostics, frameErr
	}
	terminalEvents, err := decoder.finalize(reason)
	events = append(events, terminalEvents...)
	return events, diagnostics, err
}

func (encoder *responsesStreamEncoder) Push(event llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already terminal", nil)
	}
	normalized, err := encoder.next(event)
	if err != nil {
		return nil, nil, err
	}
	event = normalized
	wire := responsesEventWire{ItemID: event.ItemID, OutputIndex: event.ItemIndex, Delta: event.Delta}
	switch event.Type {
	case llmprotocol.EventResponseStarted:
		wire.Type = "response.created"
		wire.Response = &responsesResponseWire{ID: event.ResponseID, Object: "response", Model: event.Model, Status: "in_progress"}
	case llmprotocol.EventOutputItemStarted:
		wire.Type = "response.output_item.added"
		wire.Item = &responsesItemWire{Type: "message", ID: event.ItemID, Role: "assistant", Status: "in_progress"}
		if event.ToolCall != nil {
			wire.Item.Type, wire.Item.CallID, wire.Item.Name, wire.Item.Arguments = "function_call", event.ToolCall.ID, event.ToolCall.Name, event.ToolCall.Arguments
			wire.Item.Role = ""
		} else if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
			wire.Item.Type, wire.Item.Role = "reasoning", ""
		} else if event.Content != nil {
			encoder.contentKinds[event.ItemIndex] = event.Content.Kind
		}
	case llmprotocol.EventOutputTextDelta:
		kind := llmprotocol.ContentText
		var newCitations []llmprotocol.Citation
		annotationBase := len(encoder.contentCitations[event.ItemIndex])
		if event.Content != nil && event.Content.Kind == llmprotocol.ContentRefusal {
			kind = llmprotocol.ContentRefusal
		}
		encoder.contentKinds[event.ItemIndex] = kind
		builder := encoder.contentText[event.ItemIndex]
		if builder == nil {
			builder = &strings.Builder{}
			encoder.contentText[event.ItemIndex] = builder
		}
		builder.WriteString(event.Delta)
		if event.Content != nil && len(event.Content.Citations) > 0 {
			newCitations = event.Content.Citations
			encoder.contentCitations[event.ItemIndex] = append(encoder.contentCitations[event.ItemIndex], newCitations...)
		}
		frames, frameErr := encoder.startResponsesContent(event, kind)
		if frameErr != nil {
			return nil, nil, frameErr
		}
		if event.Delta != "" {
			wire.Type = responsesTextDeltaType(kind)
			wire.Sequence = encoder.nextWireSequence()
			frame, encodeErr := encodeSSE(wire.Type, wire)
			if encodeErr != nil {
				return nil, nil, encodeErr
			}
			frames = append(frames, frame)
		}
		for annotationOffset, annotation := range encodeResponsesAnnotations(newCitations) {
			annotationIndex := annotationBase + annotationOffset
			annotationWire := responsesEventWire{
				Type: "response.output_text.annotation.added", Sequence: encoder.nextWireSequence(),
				ItemID: event.ItemID, OutputIndex: event.ItemIndex, ContentIndex: responsesContentIndex(),
				AnnotationIndex: &annotationIndex, Annotation: &annotation,
			}
			frame, encodeErr := encodeSSE(annotationWire.Type, annotationWire)
			if encodeErr != nil {
				return nil, nil, encodeErr
			}
			frames = append(frames, frame)
		}
		return frames, nil, nil
	case llmprotocol.EventReasoningDelta:
		if event.Content != nil && event.Content.Signature != "" {
			var diagnostics llmprotocol.Diagnostics
			if lossyErr := appendLossy(&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target, "reasoning.signature", "Responses cannot represent a signed reasoning delta"); lossyErr != nil {
				return nil, diagnostics, lossyErr
			}
			wire.Type = "response.reasoning_text.delta"
			wire.Sequence = encoder.nextWireSequence()
			frame, encodeErr := encodeSSE(wire.Type, wire)
			return [][]byte{frame}, diagnostics, encodeErr
		}
		wire.Type = "response.reasoning_text.delta"
	case llmprotocol.EventToolCallDelta:
		if event.ToolCall == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_event_invalid", "tool event is invalid", nil)
		}
		wire.Type, wire.ItemID, wire.Name, wire.Delta = "response.function_call_arguments.delta", event.ToolCall.ID, event.ToolCall.Name, event.ToolCall.Arguments
	case llmprotocol.EventOutputItemCompleted:
		wire.Type = "response.output_item.done"
		wire.Item = &responsesItemWire{Type: "message", ID: event.ItemID, Role: "assistant", Status: "completed"}
		if event.ToolCall != nil {
			wire.Item.Type, wire.Item.Role = "function_call", ""
			wire.Item.CallID, wire.Item.Name, wire.Item.Arguments = event.ToolCall.ID, event.ToolCall.Name, event.ToolCall.Arguments
		} else if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
			wire.Item.Type, wire.Item.Role = "reasoning", ""
		} else {
			kind := encoder.contentKinds[event.ItemIndex]
			if kind == "" {
				kind = llmprotocol.ContentText
			}
			frames, frameErr := encoder.completeResponsesContent(event, kind)
			if frameErr != nil {
				return nil, nil, frameErr
			}
			text := ""
			if builder := encoder.contentText[event.ItemIndex]; builder != nil {
				text = builder.String()
			}
			content, marshalErr := json.Marshal([]responsesContentWire{responsesContentPart(kind, text, encoder.contentCitations[event.ItemIndex])})
			if marshalErr != nil {
				return nil, nil, marshalErr
			}
			wire.Item.Content = content
			wire.Sequence = encoder.nextWireSequence()
			done, encodeErr := encodeSSE(wire.Type, wire)
			if encodeErr != nil {
				return nil, nil, encodeErr
			}
			return append(frames, done), nil, nil
		}
	case llmprotocol.EventUsageUpdated:
		return nil, nil, nil
	case llmprotocol.EventResponseCompleted:
		if event.Usage == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "terminal usage is invalid", nil)
		}
		wire.Type = "response.completed"
		wire.Response = &responsesResponseWire{ID: event.ResponseID, Object: "response", Model: event.Model, Status: "completed", Usage: encodeResponsesUsage(*event.Usage)}
		encoder.terminal = true
	case llmprotocol.EventResponseFailed:
		if event.Error == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "error_event_invalid", "error event is invalid", nil)
		}
		wire.Type = "response.failed"
		wire.Error = &responsesErrorWire{Code: event.Error.Code, Message: event.Error.Message}
		wire.Response = &responsesResponseWire{ID: event.ResponseID, Object: "response", Status: "failed", Error: wire.Error}
		encoder.terminal = true
	case llmprotocol.EventProviderOpaque:
		if encoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || encoder.context.Source != encoder.context.Target {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "opaque_event", "opaque provider event cannot cross formats", nil)
		}
		return [][]byte{append([]byte(nil), event.Opaque...)}, nil, nil
	default:
		return nil, nil, nil
	}
	wire.Sequence = encoder.nextWireSequence()
	frame, err := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, nil, err
}

func (encoder *responsesStreamEncoder) nextWireSequence() uint64 {
	encoder.wireSequence++
	return encoder.wireSequence
}

func responsesContentIndex() *int {
	value := 0
	return &value
}

func responsesContentPart(kind llmprotocol.ContentKind, text string, citations []llmprotocol.Citation) responsesContentWire {
	if kind == llmprotocol.ContentRefusal {
		return responsesContentWire{Type: "refusal", Refusal: text}
	}
	return responsesContentWire{Type: "output_text", Text: text, Annotations: encodeResponsesAnnotations(citations)}
}

func responsesTextDeltaType(kind llmprotocol.ContentKind) string {
	if kind == llmprotocol.ContentRefusal {
		return "response.refusal.delta"
	}
	return "response.output_text.delta"
}

func responsesTextDoneType(kind llmprotocol.ContentKind) string {
	if kind == llmprotocol.ContentRefusal {
		return "response.refusal.done"
	}
	return "response.output_text.done"
}

func (encoder *responsesStreamEncoder) startResponsesContent(
	event llmprotocol.Event,
	kind llmprotocol.ContentKind,
) ([][]byte, error) {
	if encoder.contentStarted[event.ItemIndex] {
		return nil, nil
	}
	encoder.contentStarted[event.ItemIndex] = true
	wire := responsesEventWire{
		Type: "response.content_part.added", Sequence: encoder.nextWireSequence(),
		ItemID: event.ItemID, OutputIndex: event.ItemIndex, ContentIndex: responsesContentIndex(),
	}
	part := responsesContentPart(kind, "", nil)
	wire.Part = &part
	frame, err := encodeSSE(wire.Type, wire)
	if err != nil {
		return nil, err
	}
	return [][]byte{frame}, nil
}

func (encoder *responsesStreamEncoder) completeResponsesContent(
	event llmprotocol.Event,
	kind llmprotocol.ContentKind,
) ([][]byte, error) {
	frames, err := encoder.startResponsesContent(event, kind)
	if err != nil {
		return nil, err
	}
	text := ""
	if builder := encoder.contentText[event.ItemIndex]; builder != nil {
		text = builder.String()
	}
	done := responsesEventWire{
		Type: responsesTextDoneType(kind), Sequence: encoder.nextWireSequence(),
		ItemID: event.ItemID, OutputIndex: event.ItemIndex, ContentIndex: responsesContentIndex(), Text: text,
	}
	doneFrame, err := encodeSSE(done.Type, done)
	if err != nil {
		return nil, err
	}
	part := responsesContentPart(kind, text, encoder.contentCitations[event.ItemIndex])
	partDone := responsesEventWire{
		Type: "response.content_part.done", Sequence: encoder.nextWireSequence(),
		ItemID: event.ItemID, OutputIndex: event.ItemIndex, ContentIndex: responsesContentIndex(), Part: &part,
	}
	partFrame, err := encodeSSE(partDone.Type, partDone)
	if err != nil {
		return nil, err
	}
	return append(frames, doneFrame, partFrame), nil
}

func (encoder *responsesStreamEncoder) Finalize(reason error) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, nil
	}
	encoder.terminal = true
	protocolError := llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "stream_incomplete", "stream ended before completion", reason)
	wire := responsesEventWire{Type: "response.failed", Error: &responsesErrorWire{Code: protocolError.Code, Message: protocolError.Message}}
	frame, err := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, nil, err
}

type anthropicStreamDecoder struct {
	streamState
	framer sseFramer
}
type anthropicStreamEncoder struct {
	streamState
	blocks map[int]llmprotocol.ContentKind
}

func (AnthropicMessagesCodec) NewDecoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamDecoder {
	return &anthropicStreamDecoder{streamState: streamState{context: context, policy: policy}, framer: newSSEFramer(policy.Limits.SSEFrameBytes)}
}

func (AnthropicMessagesCodec) NewEncoder(context llmprotocol.StreamContext, policy llmprotocol.Policy) llmprotocol.StreamEncoder {
	return &anthropicStreamEncoder{streamState: streamState{context: context, policy: policy}, blocks: make(map[int]llmprotocol.ContentKind)}
}

type anthropicEventWire struct {
	Type         string                 `json:"type"`
	Message      *anthropicResponseWire `json:"message,omitempty"`
	Index        int                    `json:"index,omitempty"`
	ContentBlock *anthropicContentWire  `json:"content_block,omitempty"`
	Delta        *anthropicDeltaWire    `json:"delta,omitempty"`
	Usage        *anthropicUsageWire    `json:"usage,omitempty"`
	Error        *anthropicErrorWire    `json:"error,omitempty"`
}

type anthropicDeltaWire struct {
	Type         string  `json:"type"`
	Text         string  `json:"text,omitempty"`
	Thinking     string  `json:"thinking,omitempty"`
	PartialJSON  string  `json:"partial_json,omitempty"`
	Signature    string  `json:"signature,omitempty"`
	StopReason   *string `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
}

func (decoder *anthropicStreamDecoder) Push(chunk []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	frames, err := decoder.framer.Push(chunk)
	if err != nil {
		return nil, nil, err
	}
	var events []llmprotocol.Event
	var diagnostics llmprotocol.Diagnostics
	for _, frame := range frames {
		decoded, frameDiagnostics, decodeErr := decoder.pushFrame(frame)
		events = append(events, decoded...)
		diagnostics = appendDiagnostics(diagnostics, frameDiagnostics, decoder.policy.Limits.Diagnostics)
		if decodeErr != nil {
			return events, diagnostics, decodeErr
		}
	}
	return events, diagnostics, nil
}

func (decoder *anthropicStreamDecoder) pushFrame(frame []byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	parsed, err := parseSSEFrame(frame, decoder.policy.Limits.SSEFrameBytes)
	if err != nil || len(parsed.Data) == 0 {
		return nil, nil, err
	}
	var wire anthropicEventWire
	if err := decodeProviderWire(parsed.Data, &wire, decoder.policy); err != nil {
		return nil, nil, err
	}
	if wire.Type == "" {
		wire.Type = parsed.Event
	}
	events := make([]llmprotocol.Event, 0, 2)
	switch wire.Type {
	case "message_start":
		event := llmprotocol.Event{Type: llmprotocol.EventResponseStarted}
		if wire.Message != nil {
			event.ResponseID, event.Model = wire.Message.ID, wire.Message.Model
			if wire.Message.Usage != nil {
				usage := decodeAnthropicStreamUsage(*wire.Message.Usage, true)
				event.Usage = &usage
			}
		}
		normalized, nextErr := decoder.next(event)
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, normalized)
	case "content_block_start":
		event := llmprotocol.Event{Type: llmprotocol.EventOutputItemStarted, ItemIndex: wire.Index, Role: llmprotocol.RoleAssistant}
		if wire.ContentBlock != nil {
			event.ItemID = wire.ContentBlock.ID
			switch wire.ContentBlock.Type {
			case "tool_use":
				event.ToolCall = &llmprotocol.ToolCall{ID: wire.ContentBlock.ID, Name: wire.ContentBlock.Name, Arguments: string(wire.ContentBlock.Input)}
			case "thinking":
				event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Signature: wire.ContentBlock.Signature}
			case "text":
				event.Content = &llmprotocol.Content{Kind: llmprotocol.ContentText}
			default:
				return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_content", "Anthropic stream content block is unsupported", nil)
			}
		}
		normalized, nextErr := decoder.next(event)
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, normalized)
	case "content_block_delta":
		if wire.Delta == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_stream_delta", "Anthropic stream delta is missing", nil)
		}
		event := llmprotocol.Event{ItemIndex: wire.Index}
		switch wire.Delta.Type {
		case "text_delta":
			event.Type, event.Delta, event.Content = llmprotocol.EventOutputTextDelta, wire.Delta.Text, &llmprotocol.Content{Kind: llmprotocol.ContentText, Text: wire.Delta.Text}
		case "thinking_delta":
			event.Type, event.Delta, event.Content = llmprotocol.EventReasoningDelta, wire.Delta.Thinking, &llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Text: wire.Delta.Thinking}
		case "input_json_delta":
			event.Type, event.ToolCall = llmprotocol.EventToolCallDelta, &llmprotocol.ToolCall{Arguments: wire.Delta.PartialJSON}
		case "signature_delta":
			event.Type, event.Content = llmprotocol.EventReasoningDelta, &llmprotocol.Content{Kind: llmprotocol.ContentReasoning, Signature: wire.Delta.Signature}
		default:
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unknown_stream_delta", "Anthropic stream delta is unsupported", nil)
		}
		normalized, nextErr := decoder.next(event)
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, normalized)
	case "content_block_stop":
		normalized, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventOutputItemCompleted, ItemIndex: wire.Index})
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, normalized)
	case "message_delta":
		if wire.Delta != nil && wire.Delta.StopReason != nil {
			decoder.stop = decodeAnthropicStop(*wire.Delta.StopReason)
		}
		if wire.Usage != nil {
			usage := decodeAnthropicStreamUsage(*wire.Usage, false)
			normalized, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventUsageUpdated, Usage: &usage})
			if nextErr != nil {
				return nil, nil, nextErr
			}
			events = append(events, normalized)
		}
	case "message_stop":
		normalized, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventResponseCompleted, StopReason: decoder.stop, Usage: &decoder.usage})
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, normalized)
	case "error":
		protocolError := llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "upstream_stream_error", "upstream stream failed", nil)
		if wire.Error != nil {
			protocolError.Code, protocolError.Message = wire.Error.Type, wire.Error.Message
		}
		normalized, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventResponseFailed, Error: protocolError, StopReason: llmprotocol.StopError})
		if nextErr != nil {
			return nil, nil, nextErr
		}
		events = append(events, normalized)
	case "ping":
		return nil, nil, nil
	default:
		if decoder.policy.UnknownFields == llmprotocol.UnknownPreserveSameFormat && decoder.context.Source == decoder.context.Target {
			normalized, nextErr := decoder.next(llmprotocol.Event{Type: llmprotocol.EventProviderOpaque, Opaque: append([]byte(nil), frame...)})
			if nextErr != nil {
				return nil, nil, nextErr
			}
			events = append(events, normalized)
		} else {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unknown_stream_event", "Anthropic stream event is unsupported", nil)
		}
	}
	return events, nil, nil
}

func decodeAnthropicStreamUsage(wire anthropicUsageWire, initial bool) llmprotocol.Usage {
	usage := llmprotocol.Usage{State: llmprotocol.UsageAvailable}
	if initial || wire.InputTokens > 0 || wire.CacheReadInputTokens > 0 || wire.CacheCreationInputTokens > 0 {
		uncached := wire.InputTokens - wire.CacheReadInputTokens - wire.CacheCreationInputTokens
		if uncached < 0 {
			uncached = 0
		}
		usage.InputUncached = authoritative(uncached)
		usage.InputCacheRead = authoritative(wire.CacheReadInputTokens)
		usage.InputCacheWrite = authoritative(wire.CacheCreationInputTokens)
		usage.InputTotal = authoritative(wire.InputTokens)
	}
	if wire.OutputTokens > 0 || !initial {
		usage.OutputOther = authoritative(wire.OutputTokens)
		usage.OutputTotal = authoritative(wire.OutputTokens)
	}
	return usage
}

func (decoder *anthropicStreamDecoder) Finalize(reason error) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	events, diagnostics, frameErr := finalizeDecoderFrames(decoder.framer.Finalize, decoder.pushFrame, decoder.policy.Limits.Diagnostics)
	if frameErr != nil {
		return events, diagnostics, frameErr
	}
	terminalEvents, err := decoder.finalize(reason)
	events = append(events, terminalEvents...)
	return events, diagnostics, err
}

func (encoder *anthropicStreamEncoder) Push(event llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already terminal", nil)
	}
	normalized, pushErr := encoder.next(event)
	if pushErr != nil {
		return nil, nil, pushErr
	}
	event = normalized
	wire := anthropicEventWire{}
	switch event.Type {
	case llmprotocol.EventResponseStarted:
		wire.Type = "message_start"
		usage := &anthropicUsageWire{}
		if event.Usage != nil && event.Usage.State == llmprotocol.UsageAvailable {
			usage = encodeAnthropicUsage(*event.Usage)
		}
		wire.Message = &anthropicResponseWire{ID: event.ResponseID, Type: "message", Role: "assistant", Model: event.Model, Content: json.RawMessage(`[]`), Usage: usage}
	case llmprotocol.EventOutputItemStarted:
		wire.Type, wire.Index = "content_block_start", event.ItemIndex
		block := &anthropicContentWire{Type: "text", Text: ""}
		if event.ToolCall != nil {
			block.Type, block.ID, block.Name, block.Input = "tool_use", event.ToolCall.ID, event.ToolCall.Name, json.RawMessage(`{}`)
			encoder.blocks[event.ItemIndex] = llmprotocol.ContentToolCall
		} else if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
			block.Type, block.Text, block.Thinking, block.Signature = "thinking", "", "", event.Content.Signature
			encoder.blocks[event.ItemIndex] = llmprotocol.ContentReasoning
		} else {
			encoder.blocks[event.ItemIndex] = llmprotocol.ContentText
		}
		wire.ContentBlock = block
	case llmprotocol.EventOutputTextDelta:
		var diagnostics llmprotocol.Diagnostics
		if event.Content != nil && len(event.Content.Citations) > 0 {
			if lossyErr := appendLossy(
				&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target,
				"content.citations", "Messages cannot represent URL citations",
			); lossyErr != nil {
				return nil, diagnostics, lossyErr
			}
		}
		if event.Content != nil && event.Content.Kind == llmprotocol.ContentRefusal {
			if lossyErr := appendLossy(&diagnostics, encoder.policy, encoder.context.Source, encoder.context.Target, "content.refusal", "Messages represents refusal as ordinary text"); lossyErr != nil {
				return nil, diagnostics, lossyErr
			}
			wire.Type, wire.Index, wire.Delta = "content_block_delta", event.ItemIndex, &anthropicDeltaWire{Type: "text_delta", Text: event.Delta}
			frame, encodeErr := encodeSSE(wire.Type, wire)
			return [][]byte{frame}, diagnostics, encodeErr
		}
		wire.Type, wire.Index, wire.Delta = "content_block_delta", event.ItemIndex, &anthropicDeltaWire{Type: "text_delta", Text: event.Delta}
		frame, encodeErr := encodeSSE(wire.Type, wire)
		return [][]byte{frame}, diagnostics, encodeErr
	case llmprotocol.EventReasoningDelta:
		wire.Type, wire.Index = "content_block_delta", event.ItemIndex
		if event.Content != nil && event.Content.Signature != "" {
			wire.Delta = &anthropicDeltaWire{Type: "signature_delta", Signature: event.Content.Signature}
		} else {
			wire.Delta = &anthropicDeltaWire{Type: "thinking_delta", Thinking: event.Delta}
		}
	case llmprotocol.EventToolCallDelta:
		if event.ToolCall == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "tool_event_invalid", "tool event is invalid", nil)
		}
		wire.Type, wire.Index, wire.Delta = "content_block_delta", event.ItemIndex, &anthropicDeltaWire{Type: "input_json_delta", PartialJSON: event.ToolCall.Arguments}
	case llmprotocol.EventOutputItemCompleted:
		wire.Type, wire.Index = "content_block_stop", event.ItemIndex
	case llmprotocol.EventUsageUpdated:
		if event.Usage == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "usage event is invalid", nil)
		}
		wire.Type, wire.Usage = "message_delta", encodeAnthropicUsage(*event.Usage)
	case llmprotocol.EventResponseCompleted:
		if event.Usage == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "usage_event_invalid", "terminal usage is invalid", nil)
		}
		stop := encodeAnthropicStop(event.StopReason)
		delta := anthropicEventWire{Type: "message_delta", Delta: &anthropicDeltaWire{Type: "message_delta", StopReason: &stop}, Usage: encodeAnthropicUsage(*event.Usage)}
		first, err := encodeSSE(delta.Type, delta)
		if err != nil {
			return nil, nil, err
		}
		wire.Type = "message_stop"
		encoder.terminal = true
		second, err := encodeSSE(wire.Type, wire)
		return [][]byte{first, second}, nil, err
	case llmprotocol.EventResponseFailed:
		if event.Error == nil {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorInternal, "error_event_invalid", "error event is invalid", nil)
		}
		wire.Type = "error"
		wire.Error = &anthropicErrorWire{Type: event.Error.Code, Message: event.Error.Message}
		encoder.terminal = true
	case llmprotocol.EventProviderOpaque:
		if encoder.policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat || encoder.context.Source != encoder.context.Target {
			return nil, nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "opaque_event", "opaque provider event cannot cross formats", nil)
		}
		return [][]byte{append([]byte(nil), event.Opaque...)}, nil, nil
	default:
		return nil, nil, nil
	}
	frame, pushErr := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, nil, pushErr
}

func (encoder *anthropicStreamEncoder) Finalize(reason error) ([][]byte, llmprotocol.Diagnostics, error) {
	if encoder.terminal {
		return nil, nil, nil
	}
	encoder.terminal = true
	wire := anthropicEventWire{Type: "error", Error: &anthropicErrorWire{Type: "stream_incomplete", Message: "stream ended before completion"}}
	frame, err := encodeSSE(wire.Type, wire)
	return [][]byte{frame}, nil, err
}
