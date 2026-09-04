package protocolcodec

import (
	"bytes"
	"encoding/json"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type sseFrame struct {
	Event   string
	Data    []byte
	HasData bool
}

// streamWireIndexes maps protocol-neutral item identities to the compact,
// contiguous indexes required by provider stream contracts. A source format
// may reserve neutral indexes; those implementation details must never leak
// onto a target wire.
type streamWireIndexes struct {
	items map[int]int
	next  int
}

type streamContentKey struct {
	item    int
	content int
}

func contentKey(event llmprotocol.Event) streamContentKey {
	return streamContentKey{item: event.ItemIndex, content: event.ContentIndex}
}

func (indexes *streamWireIndexes) translate(event llmprotocol.Event) int {
	switch event.Type {
	case llmprotocol.EventOutputItemStarted:
		if indexes.items == nil {
			indexes.items = make(map[int]int)
		}
		if wireIndex, found := indexes.items[event.ItemIndex]; found {
			return wireIndex
		}
		wireIndex := indexes.next
		indexes.next++
		indexes.items[event.ItemIndex] = wireIndex
		return wireIndex
	case llmprotocol.EventOutputTextDelta, llmprotocol.EventReasoningDelta,
		llmprotocol.EventToolCallDelta, llmprotocol.EventImageGenerationProgress,
		llmprotocol.EventOutputItemCompleted:
		if wireIndex, found := indexes.items[event.ItemIndex]; found {
			return wireIndex
		}
	}
	return event.ItemIndex
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
	return parseSSEFrameAtPosition(frame, limit, true)
}

func parseSSEFrameAtPosition(frame []byte, limit int, first bool) (sseFrame, error) {
	if err := validateSSEFrameBytes(frame, limit); err != nil {
		return sseFrame{}, err
	}
	normalized, err := normalizeSSEFrame(frame, first)
	if err != nil {
		return sseFrame{}, err
	}
	var result sseFrame
	for _, line := range bytes.Split(normalized, []byte("\n")) {
		parseSSELine(&result, line)
	}
	return result, nil
}

func validateSSEFrameBytes(frame []byte, limit int) error {
	if len(frame) == 0 || len(frame) > limit {
		return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "sse_frame_limit", "SSE frame is empty or too large", nil)
	}
	if !utf8.Valid(frame) {
		return llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_utf8", "upstream SSE frame is not valid UTF-8", nil,
		)
	}
	return nil
}

func normalizeSSEFrame(frame []byte, first bool) ([]byte, error) {
	bom := []byte{0xef, 0xbb, 0xbf}
	if !first && bytes.HasPrefix(frame, bom) {
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorUpstreamUnavailable,
			"unexpected_stream_bom",
			"SSE stream contains a UTF-8 BOM after its first event",
			nil,
		)
	}
	normalized := frame
	if first {
		normalized = bytes.TrimPrefix(normalized, bom)
	}
	normalized = bytes.ReplaceAll(normalized, []byte("\r\n"), []byte("\n"))
	normalized = bytes.ReplaceAll(normalized, []byte("\r"), []byte("\n"))
	return normalized, nil
}

func parseSSELine(result *sseFrame, line []byte) {
	if len(line) == 0 || line[0] == ':' {
		return
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
		result.HasData = true
		if len(result.Data) > 0 {
			result.Data = append(result.Data, '\n')
		}
		result.Data = append(result.Data, value...)
	}
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
