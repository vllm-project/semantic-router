package extproc

import "bytes"

const maxSSEFrameBytes = 1 << 20

var sseFrameDelimiters = [][]byte{
	[]byte("\n\n"),
	[]byte("\r\n\r\n"),
	[]byte("\r\r"),
}

// reassembleSSEFrames merges any partial SSE frame carried over from a
// prior response-body chunk (pending) with the newly arrived bytes
// (chunk), then splits the result into the leading run of COMPLETE SSE
// frames and any trailing incomplete remainder.
//
// Envoy STREAMED mode delivers the upstream response body split at
// arbitrary byte offsets, with no guarantee that a chunk boundary aligns
// to an SSE frame boundary. A single event may therefore straddle two
// chunks. Parsing a chunk that ends mid-frame drops the partial frame
// silently (json.Unmarshal fails on the truncated payload), so callers
// must hold the remainder and prepend it to the next chunk before parsing.
//
// The returned complete slice is safe for the caller to parse and forward;
// the remainder must be stored and passed back as pending on the next call.
// Both are subslices of a freshly allocated buffer, so retaining remainder
// across calls does not alias Envoy's chunk buffer.
//
// A caller should retain remainder between chunks. At an authoritative
// end-of-stream boundary it may flush that remainder as the provider's final
// event, including providers that omit the conventional trailing blank line.
func reassembleSSEFrames(pending, chunk []byte) (complete, remainder []byte) {
	batch := reassembleSSEFramesBounded(pending, chunk, false, false)
	return batch.complete, batch.remainder
}

type sseFrameBatch struct {
	complete  []byte
	remainder []byte
	dropping  bool
	invalid   bool
}

// reassembleSSEFramesBounded is the stateful streaming entrypoint shared by
// every provider path. It caps each incomplete or complete frame, discards an
// over-limit frame through its delimiter, and can recover to parse later
// frames from the same stream. EOS authoritatively flushes a final frame whose
// provider omitted the conventional blank-line delimiter.
func reassembleSSEFramesBounded(
	pending, chunk []byte,
	dropping, endOfStream bool,
) sseFrameBatch {
	result := sseFrameBatch{}
	input := chunk
	if dropping {
		_, boundary := firstSSEFrameBoundary(input)
		if boundary == 0 {
			result.dropping = !endOfStream
			result.invalid = true
			return result
		}
		input = input[boundary:]
		result.invalid = true
	}

	buf := make([]byte, 0, len(pending)+len(input))
	buf = append(buf, pending...)
	buf = append(buf, input...)

	for len(buf) > 0 {
		_, boundary := firstSSEFrameBoundary(buf)
		if boundary == 0 {
			if endOfStream {
				if len(buf) <= maxSSEFrameBytes {
					result.complete = append(result.complete, buf...)
				} else {
					result.invalid = true
				}
				return result
			}
			if len(buf) > maxSSEFrameBytes {
				result.invalid = true
				result.dropping = true
				return result
			}
			result.remainder = append(result.remainder, buf...)
			return result
		}

		if boundary > maxSSEFrameBytes {
			result.invalid = true
		} else {
			result.complete = append(result.complete, buf[:boundary]...)
		}
		buf = buf[boundary:]
	}
	return result
}

func firstSSEFrameBoundary(data []byte) (start, end int) {
	start = -1
	for _, delimiter := range sseFrameDelimiters {
		idx := bytes.Index(data, delimiter)
		if idx < 0 || (start >= 0 && idx >= start) {
			continue
		}
		start = idx
		end = idx + len(delimiter)
	}
	return start, end
}

func sseFramesContainDone(data []byte) bool {
	for _, line := range bytes.Split(data, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		if bytes.Equal(bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:"))), []byte("[DONE]")) {
			return true
		}
	}
	return false
}
