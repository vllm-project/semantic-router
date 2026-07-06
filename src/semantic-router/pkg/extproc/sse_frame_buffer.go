package extproc

import "bytes"

// sseFrameDelimiter separates two SSE events on the wire: a blank line.
// Per the SSE spec an event is terminated by a blank line, which on the
// Anthropic and OpenAI streaming surfaces is the byte sequence "\n\n".
var sseFrameDelimiter = []byte("\n\n")

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
// This relies on the well-formed-SSE contract that every event ends with a
// blank line; a stream that never emits the terminating "\n\n" for its
// final event would leave that event in remainder. That is a malformed
// upstream, out of scope here.
func reassembleSSEFrames(pending, chunk []byte) (complete, remainder []byte) {
	buf := make([]byte, 0, len(pending)+len(chunk))
	buf = append(buf, pending...)
	buf = append(buf, chunk...)

	idx := bytes.LastIndex(buf, sseFrameDelimiter)
	if idx < 0 {
		// No frame boundary yet — hold everything until more bytes arrive.
		return nil, buf
	}
	boundary := idx + len(sseFrameDelimiter)
	return buf[:boundary], buf[boundary:]
}
