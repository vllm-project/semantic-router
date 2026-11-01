package protocolcodec

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestSSEFramerMatchesReferenceAtEverySplit(t *testing.T) {
	payloads := []string{
		"", "\n", "\r", "\r\n", "\r\n\r\n", "\n\r\r\n\n",
		"data: one\n\ndata: two\n\ndata: tail",
		"\n\ndata: oversized\n\n", "\n\ndata: oversized tail",
		"data: one\r\rdata: two\r\rdata: tail\r",
		"data: one\r\n\r\ndata: two\r\n\r\n",
		"data: mixed\r\n\ndata: lines\n\r\ndata: tail\r",
		": ping\r\nid: 1\nretry: 5\revent: x\ndata: {\ndata: }\n\n",
		"\xef\xbb\xbfdata: 你好🌍\r\n\r\n",
		"data:\n\n: comment\n\n\t \r\n",
		"data: \xff\x00\r\n\r\n\xef\xbb\xbfdata: x\n\n",
		"data: {\"trailing\":", " \t\r\n\u2003",
	}
	for _, payload := range payloads {
		t.Run(fmt.Sprintf("%q", payload), func(t *testing.T) {
			body := []byte(payload)
			for _, limit := range []int{-1, 0, 1, 8, len(body) - 1, len(body), len(body) + 1} {
				for first := 0; first <= len(body); first++ {
					for second := first; second <= len(body); second++ {
						assertSSEFramerMatchesReference(t, limit, [][]byte{body[:first], body[first:second], body[second:]})
					}
				}
			}
		})
	}
}

func TestSSEFramerExhaustiveLineEndings(t *testing.T) {
	// Every string of up to six data/CR/LF bytes, at every partition. Ending
	// each string at EOF also covers all truncated line-ending combinations.
	var visit func([]byte)
	visit = func(body []byte) {
		gotEnd, gotComplete := completeSSEFrame(body)
		wantEnd, wantComplete := referenceCompleteSSEFrame(body)
		if gotEnd != wantEnd || gotComplete != wantComplete {
			t.Fatalf("one-shot frame %q: got %d/%t, want %d/%t", body, gotEnd, gotComplete, wantEnd, wantComplete)
		}
		for partition := 0; partition < 1<<max(len(body)-1, 0); partition++ {
			var chunks [][]byte
			start := 0
			for end := 1; end < len(body); end++ {
				if partition&(1<<(end-1)) != 0 {
					chunks = append(chunks, body[start:end], nil)
					start = end
				}
			}
			chunks = append(chunks, body[start:])
			for _, limit := range []int{0, 1, len(body) - 1, len(body), len(body) + 1} {
				assertSSEFramerMatchesReference(t, limit, chunks)
			}
		}
		if len(body) < 6 {
			for _, value := range []byte{'x', '\r', '\n'} {
				visit(append(body, value))
			}
		}
	}
	visit(nil)
}

func TestSSEFramerRetainsBytesAcrossBufferGrowth(t *testing.T) {
	for _, ending := range []string{"\n", "\r", "\r\n"} {
		t.Run(fmt.Sprintf("ending=%q", ending), func(t *testing.T) {
			frames := [][]byte{
				[]byte("data: first" + ending + ending),
				[]byte("data: " + strings.Repeat("你好🌍", 8192) + ending + ending),
				[]byte("data: " + strings.Repeat("last", 4096) + ending + ending),
			}
			framer := newSSEFramer(len(frames[1]))
			// Force both compaction into reused storage and growth to a new
			// allocation, while the caller retains all previously returned frames.
			framer.buffer = make([]byte, 0, 32)
			body := bytes.Join(frames, nil)
			scratch := make([]byte, 257)
			var output [][]byte
			for offset := 0; offset < len(body); {
				n := copy(scratch, body[offset:])
				batch, err := framer.Push(scratch[:n])
				if err != nil {
					t.Fatal(err)
				}
				output = append(output, batch...)
				clear(scratch)
				offset += n
			}
			final, err := framer.Finalize()
			if err != nil || !bytes.Equal(bytes.Join(append(output, final...), nil), body) {
				t.Fatalf("fragmented bytes changed after growth and scratch reuse: %v", err)
			}
		})
	}
}

func TestSSEFramerChecksEachFrameBeforeReturningBatch(t *testing.T) {
	const limit = 8
	for _, tail := range []string{"data: too long\n\n", "data: too long"} {
		framer := newSSEFramer(limit)
		frames, err := framer.Push([]byte("data:x\n\n" + tail))
		var protocolError *llmprotocol.ProtocolError
		if frames != nil || !errors.As(err, &protocolError) || protocolError.Code != "sse_frame_limit" {
			t.Fatalf("oversized tail must fail the whole batch: %q, %v", frames, err)
		}
		if string(framer.buffer) != tail {
			t.Fatalf("failed batch retained consumed frames: %q", framer.buffer)
		}
	}
	framer := newSSEFramer(limit)
	frames, err := framer.Push(bytes.Repeat([]byte("data:x\n\n"), 128))
	if err != nil || len(frames) != 128 {
		t.Fatalf("coalescing individually valid frames exceeded the limit: %d, %v", len(frames), err)
	}
}

func TestSSEFramerResetsOnFirstFinalize(t *testing.T) {
	for _, tail := range []string{"data: partial", "data: partial\r", "id: first\ndata: partial", " \t\u2003"} {
		t.Run(fmt.Sprintf("tail=%q", tail), func(t *testing.T) {
			framer := newSSEFramer(64)
			if frames, err := framer.Push([]byte(tail)); err != nil || len(frames) != 0 {
				t.Fatalf("tail completed before EOF: %q, %v", frames, err)
			}
			final, err := framer.Finalize()
			if err != nil || (strings.TrimSpace(tail) != "" && !bytes.Equal(bytes.Join(final, nil), []byte(tail))) ||
				(strings.TrimSpace(tail) == "" && len(final) != 0) {
				t.Fatalf("finalized tail changed: %q, %v", final, err)
			}
			// Do not finalize twice before reuse: a second, empty Finalize could
			// mask a missing scanner reset when the first call returned a frame.
			frames, err := framer.Push([]byte("\nx\n\n"))
			if err != nil || !reflect.DeepEqual(frames, [][]byte{[]byte("\n"), []byte("x\n\n")}) {
				t.Fatalf("scanner state survived EOF: %q, %v", frames, err)
			}
		})
	}
}

func TestSSEFramerKeepsFrameAndRemainderOwnership(t *testing.T) {
	framer := newSSEFramer(64)
	chunk := []byte("data: one\n\ndata: two\n\ndata: par")
	frames, err := framer.Push(chunk)
	want := [][]byte{[]byte("data: one\n\n"), []byte("data: two\n\n")}
	if err != nil || !reflect.DeepEqual(frames, want) {
		t.Fatalf("coalesced frames = %q, %v", frames, err)
	}
	for index := range chunk {
		chunk[index] = 'x'
	}
	next, err := framer.Push([]byte("tial\n\ndata: final"))
	if err != nil || len(next) != 1 || string(next[0]) != "data: partial\n\n" {
		t.Fatalf("remainder aliased input: %q, %v", next, err)
	}
	if !reflect.DeepEqual(frames, want) {
		t.Fatalf("later Push overwrote earlier frames: %q", frames)
	}
	frames[0][0] = 'X'
	final, err := framer.Finalize()
	if err != nil || len(final) != 1 || string(final[0]) != "data: final" ||
		string(frames[1]) != "data: two\n\n" || string(next[0]) != "data: partial\n\n" {
		t.Fatalf("frames or remainder share mutable storage: %q, %q, %q, %v", frames, next, final, err)
	}
	if _, err := framer.Push([]byte("data: reused\n\n")); err != nil {
		t.Fatal(err)
	}
	if string(final[0]) != "data: final" {
		t.Fatalf("reuse overwrote finalized frame: %q", final)
	}
}

func TestSSEFramerPreservesCRDelivery(t *testing.T) {
	framer := newSSEFramer(64)
	chunks := []string{"data: one\r", "\n", "\r", "\n", "data: two\r", "\r"}
	want := [][][]byte{
		{}, {}, {[]byte("data: one\r\n\r")}, {[]byte("\n")}, {}, {[]byte("data: two\r\r")},
	}
	for index, chunk := range chunks {
		frames, err := framer.Push([]byte(chunk))
		if err != nil || !reflect.DeepEqual(frames, want[index]) {
			t.Fatalf("Push %d = %q, %v; want %q", index, frames, err, want[index])
		}
	}
}

func FuzzSSEFramerMatchesReference(f *testing.F) {
	for _, payload := range []string{
		"data: one\r\n\r\ndata: two\n\ndata: tail\r",
		"\xef\xbb\xbfdata: 你好🌍\n\n", "\r\r\n\n\r\n", "data: \xff\n\n", " \t\u2003",
	} {
		for _, limit := range []uint16{0, 8, 64} {
			f.Add([]byte(payload), []byte{0, 1, 2, 7}, limit)
		}
	}
	f.Fuzz(func(t *testing.T, body, sizes []byte, limitSeed uint16) {
		// Bound the quadratic reference while varying bytes, limits, and every
		// read size independently (including empty reads).
		if len(body) > 4096 || len(sizes) > 128 {
			t.Skip()
		}
		var chunks [][]byte
		for offset, index := 0, 0; offset < len(body); index++ {
			size := 1
			if len(sizes) > 0 {
				size += int(sizes[index%len(sizes)])
			}
			end := min(offset+size, len(body))
			chunks = append(chunks, nil, body[offset:end])
			offset = end
		}
		assertSSEFramerMatchesReference(t, int(limitSeed)%(len(body)+2), chunks)
	})
}

func assertSSEFramerMatchesReference(t *testing.T, limit int, chunks [][]byte) {
	t.Helper()
	framer := newSSEFramer(limit)
	reference := referenceSSEFramer{limit: limit}
	compare := func(operation string, got, want [][]byte, gotErr, wantErr error) {
		t.Helper()
		var gotProtocolError, wantProtocolError *llmprotocol.ProtocolError
		sameError := gotErr == nil && wantErr == nil
		if gotErr != nil && wantErr != nil {
			sameError = errors.As(gotErr, &gotProtocolError) && errors.As(wantErr, &wantProtocolError) &&
				*gotProtocolError == *wantProtocolError
		}
		if !reflect.DeepEqual(got, want) || !sameError || !bytes.Equal(framer.buffer, reference.buffer) {
			t.Fatalf("%s limit=%d chunks=%q: got %q, %v, pending %q; want %q, %v, pending %q",
				operation, limit, chunks, got, gotErr, framer.buffer, want, wantErr, reference.buffer)
		}
	}
	for _, chunk := range chunks {
		got, gotErr := framer.Push(chunk)
		want, wantErr := reference.Push(chunk)
		compare("Push", got, want, gotErr, wantErr)
		// Even after a failure, preserve the framer's buffer and subsequent
		// behavior; stream callers independently poison their decoder state.
	}
	for range 2 {
		got, gotErr := framer.Finalize()
		want, wantErr := reference.Finalize()
		compare("Finalize", got, want, gotErr, wantErr)
	}
	got, gotErr := framer.Push([]byte("x\n\n"))
	want, wantErr := reference.Push([]byte("x\n\n"))
	compare("Push after Finalize", got, want, gotErr, wantErr)
}

// referenceSSEFramer is the pre-incremental implementation. Keep its scanner
// independent of production code so differential tests detect framing changes.
type referenceSSEFramer struct {
	buffer []byte
	limit  int
}

func (framer *referenceSSEFramer) Push(chunk []byte) ([][]byte, error) {
	if len(chunk) == 0 {
		return nil, nil
	}
	framer.buffer = append(framer.buffer, chunk...)
	frames := make([][]byte, 0, 1)
	for {
		end, complete := referenceCompleteSSEFrame(framer.buffer)
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

func (framer *referenceSSEFramer) Finalize() ([][]byte, error) {
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

func referenceCompleteSSEFrame(payload []byte) (int, bool) {
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
