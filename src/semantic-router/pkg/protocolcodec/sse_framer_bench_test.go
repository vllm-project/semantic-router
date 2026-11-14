package protocolcodec

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
)

func BenchmarkSSEFramerLargeEvent(b *testing.B) {
	for _, size := range []int{64 << 10, 128 << 10, 256 << 10} {
		frame := benchmarkSSEChatFrame(size)
		for _, chunkSize := range []int{len(frame), 256} {
			b.Run(fmt.Sprintf("%dKiB/chunk=%d", size>>10, chunkSize), func(b *testing.B) {
				benchmarkSSEFramer(b, frame, chunkSize, len(frame), 1)
			})
		}
	}
}

func BenchmarkSSEFramerCoalescedEvents(b *testing.B) {
	frame := benchmarkSSEChatFrame(8)
	for _, count := range []int{128, 512, 2048} {
		body := bytes.Repeat(frame, count)
		b.Run(fmt.Sprintf("frames=%d", count), func(b *testing.B) {
			benchmarkSSEFramer(b, body, len(body), len(frame), count)
		})
	}
}

func BenchmarkSSEFramerTokenEvents(b *testing.B) {
	frame := benchmarkSSEChatFrame(8)
	body := bytes.Repeat(frame, 512)
	benchmarkSSEFramer(b, body, len(frame), len(frame), 512)
}

func benchmarkSSEChatFrame(contentSize int) []byte {
	return []byte(`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1760000000,"model":"model","choices":[{"index":0,"delta":{"content":"` +
		strings.Repeat("x", contentSize) + `"},"finish_reason":null}]}` + "\n\n")
}

func benchmarkSSEFramer(b *testing.B, body []byte, chunkSize, limit, frameCount int) {
	b.Helper()
	b.SetBytes(int64(len(body)))
	b.ReportAllocs()
	for b.Loop() {
		framer := newSSEFramer(limit)
		count := 0
		for offset := 0; offset < len(body); offset += chunkSize {
			frames, err := framer.Push(body[offset:min(offset+chunkSize, len(body))])
			if err != nil {
				b.Fatal(err)
			}
			count += len(frames)
		}
		if count != frameCount {
			b.Fatalf("got %d frames, want %d", count, frameCount)
		}
	}
}
