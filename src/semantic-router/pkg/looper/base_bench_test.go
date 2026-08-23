package looper

import (
	"strings"
	"testing"
)

// Fixtures for the Base looper pure-function micro-benchmarks (no network, no
// logging side effects), giving stable regression baselines.
var (
	// benchChunkText is a representative aggregated answer fed to the streaming splitter.
	benchChunkText = strings.Repeat("The quick brown fox jumps over the lazy dog. ", 30)
	// benchTaggedToolCall is an inline <tool_call> payload some OpenAI-compatible
	// backends emit in message content instead of structured tool_calls.
	benchTaggedToolCall = `<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>`
)

// BenchmarkBase_SplitIntoChunks measures the streaming chunk splitter, including
// the []rune conversion it performs over the whole string.
func BenchmarkBase_SplitIntoChunks(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		splitIntoChunks(benchChunkText, 50)
	}
}

// BenchmarkBase_ParseTaggedToolCall measures the regex match + JSON parse of an
// inline tagged tool call.
func BenchmarkBase_ParseTaggedToolCall(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		parseTaggedToolCall(benchTaggedToolCall)
	}
}

// BenchmarkBase_BuildTaggedToolSemanticResponse measures tagged-tool parsing
// and construction of the neutral response published by Looper.
func BenchmarkBase_BuildTaggedToolSemanticResponse(b *testing.B) {
	usage := NewActualTokenUsage(8, 4, 12)
	b.ReportAllocs()
	for b.Loop() {
		newTaggedToolSemanticResponse("response-bench", "model-a", benchTaggedToolCall, usage)
	}
}
