package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
)

var benchmarkFastExtractResult atomic.Pointer[FastExtractResult]

type requestShape struct {
	tokens   int
	messages int
	tools    int
}

// BenchmarkExtractContentFast_ContextShape is a bounded pairwise matrix over
// the request dimensions that materially affect the ExtProc parsing hot path.
// Token counts are deterministic approximations; input_bytes is emitted as a
// custom metric so reports preserve the exact JSON size too.
func BenchmarkExtractContentFast_ContextShape(b *testing.B) {
	shapes := []requestShape{
		{tokens: 128, messages: 1, tools: 0},
		{tokens: 4096, messages: 8, tools: 8},
		{tokens: 16384, messages: 64, tools: 8},
		{tokens: 65536, messages: 8, tools: 64},
	}
	for _, shape := range shapes {
		body := benchmarkRequestBody(shape)
		name := fmt.Sprintf("tokens=%d/messages=%d/tools=%d", shape.tokens, shape.messages, shape.tools)
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				result, err := extractContentFast(body)
				if err != nil {
					b.Fatal(err)
				}
				benchmarkFastExtractResult.Store(result)
			}
			// b.Loop resets metrics on its first call, so report the request
			// size after the loop to retain it in benchmark output.
			b.ReportMetric(float64(len(body)), "input_bytes")
		})
	}
}

// BenchmarkExtractContentFast_ContextTrend holds message and tool topology
// constant so the report can show an interpretable context-length curve. The
// pairwise ContextShape benchmark above remains the production-like corner
// coverage; these points answer the narrower scaling question.
func BenchmarkExtractContentFast_ContextTrend(b *testing.B) {
	for _, tokens := range []int{128, 512, 2048, 8192, 32768, 65536} {
		shape := requestShape{tokens: tokens, messages: 1, tools: 0}
		body := benchmarkRequestBody(shape)
		name := fmt.Sprintf("context_tokens=%d/messages=1/tools=0", tokens)
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				result, err := extractContentFast(body)
				if err != nil {
					b.Fatal(err)
				}
				benchmarkFastExtractResult.Store(result)
			}
			b.ReportMetric(float64(len(body)), "input_bytes")
		})
	}
}

func BenchmarkExtractContentFast_Parallel(b *testing.B) {
	body := benchmarkRequestBody(requestShape{tokens: 4096, messages: 8, tools: 8})
	b.ReportAllocs()
	b.ReportMetric(float64(len(body)), "input_bytes")
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			result, err := extractContentFast(body)
			if err != nil {
				b.Error(err)
				return
			}
			benchmarkFastExtractResult.Store(result)
		}
	})
}

func benchmarkRequestBody(shape requestShape) []byte {
	messageCount := max(shape.messages, 1)
	wordsPerMessage := max(shape.tokens/messageCount, 1)
	messages := make([]map[string]any, 0, messageCount)
	for i := 0; i < messageCount; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		messages = append(messages, map[string]any{
			"role":    role,
			"content": strings.Repeat("token ", wordsPerMessage),
		})
	}
	tools := make([]map[string]any, 0, shape.tools)
	for i := 0; i < shape.tools; i++ {
		tools = append(tools, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        fmt.Sprintf("tool_%d", i),
				"description": "deterministic benchmark tool",
				"parameters": map[string]any{
					"type":       "object",
					"properties": map[string]any{"value": map[string]string{"type": "string"}},
				},
			},
		})
	}
	body, err := json.Marshal(map[string]any{
		"model": "auto", "stream": true, "messages": messages, "tools": tools,
	})
	if err != nil {
		panic(err)
	}
	return body
}
