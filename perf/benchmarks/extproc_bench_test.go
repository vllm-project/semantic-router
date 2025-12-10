//go:build !windows && cgo

package benchmarks

import (
	"encoding/json"
	"testing"
)

// Note: ExtProc is a complex integration component involving gRPC streaming.
// These benchmarks focus on the lightweight operations ExtProc performs:
// - JSON parsing of OpenAI requests
// - Header manipulation
// - Request/response body processing
//
// The heavy operations (classification, decision evaluation) are benchmarked
// separately in classification_bench_test.go and decision_bench_test.go

var (
	testOpenAIRequest = map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": "What is the derivative of x^2 + 3x + 5?",
			},
		},
	}

	testOpenAIResponse = map[string]interface{}{
		"id":      "chatcmpl-123",
		"object":  "chat.completion",
		"created": 1677652288,
		"model":   "gpt-4",
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "The derivative is 2x + 3",
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     20,
			"completion_tokens": 10,
			"total_tokens":      30,
		},
	}
)

// BenchmarkJSONMarshalRequest benchmarks JSON marshaling of OpenAI requests
func BenchmarkJSONMarshalRequest(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := json.Marshal(testOpenAIRequest)
		if err != nil {
			b.Fatalf("JSON marshal failed: %v", err)
		}
	}
}

// BenchmarkJSONUnmarshalRequest benchmarks JSON unmarshaling of OpenAI requests
func BenchmarkJSONUnmarshalRequest(b *testing.B) {
	// Pre-marshal the request
	data, err := json.Marshal(testOpenAIRequest)
	if err != nil {
		b.Fatalf("Setup failed: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		var req map[string]interface{}
		err := json.Unmarshal(data, &req)
		if err != nil {
			b.Fatalf("JSON unmarshal failed: %v", err)
		}
	}
}

// BenchmarkJSONMarshalResponse benchmarks JSON marshaling of OpenAI responses
func BenchmarkJSONMarshalResponse(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := json.Marshal(testOpenAIResponse)
		if err != nil {
			b.Fatalf("JSON marshal failed: %v", err)
		}
	}
}

// BenchmarkJSONUnmarshalResponse benchmarks JSON unmarshaling of OpenAI responses
func BenchmarkJSONUnmarshalResponse(b *testing.B) {
	// Pre-marshal the response
	data, err := json.Marshal(testOpenAIResponse)
	if err != nil {
		b.Fatalf("Setup failed: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		var resp map[string]interface{}
		err := json.Unmarshal(data, &resp)
		if err != nil {
			b.Fatalf("JSON unmarshal failed: %v", err)
		}
	}
}

// BenchmarkHeaderManipulation benchmarks header map operations
func BenchmarkHeaderManipulation(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		headers := make(map[string]string, 10)
		headers["content-type"] = "application/json"
		headers["x-request-id"] = "test-123"
		headers["x-selected-model"] = "gpt-4"
		headers["x-decision"] = "math-reasoning"
		headers["x-category"] = "math"
		headers["x-confidence"] = "0.95"

		// Simulate header read operations
		_ = headers["content-type"]
		_ = headers["x-selected-model"]
		_ = headers["x-decision"]
	}
}

// BenchmarkRequestBodyParsing benchmarks parsing OpenAI request body
func BenchmarkRequestBodyParsing(b *testing.B) {
	// Create test request body
	reqBody := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": "What is the derivative of x^2 + 3x + 5?",
			},
		},
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		b.Fatalf("Setup failed: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		var parsed map[string]interface{}
		err := json.Unmarshal(data, &parsed)
		if err != nil {
			b.Fatalf("Parse failed: %v", err)
		}

		// Simulate extracting fields
		_ = parsed["model"]
		_ = parsed["messages"]
	}
}
