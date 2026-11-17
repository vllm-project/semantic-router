/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

func looperResponseWithLatencyAndUsage() *looper.Response {
	return &looper.Response{
		Body:          []byte(`{"id":"chatcmpl-looper","model":"model-b","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`),
		ContentType:   "application/json",
		Model:         "model-b",
		ModelsUsed:    []string{"model-a", "model-b"},
		Iterations:    2,
		AlgorithmType: "elo",
		LatencyMs:     123,
		Usage: looper.TokenUsage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}
}

// The Looper's aggregate latency and token usage are intermediate
// observability, so they ride in the trace's looper member.
func TestLooperTraceIncludesLatencyAndUsage(t *testing.T) {
	trace := looperTraceFor(t, looperResponseWithLatencyAndUsage(), &RequestContext{})

	want := `("model-a" "model-b");algorithm=elo;model="model-b";iterations=2;latency_ms=123;prompt_tokens=10;completion_tokens=20;total_tokens=30`
	assert.Equal(t, want, trace[headers.TraceLooper])
}

func TestLooperResponseKeepsTheTraceFromOtherCallers(t *testing.T) {
	reqCtx := &RequestContext{}
	response := (&OpenAIRouter{}).createLooperResponse(looperResponseWithLatencyAndUsage(), reqCtx)
	appendRoutedResponseHeaders(response, reqCtx)

	headerMap := headerValuesByName(response.GetImmediateResponse().Headers.SetHeaders)
	assert.NotContains(t, headerMap, headers.VSRTrace)
	assert.Equal(t, headers.ResponsePathLooper, headerMap[headers.VSRResponsePath])
}

// A zero LatencyMs or Usage field means "not measured" (e.g. a caller that
// bypasses looper.ExecuteWithLatency, such as cmd/fusioneval) rather than a
// genuine zero-cost execution. Leaving it out keeps "present" meaning
// "actually measured" instead of a bypassed measurement being indistinguishable
// from a real zero.
func TestLooperTraceOmitsUnmeasuredZeroValues(t *testing.T) {
	resp := &looper.Response{
		Body:          []byte(`{"id":"chatcmpl-looper","model":"model-b","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`),
		ContentType:   "application/json",
		Model:         "model-b",
		ModelsUsed:    []string{"model-a", "model-b"},
		Iterations:    2,
		AlgorithmType: "elo",
		// LatencyMs and Usage left at their zero values.
	}
	trace := looperTraceFor(t, resp, &RequestContext{})

	assert.Equal(t, `("model-a" "model-b");algorithm=elo;model="model-b";iterations=2`, trace[headers.TraceLooper])
}
