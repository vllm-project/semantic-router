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
		Body:          []byte(`{"ok":true}`),
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

// The looper aggregate latency and token usage (#2694) are intermediate
// observability, so they must be demoted to the x-vsr-debug surface next to
// the existing x-vsr-looper-* trace headers (#2205).
func TestCreateLooperResponseIncludesLatencyAndUsageHeaders(t *testing.T) {
	resp := looperResponseWithLatencyAndUsage()
	reqCtx := &RequestContext{
		Headers: map[string]string{headers.VSRDebug: "true"},
	}

	response := (&OpenAIRouter{}).createLooperResponse(resp, reqCtx)
	headerMap := headerValuesByName(response.GetImmediateResponse().Headers.SetHeaders)

	assert.Equal(t, "123", headerMap[headers.VSRLooperLatencyMs])
	assert.Equal(t, "10", headerMap[headers.VSRLooperPromptTokens])
	assert.Equal(t, "20", headerMap[headers.VSRLooperCompletionTokens])
	assert.Equal(t, "30", headerMap[headers.VSRLooperTotalTokens])
}

func TestCreateLooperResponseDefaultSurfaceExcludesLatencyAndUsageHeaders(t *testing.T) {
	resp := looperResponseWithLatencyAndUsage()
	reqCtx := &RequestContext{}

	response := (&OpenAIRouter{}).createLooperResponse(resp, reqCtx)
	headerMap := headerValuesByName(response.GetImmediateResponse().Headers.SetHeaders)

	assert.NotContains(t, headerMap, headers.VSRLooperLatencyMs)
	assert.NotContains(t, headerMap, headers.VSRLooperPromptTokens)
	assert.NotContains(t, headerMap, headers.VSRLooperCompletionTokens)
	assert.NotContains(t, headerMap, headers.VSRLooperTotalTokens)
}
