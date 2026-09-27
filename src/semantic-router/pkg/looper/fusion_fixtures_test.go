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

package looper

// Shared fixtures for the Fusion test suites: the canonical request, the model
// metadata the runtime contract depends on, and the stub backend.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func newFusionTestRequest() *Request {
	params := openai.ChatCompletionNewParams{
		Model: "vllm-sr/fusion",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("compare the options"),
		},
	}
	return &Request{
		OriginalRequest: &params,
		DecisionName:    "fusion-test",
		ModelParams:     fusionTestModelParams(),
	}
}

// fusionTestModelParams declares chat capability for every panel model, judge,
// and fallback target the Fusion suites use, as a loaded configuration would.
func fusionTestModelParams() map[string]config.ModelParams {
	return map[string]config.ModelParams{
		"panel-a":         {Capabilities: []string{"chat"}},
		"panel-b":         {Capabilities: []string{"chat"}},
		"panel-c":         {Capabilities: []string{"chat"}},
		"panel-d":         {Capabilities: []string{"chat"}},
		"panel-e":         {Capabilities: []string{"chat"}},
		"panel-slow":      {Capabilities: []string{"chat"}},
		"judge":           {Capabilities: []string{"chat"}},
		"backup-model":    {Capabilities: []string{"chat"}},
		"fallback-target": {Capabilities: []string{"chat"}},
	}
}

func newFusionStubServer(
	t *testing.T,
	respond func(model string, prompt string) (content string, status int),
) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "1", r.Header.Get("x-vsr-fusion-depth"))
		var payload struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		prompt := ""
		if len(payload.Messages) > 0 {
			prompt = payload.Messages[len(payload.Messages)-1].Content
		}
		content, status := respond(payload.Model, prompt)
		writeFusionTestCompletion(w, payload.Model, content, status)
	}))
}
