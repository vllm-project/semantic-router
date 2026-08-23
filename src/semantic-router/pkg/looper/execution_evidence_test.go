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

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestPartialExecutionErrorPreservesTypedAggregateEvidence(t *testing.T) {
	cause := errors.New("algorithm failed")
	err := newPartialExecutionError(cause, ExecutionEvidence{
		ModelsUsed: []string{"model-a"},
		Iterations: 1,
		Usage:      TokenUsage{PromptTokens: 8, CompletionTokens: 4, TotalTokens: 12},
	})

	require.ErrorIs(t, err, cause)
	evidence, ok := ExecutionEvidenceFromError(err)
	require.True(t, ok)
	assert.Equal(t, ExecutionEvidence{
		ModelsUsed: []string{"model-a"},
		Iterations: 1,
		Usage:      TokenUsage{PromptTokens: 8, CompletionTokens: 4, TotalTokens: 12},
	}, evidence)

	evidence.ModelsUsed[0] = "mutated"
	again, ok := ExecutionEvidenceFromError(err)
	require.True(t, ok)
	assert.Equal(t, []string{"model-a"}, again.ModelsUsed)
}

func TestRatingsFailureCarriesSuccessfulSiblingUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		if request.Model == "failed-model" {
			http.Error(w, "upstream failed", http.StatusBadGateway)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		require.NoError(t, json.NewEncoder(w).Encode(map[string]interface{}{
			"id": "chatcmpl-ratings-partial", "object": "chat.completion", "created": 1,
			"model": request.Model,
			"choices": []map[string]interface{}{{
				"index":         0,
				"message":       map[string]interface{}{"role": "assistant", "content": "ok"},
				"finish_reason": "stop",
			}},
			"usage": map[string]int{"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
		}))
	}))
	defer server.Close()

	request := openai.ChatCompletionNewParams{
		Model:    "mom",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("test")},
	}
	_, err := NewRatingsLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), &Request{
		executionRequest: &request,
		ModelRefs: []config.ModelRef{
			{Model: "successful-model"},
			{Model: "failed-model"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmRatings,
			Ratings: &config.RatingsAlgorithmConfig{
				OnError: "fail",
			},
		},
	})
	require.Error(t, err)

	evidence, ok := ExecutionEvidenceFromError(err)
	require.True(t, ok)
	assert.Equal(t, []string{"successful-model"}, evidence.ModelsUsed)
	assert.Equal(t, 2, evidence.Iterations)
	assert.Equal(t, TokenUsage{PromptTokens: 8, CompletionTokens: 4, TotalTokens: 12}, evidence.Usage)
}

func TestAlgorithmEvidenceHelpersPreserveActualUsage(t *testing.T) {
	panel := &ModelResponse{Model: "panel", Usage: TokenUsage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7}}
	judge := &ModelResponse{Model: "judge", Usage: TokenUsage{PromptTokens: 6, CompletionTokens: 3, TotalTokens: 9}}

	t.Run("fusion", func(t *testing.T) {
		evidence := fusionExecutionEvidence([]*ModelResponse{panel}, judge, nil)
		assert.Equal(t, []string{"panel", "judge"}, evidence.ModelsUsed)
		assert.Equal(t, 2, evidence.Iterations)
		assert.Equal(t, TokenUsage{PromptTokens: 11, CompletionTokens: 5, TotalTokens: 16}, evidence.Usage)
	})

	t.Run("remom", func(t *testing.T) {
		evidence := remomExecutionEvidence(&remomScheduleResult{
			modelsUsed:      map[string]bool{"panel": true, "judge": true},
			totalIterations: 2,
			usage:           TokenUsage{PromptTokens: 11, CompletionTokens: 5, TotalTokens: 16},
		})
		assert.Equal(t, []string{"judge", "panel"}, evidence.ModelsUsed)
		assert.Equal(t, 2, evidence.Iterations)
		assert.Equal(t, TokenUsage{PromptTokens: 11, CompletionTokens: 5, TotalTokens: 16}, evidence.Usage)
	})

	t.Run("workflows", func(t *testing.T) {
		evidence := workflowExecutionEvidence(
			workflowsExecutionConfig{PlannerModel: "planner"},
			panel,
			[]workflowStepResult{{responses: []*ModelResponse{judge}}},
		)
		assert.Equal(t, []string{"planner", "panel", "judge"}, evidence.ModelsUsed)
		assert.Equal(t, 2, evidence.Iterations)
		assert.Equal(t, TokenUsage{PromptTokens: 11, CompletionTokens: 5, TotalTokens: 16}, evidence.Usage)
	})
}
