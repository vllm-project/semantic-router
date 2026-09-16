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
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestIsUsableFusionPanelResponse(t *testing.T) {
	tests := []struct {
		name     string
		response *ModelResponse
		want     bool
	}{
		{name: "nil response", response: nil, want: false},
		{name: "empty response", response: &ModelResponse{}, want: false},
		{name: "empty content", response: &ModelResponse{Content: ""}, want: false},
		{name: "whitespace content", response: &ModelResponse{Content: " \n\t "}, want: false},
		{name: "whitespace reasoning", response: &ModelResponse{ReasoningContent: " \n\t "}, want: false},
		{name: "tool only", response: &ModelResponse{HasToolCalls: true}, want: false},
		{name: "content", response: &ModelResponse{Content: "answer"}, want: true},
		{name: "reasoning only", response: &ModelResponse{ReasoningContent: "reasoning"}, want: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assert.Equal(t, test.want, isUsableFusionPanelResponse(test.response))
		})
	}
}

func TestFusionQuorumEvidenceIsContentFreeAndDefensive(t *testing.T) {
	cause := context.DeadlineExceeded
	quorumErr := newFusionQuorumError(2, fusionPanelOutcome{
		responses: []*ModelResponse{{Content: "private panel answer"}},
		attempts: []FusionPanelAttemptEvidence{
			{
				Model: "panel-a",
				State: FusionPanelAttemptUsable,
				Usage: TokenUsage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5},
			},
			{
				Model: "panel-b",
				State: FusionPanelAttemptTimedOut,
				Error: "context deadline exceeded",
			},
		},
		usage: TokenUsage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5},
	}, cause)

	assert.ErrorIs(t, quorumErr, cause)
	assert.Equal(t, "fusion panel quorum not met: got 1 usable response, require 2: context deadline exceeded", quorumErr.Error())

	evidence, ok := FusionQuorumEvidenceFromError(fmt.Errorf("wrapped: %w", quorumErr))
	require.True(t, ok)
	assert.Equal(t, 2, evidence.RequiredCount)
	assert.Equal(t, 1, evidence.UsableCount)
	assert.Equal(t, TokenUsage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5}, evidence.Usage)
	assert.NotContains(t, fmt.Sprintf("%+v", evidence), "private panel answer")

	evidence.Attempts[0].Model = "mutated"
	again, ok := FusionQuorumEvidenceFromError(quorumErr)
	require.True(t, ok)
	assert.Equal(t, "panel-a", again.Attempts[0].Model)
}

func TestFusionLooperRejectsPanelBelowUsableQuorum(t *testing.T) {
	var judgeCalls atomic.Int64
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		switch model {
		case "panel-a":
			return "panel a answer", http.StatusOK
		case "panel-b", "panel-c":
			return "failed", http.StatusBadGateway
		case "judge":
			judgeCalls.Add(1)
			if strings.Contains(prompt, "return only valid JSON") {
				return `{"consensus":["a"],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`, http.StatusOK
			}
			return "unexpected synthesis", http.StatusOK
		default:
			return "unexpected model", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                  "judge",
			AnalysisModels:         []string{"panel-a", "panel-b", "panel-c"},
			MaxConcurrent:          3,
			MinSuccessfulResponses: 2,
			OnError:                config.FusionOnErrorSkip,
		},
	}

	_, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "fusion panel quorum not met")
	assert.Zero(t, judgeCalls.Load(), "judge must not run below panel quorum")
}

func TestFusionLooperEmptyPanelResponseDoesNotMeetQuorum(t *testing.T) {
	var judgeCalls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		switch payload.Model {
		case "panel-a":
			writeFusionTestCompletion(w, payload.Model, "panel a answer", http.StatusOK)
		case "panel-b":
			writeFusionEmptyCompletion(w, payload.Model)
		case "panel-c":
			writeFusionTestCompletion(w, payload.Model, "failed", http.StatusBadGateway)
		case "judge":
			judgeCalls.Add(1)
			writeFusionTestCompletion(w, payload.Model, "unexpected judge response", http.StatusOK)
		default:
			t.Errorf("unexpected model call: %s", payload.Model)
			writeFusionTestCompletion(w, payload.Model, "unexpected model", http.StatusInternalServerError)
		}
	}))
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                  "judge",
			AnalysisModels:         []string{"panel-a", "panel-b", "panel-c"},
			MaxConcurrent:          3,
			MinSuccessfulResponses: 2,
			OnError:                config.FusionOnErrorSkip,
		},
	}

	_, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.Error(t, err)
	assert.Equal(t, "fusion panel quorum not met: got 1 usable response, require 2", err.Error())
	assert.Zero(t, judgeCalls.Load(), "judge must not run below panel quorum")

	evidence, ok := FusionQuorumEvidenceFromError(err)
	require.True(t, ok)
	assert.Equal(t, 2, evidence.RequiredCount)
	assert.Equal(t, 1, evidence.UsableCount)
	assert.Equal(t, TokenUsage{PromptTokens: 30, CompletionTokens: 5, TotalTokens: 35}, evidence.Usage)
	require.Len(t, evidence.Attempts, 3)
	assert.Equal(t, []FusionPanelAttemptState{
		FusionPanelAttemptUsable,
		FusionPanelAttemptUnusable,
		FusionPanelAttemptFailed,
	}, []FusionPanelAttemptState{
		evidence.Attempts[0].State,
		evidence.Attempts[1].State,
		evidence.Attempts[2].State,
	})
	assert.Equal(t, fusionTestTokenUsage("panel-b"), evidence.Attempts[1].Usage)
}

func TestFusionLooperRecordsUnusablePanelResponseWhenQuorumMet(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model    string `json:"model"`
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		switch payload.Model {
		case "panel-a", "panel-b":
			time.Sleep(50 * time.Millisecond)
			writeFusionTestCompletion(w, payload.Model, payload.Model+" answer", http.StatusOK)
		case "panel-c":
			writeFusionEmptyCompletion(w, payload.Model)
		case "judge":
			prompt := payload.Messages[len(payload.Messages)-1].Content
			if strings.Contains(prompt, "return only valid JSON") {
				writeFusionTestCompletion(w, payload.Model, `{"consensus":["a+b"],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`, http.StatusOK)
				return
			}
			writeFusionTestCompletion(w, payload.Model, "final from usable panel", http.StatusOK)
		default:
			t.Errorf("unexpected model call: %s", payload.Model)
		}
	}))
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                  "judge",
			AnalysisModels:         []string{"panel-a", "panel-b", "panel-c"},
			MaxConcurrent:          3,
			MinSuccessfulResponses: 2,
			OnError:                config.FusionOnErrorSkip,
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "final from usable panel", extractMessageContent(t, resp.Body))
	assert.Equal(t, TokenUsage{PromptTokens: 91, CompletionTokens: 14, TotalTokens: 105}, resp.Usage)

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body, &body))
	fusionTrace := body["fusion"].(map[string]interface{})
	require.Len(t, fusionTrace["responses"], 2)
	require.Len(t, fusionTrace["failed_models"], 1)
	failed := fusionTrace["failed_models"].([]interface{})[0].(map[string]interface{})
	assert.Equal(t, "panel-c", failed["model"])
	assert.Contains(t, failed["error"], "no usable assistant content")
}

func TestFusionLooperOnErrorFailStopsAtUnusableResponse(t *testing.T) {
	var judgeCalls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		switch payload.Model {
		case "panel-a":
			select {
			case <-r.Context().Done():
				return
			case <-time.After(time.Second):
				writeFusionTestCompletion(w, payload.Model, "too late", http.StatusOK)
			}
		case "panel-b":
			writeFusionEmptyCompletion(w, payload.Model)
		case "judge":
			judgeCalls.Add(1)
			writeFusionTestCompletion(w, payload.Model, "unexpected judge response", http.StatusOK)
		default:
			t.Errorf("unexpected model call: %s", payload.Model)
		}
	}))
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                  "judge",
			AnalysisModels:         []string{"panel-a", "panel-b"},
			MaxConcurrent:          2,
			MinSuccessfulResponses: 1,
			OnError:                config.FusionOnErrorFail,
		},
	}

	_, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `fusion panel model "panel-b" returned no usable assistant content`)
	assert.Zero(t, judgeCalls.Load())
}

func TestFusionLooperTimeoutBelowQuorumPreservesPanelEvidence(t *testing.T) {
	var judgeCalls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		switch payload.Model {
		case "panel-a":
			writeFusionTestCompletion(w, payload.Model, "panel a answer", http.StatusOK)
		case "panel-b":
			<-r.Context().Done()
		case "judge":
			judgeCalls.Add(1)
			writeFusionTestCompletion(w, payload.Model, "unexpected judge response", http.StatusOK)
		default:
			t.Errorf("unexpected model call: %s", payload.Model)
		}
	}))
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                  "judge",
			AnalysisModels:         []string{"panel-a", "panel-b"},
			MaxConcurrent:          2,
			MinSuccessfulResponses: 2,
			OnError:                config.FusionOnErrorSkip,
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
	defer cancel()

	_, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(ctx, req)
	require.Error(t, err)
	assert.ErrorIs(t, err, context.DeadlineExceeded)
	assert.Zero(t, judgeCalls.Load(), "judge must not run below panel quorum")

	evidence, ok := FusionQuorumEvidenceFromError(err)
	require.True(t, ok)
	assert.Equal(t, 1, evidence.UsableCount)
	assert.Equal(t, fusionTestTokenUsage("panel-a"), evidence.Usage)
	require.Len(t, evidence.Attempts, 2)
	assert.Equal(t, FusionPanelAttemptUsable, evidence.Attempts[0].State)
	assert.Equal(t, FusionPanelAttemptTimedOut, evidence.Attempts[1].State)
}

func writeFusionEmptyCompletion(w http.ResponseWriter, model string) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"id":      "chatcmpl-empty",
		"object":  "chat.completion",
		"created": 1,
		"model":   model,
		"choices": []interface{}{},
		"usage":   fusionTestUsage(model),
	})
}

func extractMessageContent(t *testing.T, body []byte) string {
	t.Helper()
	var payload struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(body, &payload))
	require.NotEmpty(t, payload.Choices)
	return payload.Choices[0].Message.Content
}

func fusionTestTokenUsage(model string) TokenUsage {
	usage := fusionTestUsage(model)
	return TokenUsage{
		PromptTokens:     usage["prompt_tokens"],
		CompletionTokens: usage["completion_tokens"],
		TotalTokens:      usage["total_tokens"],
	}
}
