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
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type recordingCallObserver struct {
	mu       sync.Mutex
	started  []CallInfo
	finished []CallResult
}

func (o *recordingCallObserver) BeforeCall(_ context.Context, info CallInfo) (*CallReservation, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.started = append(o.started, info)
	return &CallReservation{ID: "recorded", EstimatedTokens: info.EstimatedTotalTokens}, nil
}

func (o *recordingCallObserver) AfterCall(_ context.Context, _ CallInfo, _ *CallReservation, result CallResult) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.finished = append(o.finished, result)
}

func TestCallObserverSeesSuccessfulCallAndUsagePresence(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id": "observer-test", "object": "chat.completion", "created": 1,
			"choices": []map[string]interface{}{{
				"index":         0,
				"message":       map[string]interface{}{"role": "assistant", "content": "ok"},
				"finish_reason": "stop",
			}},
			"usage": map[string]interface{}{"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
		})
	}))
	defer server.Close()

	observer := &recordingCallObserver{}
	client := NewClient(&config.LooperConfig{Endpoint: server.URL})
	response, err := client.CallModelWithOptions(
		context.Background(),
		openai.ChatCompletionNewParams{Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")}},
		ModelTarget{Name: "model-a"},
		CallOptions{Iteration: 1, Mode: ResponseJSON, Stage: CallStageJudge, Role: "judge", Observer: observer},
	)
	if err != nil {
		t.Fatalf("CallModelWithOptions() error = %v", err)
	}
	if response == nil {
		t.Fatal("response is nil")
	}
	if !response.UsagePresent.Any() || !response.UsagePresent.TotalTokens {
		t.Fatalf("usage presence = %+v, want all fields present", response.UsagePresent)
	}
	if len(observer.started) != 1 || len(observer.finished) != 1 {
		t.Fatalf("observer callbacks = %d/%d, want 1/1", len(observer.started), len(observer.finished))
	}
	if observer.started[0].Stage != CallStageJudge || observer.started[0].Role != "judge" {
		t.Fatalf("call info = %+v, want judge metadata", observer.started[0])
	}
	if observer.finished[0].Response != response || observer.finished[0].Err != nil {
		t.Fatalf("call result = %+v, want successful response", observer.finished[0])
	}
}

func TestBudgetControllerRejectsBeforeSecondDispatch(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer server.Close()

	controller := NewBudgetController(BudgetLimits{MaxCalls: 1, MaxTotalTokens: 100})
	client := NewClient(&config.LooperConfig{Endpoint: server.URL})
	request := openai.ChatCompletionNewParams{Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")}}
	options := CallOptions{Iteration: 1, Mode: ResponseJSON, Observer: controller}
	if _, err := client.CallModelWithOptions(context.Background(), request, ModelTarget{Name: "model-a"}, options); err != nil {
		t.Fatalf("first call error = %v", err)
	}
	if _, err := client.CallModelWithOptions(context.Background(), request, ModelTarget{Name: "model-a"}, options); !IsBudgetExhausted(err) {
		t.Fatalf("second call error = %v, want budget exhaustion", err)
	}
	if requests != 1 {
		t.Fatalf("upstream requests = %d, want 1", requests)
	}
	snapshot := controller.Snapshot()
	if snapshot.Calls != 1 || !snapshot.Exhausted || snapshot.ActiveCalls != 0 {
		t.Fatalf("budget snapshot = %+v", snapshot)
	}
}

func TestCallObserverAttachedToContextReachesClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer server.Close()

	observer := &recordingCallObserver{}
	ctx := WithCallObserver(context.Background(), observer)
	client := NewClient(&config.LooperConfig{Endpoint: server.URL})
	_, err := client.CallModelWithOptions(
		ctx,
		openai.ChatCompletionNewParams{},
		ModelTarget{Name: "model-a"},
		CallOptions{Iteration: 1, Mode: ResponseJSON, Stage: CallStageGenerate},
	)
	if err != nil {
		t.Fatalf("CallModelWithOptions() error = %v", err)
	}
	if len(observer.started) != 1 || len(observer.finished) != 1 {
		t.Fatalf("context observer callbacks = %d/%d, want 1/1", len(observer.started), len(observer.finished))
	}
}

func TestUsageKnownDistinguishesPartialAndExplicitZeroUsage(t *testing.T) {
	partial := &ModelResponse{
		Usage:        TokenUsage{CompletionTokens: 2},
		UsagePresent: UsagePresence{CompletionTokens: true},
	}
	if partial.UsageKnown() {
		t.Fatal("partial usage must remain unknown")
	}
	if tokens, known := usageTokensForBudget(partial); known || tokens != 0 {
		t.Fatalf("partial budget usage = %d, %t; want unknown", tokens, known)
	}
	zero := &ModelResponse{
		Usage:        TokenUsage{TotalTokens: 0},
		UsagePresent: UsagePresence{TotalTokens: true},
	}
	if !zero.UsageKnown() {
		t.Fatal("explicit zero total must be known")
	}
	if tokens, known := usageTokensForBudget(zero); !known || tokens != 0 {
		t.Fatalf("zero budget usage = %d, %t; want known zero", tokens, known)
	}
}
