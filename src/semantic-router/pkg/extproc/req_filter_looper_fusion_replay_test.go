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
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

const (
	fusionReplayAnswerCanary    = "candidate-answer-canary-must-not-escape"
	fusionReplayReasoningCanary = "candidate-reasoning-canary-must-not-escape"
	fusionReplayErrorCanary     = "provider-error-body-canary-must-not-escape"
)

func TestHandleFusionQuorumFailurePersistsAccountingAndAttempts(t *testing.T) {
	core, observedLogs := observer.New(zapcore.DebugLevel)
	restoreLogger := zap.ReplaceGlobals(zap.New(core))
	defer restoreLogger()

	var judgeCalls atomic.Int64
	server := newFusionQuorumReplayServer(t, &judgeCalls)
	defer server.Close()

	replayConfig := config.DefaultRouterReplayPluginConfig()
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{Endpoint: server.URL},
		},
		ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(10, 0)),
	}
	decision := &config.Decision{
		Name: "fusion-quorum-route",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:                  "judge",
				AnalysisModels:         []string{"panel-a", "panel-b", "panel-c"},
				MaxConcurrent:          3,
				MinSuccessfulResponses: 2,
				OnError:                config.FusionOnErrorSkip,
			},
		},
	}
	request := testNeutralRequest("router-entrypoint", "hello")
	requestContext := &RequestContext{
		RequestID:                "replay-fusion-quorum",
		Headers:                  map[string]string{},
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          request,
		RouterReplayPluginConfig: &replayConfig,
		VSRSelectedDecision:      decision,
	}

	response, err := router.handleLooperExecution(context.Background(), request, decision, requestContext)
	if err != nil {
		t.Fatalf("handleLooperExecution: %v", err)
	}
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != typev3.StatusCode_InternalServerError {
		t.Fatalf("Fusion quorum failure status = %v, want %v", got, typev3.StatusCode_InternalServerError)
	}
	if got := judgeCalls.Load(); got != 0 {
		t.Fatalf("judge calls = %d, want 0 below panel quorum", got)
	}

	record, found := router.ReplayRecorder.GetRecord(requestContext.RouterReplayID)
	if !found {
		t.Fatalf("Replay record %q not found", requestContext.RouterReplayID)
	}
	assertFusionQuorumFailureRecord(t, record)

	entries := observedLogs.FilterMessage("looper_execution_failed").All()
	if len(entries) != 1 {
		t.Fatalf("looper_execution_failed entries = %d, want 1", len(entries))
	}
	logFields := jsonObject(t, entries[0].ContextMap())
	assertFusionQuorumDiagnostics(t, logFields, "fusion_quorum")
	assertJSONNumber(t, logFields, "prompt_tokens", 30)
	assertJSONNumber(t, logFields, "completion_tokens", 5)
	assertJSONNumber(t, logFields, "total_tokens", 35)

	assertFusionFailureDoesNotLeakPanelContent(t, response, record, logFields)
}

func newFusionQuorumReplayServer(t *testing.T, judgeCalls *atomic.Int64) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			t.Errorf("decode Fusion request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}

		switch payload.Model {
		case "panel-a":
			writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
				"index": 0,
				"message": map[string]interface{}{
					"role":              "assistant",
					"content":           fusionReplayAnswerCanary,
					"reasoning_content": fusionReplayReasoningCanary,
				},
				"finish_reason": "stop",
			}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
		case "panel-b":
			writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{}, map[string]int64{
				"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23,
			})
		case "panel-c":
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte(fusionReplayErrorCanary))
		case "judge":
			judgeCalls.Add(1)
			writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "unexpected synthesized answer",
				},
				"finish_reason": "stop",
			}}, map[string]int64{"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
		default:
			t.Errorf("unexpected Fusion model call %q", payload.Model)
			http.Error(w, "unexpected model", http.StatusInternalServerError)
		}
	}))
}

func writeFusionReplayCompletion(
	w http.ResponseWriter,
	model string,
	choices []map[string]interface{},
	usage map[string]int64,
) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"id":      "chatcmpl-fusion-replay",
		"object":  "chat.completion",
		"created": 1,
		"model":   model,
		"choices": choices,
		"usage":   usage,
	})
}

func assertFusionQuorumFailureRecord(t *testing.T, record store.Record) {
	t.Helper()
	assertFusionQuorumReplayTerminal(t, record)
	assertFusionQuorumReplayUsage(t, record)
	assertFusionQuorumReplayDiagnostics(t, record)
}

func assertFusionQuorumReplayTerminal(t *testing.T, record store.Record) {
	t.Helper()
	if record.ResponseStatus != http.StatusInternalServerError || record.LifecycleState != routerreplay.LifecycleFailed {
		t.Fatalf("Replay terminal state = status %d / %q, want 500 / %q", record.ResponseStatus, record.LifecycleState, routerreplay.LifecycleFailed)
	}
	if record.EndedAt == nil || record.TerminalReason != "looper_execution_failed" {
		t.Fatalf("Replay terminal metadata = ended_at %v / reason %q", record.EndedAt, record.TerminalReason)
	}
	if record.SelectedModel != "" {
		t.Fatalf("Fusion quorum failure recorded selected model %q", record.SelectedModel)
	}
}

func assertFusionQuorumReplayUsage(t *testing.T, record store.Record) {
	t.Helper()
	assertReplayTokenCount(t, "prompt", record.PromptTokens, 30)
	assertReplayTokenCount(t, "completion", record.CompletionTokens, 5)
	assertReplayTokenCount(t, "total", record.TotalTokens, 35)
	if record.ActualCost != nil || record.BaselineCost != nil || record.CostSavings != nil {
		t.Fatalf("Fusion panel usage invented aggregate cost: actual %v baseline %v savings %v", record.ActualCost, record.BaselineCost, record.CostSavings)
	}
}

func assertReplayTokenCount(t *testing.T, name string, got *int, want int) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("Replay %s tokens = %v, want %d", name, got, want)
	}
}

func assertFusionQuorumReplayDiagnostics(t *testing.T, record store.Record) {
	t.Helper()
	if record.SelectionMethod != config.DecisionAlgorithmFusion {
		t.Fatalf("Replay selection method = %q, want fusion", record.SelectionMethod)
	}

	recordJSON := jsonObject(t, record)
	routeDiagnostics := requiredJSONObject(t, recordJSON, "route_diagnostics")
	if got, _ := routeDiagnostics["selection_method"].(string); got != config.DecisionAlgorithmFusion {
		t.Fatalf("route_diagnostics.selection_method = %q, want fusion", got)
	}
	if reasoning, _ := routeDiagnostics["selection_reasoning"].(string); !strings.Contains(reasoning, "1/2 usable responses") {
		t.Fatalf("route_diagnostics.selection_reasoning = %q, want bounded quorum summary", reasoning)
	}
	assertFusionQuorumDiagnostics(t, routeDiagnostics, "fusion_quorum")
}

func assertFusionQuorumDiagnostics(t *testing.T, parent map[string]interface{}, key string) {
	t.Helper()
	diagnostics := requiredJSONObject(t, parent, key)
	assertJSONNumber(t, diagnostics, "required_count", 2)
	assertJSONNumber(t, diagnostics, "usable_count", 1)
	attempts, ok := diagnostics["attempts"].([]interface{})
	if !ok || len(attempts) != 3 {
		t.Fatalf("%s.attempts = %#v, want three ordered attempts", key, diagnostics["attempts"])
	}

	wants := []struct {
		model      string
		state      string
		prompt     float64
		completion float64
		total      float64
		hasUsage   bool
	}{
		{model: "panel-a", state: "usable", prompt: 10, completion: 2, total: 12, hasUsage: true},
		{model: "panel-b", state: "unusable", prompt: 20, completion: 3, total: 23, hasUsage: true},
		{model: "panel-c", state: "failed"},
	}
	for index, want := range wants {
		attempt, ok := attempts[index].(map[string]interface{})
		if !ok {
			t.Fatalf("%s.attempts[%d] = %#v, want object", key, index, attempts[index])
		}
		if got, _ := attempt["model"].(string); got != want.model {
			t.Fatalf("%s.attempts[%d].model = %q, want %q", key, index, got, want.model)
		}
		if got, _ := attempt["state"].(string); got != want.state {
			t.Fatalf("%s.attempts[%d].state = %q, want %q", key, index, got, want.state)
		}
		if want.hasUsage {
			assertJSONNumber(t, attempt, "prompt_tokens", want.prompt)
			assertJSONNumber(t, attempt, "completion_tokens", want.completion)
			assertJSONNumber(t, attempt, "total_tokens", want.total)
		} else {
			assertAbsentOrZeroJSONNumber(t, attempt, "prompt_tokens")
			assertAbsentOrZeroJSONNumber(t, attempt, "completion_tokens")
			assertAbsentOrZeroJSONNumber(t, attempt, "total_tokens")
		}
		if _, exists := attempt["error"]; exists {
			t.Fatalf("%s.attempts[%d] exposed error text: %#v", key, index, attempt["error"])
		}
	}
}

func assertFusionFailureDoesNotLeakPanelContent(
	t *testing.T,
	response interface{},
	record store.Record,
	logFields map[string]interface{},
) {
	t.Helper()
	combined := fmt.Sprintf("response=%s\nrecord=%s\nlogs=%s", jsonBytes(t, response), jsonBytes(t, record), jsonBytes(t, logFields))
	for _, canary := range []string{fusionReplayAnswerCanary, fusionReplayReasoningCanary, fusionReplayErrorCanary} {
		if strings.Contains(combined, canary) {
			t.Fatalf("Fusion quorum failure exposed private canary %q: %s", canary, combined)
		}
	}
}

func jsonObject(t *testing.T, value interface{}) map[string]interface{} {
	t.Helper()
	var object map[string]interface{}
	if err := json.Unmarshal(jsonBytes(t, value), &object); err != nil {
		t.Fatalf("decode JSON object: %v", err)
	}
	return object
}

func jsonBytes(t *testing.T, value interface{}) []byte {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("encode JSON: %v", err)
	}
	return encoded
}

func requiredJSONObject(t *testing.T, parent map[string]interface{}, key string) map[string]interface{} {
	t.Helper()
	value, ok := parent[key].(map[string]interface{})
	if !ok {
		t.Fatalf("%s = %#v, want object", key, parent[key])
	}
	return value
}

func assertJSONNumber(t *testing.T, object map[string]interface{}, key string, want float64) {
	t.Helper()
	got, ok := object[key].(float64)
	if !ok || got != want {
		t.Fatalf("%s = %#v, want %.0f", key, object[key], want)
	}
}

func assertAbsentOrZeroJSONNumber(t *testing.T, object map[string]interface{}, key string) {
	t.Helper()
	value, exists := object[key]
	if !exists {
		return
	}
	if got, ok := value.(float64); !ok || got != 0 {
		t.Fatalf("%s = %#v, want absent or zero", key, value)
	}
}
