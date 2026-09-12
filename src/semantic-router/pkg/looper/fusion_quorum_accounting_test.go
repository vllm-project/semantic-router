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

// Accounting tests for the Fusion panel: which calls the router attempted, and
// whether the response and its per-call ordinals describe those same calls.

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A worker still queued behind max_concurrent when the round ends never reaches
// a backend, so the fallback's accounting must name dispatched calls rather than
// the configured panel. Which worker wins the slot is up to the scheduler, so the
// assertions are on counts.
func TestFallbackAccountingCountsOnlyDispatchedPanelCalls(t *testing.T) {
	panelModels := []string{"panel-a", "panel-b", "panel-c"}
	var mu sync.Mutex
	dispatchedPanel := map[string]bool{}
	// Capture the per-call ordinal every backend actually saw. Asserting only on
	// Response.Iterations cannot detect an ordinal that goes backwards or repeats.
	iterations := map[string]int{}
	release := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		ordinal, _ := strconv.Atoi(r.Header.Get("x-vsr-looper-iteration"))
		mu.Lock()
		iterations[payload.Model] = ordinal
		mu.Unlock()
		if payload.Model == "backup-model" {
			writeCanaryReasoningCompletion(w, payload.Model, canaryFallbackAnswer, "")
			return
		}
		mu.Lock()
		dispatchedPanel[payload.Model] = true
		mu.Unlock()
		// Hold the only concurrency slot past the round deadline.
		<-release
		writeCanaryReasoningCompletion(w, payload.Model, "too late", "")
	}))
	defer server.Close()
	defer close(release)

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                  "judge",
			AnalysisModels:         panelModels,
			MaxConcurrent:          1,
			MinSuccessfulResponses: 2,
			RoundTimeoutSeconds:    1,
			OnError:                config.FusionOnErrorSkip,
			QuorumFailurePolicy:    config.FusionQuorumFailurePolicyFallback,
			QuorumFallbackTarget:   "backup-model",
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).
		Execute(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)

	mu.Lock()
	dispatchedCount := len(dispatchedPanel)
	mu.Unlock()
	require.Equal(t, 1, dispatchedCount,
		"one concurrency slot means exactly one panel model reaches the backend")

	assert.Equal(t, dispatchedCount+1, resp.Iterations,
		"Iterations must count calls made, not the configured panel size")
	assert.Len(t, resp.ModelsUsed, dispatchedCount+1,
		"ModelsUsed must not name panel models that were still queued")
	assert.Equal(t, "backup-model", resp.ModelsUsed[len(resp.ModelsUsed)-1])

	mu.Lock()
	for _, model := range panelModels {
		if !dispatchedPanel[model] {
			assert.NotContains(t, resp.ModelsUsed, model,
				"a queued model that never attempted a call must not be reported as used")
		}
	}

	mu.Unlock()

	assertCallOrdinalContract(t, iterations, dispatchedCount, resp.Iterations)
}

// assertCallOrdinalContract checks that every backend call carried a distinct
// ordinal, that the fallback came after every dispatched panel call, and that
// the ordinals count actual calls.
//
// A slot-derived panel ordinal disagrees with a count-derived fallback ordinal
// whenever the dispatched subset is not a prefix of the configured panel, which
// lets the fallback appear to run before a panel call or reuse its number.
func assertCallOrdinalContract(t *testing.T, iterations map[string]int, dispatchedCount, aggregate int) {
	t.Helper()
	seen := map[int]string{}
	for model, ordinal := range iterations {
		require.Positive(t, ordinal, "%s carried no call ordinal", model)
		if previous, duplicate := seen[ordinal]; duplicate {
			t.Fatalf("call ordinal %d used by both %s and %s", ordinal, previous, model)
		}
		seen[ordinal] = model
	}
	fallbackOrdinal := iterations["backup-model"]
	for model, ordinal := range iterations {
		if model == "backup-model" {
			continue
		}
		assert.Less(t, ordinal, fallbackOrdinal,
			"the fallback must be ordered after every dispatched panel call")
	}
	assert.Len(t, seen, dispatchedCount+1)
	assert.Equal(t, dispatchedCount+1, fallbackOrdinal,
		"the fallback ordinal must equal the aggregate call count")
	assert.Equal(t, aggregate, fallbackOrdinal,
		"per-call telemetry must agree with the aggregate Iterations")
}

// The ordinary success path must account for dispatched calls too, not only the
// fallback path: early quorum strands queued workers there as well.
//
// The exact dispatched count is not asserted because it is legitimately racy. A
// worker can win the semaphore in the window between the deciding result being
// sent and cancellation becoming observable. What must hold is that the reported
// accounting describes calls actually attempted, never the whole plan.
func TestSuccessAccountingCountsOnlyDispatchedPanelCalls(t *testing.T) {
	panelModels := []string{"panel-a", "panel-b", "panel-c", "panel-d", "panel-e"}
	recorder := &fusionCallRecorder{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		ordinal, _ := strconv.Atoi(r.Header.Get("x-vsr-looper-iteration"))
		recorder.record(payload.Model, ordinal)

		writeCanaryReasoningCompletion(w, payload.Model, "usable answer from "+payload.Model, "")
	}))
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: panelModels,
			// One slot and a quorum of one: the first usable response decides the
			// panel while the remaining workers are still queued.
			MaxConcurrent:          1,
			MinSuccessfulResponses: 1,
			OnError:                config.FusionOnErrorSkip,
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).
		Execute(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)

	calls := recorder.snapshot()
	observed := map[string]bool{}
	judgeCalls := 0
	for _, call := range calls {
		if call.model == "judge" {
			judgeCalls++
			continue
		}
		observed[call.model] = true
	}

	require.Positive(t, judgeCalls, "a successful panel must reach judge synthesis")

	reportedPanel := []string{}
	for _, model := range resp.ModelsUsed {
		if model != "judge" {
			reportedPanel = append(reportedPanel, model)
		}
	}

	assert.Less(t, len(reportedPanel), len(panelModels),
		"early quorum left workers queued, so not every configured model was called")
	assert.Contains(t, resp.ModelsUsed, "judge")

	// Judge count comes from the backend rather than a literal, so this stays
	// correct if the recipe's analysis_mode changes.
	assert.Equal(t, len(reportedPanel)+judgeCalls, resp.Iterations,
		"Iterations and ModelsUsed must describe the same calls")

	for model := range observed {
		assert.Contains(t, reportedPanel, model,
			"a model the backend served must be reported as used")
	}

	// Every call the backend saw, judge stages included, carried a distinct
	// ordinal. Asserting over the panel subset alone would miss a judge call
	// renumbered onto an ordinal a panel call already used.
	//
	// Contiguity is not asserted here: a worker can pass the dispatch gate and
	// take an ordinal just before cancellation becomes observable, and its call
	// is then cancelled in flight without reaching this backend. The
	// deterministic sequence is pinned by
	// TestJudgeOrdinalsContinuePanelAttemptSequence instead.
	assertUniqueCallOrdinals(t, calls, resp.Iterations)
	assert.LessOrEqual(t, len(calls), resp.Iterations,
		"the backend cannot have seen more calls than the aggregate reports")
}

// fusionCallRecorder captures every request the test backend handler observes,
// judge calls included, in handler-observation order. That is not necessarily
// dispatch order, and no assertion here depends on the order.
//
// Judge calls are recorded because a recorder keyed by model name collapses the
// two judge stages and cannot see an ordinal collision with a panel call.
type fusionCallRecorder struct {
	mu    sync.Mutex
	calls []fusionRecordedCall
}

type fusionRecordedCall struct {
	model   string
	ordinal int
}

func (r *fusionCallRecorder) record(model string, ordinal int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, fusionRecordedCall{model: model, ordinal: ordinal})
}

func (r *fusionCallRecorder) snapshot() []fusionRecordedCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]fusionRecordedCall(nil), r.calls...)
}

// assertUniqueCallOrdinals checks that every call the backend saw carried a
// distinct ordinal within the aggregate count.
//
// It deliberately does not require contiguity. An ordinal is allocated when a
// worker passes the dispatch gate, but cancellation can land while that call is
// in flight, so a genuinely attempted call can be numbered without ever reaching
// a backend. That leaves a gap in what the backend observed, which is accurate
// rather than a defect.
func assertUniqueCallOrdinals(t *testing.T, calls []fusionRecordedCall, aggregate int) map[int]string {
	t.Helper()
	seen := make(map[int]string, len(calls))
	for _, call := range calls {
		require.Positive(t, call.ordinal, "%s carried no call ordinal", call.model)
		require.LessOrEqual(t, call.ordinal, aggregate,
			"%s carried an ordinal outside the reported aggregate", call.model)
		if previous, duplicate := seen[call.ordinal]; duplicate {
			t.Fatalf("ordinal %d reused by %s and %s", call.ordinal, previous, call.model)
		}
		seen[call.ordinal] = call.model
	}
	return seen
}

// assertContiguousCallOrdinals additionally requires the observed ordinals to
// cover 1..len(calls) with no gap. Only valid where every dispatched call is
// guaranteed to reach the backend, which makes a gap real evidence of
// renumbering rather than of an in-flight cancellation.
func assertContiguousCallOrdinals(t *testing.T, calls []fusionRecordedCall) {
	t.Helper()
	seen := assertUniqueCallOrdinals(t, calls, len(calls))
	for ordinal := 1; ordinal <= len(calls); ordinal++ {
		assert.Contains(t, seen, ordinal,
			"ordinal %d was never emitted, so the sequence has a gap", ordinal)
	}
}

// The judge stages must continue the ordinal sequence the panel emitted, not
// restart from the size of the response slice they are handed.
//
// That slice is the usable, grounding-filtered panel. Whenever it is smaller than
// the number of dispatched attempts -- an unusable reply here, a grounding filter
// in production -- inferring an ordinal from its length renumbers the judge onto
// an ordinal the panel already used. Pinning it at this seam covers every such
// filter, because the judge stages no longer see the slice length at all.
//
// Every analysis mode is covered: each dispatches its judge stages through a
// different path, so pinning only the two-stage mode would leave the single-call
// paths free to re-derive an ordinal from the panel slice.
func TestJudgeOrdinalsContinuePanelAttemptSequence(t *testing.T) {
	tests := []struct {
		name          string
		analysisMode  string
		judgeOrdinals []int
	}{
		{
			name:          "separate runs analysis then final",
			analysisMode:  config.FusionAnalysisModeSeparate,
			judgeOrdinals: []int{3, 4},
		},
		{
			name:          "one_call runs a single terminal judge",
			analysisMode:  config.FusionAnalysisModeOneCall,
			judgeOrdinals: []int{3},
		},
		{
			name:          "none runs a single terminal judge",
			analysisMode:  config.FusionAnalysisModeNone,
			judgeOrdinals: []int{3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := &fusionCallRecorder{}
			unusableServed := make(chan struct{})

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var payload struct {
					Model string `json:"model"`
				}
				require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
				ordinal, _ := strconv.Atoi(r.Header.Get("x-vsr-looper-iteration"))
				recorder.record(payload.Model, ordinal)

				switch payload.Model {
				case "panel-b":
					// Dispatched and paid for, but filtered out of the judge's panel.
					writeCanaryReasoningCompletion(w, payload.Model, "", "")
					close(unusableServed)
				case "panel-a":
					<-unusableServed
					writeCanaryReasoningCompletion(w, payload.Model, "panel a answer", "")
				default:
					writeCanaryReasoningCompletion(w, payload.Model, "judge output", "")
				}
			}))
			defer server.Close()

			req := newFusionTestRequest()
			req.Algorithm = &config.AlgorithmConfig{
				Type: "fusion",
				Fusion: &config.FusionAlgorithmConfig{
					Model:                  "judge",
					AnalysisModels:         []string{"panel-a", "panel-b"},
					AnalysisMode:           tt.analysisMode,
					MaxConcurrent:          2,
					MinSuccessfulResponses: 1,
					OnError:                config.FusionOnErrorSkip,
				},
			}

			resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).
				Execute(context.Background(), req)
			require.NoError(t, err)
			require.NotNil(t, resp)

			calls := recorder.snapshot()
			require.Len(t, calls, 2+len(tt.judgeOrdinals),
				"two panel attempts plus this mode's judge calls")

			assertContiguousCallOrdinals(t, calls)
			assert.Equal(t, len(calls), resp.Iterations,
				"aggregate Iterations must equal the number of calls actually made")

			// The judge stages come last, on the ordinals after the panel attempts.
			judgeOrdinals := []int{}
			for _, call := range calls {
				if call.model == "judge" {
					judgeOrdinals = append(judgeOrdinals, call.ordinal)
				}
			}
			assert.ElementsMatch(t, tt.judgeOrdinals, judgeOrdinals,
				"the judge calls follow the two dispatched panel attempts")
		})
	}
}
