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
	"net/http"
	"net/http/httptest"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// quorumFallbackServer serves a below-quorum panel whose fallback target
// answers. One panel member is usable, one fails, so quorum of two is missed.
func quorumFallbackServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(request.Body).Decode(&payload))
		switch payload.Model {
		case "panel-a":
			writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
				"index":         0,
				"message":       map[string]interface{}{"role": "assistant", "content": "panel a answer"},
				"finish_reason": "stop",
			}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
		case "panel-b":
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte("panel b upstream failure"))
		case "backup-model":
			writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
				"index":         0,
				"message":       map[string]interface{}{"role": "assistant", "content": "fallback answer"},
				"finish_reason": "stop",
			}}, map[string]int64{"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10})
		default:
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
}

func quorumFallbackDecision() *config.Decision {
	return &config.Decision{
		Name: "fusion-quorum-fallback-route",
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model:                  "judge",
				AnalysisModels:         []string{"panel-a", "panel-b"},
				MaxConcurrent:          2,
				MinSuccessfulResponses: 2,
				OnError:                config.FusionOnErrorSkip,
				QuorumFailurePolicy:    config.FusionQuorumFailurePolicyFallback,
				QuorumFallbackTarget:   "backup-model",
			},
		},
	}
}

// quorumFallbackRouter wires a real Replay recorder so persistence is asserted
// rather than assumed.
func quorumFallbackRouter(endpoint string) (*OpenAIRouter, *RequestContext, *llmprotocol.Request) {
	replayConfig := config.DefaultRouterReplayPluginConfig()
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{Endpoint: endpoint},
			// The fallback needs resolvable metadata for the panel and target.
			BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
				"panel-a":      {Capabilities: []string{"chat"}},
				"panel-b":      {Capabilities: []string{"chat"}},
				"judge":        {Capabilities: []string{"chat"}},
				"backup-model": {Capabilities: []string{"chat"}},
				"broken-model": {Capabilities: []string{"chat"}},
			}},
		},
		ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(10, 0)),
	}
	request := testNeutralRequest("router-entrypoint", "hello")
	requestContext := &RequestContext{
		RequestID:                "replay-fusion-fallback",
		Headers:                  map[string]string{},
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          request,
		RouterReplayPluginConfig: &replayConfig,
		// A preliminary panel candidate is already selected by runtime model
		// selection. The fallback target must replace it in the client-visible
		// routing facts, which is only true if the looper's final model lands on
		// the context before response headers are built.
		VSRSelectedModel: "panel-a",
	}
	return router, requestContext, request
}

// A served fallback must persist its bounded evidence and aggregate usage, and
// its client-visible routing facts must name the model that answered.
func TestHandleServedFallbackPersistsOutcomeAndRoutingFacts(t *testing.T) {
	server := quorumFallbackServer(t)
	defer server.Close()

	router, requestContext, request := quorumFallbackRouter(server.URL)
	decision := quorumFallbackDecision()

	response, err := router.handleLooperExecution(context.Background(), request, decision, requestContext)
	require.NoError(t, err)
	require.Equal(t, typev3.StatusCode_OK, response.GetImmediateResponse().GetStatus().GetCode())

	// The fallback target answered, so it must be the advertised selected model
	// even though a panel candidate was selected first.
	assert.Equal(t, "backup-model", requestContext.VSRSelectedModel)
	assert.Equal(t, "backup-model", immediateHeaderValue(response, headers.VSRSelectedModel))

	// Replay must exist and its ID must reach the caller.
	require.NotEmpty(t, requestContext.RouterReplayID)
	assert.Equal(t, requestContext.RouterReplayID,
		immediateHeaderValue(response, headers.RouterReplayID))

	record, found := router.ReplayRecorder.GetRecord(requestContext.RouterReplayID)
	require.True(t, found, "a served fallback must be recorded in Replay")
	quorum := record.RouteDiagnostics.FusionQuorum
	require.NotNil(t, quorum, "the served fallback outcome must be persisted")
	assert.Equal(t, 2, quorum.RequiredCount)
	assert.Equal(t, 1, quorum.UsableCount)
	assert.Equal(t, "backup-model", quorum.FallbackTarget)
	assert.Equal(t, "fallback_served", quorum.Disposition,
		"the terminal disposition must reflect that encoding succeeded")
	require.Len(t, quorum.Attempts, 2)

	// Aggregate accounting covers the panel plus the fallback call.
	assert.Equal(t, 10+7, derefInt(record.PromptTokens))
	assert.Equal(t, 12+10, derefInt(record.TotalTokens))

	// The encoded response carries no router extension.
	body := string(response.GetImmediateResponse().GetBody())
	assert.Contains(t, body, "fallback answer")
	assert.NotContains(t, body, `"fusion"`)
}

// When protocol translation fails after a fallback is ready, the caller gets a
// 502 and the evidence must still be persisted. Recording the outcome without
// starting Replay would silently persist nothing.
func TestHandleFallbackEncodeFailurePersistsFailedReplay(t *testing.T) {
	server := quorumFallbackServer(t)
	defer server.Close()

	router, requestContext, request := quorumFallbackRouter(server.URL)
	// An unsupported target format makes prepareLooperResponse fail after the
	// fallback has already answered and formatted.
	requestContext.SourceFormat = llmprotocol.WireFormat("definitely.not.a.codec")
	decision := quorumFallbackDecision()

	response, err := router.handleLooperExecution(context.Background(), request, decision, requestContext)
	require.NoError(t, err)
	require.Equal(t, typev3.StatusCode_BadGateway, response.GetImmediateResponse().GetStatus().GetCode())

	require.NotEmpty(t, requestContext.RouterReplayID, "a failed encode must still create a Replay record")
	assert.Equal(t, requestContext.RouterReplayID,
		immediateHeaderValue(response, headers.RouterReplayID))

	record, found := router.ReplayRecorder.GetRecord(requestContext.RouterReplayID)
	require.True(t, found)
	assert.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	assert.Equal(t, "looper_response_encode_failed", record.TerminalReason)

	quorum := record.RouteDiagnostics.FusionQuorum
	require.NotNil(t, quorum, "the failed encode must still persist the quorum outcome")
	assert.Equal(t, "response_encode_failed", quorum.Disposition,
		"the disposition must say the response was never encoded")
	assert.Equal(t, 12+10, derefInt(record.TotalTokens), "tokens were spent and must be persisted")

	// No canary text from the failing panel member may reach the client body.
	assert.NotContains(t, string(response.GetImmediateResponse().GetBody()), "panel b upstream failure")
}

// The served-fallback path emits one sample of each metric, with the exact label
// set #3375 requires. Deltas rather than absolute values, because the vectors
// are process-global.
//
// Exactly-once across the router rests on the three finalizing paths being
// mutually exclusive, not on a guard inside the outcome. This asserts the served
// path; the encode-failure and execution-failure paths are asserted separately.
func TestFinalizeLooperQuorumOutcomeEmitsOneSamplePerServedFallback(t *testing.T) {
	decision := &config.Decision{Name: "quorum-metric-served"}
	outcome := &looper.FusionQuorumOutcome{
		RequiredCount:  2,
		UsableCount:    1,
		SelectedPolicy: "fallback",
		FallbackTarget: "backup-model",
		Disposition:    looper.FusionQuorumFallbackReady,
		Attempts: []looper.FusionQuorumAttemptOutcome{
			{Model: "panel-a", State: looper.FusionPanelAttemptUsable},
			{Model: "panel-b", State: looper.FusionPanelAttemptFailed},
		},
	}
	failure := quorumFailureSeries(decision.Name, "fallback", "fallback_served")
	fallback := quorumFallbackSeries(decision.Name, "backup-model", "fallback_served")
	required := quorumHistogramSeries(decision.Name, "llm_fusion_quorum_required_responses")
	usable := quorumHistogramSeries(decision.Name, "llm_fusion_quorum_usable_responses")
	usableState := quorumAttemptSeries(decision.Name, "usable")
	failedState := quorumAttemptSeries(decision.Name, "failed")
	before := captureQuorumSeries(t, failure, fallback, required, usable, usableState, failedState)

	finalizeLooperQuorumOutcome(&RequestContext{RequestID: "m1"}, outcome, decision, true)

	assert.Equal(t, 1.0, before.delta(t, failure),
		"one quorum sample, carrying the selected policy")
	assert.Equal(t, 1.0, before.delta(t, fallback),
		"one fallback sample, carrying the fallback target")
	assert.Equal(t, 1.0, before.delta(t, required))
	assert.Equal(t, 1.0, before.delta(t, usable))
	assert.Equal(t, 1.0, before.delta(t, usableState))
	assert.Equal(t, 1.0, before.delta(t, failedState))
	assert.NotContains(t, quorumDispositionsSeen(t, decision.Name), string(looper.FusionQuorumFallbackReady))
}

// A response that cannot be encoded is counted as such, and never as served.
func TestEncodeFailureEmitsResponseEncodeFailedSample(t *testing.T) {
	server := quorumFallbackServer(t)
	defer server.Close()

	router, requestContext, request := quorumFallbackRouter(server.URL)
	requestContext.SourceFormat = llmprotocol.WireFormat("definitely.not.a.codec")
	decision := quorumFallbackDecision()
	decision.Name = "quorum-metric-encode-failure"

	encodeFailed := quorumFailureSeries(decision.Name, "fallback", "response_encode_failed")
	served := quorumFailureSeries(decision.Name, "fallback", "fallback_served")
	before := captureQuorumSeries(t, encodeFailed, served)

	_, err := router.handleLooperExecution(context.Background(), request, decision, requestContext)
	require.NoError(t, err)

	assert.Equal(t, 1.0, before.delta(t, encodeFailed))
	assert.Zero(t, before.delta(t, served),
		"a response the caller never received must not be counted as served")
	assert.NotContains(t, quorumDispositionsSeen(t, decision.Name), string(looper.FusionQuorumFallbackReady))
}

// A served fallback is counted once end to end, with policy and target labels.
func TestServedFallbackEmitsOneServedSample(t *testing.T) {
	server := quorumFallbackServer(t)
	defer server.Close()

	router, requestContext, request := quorumFallbackRouter(server.URL)
	decision := quorumFallbackDecision()
	decision.Name = "quorum-metric-served"

	failure := quorumFailureSeries(decision.Name, "fallback", "fallback_served")
	fallback := quorumFallbackSeries(decision.Name, "backup-model", "fallback_served")
	before := captureQuorumSeries(t, failure, fallback)

	_, err := router.handleLooperExecution(context.Background(), request, decision, requestContext)
	require.NoError(t, err)

	assert.Equal(t, 1.0, before.delta(t, failure))
	assert.Equal(t, 1.0, before.delta(t, fallback))
	assert.NotContains(t, quorumDispositionsSeen(t, decision.Name), string(looper.FusionQuorumFallbackReady))
}

// Terminal outcomes must be counted exactly once, by the Looper that produced
// them, with their real policy and target labels. Driving them through real
// policy paths is what proves the first sample exists at all; a fabricated
// already-terminal response only shows the finalizer adds no second one.
func TestLooperTerminalOutcomesAreCountedOnceEndToEnd(t *testing.T) {
	for _, test := range []struct {
		name        string
		disposition string
		policy      string
		target      string
		configure   func(*config.Decision, *RequestContext)
	}{
		{
			name:        "quorum failed with no fallback configured",
			disposition: "quorum_failed",
			policy:      "fail",
			configure: func(d *config.Decision, _ *RequestContext) {
				d.Algorithm.Fusion.QuorumFailurePolicy = ""
				d.Algorithm.Fusion.QuorumFallbackTarget = ""
			},
		},
		{
			name:        "fallback target itself fails",
			disposition: "fallback_failed",
			policy:      "fallback",
			target:      "broken-model",
			configure: func(d *config.Decision, _ *RequestContext) {
				d.Algorithm.Fusion.QuorumFallbackTarget = "broken-model"
			},
		},
		{
			name:        "caller cancelled before the fallback",
			disposition: "cancelled",
			policy:      "fallback",
			configure:   func(_ *config.Decision, _ *RequestContext) {},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := quorumFallbackServer(t)
			defer server.Close()

			router, requestContext, request := quorumFallbackRouter(server.URL)
			decision := quorumFallbackDecision()
			decision.Name = "quorum-terminal-" + test.disposition
			test.configure(decision, requestContext)

			failure := quorumFailureSeries(decision.Name, test.policy, test.disposition)
			series := []quorumSeries{failure}
			var fallbackSeries quorumSeries
			if test.target != "" {
				fallbackSeries = quorumFallbackSeries(decision.Name, test.target, test.disposition)
				series = append(series, fallbackSeries)
			}
			before := captureQuorumSeries(t, series...)

			ctx := context.Background()
			if test.disposition == "cancelled" {
				cancelled, cancel := context.WithCancel(ctx)
				cancel()
				ctx = cancelled
			}
			_, err := router.handleLooperExecution(ctx, request, decision, requestContext)
			require.NoError(t, err)

			assert.Equal(t, 1.0, before.delta(t, failure),
				"a terminal outcome must be counted exactly once, with its real policy label")
			assert.NotContains(t, quorumDispositionsSeen(t, decision.Name), string(looper.FusionQuorumFallbackReady))

			if test.target != "" {
				assert.Equal(t, 1.0, before.delta(t, fallbackSeries),
					"the fallback counter must carry the real target label")
			}
		})
	}
}

// A disposition that was already terminal in the Looper is counted here, once,
// and is never rewritten. The Looper decided what happened; this layer only
// decides when the sample is emitted.
func TestFinalizeLooperQuorumOutcomeCountsLooperTerminalsOnce(t *testing.T) {
	for _, disposition := range []looper.FusionQuorumDisposition{
		looper.FusionQuorumFailed,
		looper.FusionQuorumFallbackFailed,
		looper.FusionQuorumFallbackResponseFailed,
		looper.FusionQuorumBudgetExhausted,
		looper.FusionQuorumCancelled,
	} {
		t.Run(string(disposition), func(t *testing.T) {
			decision := &config.Decision{Name: "quorum-terminal-" + string(disposition)}
			failure := quorumFailureSeries(decision.Name, "fallback", string(disposition))
			before := captureQuorumSeries(t, failure)
			outcome := &looper.FusionQuorumOutcome{
				SelectedPolicy: "fallback",
				Disposition:    disposition,
			}

			finalizeLooperQuorumOutcome(&RequestContext{}, outcome, decision, true)

			assert.Equal(t, disposition, outcome.Disposition,
				"a terminal disposition must not be rewritten")
			assert.Equal(t, 1.0, before.delta(t, failure),
				"a terminal outcome must be counted exactly once, at this layer")
		})
	}
}

func derefInt(value *int) int {
	if value == nil {
		return 0
	}
	return *value
}

// A streaming caller must receive the fallback as SSE through the full ExtProc
// boundary, not merely from the Looper formatter. The formatter tests cannot see
// protocol translation, header construction, or Replay.
func TestHandleStreamingFallbackThroughExtProc(t *testing.T) {
	server := quorumFallbackServer(t)
	defer server.Close()

	router, requestContext, request := quorumFallbackRouter(server.URL)
	// Streaming is driven by the request context, not the neutral request.
	requestContext.ExpectStreamingResponse = true
	decision := quorumFallbackDecision()
	decision.Name = "quorum-streaming-fallback"

	response, err := router.handleLooperExecution(context.Background(), request, decision, requestContext)
	require.NoError(t, err)
	immediate := response.GetImmediateResponse()
	require.NotNil(t, immediate)
	require.Equal(t, typev3.StatusCode_OK, immediate.GetStatus().GetCode())

	assert.Equal(t, "text/event-stream", immediateHeaderValue(response, "content-type"))

	body := string(immediate.GetBody())
	assert.Contains(t, body, "fallback answer", "the caller must receive the fallback answer")
	assert.Contains(t, body, "data: [DONE]", "the stream must terminate")
	assert.NotContains(t, body, `"fusion"`, "a served fallback must carry no Fusion extension")

	// Routing facts and Replay behave the same as the buffered path.
	assert.Equal(t, "backup-model", immediateHeaderValue(response, headers.VSRSelectedModel))
	require.NotEmpty(t, requestContext.RouterReplayID)
	assert.Equal(t, requestContext.RouterReplayID,
		immediateHeaderValue(response, headers.RouterReplayID))

	record, found := router.ReplayRecorder.GetRecord(requestContext.RouterReplayID)
	require.True(t, found)
	quorum := record.RouteDiagnostics.FusionQuorum
	require.NotNil(t, quorum)
	assert.Equal(t, "fallback_served", quorum.Disposition)
	assert.Equal(t, 12+10, derefInt(record.TotalTokens))
}
