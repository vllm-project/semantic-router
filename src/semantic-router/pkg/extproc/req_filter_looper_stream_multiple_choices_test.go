package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
)

const (
	looperStreamedTwoChoices  = `{"model":"vllm-sr/auto","stream":true,"n":2,"messages":[{"role":"user","content":"What is the capital of France?"}]}`
	looperStreamedOneChoice   = `{"model":"vllm-sr/auto","stream":true,"n":1,"messages":[{"role":"user","content":"What is the capital of France?"}]}`
	looperBufferedTwoChoices  = `{"model":"vllm-sr/auto","n":2,"messages":[{"role":"user","content":"What is the capital of France?"}]}`
	looperRatingsSkipDecision = `
      modelRefs: [{model: model-a}, {model: model-b}, {model: model-c}]
      algorithm:
        type: ratings
        on_error: skip
        ratings: {max_concurrent: 3}`
	looperFusionDecision = `
      modelRefs: [{model: model-a}, {model: model-b}]
      algorithm:
        type: fusion
        on_error: skip
        fusion:
          model: judge
          analysis_models: [model-a, model-b, model-c]
          analysis_mode: separate
          max_concurrent: 3
          min_successful_responses: 2`
)

// A Looper run whose every hop dispatch refuses for a stream of several
// choices answers the client with that refusal, as a single-model route
// does, instead of a server error a client would retry.
func TestLooperRunRefusedAtEveryHopAnswersStreamMultipleChoices(t *testing.T) {
	decisions := []struct{ name, decision string }{
		{"ratings-skip", looperRatingsSkipDecision},
		{"ratings-fail", `
      modelRefs: [{model: model-a}, {model: model-b}, {model: model-c}]
      algorithm:
        type: ratings
        on_error: fail
        ratings: {max_concurrent: 3}`},
		{"escalate-self-verify", `
      modelRefs: [{model: model-a}, {model: model-b}]
      algorithm:
        type: escalate
        escalate:
          order: size
          accept: {confidence: {method: self_verify, gte: 0.6}}`},
	}
	for _, d := range decisions {
		for adapter, engine := range looperAdapters {
			t.Run(d.name+"/"+adapter, func(t *testing.T) {
				harness := newLooperHarness(t, looperFixture{name: d.name, decision: d.decision})
				record := harness.serve(t, engine(harness.router), looperStreamedTwoChoices)

				require.Equal(t, 400, record.Response.Status, "body: %s", record.Response.Body)
				var body struct {
					Error struct {
						Code string `json:"code"`
						Type string `json:"type"`
					} `json:"error"`
				}
				require.NoError(t, json.Unmarshal(record.Response.Body, &body))
				require.Equal(t, "stream_multiple_choices", body.Error.Code)
				require.Equal(t, "invalid_request_error", body.Error.Type)
				require.Empty(t, record.Calls, "a refused hop must not reach a backend")
			})
		}
	}
}

// The refusal answers only a request no hop can serve: a stream of one
// choice, a buffered request for several, and Fusion, whose hops are
// buffered and keep n, still reach the backends.
func TestLooperKeepsRequestsItsHopsCanServe(t *testing.T) {
	cases := []struct {
		name, decision, request string
		wantStream              bool
		wantN                   int
	}{
		{"ratings-streamed-n1", looperRatingsSkipDecision, looperStreamedOneChoice, true, 1},
		{"ratings-buffered-n2", looperRatingsSkipDecision, looperBufferedTwoChoices, false, 2},
		{"fusion-streamed-n2", looperFusionDecision, looperStreamedTwoChoices, false, 2},
	}
	for _, tc := range cases {
		for adapter, engine := range looperAdapters {
			t.Run(tc.name+"/"+adapter, func(t *testing.T) {
				harness := newLooperHarness(t, looperFixture{name: tc.name, decision: tc.decision})
				record := harness.serve(t, engine(harness.router), tc.request)

				require.Equal(t, 200, record.Response.Status, "body: %s", record.Response.Body)
				require.NotEmpty(t, record.Calls)
				for _, call := range record.Calls {
					var sent struct {
						Stream bool `json:"stream"`
						N      int  `json:"n"`
					}
					require.NoError(t, json.Unmarshal(call.Body, &sent))
					require.Equal(t, tc.wantStream, sent.Stream, "%s: %s", call.Route, call.Body)
					require.Equal(t, tc.wantN, sent.N, "%s: %s", call.Route, call.Body)
				}
			})
		}
	}
}

// The run answers with the refusal only when it refused every hop the run
// sent, each hop counted once.
func TestLooperHopRefusalsAnswerOnlyWhenEveryHopIsRefused(t *testing.T) {
	runCtx, refusals := withLooperHopRefusals(context.Background())
	refusal := llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "stream_multiple_choices", "streaming multiple choices is unsupported", nil)
	hop := func() *RequestContext { return &RequestContext{LooperRequest: true, TraceContext: runCtx} }

	first, second := hop(), hop()
	noteLooperHopRefusal(first, refusal)
	noteLooperHopRefusal(first, refusal)
	noteLooperHopRefusal(second, refusal)
	noteLooperHopRefusal(hop(), nil)
	noteLooperHopRefusal(&RequestContext{TraceContext: runCtx}, refusal)

	require.Nil(t, refusals.ofEveryHop(&graph.Outcome{Hops: 3}), "a hop got past the refusal")
	require.Nil(t, refusals.ofEveryHop(&graph.Outcome{}), "a run that sent no hop")
	require.Same(t, refusal, refusals.ofEveryHop(&graph.Outcome{Hops: 2}))

	runErr := explainLooperRunError(context.Canceled, refusals, &graph.Outcome{Hops: 2})
	var refused *looperHopsRefusedError
	require.ErrorAs(t, runErr, &refused)
	require.ErrorIs(t, runErr, context.Canceled, "the run's own error stays readable")
	require.Equal(t, context.Canceled, explainLooperRunError(context.Canceled, refusals, &graph.Outcome{Hops: 3}))
}
