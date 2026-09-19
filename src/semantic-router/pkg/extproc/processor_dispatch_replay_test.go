package extproc

import (
	"encoding/json"
	"net/http"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// Exercise the routing callers, not just the snapshot builder: Replay must
// persist the final demand after dispatch has materialized automatic output.
func TestProviderDispatchReplayPersistsFinalDemand(t *testing.T) {
	for _, tc := range []struct {
		name      string
		specified bool
		explicit  bool
		stream    bool
	}{
		{name: "entrypoint_auto"},
		{name: "entrypoint_auto_stream", stream: true},
		{name: "entrypoint_explicit", explicit: true},
		{name: "specified_explicit", specified: true, explicit: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			r, ctx, model := dispatchReplayFixture(t, tc.explicit, renderMock(t, &calls, 0, 0))
			ctx.SemanticRequest.Stream = tc.stream
			ctx.ExpectStreamingResponse = tc.stream
			var response *ext_proc.ProcessingResponse
			var err error
			if tc.specified {
				response, err = r.handleSpecifiedModelRouting(ctx.SemanticRequest, model, "", ctx)
			} else {
				response, err = r.handleEntrypointModelRouting(ctx.SemanticRequest, "auto", ctx.VSRSelectedDecision.Name, entropy.ReasoningDecision{}, model, ctx)
			}
			require.NoError(t, err)
			var wire map[string]any
			require.NoError(t, json.Unmarshal(response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), &wire))
			if tc.stream {
				require.Equal(t, true, wire["stream"])
			}
			budget := int64(8192)
			if tc.explicit {
				budget = 12
				require.Zero(t, calls)
			} else {
				require.Equal(t, 3, calls, "selection, preparation and final dispatch must each render")
			}
			wireBudget := wire["max_tokens"]
			if wireBudget == nil {
				wireBudget = wire["max_completion_tokens"]
			}
			require.EqualValues(t, budget, wireBudget)
			require.NotEmpty(t, ctx.RouterReplayID)
			record, ok := r.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			require.True(t, ok)
			require.NotNil(t, record.RouteDiagnostics)
			snapshots := record.RouteDiagnostics.RequestDemandSnapshots
			require.Len(t, snapshots, 4, "persisted Replay must include provider_bound")
			require.Equal(t, requestDemandStageProviderBound, snapshots[3].Stage)
			require.Equal(t, model, snapshots[3].Model)
			require.EqualValues(t, budget, snapshots[3].ReservedOutputTokens)
			require.True(t, snapshots[3].TotalDemandKnown)
			if !tc.explicit {
				require.False(t, snapshots[0].TotalDemandKnown, "ingress must retain the original unknown reserve")
				require.Zero(t, snapshots[0].ReservedOutputTokens)
			}

			// Client headers are emitted after dispatch, so deferring Replay
			// creation must still expose the ID of the persisted record.
			contentType := "application/json"
			if tc.stream {
				contentType = "text/event-stream"
			}
			headerResponse, err := r.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
				ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
					{Key: ":status", RawValue: []byte("200")},
					{Key: "content-type", RawValue: []byte(contentType)},
				}}},
			}, ctx)
			require.NoError(t, err)
			replayHeaders := 0
			for _, option := range headerResponse.GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders() {
				if option.GetHeader().GetKey() == headers.RouterReplayID {
					replayHeaders++
					require.Equal(t, record.ID, string(option.GetHeader().GetRawValue()))
				}
			}
			require.Equal(t, 1, replayHeaders)
			r.attachRouterReplayResponse(ctx, []byte(`{"choices":[{"message":{"content":"hello"}}]}`), true)
			finished, ok := r.ReplayRecorder.GetRecord(record.ID)
			require.True(t, ok)
			require.Equal(t, routerreplay.LifecycleCompleted, finished.LifecycleState)
			require.Equal(t, snapshots, finished.RouteDiagnostics.RequestDemandSnapshots)
		})
	}
}

func TestProviderDispatchReplaySurvivesFinalizerFailure(t *testing.T) {
	calls := 0
	normalRender := renderMock(t, &calls, 0, 0)
	r, ctx, model := dispatchReplayFixture(t, false, func(w http.ResponseWriter, req *http.Request) {
		if calls == 2 {
			calls++
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		normalRender(w, req)
	})
	response, err := r.handleEntrypointModelRouting(ctx.SemanticRequest, "auto", ctx.VSRSelectedDecision.Name, entropy.ReasoningDecision{}, model, ctx)
	require.Nil(t, response)
	require.ErrorIs(t, err, selection.ErrNoEligibleCandidates)
	require.Equal(t, 3, calls, "failure must occur in the finalizer, after dispatch preparation")
	require.NotEmpty(t, ctx.RouterReplayID, "finalizer failure must not discard the Replay record")

	// Process owns the terminal error transition after the routing caller returns.
	state, reason := replayLifecycleForProcessError(err)
	r.finalizeRouterReplay(ctx, state, reason)
	record, ok := r.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	require.Equal(t, "request_processing_failed", record.TerminalReason)
	require.Len(t, record.RouteDiagnostics.RequestDemandSnapshots, 3, "failed rendering must not fabricate provider_bound")
}

func dispatchReplayFixture(t *testing.T, explicit bool, handler http.HandlerFunc) (*OpenAIRouter, *RequestContext, string) {
	t.Helper()
	r, ctx := automaticFixture(t, "hello", handler)
	model := ctx.VSRSelectedDecision.ModelRefs[0].Model
	params := r.Config.ModelConfig[model]
	params.MaxOutputTokens = 8192
	r.Config.ModelConfig[model] = params
	r.ReplayRecorder = routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	ctx.RouterReplayPluginConfig = &config.RouterReplayPluginConfig{Enabled: true}
	if explicit {
		ctx.SemanticRequest.Sampling.MaxOutputTokens = llmprotocol.Int64(12)
	}
	captureRequestDemand(ctx, requestDemandStageOriginal, ctx.SemanticRequest, "auto")
	require.NoError(t, r.prepareDecisionContextOverflow(ctx, "auto"))
	captureRequestDemand(ctx, requestDemandStagePostContext, ctx.SemanticRequest, "auto")
	captureRequestDemand(ctx, requestDemandStagePostToolPolicy, ctx.SemanticRequest, "auto")
	return r, ctx, model
}
