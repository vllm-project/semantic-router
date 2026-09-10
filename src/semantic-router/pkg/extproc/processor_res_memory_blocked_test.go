package extproc

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestBlockedResponseRecordsMemoryPersistenceReceipt(t *testing.T) {
	for _, tc := range []struct {
		name, status, reason string
		classifyFailure      bool
		autoStoreOff         bool
		noExtractor          bool
	}{
		{name: "detected", status: "policy_blocked", reason: "response_jailbreak"},
		{name: "unverified", status: "policy_blocked", reason: "jailbreak_unverified", classifyFailure: true},
		{name: "opt_out", status: "disabled", reason: "auto_store_off", autoStoreOff: true},
		{name: "no_extractor", status: "disabled", reason: "no_extractor", noExtractor: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := newJailbreakScoreServer(t, 0.95, 0.05)
			if tc.classifyFailure {
				server = newJailbreakFailingServer(t)
			}
			router, ctx := newResponseStageRouter(t, server, config.OnErrorBlock, "block")
			router.Config.Memory.AutoStore = true
			if !tc.noExtractor {
				router.MemoryExtractor = memory.NewMemoryChunkStore(&noopMemoryStore{})
			}
			if tc.autoStoreOff {
				off := false
				ctx.RequestAutoStore = &off
			}
			// No runner is installed: a blocked response must never submit a write.
			recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
			t.Cleanup(func() { assert.NoError(t, recorder.Close()) })
			ctx.RequestID, ctx.RouterReplayID = tc.name, tc.name
			ctx.RouterReplayRecorder = recorder
			ctx.SourceFormat, ctx.TargetFormat = llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1
			_, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: tc.name})
			require.NoError(t, err)
			counter := metrics.PluginExecutionTotal.WithLabelValues("memory_persistence", requestDecisionStateKey(ctx), tc.status)
			before := testutil.ToFloat64(counter)

			response := router.handleNonStreamingResponseBody(extProcResponseFixture(ctx.SourceFormat), ctx, 0)
			require.NotNil(t, response.GetImmediateResponse(), "the safety policy must still block the response")
			assert.EqualValues(t, 403, response.GetImmediateResponse().GetStatus().GetCode())
			require.NoError(t, recorder.DrainOutcomes())
			record, found := recorder.GetRecord(tc.name)
			require.True(t, found)
			var receipts []routerreplay.Outcome
			for _, outcome := range record.Outcomes {
				if outcome.TargetRef == "memory_persistence" {
					receipts = append(receipts, outcome)
				}
			}
			require.Len(t, receipts, 1)
			assert.Equal(t, tc.status, receipts[0].Verdict)
			assert.Equal(t, tc.reason, receipts[0].Reason)
			assert.Equal(t, "terminal", receipts[0].Metadata["phase"])
			assert.Equal(t, "false", receipts[0].Metadata["fail_open"])
			assert.Equal(t, before+1, testutil.ToFloat64(counter))
		})
	}
}
