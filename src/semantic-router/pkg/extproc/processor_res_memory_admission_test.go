package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestMemorySchedulingRejectsLargeHistoryBeforePreparation(t *testing.T) {
	for _, mode := range []string{"queue_full", "receipt_queue_full", "shutting_down", "missing_user", "history_too_large"} {
		t.Run(mode, func(t *testing.T) {
			recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
			_, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: mode})
			require.NoError(t, err)
			runner := memory.NewPersistenceRunner(time.Second, 1, 1)
			release := make(chan struct{})
			var heldReceipts []*routerreplay.OutcomeReservation
			t.Cleanup(func() {
				close(release)
				assert.NoError(t, runner.RetireAndWait(time.Second))
				for _, held := range heldReceipts {
					held.Finish(routerreplay.Outcome{Verdict: "cancelled"})
				}
				assert.NoError(t, recorder.DrainOutcomes())
			})
			switch mode {
			case "queue_full":
				started := make(chan struct{}, 1)
				job := memory.PersistenceJob{
					Run: func(context.Context) (memory.PersistenceOutcome, error) {
						started <- struct{}{}
						<-release
						return memory.PersistenceOutcome{}, nil
					},
					Report: func(string, string, bool, error) {},
				}
				runner.Submit(context.Background(), job)
				requireReceiptSignal(t, started)
				runner.Submit(context.Background(), job)
			case "receipt_queue_full":
				for i := 0; i < routerreplay.DefaultOutcomeQueueCapacity; i++ {
					held := recorder.TryReserveOutcome(mode)
					require.NotNil(t, held)
					heldReceipts = append(heldReceipts, held)
				}
			case "shutting_down":
				require.NoError(t, runner.RetireAndWait(time.Second))
			}
			router := &OpenAIRouter{
				Config:          &config.RouterConfig{Memory: config.MemoryConfig{AutoStore: true}},
				MemoryExtractor: memory.NewMemoryChunkStore(&noopMemoryStore{}), memoryPersistence: runner, ReplayRecorder: recorder,
			}
			ctx := persistenceRegressionContext(mode)
			ctx.RouterReplayID = mode
			// This would be rejected by the history validator if scheduling touched
			// history before checking identity and reserving runner capacity.
			ctx.SemanticRequest.Messages = make([]llmprotocol.Message, maxMemorySnapshotMessages+1)
			if mode == "missing_user" {
				ctx.Headers = nil
			}
			router.scheduleResponseMemoryStoreText(ctx, "response")
			require.Eventually(t, func() bool {
				record, _ := recorder.GetRecord(mode)
				for _, outcome := range record.Outcomes {
					if outcome.Metadata["phase"] == "terminal" {
						return true
					}
				}
				return false
			}, time.Second, time.Millisecond)
			record, found := recorder.GetRecord(mode)
			require.True(t, found)
			want := mode
			if mode == "missing_user" {
				want = "memory_info_unavailable"
			}
			assert.Equal(t, want, record.Outcomes[len(record.Outcomes)-1].Reason)
			if mode != "history_too_large" {
				assert.Len(t, record.Outcomes, 1)
			}
		})
	}
}
