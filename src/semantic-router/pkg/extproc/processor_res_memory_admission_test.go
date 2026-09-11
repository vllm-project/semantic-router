package extproc

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
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
			// Capacity rejections still avoid cloning a large, valid snapshot.
			ctx.SemanticRequest.Messages[0].Content[0].Text = strings.Repeat("x", maxMemorySnapshotBytes/2)
			if mode == "missing_user" || mode == "history_too_large" {
				ctx.SemanticRequest.Messages = make([]llmprotocol.Message, maxMemorySnapshotMessages+1)
			}
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
			assert.Len(t, record.Outcomes, 1)
		})
	}
}

func TestMemorySchedulingOversizedResponseDoesNotReserveCapacity(t *testing.T) {
	for _, mode := range []string{"response", "combined_history", "retained_history", "think_tags"} {
		t.Run(mode, func(t *testing.T) {
			recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
			t.Cleanup(func() { assert.NoError(t, recorder.Close()) })
			runner, unblock := blockedPersistenceRunner(t)
			router := &OpenAIRouter{
				Config:            &config.RouterConfig{Memory: config.MemoryConfig{AutoStore: true}},
				MemoryExtractor:   memory.NewMemoryChunkStore(&noopMemoryStore{}),
				memoryPersistence: runner,
				ReplayRecorder:    recorder,
			}
			ctx := persistenceRegressionContext(mode)
			ctx.RouterReplayID = mode
			_, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: mode})
			require.NoError(t, err)
			text := strings.Repeat("x", maxMemorySnapshotBytes+1)
			switch mode {
			case "combined_history":
				text = strings.Repeat("x", maxMemorySnapshotBytes/2)
				ctx.SemanticRequest.Messages[0].Content[0].Text = text
			case "retained_history":
				text = strings.Repeat("x", maxMemorySnapshotBytes/2)
				ctx.ResponseObjectState = &ResponseObjectState{
					ConversationHistory: []*responseapi.StoredResponse{{OutputText: text}},
				}
			case "think_tags":
				text = "<think>" + text + "</think>Short answer."
			}
			response := memoryTestResponse(text)
			router.scheduleSemanticResponseMemoryStore(ctx, response)
			assert.Equal(t, text, response.Output[0].Content[0].Text, "the delivered response must remain intact")
			require.Eventually(t, func() bool {
				record, _ := recorder.GetRecord(mode)
				return len(record.Outcomes) == 1 && record.Outcomes[0].Verdict == "skipped"
			}, time.Second, time.Millisecond, "oversized input must be rejected while the worker is still blocked")
			rejected, _ := recorder.GetRecord(mode)
			assert.Equal(t, "history_too_large", rejected.Outcomes[0].Reason)
			assert.Equal(t, "terminal", rejected.Outcomes[0].Metadata["phase"])
			assert.Equal(t, "true", rejected.Outcomes[0].Metadata["fail_open"])

			// Both original queue slots must remain available despite the rejection.
			for _, id := range []string{"valid_one", "valid_two"} {
				_, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: id})
				require.NoError(t, err)
				valid := persistenceRegressionContext(id)
				valid.RouterReplayID = id
				router.scheduleSemanticResponseMemoryStore(valid, memoryTestResponse("Deploy the service in the preferred region."))
				require.Eventually(t, func() bool {
					record, _ := recorder.GetRecord(id)
					return len(record.Outcomes) == 1 && record.Outcomes[0].Verdict == "scheduled"
				}, time.Second, time.Millisecond)
			}
			unblock()
			require.NoError(t, runner.RetireAndWait(5*time.Second))
			require.NoError(t, recorder.DrainOutcomes())
			for _, id := range []string{"valid_one", "valid_two"} {
				record, _ := recorder.GetRecord(id)
				require.Len(t, record.Outcomes, 2)
				assert.Equal(t, "completed", record.Outcomes[1].Verdict)
			}
			rejected, _ = recorder.GetRecord(mode)
			assert.Len(t, rejected.Outcomes, 1, "rejected input must not execute after the worker is released")
		})
	}
}

func TestOversizedMemoryResponseStillReachesClient(t *testing.T) {
	server := newJailbreakScoreServer(t, 0.01, 0.99)
	router, ctx := newResponseStageRouter(t, server, "", "block")
	router.Config.Memory.AutoStore = true
	router.MemoryExtractor = memory.NewMemoryChunkStore(&noopMemoryStore{})
	request := persistenceRegressionContext("oversized-response")
	ctx.Headers, ctx.SemanticRequest = request.Headers, request.SemanticRequest
	ctx.RequestID, ctx.RouterReplayID = "oversized-response", "oversized-response"
	ctx.SourceFormat, ctx.TargetFormat = llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	t.Cleanup(func() { assert.NoError(t, recorder.Close()) })
	ctx.RouterReplayRecorder = recorder
	_, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: ctx.RouterReplayID})
	require.NoError(t, err)
	// No runner is installed: oversized replies must be rejected before admission.
	text := strings.Repeat("x", maxMemorySnapshotBytes+1)
	body := []byte(strings.Replace(string(extProcResponseFixture(ctx.SourceFormat)), "hello", text, 1))
	response := router.handleNonStreamingResponseBody(body, ctx, 0)
	require.NotNil(t, response.GetResponseBody(), "the reply must continue to the client")
	assert.Nil(t, response.GetResponseBody().GetResponse().GetBodyMutation())
	assert.Equal(t, text, semanticAssistantContent(ctx.SemanticResponse))
	require.NoError(t, recorder.DrainOutcomes())
	record, found := recorder.GetRecord(ctx.RouterReplayID)
	require.True(t, found)
	var receipts []routerreplay.Outcome
	for _, outcome := range record.Outcomes {
		if outcome.TargetRef == "memory_persistence" {
			receipts = append(receipts, outcome)
		}
	}
	require.Len(t, receipts, 1)
	assert.Equal(t, "skipped", receipts[0].Verdict)
	assert.Equal(t, "history_too_large", receipts[0].Reason)
	assert.Equal(t, "true", receipts[0].Metadata["fail_open"])
}
