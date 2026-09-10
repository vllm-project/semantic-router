package extproc

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

type slowReceiptStore struct {
	store.Storage
	entered     chan struct{}
	release     chan struct{}
	appendCalls atomic.Int64
}

func (s *slowReceiptStore) AppendOutcome(ctx context.Context, id string, outcome store.Outcome) error {
	s.appendCalls.Add(1)
	select {
	case s.entered <- struct{}{}:
	default:
	}
	select {
	case <-s.release:
		return s.Storage.AppendOutcome(ctx, id, outcome)
	case <-ctx.Done():
		return ctx.Err()
	}
}

func requireReceiptSignal(t *testing.T, signal <-chan struct{}) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(time.Second):
		t.Fatal("operation did not complete while recorder was blocked")
	}
}

func TestMemoryReceipts_SlowRecorderDoesNotBlockResponseScheduling(t *testing.T) {
	for _, tc := range []struct {
		name, status, reason string
		failOpen             bool
	}{
		{"no_extractor", "disabled", "no_extractor", false},
		{"auto_store_off", "disabled", "auto_store_off", false},
		{"policy_blocked", "policy_blocked", "response_jailbreak", false},
		{"queue_full", "rejected", "queue_full", true},
		{"shutting_down", "rejected", "shutting_down", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend := &slowReceiptStore{Storage: store.NewMemoryStore(10, 0), entered: make(chan struct{}, 1), release: make(chan struct{})}
			var releaseOnce sync.Once
			release := func() { releaseOnce.Do(func() { close(backend.release) }) }
			t.Cleanup(release)
			recorder := routerreplay.NewRecorder(backend)
			_, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: tc.name})
			require.NoError(t, err)
			t.Cleanup(func() { assert.NoError(t, recorder.DrainOutcomes()) })
			// Release the blocked store before cleanup drains the dispatcher.
			t.Cleanup(release)
			runner := memory.NewPersistenceRunner(time.Second, 1, 1)
			workRelease := make(chan struct{})
			var workOnce sync.Once
			unblockWork := func() { workOnce.Do(func() { close(workRelease) }) }
			t.Cleanup(func() {
				unblockWork()
				assert.NoError(t, runner.RetireAndWait(time.Second))
			})
			router := &OpenAIRouter{
				Config:            &config.RouterConfig{Memory: config.MemoryConfig{AutoStore: true}},
				MemoryExtractor:   memory.NewMemoryChunkStore(&noopMemoryStore{}),
				memoryPersistence: runner, ReplayRecorder: recorder,
			}
			req := persistenceRegressionContext(tc.name)
			req.RouterReplayID = tc.name
			switch tc.name {
			case "no_extractor":
				router.MemoryExtractor = nil
			case "auto_store_off":
				router.Config.Memory.AutoStore = false
			case "policy_blocked":
				req.ResponseJailbreakDetected = true
			case "shutting_down":
				require.NoError(t, runner.RetireAndWait(time.Second))
			case "queue_full":
				started := make(chan struct{}, 2)
				job := memory.PersistenceJob{
					Run: func(context.Context) (memory.PersistenceOutcome, error) {
						started <- struct{}{}
						<-workRelease
						return memory.PersistenceOutcome{}, nil
					},
					Report: func(string, string, bool, error) {},
				}
				runner.Submit(context.Background(), job)
				requireReceiptSignal(t, started)
				runner.Submit(context.Background(), job)
			}
			returned := make(chan struct{})
			go func() {
				router.scheduleResponseMemoryStoreText(req, "response")
				close(returned)
			}()
			requireReceiptSignal(t, backend.entered)
			requireReceiptSignal(t, returned)
			release()
			require.NoError(t, recorder.DrainOutcomes())
			record, found := recorder.GetRecord(tc.name)
			require.True(t, found)
			require.Len(t, record.Outcomes, 1)
			assert.Equal(t, tc.status, record.Outcomes[0].Verdict)
			assert.Equal(t, tc.reason, record.Outcomes[0].Reason)
			assert.Equal(t, map[bool]string{true: "true", false: "false"}[tc.failOpen], record.Outcomes[0].Metadata["fail_open"])
		})
	}
}

func TestMemoryReceipts_SaturationAndShutdownNeverWriteInline(t *testing.T) {
	backend := &slowReceiptStore{Storage: store.NewMemoryStore(10, 0), entered: make(chan struct{}, 1), release: make(chan struct{})}
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(backend.release) }) }
	recorder := routerreplay.NewRecorder(backend)
	_, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: "saturated"})
	require.NoError(t, err)
	t.Cleanup(func() {
		release()
		assert.NoError(t, recorder.DrainOutcomes())
	})
	receipt := memoryPersistenceReceipt{replayID: "saturated", recorder: recorder, decisionKey: "receipt-saturation"}
	dropped := metrics.PluginExecutionTotal.WithLabelValues("memory_persistence_receipt", receipt.decisionKey, "dropped")
	before := testutil.ToFloat64(dropped)
	receipt.record("scheduled", "queue_accepted", false, nil)
	requireReceiptSignal(t, backend.entered)
	returned := make(chan struct{})
	go func() {
		for i := 0; i < routerreplay.DefaultOutcomeQueueCapacity; i++ {
			receipt.record("disabled", "auto_store_off", false, nil)
		}
		for i := 0; i < 100; i++ {
			receipt.record("disabled", "auto_store_off", false, nil)
		}
		close(returned)
	}()
	requireReceiptSignal(t, returned)
	assert.Equal(t, before+100, testutil.ToFloat64(dropped))
	release()
	require.NoError(t, recorder.DrainOutcomes())
	callsAfterDrain := backend.appendCalls.Load()
	receipt.record("rejected", "shutting_down", true, nil)
	assert.Equal(t, callsAfterDrain, backend.appendCalls.Load(),
		"a closed receipt queue must not fall back to a synchronous backend write")
	assert.Equal(t, before+101, testutil.ToFloat64(dropped))
}

func TestMemoryReceipts_SlowScheduledReceiptDoesNotOccupyPersistenceWorker(t *testing.T) {
	backend := &slowReceiptStore{Storage: store.NewMemoryStore(10, 0), entered: make(chan struct{}, 1), release: make(chan struct{})}
	recorder := routerreplay.NewRecorder(backend)
	_, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: "worker-progress"})
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, recorder.DrainOutcomes()) })
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(backend.release) }) }
	t.Cleanup(release)
	receipt := memoryPersistenceReceipt{replayID: "worker-progress", recorder: recorder}
	runner := memory.NewPersistenceRunner(time.Second, 1, 1)
	t.Cleanup(func() { assert.NoError(t, runner.RetireAndWait(time.Second)) })
	ran := make(chan struct{})
	runner.Submit(context.Background(), memory.PersistenceJob{
		Run: func(context.Context) (memory.PersistenceOutcome, error) {
			close(ran)
			return memory.PersistenceOutcome{}, nil
		},
		Report: receipt.record,
	})
	requireReceiptSignal(t, backend.entered)
	requireReceiptSignal(t, ran)
	require.NoError(t, runner.RetireAndWait(time.Second))
	release()
	require.NoError(t, recorder.DrainOutcomes())
	record, found := recorder.GetRecord("worker-progress")
	require.True(t, found)
	require.Len(t, record.Outcomes, 2)
	assert.Equal(t, "scheduled", record.Outcomes[0].Verdict)
	assert.Equal(t, "completed", record.Outcomes[1].Verdict)
}

func TestMemoryReceipts_AcceptedTerminalSurvivesSaturation(t *testing.T) {
	for _, verdict := range []string{"completed", "timeout"} {
		t.Run(verdict, func(t *testing.T) {
			backend := &slowReceiptStore{Storage: store.NewMemoryStore(10, 0), entered: make(chan struct{}, 1), release: make(chan struct{})}
			recorder := routerreplay.NewRecorder(backend)
			_, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: verdict})
			require.NoError(t, err)
			runner := memory.NewPersistenceRunner(100*time.Millisecond, 1, 1)
			workRelease := make(chan struct{})
			var workOnce, replayOnce sync.Once
			releaseWork := func() { workOnce.Do(func() { close(workRelease) }) }
			releaseReplay := func() { replayOnce.Do(func() { close(backend.release) }) }
			t.Cleanup(func() {
				releaseWork()
				releaseReplay()
				assert.NoError(t, runner.RetireAndWait(time.Second))
				assert.NoError(t, recorder.DrainOutcomes())
			})
			started := make(chan struct{})
			runner.Submit(context.Background(), memory.PersistenceJob{
				Run: func(context.Context) (memory.PersistenceOutcome, error) {
					close(started)
					<-workRelease
					return memory.PersistenceOutcome{}, nil
				}, Report: func(string, string, bool, error) {},
			})
			requireReceiptSignal(t, started)
			router := &OpenAIRouter{
				Config:          &config.RouterConfig{Memory: config.MemoryConfig{AutoStore: true}},
				MemoryExtractor: memory.NewMemoryChunkStore(&noopMemoryStore{}), memoryPersistence: runner, ReplayRecorder: recorder,
			}
			ctx := persistenceRegressionContext("reserved-" + verdict)
			ctx.RouterReplayID = verdict
			terminalMetric := metrics.PluginExecutionTotal.WithLabelValues("memory_persistence", "reserved-"+verdict, verdict)
			before := testutil.ToFloat64(terminalMetric)
			router.scheduleResponseMemoryStoreText(ctx, "Deploy a regional cluster and use a load balancer for the service.")
			requireReceiptSignal(t, backend.entered)
			for i := 0; i < routerreplay.DefaultOutcomeQueueCapacity; i++ {
				require.True(t, recorder.TryAppendOutcome(verdict, routerreplay.Outcome{Verdict: "ordinary"}))
			}
			require.False(t, recorder.TryAppendOutcome(verdict, routerreplay.Outcome{}))
			if verdict == "completed" {
				releaseWork()
			}
			require.Eventually(t, func() bool { return testutil.ToFloat64(terminalMetric) == before+1 }, time.Second, time.Millisecond)
			releaseWork()
			require.NoError(t, runner.RetireAndWait(time.Second))
			releaseReplay()
			require.NoError(t, recorder.DrainOutcomes())
			record, found := recorder.GetRecord(verdict)
			require.True(t, found)
			var receipts []string
			for _, outcome := range record.Outcomes {
				if outcome.Metadata["kind"] == "memory_persistence_receipt" {
					receipts = append(receipts, outcome.Verdict)
				}
			}
			assert.Equal(t, []string{"scheduled", verdict}, receipts)
		})
	}
}
