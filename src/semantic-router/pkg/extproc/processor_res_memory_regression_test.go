package extproc

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

type persistenceRegressionStore struct {
	*memory.InMemoryStore
	failSession bool
}

func (s *persistenceRegressionStore) Store(ctx context.Context, mem *memory.Memory) error {
	if s.failSession && mem.Source == "session_window" {
		return errors.New("session write failed")
	}
	// Persistence tests do not need model inference.
	mem.Embedding = []float32{1}
	return s.InMemoryStore.Store(ctx, mem)
}

func persistenceRegressionContext(decision string) *RequestContext {
	return &RequestContext{
		RequestID:               "original-request",
		SessionID:               "original-session",
		Headers:                 map[string]string{headers.AuthzUserID: "original-user"},
		TraceContext:            context.Background(),
		VSRSelectedDecisionName: decision,
		SemanticRequest: &llmprotocol.Request{Messages: []llmprotocol.Message{
			neutralMemoryMessage("user", "My preferred deployment region is eu-central-1."),
			neutralMemoryMessage("assistant", "I will remember your preferred deployment region."),
			neutralMemoryMessage("user", "Explain how to deploy a backend service in that region."),
		}},
	}
}

func TestScheduleResponseMemoryStore_QueuedJobUsesSubmissionSnapshot(t *testing.T) {
	backend := &persistenceRegressionStore{InMemoryStore: memory.NewInMemoryStore()}
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	_, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: "original-replay"})
	require.NoError(t, err)
	replacement := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	_, err = replacement.AddRecord(routerreplay.RoutingRecord{ID: "changed-replay"})
	require.NoError(t, err)

	runner, unblock := blockedPersistenceRunner(t)
	router := &OpenAIRouter{
		Config:            &config.RouterConfig{Memory: config.MemoryConfig{AutoStore: true}},
		MemoryExtractor:   memory.NewMemoryChunkStore(backend),
		ReplayRecorder:    recorder,
		memoryPersistence: runner,
	}
	reqCtx := persistenceRegressionContext("snapshot-original")
	reqCtx.RouterReplayID = "original-replay"
	reqCtx.RouterReplayRecorder = recorder
	originalMetric := metrics.PluginExecutionTotal.WithLabelValues("memory_persistence", requestDecisionStateKey(reqCtx), "completed")
	before := testutil.ToFloat64(originalMetric)
	router.scheduleResponseMemoryStoreText(reqCtx, "Deploy the service using a regional cluster and a load balancer.")

	mutatePersistenceRequest(reqCtx, replacement)
	unblock()
	require.NoError(t, runner.RetireAndWait(5*time.Second))

	stored, err := backend.List(context.Background(), memory.ListOptions{UserID: "original-user", Limit: 10})
	require.NoError(t, err)
	require.Len(t, stored.Memories, 2)
	var sessionContent string
	for _, mem := range stored.Memories {
		assert.NotContains(t, mem.Content, "changed conversation")
		assert.Contains(t, mem.Content, "Explain how to deploy")
		if mem.Source == "session_window" {
			sessionContent = mem.Content
		}
	}
	assert.Contains(t, sessionContent, "eu-central-1")
	assert.Equal(t, before+1, testutil.ToFloat64(originalMetric))
	record, found := recorder.GetRecord("original-replay")
	require.True(t, found)
	require.Len(t, record.Outcomes, 2)
	assert.Equal(t, "scheduled", record.Outcomes[0].Verdict)
	assert.Equal(t, "completed", record.Outcomes[1].Verdict)
	changed, found := replacement.GetRecord("changed-replay")
	require.True(t, found)
	assert.Empty(t, changed.Outcomes)
}

func TestScheduleResponseMemoryStore_PartialWriteReportsFailure(t *testing.T) {
	backend := &persistenceRegressionStore{InMemoryStore: memory.NewInMemoryStore(), failSession: true}
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	_, err := recorder.AddRecord(routerreplay.RoutingRecord{ID: "partial-write"})
	require.NoError(t, err)
	runner := memory.NewPersistenceRunner(10*time.Second, 1, 1)
	t.Cleanup(func() { assert.NoError(t, runner.RetireAndWait(5*time.Second)) })
	router := &OpenAIRouter{
		Config:            &config.RouterConfig{Memory: config.MemoryConfig{AutoStore: true}},
		MemoryExtractor:   memory.NewMemoryChunkStore(backend),
		ReplayRecorder:    recorder,
		memoryPersistence: runner,
	}
	reqCtx := persistenceRegressionContext("partial-write")
	reqCtx.RouterReplayID = "partial-write"
	failed := metrics.PluginExecutionTotal.WithLabelValues("memory_persistence", requestDecisionStateKey(reqCtx), "store_failed")
	completed := metrics.PluginExecutionTotal.WithLabelValues("memory_persistence", requestDecisionStateKey(reqCtx), "completed")
	beforeFailed, beforeCompleted := testutil.ToFloat64(failed), testutil.ToFloat64(completed)

	router.scheduleResponseMemoryStoreText(reqCtx, "Deploy the service using a regional cluster and a load balancer.")
	require.NoError(t, runner.RetireAndWait(5*time.Second))

	stored, err := backend.List(context.Background(), memory.ListOptions{UserID: "original-user", Limit: 10})
	require.NoError(t, err)
	require.Len(t, stored.Memories, 1)
	assert.Equal(t, "conversation", stored.Memories[0].Source)
	assert.Equal(t, beforeFailed+1, testutil.ToFloat64(failed))
	assert.Equal(t, beforeCompleted, testutil.ToFloat64(completed))
	record, found := recorder.GetRecord("partial-write")
	require.True(t, found)
	require.Len(t, record.Outcomes, 2)
	assert.Equal(t, "scheduled", record.Outcomes[0].Verdict)
	assert.Equal(t, "store_failed", record.Outcomes[1].Verdict)
	assert.Equal(t, "persist_error", record.Outcomes[1].Reason)
	assert.Equal(t, "terminal", record.Outcomes[1].Metadata["phase"])
	assert.Equal(t, "true", record.Outcomes[1].Metadata["fail_open"])
}

func blockedPersistenceRunner(t *testing.T) (*memory.PersistenceRunner, func()) {
	t.Helper()
	runner := memory.NewPersistenceRunner(10*time.Second, 1, 2)
	started, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(func() {
		unblock()
		assert.NoError(t, runner.RetireAndWait(5*time.Second))
	})
	// Occupy the only worker so the request can change before persistence starts.
	runner.Submit(context.Background(), memory.PersistenceJob{
		Run: func(ctx context.Context) (memory.PersistenceOutcome, error) {
			close(started)
			select {
			case <-release:
				return memory.PersistenceOutcome{}, nil
			case <-ctx.Done():
				return memory.PersistenceOutcome{}, ctx.Err()
			}
		},
		Report: func(string, string, bool, error) {},
	})
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("blocking job did not start")
	}
	return runner, unblock
}

func mutatePersistenceRequest(reqCtx *RequestContext, replacement *routerreplay.Recorder) {
	reqCtx.Headers[headers.AuthzUserID] = "changed-user"
	reqCtx.SessionID = "changed-session"
	reqCtx.RequestID = "changed-request"
	reqCtx.VSRSelectedDecisionName = "snapshot-changed"
	reqCtx.RouterReplayID = "changed-replay"
	reqCtx.RouterReplayRecorder = replacement
	for i := range reqCtx.SemanticRequest.Messages {
		reqCtx.SemanticRequest.Messages[i].Content[0].Text = "changed conversation content"
	}
}
