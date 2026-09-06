package looper

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestResumeWorkflowToolStateRestoreAfterCanceledOrDeadline(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		err  error
	}{
		{name: "canceled", err: context.Canceled},
		{name: "deadline", err: context.DeadlineExceeded},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assertResumeRestoresAfterRequestStop(t, tc.name, tc.err)
		})
	}
}

func assertResumeRestoresAfterRequestStop(t *testing.T, name string, stopErr error) {
	t.Helper()
	redisStore := setupDurableRedisStore(t)
	stateID := "resume-restore-" + strings.ReplaceAll(name, " ", "-")
	reqCtx := newLatchContext()
	wrapped := &restoreProbeStore{workflowToolStateStore: redisStore}
	wrapped.afterTake = func() { reqCtx.stop(stopErr) }

	putWorkflowState(t, redisStore, stateID)
	err := executeWorkflowResumeWithStore(t, wrapped, reqCtx, stateID)
	requireResumeFailedWithoutRestoreError(t, err)
	if reqCtx.Err() == nil {
		t.Fatal("request context should be done before restore")
	}
	requireIndependentRestorePut(t, wrapped)
	requireRedisStatePresent(t, redisStore, stateID, name)
}

func TestResumeWorkflowToolStateRestoreSurfacesPutError(t *testing.T) {
	t.Parallel()
	redisStore := setupDurableRedisStore(t)
	const stateID = "resume-restore-put-error"
	injected := errors.New("injected restore failure")
	reqCtx := newLatchContext()
	wrapped := &restoreProbeStore{
		workflowToolStateStore: redisStore,
		putErr:                 injected,
	}
	wrapped.afterTake = func() { reqCtx.stop(context.Canceled) }

	putWorkflowState(t, redisStore, stateID)
	err := executeWorkflowResumeWithStore(t, wrapped, reqCtx, stateID)
	if err == nil || !strings.Contains(err.Error(), "restore workflow tool state") {
		t.Fatalf("expected restore Put failure, got %v", err)
	}
	if !errors.Is(err, injected) {
		t.Fatalf("resume error missing injected Put cause: %v", err)
	}
	requireIndependentRestorePut(t, wrapped)
	requireRedisStateAbsent(t, redisStore, stateID)
}

type restoreProbeStore struct {
	workflowToolStateStore
	afterTake func()
	putErr    error

	mu             sync.Mutex
	sawPut         bool
	putCtxErr      error
	putHasDeadline bool
}

func (s *restoreProbeStore) Take(ctx context.Context, id string) (*workflowPendingToolState, bool, error) {
	state, ok, err := s.workflowToolStateStore.Take(ctx, id)
	if s.afterTake != nil {
		s.afterTake()
	}
	return state, ok, err
}

func (s *restoreProbeStore) Put(ctx context.Context, state *workflowPendingToolState) (string, error) {
	s.mu.Lock()
	s.sawPut = true
	s.putCtxErr = ctx.Err()
	_, s.putHasDeadline = ctx.Deadline()
	putErr := s.putErr
	s.mu.Unlock()
	if putErr != nil {
		return "", putErr
	}
	return s.workflowToolStateStore.Put(ctx, state)
}

func (s *restoreProbeStore) putCalled() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sawPut
}

func (s *restoreProbeStore) putErrAtCall() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.putCtxErr
}

func (s *restoreProbeStore) putHadDeadline() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.putHasDeadline
}

type latchContext struct {
	done chan struct{}

	mu  sync.Mutex
	err error
}

func newLatchContext() *latchContext {
	return &latchContext{done: make(chan struct{})}
}

func (c *latchContext) Deadline() (time.Time, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if errors.Is(c.err, context.DeadlineExceeded) {
		return time.Now(), true
	}
	return time.Time{}, false
}

func (c *latchContext) Done() <-chan struct{} {
	return c.done
}

func (c *latchContext) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.err
}

func (c *latchContext) Value(any) any {
	return nil
}

func (c *latchContext) stop(err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.err != nil {
		return
	}
	c.err = err
	close(c.done)
}

func putWorkflowState(t *testing.T, store workflowToolStateStore, stateID string) {
	t.Helper()
	if _, err := store.Put(context.Background(), makeTestState(stateID)); err != nil {
		t.Fatalf("Put: %v", err)
	}
}

func requireResumeFailedWithoutRestoreError(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("expected resume to fail after taking state")
	}
	if strings.Contains(err.Error(), "restore workflow tool state") {
		t.Fatalf("restore failure should not be returned when Put succeeds: %v", err)
	}
}

func requireIndependentRestorePut(t *testing.T, store *restoreProbeStore) {
	t.Helper()
	if !store.putCalled() {
		t.Fatal("restore did not Put the taken Redis state")
	}
	if err := store.putErrAtCall(); err != nil {
		t.Fatalf("restore reused a done request context: %v", err)
	}
	if !store.putHadDeadline() {
		t.Fatal("restore Put context is unbounded")
	}
}

func requireRedisStatePresent(t *testing.T, store workflowToolStateStore, stateID, resumeKind string) {
	t.Helper()
	taken, ok, err := store.Take(context.Background(), stateID)
	if err != nil {
		t.Fatalf("Take after restore: %v", err)
	}
	if !ok || taken == nil || taken.ID != stateID {
		t.Fatalf("GETDEL state was lost after %s resume; ok=%v taken=%v", resumeKind, ok, taken)
	}
}

func requireRedisStateAbsent(t *testing.T, store workflowToolStateStore, stateID string) {
	t.Helper()
	_, ok, err := store.Take(context.Background(), stateID)
	if err != nil {
		t.Fatalf("Take after failed restore: %v", err)
	}
	if ok {
		t.Fatal("failed restore should not have rewritten GETDEL state")
	}
}

func executeWorkflowResumeWithStore(t *testing.T, store workflowToolStateStore, ctx context.Context, stateID string) error {
	t.Helper()
	cfg := workflowToolLooperConfig("http://127.0.0.1:1", t.TempDir())
	looper := &WorkflowsLooper{
		BaseLooper: NewBaseLooper(cfg),
		toolStates: store,
		ownsStore:  false,
	}
	toolCallID := workflowToolCallIDPrefix + stateID + workflowToolCallIDSeparator + "call_lookup"
	assistant := map[string]interface{}{
		"role": "assistant",
		"tool_calls": []interface{}{
			map[string]interface{}{
				"id":   toolCallID,
				"type": "function",
				"function": map[string]interface{}{
					"name":      "lookup",
					"arguments": `{"query":"flow"}`,
				},
			},
		},
	}
	req := workflowToolLooperRequest(workflowToolResumeRequest(t, assistant, toolCallID))
	_, err := looper.Execute(ctx, req)
	return err
}

func setupDurableRedisStore(t *testing.T) *workflowRedisToolStateStore {
	t.Helper()
	mr, err := miniredis.Run()
	if err != nil {
		t.Fatalf("miniredis.Run: %v", err)
	}
	store := newWorkflowRedisToolStateStore(config.WorkflowStateRedisConfig{
		Address:   mr.Addr(),
		KeyPrefix: "test-restore:",
	}, time.Hour)
	t.Cleanup(func() {
		_ = store.Close()
		mr.Close()
	})
	return store
}

func TestWorkflowStateRestoreContextIsIndependentAndBounded(t *testing.T) {
	t.Parallel()
	parent, cancel := context.WithCancel(context.Background())
	cancel()
	ctx, stop := workflowStateRestoreContext()
	defer stop()
	if ctx.Err() != nil {
		t.Fatalf("restore context inherited parent cancellation: %v", ctx.Err())
	}
	if parent.Err() == nil {
		t.Fatal("parent context should already be canceled")
	}
	deadline, ok := ctx.Deadline()
	if !ok {
		t.Fatal("restore context missing deadline")
	}
	remaining := time.Until(deadline)
	if remaining <= 0 || remaining > workflowStateRestoreTimeout {
		t.Fatalf("restore deadline remaining %v, want (0, %v]", remaining, workflowStateRestoreTimeout)
	}
}

func TestRestoreWorkflowToolStateSurfacesPutError(t *testing.T) {
	t.Parallel()
	injected := fmt.Errorf("disk full")
	mem := newWorkflowMemoryToolStateStore(time.Hour)
	t.Cleanup(func() { _ = mem.Close() })
	looper := &WorkflowsLooper{
		toolStates: &restoreProbeStore{
			workflowToolStateStore: mem,
			putErr:                 injected,
		},
	}
	restore := true
	err := looper.restoreWorkflowToolState(makeTestState("restore-error"), &restore)
	if !errors.Is(err, injected) {
		t.Fatalf("restore error = %v, want injected cause", err)
	}
}
