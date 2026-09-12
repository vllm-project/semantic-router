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
	redisStore, _ := setupDurableRedisStore(t)
	stateID := "resume-restore-" + strings.ReplaceAll(name, " ", "-")
	reqCtx := newLatchContext()
	wrapped := &restoreProbeStore{workflowToolStateStore: redisStore}
	wrapped.afterClaim = func() { reqCtx.stop(stopErr) }

	putWorkflowState(t, redisStore, stateID)
	err := executeWorkflowResumeWithStore(t, wrapped, reqCtx, stateID)
	requireResumeFailedWithoutRestoreError(t, err)
	if reqCtx.Err() == nil {
		t.Fatal("request context should be done before restore")
	}
	requireIndependentRestoreRelease(t, wrapped)
	requireRedisStatePresent(t, redisStore, stateID, name)
}

func TestResumeWorkflowToolStateReleaseFailureKeepsClaimUntilLeaseExpiry(t *testing.T) {
	setWorkflowStateClaimLease(t, 80*time.Millisecond)
	redisStore, mr := setupDurableRedisStore(t)
	const stateID = "resume-restore-release-error"
	injected := errors.New("injected release failure")
	reqCtx := newLatchContext()
	wrapped := &restoreProbeStore{
		workflowToolStateStore: redisStore,
		releaseErr:             injected,
	}
	wrapped.afterClaim = func() { reqCtx.stop(context.Canceled) }

	putWorkflowState(t, redisStore, stateID)
	err := executeWorkflowResumeWithStore(t, wrapped, reqCtx, stateID)
	if err == nil || !strings.Contains(err.Error(), "release workflow tool state") {
		t.Fatalf("expected release failure, got %v", err)
	}
	if !errors.Is(err, injected) {
		t.Fatalf("resume error missing injected release cause: %v", err)
	}
	requireIndependentRestoreRelease(t, wrapped)

	claim, ok, claimErr := redisStore.Claim(context.Background(), config.DefaultRecipeName, stateID)
	if claimErr != nil {
		t.Fatalf("Claim while lease held: %v", claimErr)
	}
	if ok || claim != nil {
		t.Fatal("failed release should leave the durable claim held")
	}

	mr.FastForward(120 * time.Millisecond)
	requireRedisStatePresent(t, redisStore, stateID, "lease-expiry")
}

type restoreProbeStore struct {
	workflowToolStateStore
	afterClaim func()
	releaseErr error

	mu                 sync.Mutex
	sawRelease         bool
	releaseCtxErr      error
	releaseHasDeadline bool
}

func (s *restoreProbeStore) Claim(ctx context.Context, recipe config.RecipeName, id string) (*workflowStateClaim, bool, error) {
	claim, ok, err := s.workflowToolStateStore.Claim(ctx, recipe, id)
	if s.afterClaim != nil {
		s.afterClaim()
	}
	return claim, ok, err
}

func (s *restoreProbeStore) Release(ctx context.Context, recipe config.RecipeName, id, token string) error {
	s.mu.Lock()
	s.sawRelease = true
	s.releaseCtxErr = ctx.Err()
	_, s.releaseHasDeadline = ctx.Deadline()
	releaseErr := s.releaseErr
	s.mu.Unlock()
	if releaseErr != nil {
		return releaseErr
	}
	return s.workflowToolStateStore.Release(ctx, recipe, id, token)
}

func (s *restoreProbeStore) releaseCalled() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sawRelease
}

func (s *restoreProbeStore) releaseErrAtCall() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.releaseCtxErr
}

func (s *restoreProbeStore) releaseHadDeadline() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.releaseHasDeadline
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
		t.Fatal("expected resume to fail after claiming state")
	}
	if strings.Contains(err.Error(), "release workflow tool state") {
		t.Fatalf("release failure should not be returned when Release succeeds: %v", err)
	}
}

func requireIndependentRestoreRelease(t *testing.T, store *restoreProbeStore) {
	t.Helper()
	if !store.releaseCalled() {
		t.Fatal("interrupted resume did not Release the claimed state")
	}
	if err := store.releaseErrAtCall(); err != nil {
		t.Fatalf("release reused a done request context: %v", err)
	}
	if !store.releaseHadDeadline() {
		t.Fatal("release context is unbounded")
	}
}

func requireRedisStatePresent(t *testing.T, store workflowToolStateStore, stateID, resumeKind string) {
	t.Helper()
	claim, ok, err := store.Claim(context.Background(), config.DefaultRecipeName, stateID)
	if err != nil {
		t.Fatalf("Claim after restore: %v", err)
	}
	if !ok || claim == nil || claim.State == nil || claim.State.ID != stateID {
		t.Fatalf("claimed state was lost after %s resume; ok=%v claim=%v", resumeKind, ok, claim)
	}
	if err := store.Release(context.Background(), config.DefaultRecipeName, stateID, claim.Token); err != nil {
		t.Fatalf("Release after presence check: %v", err)
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

func setupDurableRedisStore(t *testing.T) (*workflowRedisToolStateStore, *miniredis.Miniredis) {
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
	return store, mr
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

func TestReleaseWorkflowToolStateSurfacesError(t *testing.T) {
	t.Parallel()
	injected := fmt.Errorf("disk full")
	mem := newWorkflowMemoryToolStateStore(time.Hour)
	t.Cleanup(func() { _ = mem.Close() })
	looper := &WorkflowsLooper{
		toolStates: &restoreProbeStore{
			workflowToolStateStore: mem,
			releaseErr:             injected,
		},
	}
	restore := true
	err := looper.releaseWorkflowToolState(&workflowStateClaim{
		Recipe: config.DefaultRecipeName,
		ID:     "restore-error",
		Token:  "token",
	}, &restore)
	if !errors.Is(err, injected) {
		t.Fatalf("release error = %v, want injected cause", err)
	}
}
