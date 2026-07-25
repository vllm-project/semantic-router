package looper

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ---- helpers ----------------------------------------------------------------

func makeTestState(id string) *workflowPendingToolState {
	return &workflowPendingToolState{
		ID:        id,
		CreatedAt: time.Now().UTC(),
	}
}

func makeOversizedState() *workflowPendingToolState {
	// Build a state whose JSON serialisation exceeds maxStatePayloadBytes.
	big := make([]byte, maxStatePayloadBytes+1)
	for i := range big {
		big[i] = 'x'
	}
	return &workflowPendingToolState{
		ID:           "oversized",
		CreatedAt:    time.Now().UTC(),
		AssistantRaw: big,
	}
}

// backends returns the three backend constructors for table-driven tests.
// Each entry yields a fresh, ready-to-use store and a cleanup function.
func backends(t *testing.T) []struct {
	name  string
	store func() workflowToolStateStore
} {
	t.Helper()
	return []struct {
		name  string
		store func() workflowToolStateStore
	}{
		{
			name: "memory",
			store: func() workflowToolStateStore {
				return newWorkflowMemoryToolStateStore(30 * time.Minute)
			},
		},
		{
			name: "file",
			store: func() workflowToolStateStore {
				dir := filepath.Join(t.TempDir(), "state")
				return newWorkflowFileToolStateStore(dir, 30*time.Minute)
			},
		},
		// Redis is excluded from unit tests because it requires an
		// external server. The Redis backend shares the same interface
		// contract; integration tests cover it via the E2E harness.
	}
}

// ---- Pause / Resume (two-request conformance) -------------------------------

func TestStateStore_PauseResume(t *testing.T) {
	for _, backend := range backends(t) {
		t.Run(backend.name, func(t *testing.T) {
			s := backend.store()
			defer s.Close()
			ctx := context.Background()

			// Request 1: pause — put state.
			state := makeTestState("resume-test")
			id, err := s.Put(ctx, state)
			if err != nil {
				t.Fatalf("Put: %v", err)
			}

			// Request 2: resume — take state by ID from a fresh context.
			got, ok, err := s.Take(ctx, id)
			if err != nil {
				t.Fatalf("Take: %v", err)
			}
			if !ok {
				t.Fatal("Take returned ok=false; expected to find the paused state")
			}
			if got.ID != state.ID {
				t.Fatalf("Take ID = %q, want %q", got.ID, state.ID)
			}

			// Second take must return nothing (already consumed).
			_, ok2, err := s.Take(ctx, id)
			if err != nil {
				t.Fatalf("second Take: %v", err)
			}
			if ok2 {
				t.Fatal("second Take returned ok=true; state should have been consumed")
			}
		})
	}
}

// ---- Concurrent Take (exactly-once) -----------------------------------------

func TestStateStore_ConcurrentTakeExactlyOnce(t *testing.T) {
	for _, backend := range backends(t) {
		t.Run(backend.name, func(t *testing.T) {
			s := backend.store()
			defer s.Close()
			ctx := context.Background()

			state := makeTestState("race-test")
			_, err := s.Put(ctx, state)
			if err != nil {
				t.Fatalf("Put: %v", err)
			}

			const goroutines = 20
			var won int64
			var wg sync.WaitGroup
			wg.Add(goroutines)

			for i := 0; i < goroutines; i++ {
				go func() {
					defer wg.Done()
					_, ok, takeErr := s.Take(ctx, "race-test")
					if takeErr != nil {
						t.Errorf("Take: %v", takeErr)
						return
					}
					if ok {
						atomic.AddInt64(&won, 1)
					}
				}()
			}
			wg.Wait()

			if won != 1 {
				t.Fatalf("concurrent Take: %d goroutines got the state, want exactly 1", won)
			}
		})
	}
}

// ---- Payload size cap -------------------------------------------------------

func TestStateStore_PayloadSizeCap(t *testing.T) {
	for _, backend := range backends(t) {
		t.Run(backend.name, func(t *testing.T) {
			s := backend.store()
			defer s.Close()
			ctx := context.Background()

			_, err := s.Put(ctx, makeOversizedState())
			if err == nil {
				t.Fatal("Put should reject oversized payload")
			}
			if !strings.Contains(err.Error(), "exceeds limit") {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

// ---- Memory cardinality cap -------------------------------------------------

func TestMemoryStateStore_CardinalityCap(t *testing.T) {
	s := newWorkflowMemoryToolStateStore(30 * time.Minute)
	defer s.Close()
	ctx := context.Background()

	// Fill to capacity.
	for i := 0; i < maxMemoryStateEntries; i++ {
		state := &workflowPendingToolState{CreatedAt: time.Now().UTC()}
		if _, err := s.Put(ctx, state); err != nil {
			t.Fatalf("Put[%d]: %v", i, err)
		}
	}

	// One more should fail.
	state := &workflowPendingToolState{CreatedAt: time.Now().UTC()}
	_, err := s.Put(ctx, state)
	if err == nil {
		t.Fatal("Put should reject when at cardinality cap")
	}
	if !strings.Contains(err.Error(), "capacity") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// ---- Memory aggregate byte cap ----------------------------------------------

func TestMemoryStateStore_AggregateByteCap(t *testing.T) {
	s := newWorkflowMemoryToolStateStore(30 * time.Minute)
	defer s.Close()
	ctx := context.Background()

	// Fill to just under capacity.
	// We'll generate a payload that fits easily under the single-item payload limit (maxStatePayloadBytes).
	chunkSize := maxStatePayloadBytes / 4
	large := make([]byte, chunkSize)
	for i := range large {
		large[i] = 'y'
	}

	// Keep putting until we hit the capacity error.
	var hitCap bool
	for i := 0; i < 1000; i++ {
		state := &workflowPendingToolState{ID: fmt.Sprintf("big-%d", i), CreatedAt: time.Now().UTC(), AssistantRaw: large}
		if _, err := s.Put(ctx, state); err != nil {
			if strings.Contains(err.Error(), "max 104857600") { // 100 MiB
				hitCap = true
				break
			}
			t.Fatalf("Put[%d] unexpected error: %v", i, err)
		}
	}
	if !hitCap {
		t.Fatal("Put should reject when at aggregate byte cap")
	}
}

// ---- TTL expiry -------------------------------------------------------------

func TestStateStore_TTLExpiry(t *testing.T) {
	// Use a very short TTL so expiry happens inside the test.
	for _, backend := range []struct {
		name  string
		store func() workflowToolStateStore
	}{
		{
			name: "memory",
			store: func() workflowToolStateStore {
				return newWorkflowMemoryToolStateStore(1 * time.Millisecond)
			},
		},
		{
			name: "file",
			store: func() workflowToolStateStore {
				return newWorkflowFileToolStateStore(
					filepath.Join(t.TempDir(), "ttl"), 1*time.Millisecond,
				)
			},
		},
	} {
		t.Run(backend.name, func(t *testing.T) {
			s := backend.store()
			defer s.Close()
			ctx := context.Background()

			state := makeTestState("ttl-test")
			_, err := s.Put(ctx, state)
			if err != nil {
				t.Fatalf("Put: %v", err)
			}

			// Wait for TTL to pass.
			time.Sleep(10 * time.Millisecond)

			_, ok, err := s.Take(ctx, "ttl-test")
			if err != nil {
				t.Fatalf("Take: %v", err)
			}
			if ok {
				t.Fatal("Take returned expired state; TTL not enforced")
			}
		})
	}
}

// ---- Close idempotency ------------------------------------------------------

func TestStateStore_CloseIdempotent(t *testing.T) {
	for _, backend := range backends(t) {
		t.Run(backend.name, func(t *testing.T) {
			s := backend.store()
			// Calling Close twice must not panic (sync.Once guard).
			if err := s.Close(); err != nil {
				t.Fatalf("first Close: %v", err)
			}
			if err := s.Close(); err != nil {
				t.Fatalf("second Close: %v", err)
			}
		})
	}
}

// ---- Redis Integration ------------------------------------------------------

func setupRedisStore(t *testing.T) (*miniredis.Miniredis, *workflowRedisToolStateStore) {
	t.Helper()
	mr, err := miniredis.Run()
	if err != nil {
		t.Fatalf("miniredis.Run: %v", err)
	}

	cfg := config.WorkflowStateRedisConfig{
		Address:   mr.Addr(),
		KeyPrefix: "test-redis:",
	}
	s := newWorkflowRedisToolStateStore(cfg, 100*time.Millisecond)
	return mr, s
}

func TestWorkflowRedisToolStateStore_PauseResume(t *testing.T) {
	mr, s := setupRedisStore(t)
	defer mr.Close()
	defer s.Close()

	ctx := context.Background()
	state := makeTestState("redis-test")
	id, putErr := s.Put(ctx, state)
	if putErr != nil {
		t.Fatalf("Put: %v", putErr)
	}

	got, ok, takeErr := s.Take(ctx, id)
	if takeErr != nil {
		t.Fatalf("Take: %v", takeErr)
	}
	if !ok || got.ID != state.ID {
		t.Fatalf("Take failed. ok=%v, got=%v", ok, got)
	}
}

func TestWorkflowRedisToolStateStore_TTLExpiry(t *testing.T) {
	mr, s := setupRedisStore(t)
	defer mr.Close()
	defer s.Close()

	ctx := context.Background()
	state2 := makeTestState("redis-ttl")
	_, putErr := s.Put(ctx, state2)
	if putErr != nil {
		t.Fatalf("Put: %v", putErr)
	}
	mr.FastForward(200 * time.Millisecond) // fast-forward miniredis time
	time.Sleep(10 * time.Millisecond)      // wait for local TTL logic just in case

	_, ok, takeErr := s.Take(ctx, "redis-ttl")
	if takeErr != nil {
		t.Fatalf("Take: %v", takeErr)
	}
	if ok {
		t.Fatal("Take returned expired state; TTL not enforced")
	}
}

func TestWorkflowRedisToolStateStore_Clear(t *testing.T) {
	mr, s := setupRedisStore(t)
	defer mr.Close()
	defer s.Close()

	ctx := context.Background()
	state3 := makeTestState("redis-clear1")
	state4 := makeTestState("redis-clear2")
	if _, putErr := s.Put(ctx, state3); putErr != nil {
		t.Fatalf("Put: %v", putErr)
	}
	if _, putErr := s.Put(ctx, state4); putErr != nil {
		t.Fatalf("Put: %v", putErr)
	}

	if clearErr := s.Clear(ctx); clearErr != nil {
		t.Fatalf("Clear: %v", clearErr)
	}

	_, ok, _ := s.Take(ctx, "redis-clear1")
	if ok {
		t.Fatal("Clear failed to remove state")
	}
}
