package agentruntime

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

type contendedClaimStore struct {
	agentmanagement.Store
	calls atomic.Int32
}

func (store *contendedClaimStore) ClaimNextTurn(
	context.Context, string, time.Time,
) (agentmanagement.TurnLease, error) {
	if store.calls.Add(1) == 1 {
		return agentmanagement.TurnLease{}, agentmanagement.ErrConflict
	}
	return agentmanagement.TurnLease{}, agentmanagement.ErrNotFound
}

func TestRunClaimLoopKeepsPollingAfterQueueContention(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	store := &contendedClaimStore{}
	worker := &Worker{
		store: store, pollInterval: time.Millisecond, leaseDuration: time.Minute, now: time.Now,
	}
	done := make(chan error, 1)
	go func() { done <- worker.runClaimLoop(ctx, "worker/1") }()

	deadline := time.NewTimer(time.Second)
	defer deadline.Stop()
	poll := time.NewTicker(time.Millisecond)
	defer poll.Stop()
	for store.calls.Load() < 2 {
		select {
		case <-deadline.C:
			t.Fatal("worker stopped polling after queue contention")
		case <-poll.C:
		}
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("runClaimLoop() error = %v, want context cancellation", err)
	}
}
