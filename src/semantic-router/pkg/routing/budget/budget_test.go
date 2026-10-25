package budget

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
)

func TestConcurrentConsumersCannotExceedRequestBudget(t *testing.T) {
	ctx, ledger := WithLimit(context.Background(), 7)
	var successes atomic.Int64
	var group sync.WaitGroup
	for range 100 {
		group.Go(func() {
			if err := Consume(ctx); err == nil {
				successes.Add(1)
			} else if !errors.Is(err, ErrExhausted) {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
	group.Wait()
	if successes.Load() != 7 || ledger.Used() != 7 || ledger.Remaining() != 0 {
		t.Fatalf("successes=%d used=%d remaining=%d", successes.Load(), ledger.Used(), ledger.Remaining())
	}
}

func TestNestedAndCanceledContextsCannotReplenishBudget(t *testing.T) {
	ctx, ledger := WithLimit(context.Background(), 1)
	child, other := WithLimit(ctx, 100)
	if other != ledger {
		t.Fatal("nested request acquired a new budget")
	}
	if err := Consume(child); err != nil {
		t.Fatal(err)
	}
	if err := Consume(ctx); !errors.Is(err, ErrExhausted) {
		t.Fatalf("expected exhaustion, got %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := Consume(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected cancellation, got %v", err)
	}
}
