package extproc

import (
	"errors"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
)

func TestResourceScopeClosesInReverseOrderOnceAndJoinsErrors(t *testing.T) {
	errA := errors.New("a")
	errB := errors.New("b")
	var order []int
	scope := newResourceScope()
	scope.add(func() error {
		order = append(order, 1)
		return errA
	})
	scope.add(func() error {
		order = append(order, 2)
		return errB
	})

	err := scope.close()
	if !errors.Is(err, errA) || !errors.Is(err, errB) {
		t.Fatalf("close() error = %v, want both errors", err)
	}
	if !slices.Equal(order, []int{2, 1}) {
		t.Fatalf("close order = %v, want [2 1]", order)
	}
	if err := scope.close(); err != nil {
		t.Fatalf("second close() error = %v", err)
	}
	if !slices.Equal(order, []int{2, 1}) {
		t.Fatalf("second close changed order to %v", order)
	}
}

func TestResourceScopeConcurrentCloseIsIdempotent(t *testing.T) {
	var calls atomic.Int32
	scope := newResourceScope()
	scope.add(func() error {
		calls.Add(1)
		return nil
	})

	var wg sync.WaitGroup
	wg.Add(8)
	for i := 0; i < 8; i++ {
		go func() {
			defer wg.Done()
			_ = scope.close()
		}()
	}
	wg.Wait()
	if got := calls.Load(); got != 1 {
		t.Fatalf("closer calls = %d, want 1", got)
	}
}
