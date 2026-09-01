package admission

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestNoopAdmitsImmediately(t *testing.T) {
	ticket, err := Noop{}.Acquire(context.Background())
	if err != nil || ticket == nil {
		t.Fatalf("ticket = %v, err = %v", ticket, err)
	}
	ticket()
}

func TestDoRunsUnderTicketAndPropagatesResult(t *testing.T) {
	gate := NewSemaphore(1, 0, 0, OverflowShed)
	value, err := Do(context.Background(), gate, func() (int, error) { return 42, nil })
	if err != nil || value != 42 {
		t.Fatalf("value = %d, err = %v", value, err)
	}

	wantErr := errors.New("inference failed")
	_, err = Do(context.Background(), gate, func() (int, error) { return 0, wantErr })
	if !errors.Is(err, wantErr) {
		t.Fatalf("err = %v", err)
	}

	if _, err := Do(context.Background(), gate, func() (int, error) { return 0, nil }); err != nil {
		t.Fatalf("slot not released after Do: %v", err)
	}
}

func TestDoReturnsZeroValueWhenShed(t *testing.T) {
	gate := NewSemaphore(1, 0, 0, OverflowShed)
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket()

	value, err := Do(context.Background(), gate, func() (string, error) { return "ran", nil })
	if !errors.Is(err, ErrQueueFull) || value != "" {
		t.Fatalf("value = %q, err = %v", value, err)
	}
}

func TestSemaphoreBoundsConcurrency(t *testing.T) {
	const limit = 4
	gate := NewSemaphore(limit, 64, 0, OverflowShed)
	var inFlight, peak atomic.Int64
	var waitGroup sync.WaitGroup
	for range 32 {
		waitGroup.Add(1)
		go func() {
			defer waitGroup.Done()
			_, err := Do(context.Background(), gate, func() (struct{}, error) {
				current := inFlight.Add(1)
				for {
					observed := peak.Load()
					if current <= observed || peak.CompareAndSwap(observed, current) {
						break
					}
				}
				time.Sleep(time.Millisecond)
				inFlight.Add(-1)
				return struct{}{}, nil
			})
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		}()
	}
	waitGroup.Wait()
	if peak.Load() > limit {
		t.Fatalf("peak concurrency = %d, limit %d", peak.Load(), limit)
	}
}

func TestSemaphoreShedsWhenQueueFull(t *testing.T) {
	gate := NewSemaphore(1, 0, 0, OverflowShed)
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket()

	if _, err := gate.Acquire(context.Background()); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("err = %v, want ErrQueueFull", err)
	}
}

func TestSemaphoreFailOpenAdmitsWithoutSlot(t *testing.T) {
	gate := NewSemaphore(1, 0, 0, OverflowFailOpen)
	first, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	open, err := gate.Acquire(context.Background())
	if err != nil || open == nil {
		t.Fatalf("ticket = %v, err = %v", open, err)
	}
	open()

	first()
	if _, err := gate.Acquire(context.Background()); err != nil {
		t.Fatalf("slot leaked by fail_open ticket: %v", err)
	}
}

func TestSemaphoreWaitBlocksForQueueSlot(t *testing.T) {
	gate := NewSemaphore(1, 0, 0, OverflowWait)
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	acquired := make(chan error, 1)
	go func() {
		waited, err := gate.Acquire(context.Background())
		if err == nil {
			waited()
		}
		acquired <- err
	}()

	select {
	case err := <-acquired:
		t.Fatalf("acquire returned before release: %v", err)
	case <-time.After(20 * time.Millisecond):
	}

	ticket()
	if err := <-acquired; err != nil {
		t.Fatalf("waiter failed after release: %v", err)
	}
}

func TestSemaphoreWaitBoundsQueueOccupancy(t *testing.T) {
	gate := NewSemaphore(1, 1, 0, OverflowWait)
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	const callers = 3
	done := make(chan error, callers)
	for range callers {
		go func() {
			waited, err := gate.Acquire(context.Background())
			if err == nil {
				waited()
			}
			done <- err
		}()
	}

	deadline := time.After(100 * time.Millisecond)
	for {
		if occupancy := len(gate.waiters); occupancy > cap(gate.waiters) {
			t.Fatalf("queue occupancy = %d, cap %d", occupancy, cap(gate.waiters))
		}
		select {
		case <-deadline:
			ticket()
			for range callers {
				if err := <-done; err != nil {
					t.Fatalf("waiter failed after release: %v", err)
				}
			}
			return
		default:
			time.Sleep(time.Millisecond)
		}
	}
}

func TestSemaphoreWaitHonorsContextWhileQueueFull(t *testing.T) {
	gate := NewSemaphore(1, 1, 0, OverflowWait)
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket()

	queued := make(chan struct{})
	go func() {
		close(queued)
		waited, err := gate.Acquire(context.Background())
		if err == nil {
			defer waited()
		}
	}()
	<-queued
	for len(gate.waiters) == 0 {
		time.Sleep(time.Millisecond)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if _, err := gate.Acquire(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want DeadlineExceeded", err)
	}
}

func TestSemaphoreWaitQueueTimeoutWhileQueueFull(t *testing.T) {
	gate := NewSemaphore(1, 1, 10*time.Millisecond, OverflowWait)
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket()

	go func() {
		waited, err := gate.Acquire(context.Background())
		if err == nil {
			defer waited()
		}
	}()
	for len(gate.waiters) == 0 {
		time.Sleep(time.Millisecond)
	}

	if _, err := gate.Acquire(context.Background()); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("err = %v, want ErrQueueFull", err)
	}
}

func TestSemaphoreQueuedWaiterHonorsContext(t *testing.T) {
	gate := NewSemaphore(1, 1, 0, OverflowShed)
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if _, err := gate.Acquire(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want DeadlineExceeded", err)
	}

	ctx2, cancel2 := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel2()
	if _, err := gate.Acquire(ctx2); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want DeadlineExceeded; ErrQueueFull means the queue slot leaked", err)
	}
}

func TestSemaphoreQueueTimeoutSheds(t *testing.T) {
	gate := NewSemaphore(1, 4, 10*time.Millisecond, OverflowShed)
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer ticket()

	start := time.Now()
	if _, err := gate.Acquire(context.Background()); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("err = %v, want ErrQueueFull", err)
	}
	if time.Since(start) > time.Second {
		t.Fatal("queue timeout did not bound the wait")
	}
}

func TestTicketReleaseIsIdempotent(t *testing.T) {
	gate := NewSemaphore(1, 0, 0, OverflowShed)
	ticket, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	ticket()
	ticket()

	first, err := gate.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer first()
	if _, err := gate.Acquire(context.Background()); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("double release added a slot: %v", err)
	}
}
