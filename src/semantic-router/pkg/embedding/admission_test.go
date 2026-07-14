package embedding

import (
	"context"
	"errors"
	"sync"
	"testing"
)

func TestProcessAdmissionCapacityCancellationAndRelease(t *testing.T) {
	admission := NewProcessAdmission(2)
	releaseFirst, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("acquire first slot: %v", err)
	}
	releaseSecond, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("acquire second slot: %v", err)
	}
	if _, overloadErr := admission.TryAcquire(context.Background()); !errors.Is(overloadErr, ErrOverloaded) {
		t.Fatalf("third acquire error = %v, want overload", overloadErr)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, cancellationErr := admission.TryAcquire(canceled); !errors.Is(cancellationErr, context.Canceled) {
		t.Fatalf("canceled acquire error = %v, want context cancellation", cancellationErr)
	}

	releaseFirst()
	releaseFirst()
	reacquired, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("reacquire released slot: %v", err)
	}
	reacquired()
	releaseSecond()
}

func TestProcessAdmissionConcurrentReleaseIsIdempotent(t *testing.T) {
	admission := NewProcessAdmission(1)
	release, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("acquire slot: %v", err)
	}

	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			release()
		}()
	}
	wg.Wait()

	reacquired, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("slot leaked after concurrent release: %v", err)
	}
	reacquired()
}
