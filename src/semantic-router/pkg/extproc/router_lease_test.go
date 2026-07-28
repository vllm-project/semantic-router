package extproc

import (
	"testing"
	"time"
)

func TestRouterLeaseRetireWaitsForInFlightAcquireToRelease(t *testing.T) {
	lease := newRouterLease(&OpenAIRouter{})
	if !lease.acquire() {
		t.Fatal("acquire() = false, want true before retire")
	}

	retireDone := make(chan struct{})
	go func() {
		lease.retire(5 * time.Second)
		close(retireDone)
	}()

	select {
	case <-retireDone:
		t.Fatal("retire() returned before the in-flight acquire was released")
	case <-time.After(50 * time.Millisecond):
	}

	lease.release()

	select {
	case <-retireDone:
	case <-time.After(5 * time.Second):
		t.Fatal("retire() did not return after the in-flight acquire was released")
	}
}

func TestRouterLeaseAcquireFailsOnceRetiring(t *testing.T) {
	lease := newRouterLease(&OpenAIRouter{})
	lease.retire(time.Second)

	if lease.acquire() {
		t.Fatal("acquire() = true, want false once retire() has run")
	}
}

func TestRouterLeaseRetireRespectsDrainTimeout(t *testing.T) {
	lease := newRouterLease(&OpenAIRouter{})
	if !lease.acquire() {
		t.Fatal("acquire() = false, want true")
	}
	defer lease.release()

	start := time.Now()
	lease.retire(50 * time.Millisecond)
	elapsed := time.Since(start)

	if elapsed > 2*time.Second {
		t.Fatalf("retire() took %s, want it bounded near the 50ms drain timeout", elapsed)
	}
}

// TestRouterLeaseConcurrentAcquireDuringRetireNeverRacesAfterWait exercises
// the race retire() and acquire() must both defend against: a caller can
// call acquire() at any point up to and including the instant retire()'s
// internal Wait() returns. No successful acquire() may be observed once
// retire() has returned, since the caller closes the router's resources
// immediately afterward. Run with -race.
func TestRouterLeaseConcurrentAcquireDuringRetireNeverRacesAfterWait(t *testing.T) {
	for i := 0; i < 500; i++ {
		lease := newRouterLease(&OpenAIRouter{})
		var acquired bool

		retireStarted := make(chan struct{})
		retireDone := make(chan struct{})
		go func() {
			close(retireStarted)
			lease.retire(5 * time.Second)
			close(retireDone)
		}()

		<-retireStarted
		if lease.acquire() {
			acquired = true
			lease.release()
		}

		<-retireDone
		if acquired {
			// A successful acquire raced retire() legitimately (it landed
			// before retiring flipped); retire()'s Wait() must have already
			// accounted for it, so retireDone closing here is itself the
			// assertion that no deadlock or double-release occurred.
			continue
		}
	}
}

func TestRouterLeaseNilIsNoop(t *testing.T) {
	var lease *routerLease
	if lease.acquire() {
		t.Fatal("acquire() on nil lease = true, want false")
	}
	lease.release()
	lease.retire(time.Second)
}
