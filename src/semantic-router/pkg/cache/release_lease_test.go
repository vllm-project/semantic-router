package cache

import (
	"context"
	"testing"
	"time"
)

// A leader whose request will not store releases its lease, so the next
// lookup leads immediately instead of waiting out the 30 s lease.
func TestReleaseLeaseLetsTheNextRequestLead(t *testing.T) {
	service := NewResponseCacheService(nil, ResponseCacheServiceOptions{
		SingleflightWait: 250 * time.Millisecond,
	})
	lookup := ExactLookup{Identity: CacheIdentity{ExactFingerprint: "fp"}}
	_, firstLease, err := service.LookupExactCoalesced(context.Background(), lookup)
	if err != nil || !firstLease.Leader {
		t.Fatalf("first lookup: lease=%+v err=%v, want the leader's", firstLease, err)
	}

	released := make(chan struct{})
	go func() {
		service.ReleaseLease(firstLease)
		close(released)
	}()
	select {
	case <-released:
	case <-time.After(time.Second):
		t.Fatal("ReleaseLease did not return")
	}

	done := make(chan SingleflightLease, 1)
	go func() {
		_, lease, _ := service.LookupExactCoalesced(context.Background(), lookup)
		done <- lease
	}()
	select {
	case lease := <-done:
		if !lease.Leader {
			t.Fatal("the next lookup still follows a released lease")
		}
	case <-time.After(time.Second):
		t.Fatal("the next lookup waited on the released lease")
	}
}
