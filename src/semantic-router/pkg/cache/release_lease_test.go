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

// A leader whose lease expired no longer holds the key: a new leader took it.
// Releasing the old lease must leave the new leader's flight running instead of
// waking its waiters before it stores.
func TestReleasingAnExpiredLeaseLeavesTheNewLeaderAlone(t *testing.T) {
	service := NewResponseCacheService(nil, ResponseCacheServiceOptions{
		SingleflightWait:  250 * time.Millisecond,
		SingleflightLease: 30 * time.Second,
	})
	lookup := ExactLookup{Identity: CacheIdentity{ExactFingerprint: "fp"}}
	_, stale, err := service.LookupExactCoalesced(context.Background(), lookup)
	if err != nil || !stale.Leader {
		t.Fatalf("first lookup: lease=%+v err=%v, want the leader's", stale, err)
	}
	service.mu.Lock()
	service.inflight[stale.Key].expiresAt = time.Now().Add(-time.Second)
	service.mu.Unlock()

	_, next, err := service.LookupExactCoalesced(context.Background(), lookup)
	if err != nil || !next.Leader {
		t.Fatalf("second lookup: lease=%+v err=%v, want the new leader's", next, err)
	}
	service.ReleaseLease(stale)

	service.mu.Lock()
	active := service.inflight[next.Key]
	service.mu.Unlock()
	if active != next.entry {
		t.Fatal("releasing an expired lease ended the new leader's flight")
	}
}
