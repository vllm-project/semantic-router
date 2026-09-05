package responsestore

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestConversationIndexScanLeaseSerializesConcurrentReadersSameConversation
// is the direct regression test for the global lease's purpose: many
// concurrent readers missing the same conversation's index must cause
// exactly one legacy scan, even when that scan runs long enough that the
// superseded per-conversation lock's fixed backoff-then-scan-anyway
// fallback would have let every one of them scan independently.
func TestConversationIndexScanLeaseSerializesConcurrentReadersSameConversation(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	var scanning atomic.Bool
	store.client.AddHook(&beforeCommandHook{name: "scan", before: func() {
		if !scanning.CompareAndSwap(false, true) {
			t.Error("a second scan started while one was still in flight")
		}
		time.Sleep(600 * time.Millisecond)
		scanning.Store(false)
	}})

	const readers = 20
	var wg sync.WaitGroup
	errs := make([]error, readers)
	for i := 0; i < readers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := store.ListResponsesByConversation(ctx, "conv_lease_same", ListOptions{})
			errs[i] = err
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		assert.NoErrorf(t, err, "reader %d", i)
	}
	assert.Equal(t, int64(1), store.scanInvocations.Load(),
		"the global lease must serialize every reader onto exactly one scan")
}

// TestConversationIndexScanLeaseSerializesConcurrentNovelConversations
// covers the same guarantee across *different* conversation IDs: the lease
// is global, not per-conversation, so concurrent lookups for entirely
// distinct, never-seen IDs must still never run more than one scan at a
// time system-wide (they run sequentially, one scan per ID, rather than
// each independently in parallel).
func TestConversationIndexScanLeaseSerializesConcurrentNovelConversations(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	var inFlight atomic.Int32
	var maxInFlight atomic.Int32
	store.client.AddHook(&beforeCommandHook{name: "scan", before: func() {
		current := inFlight.Add(1)
		for {
			observed := maxInFlight.Load()
			if current <= observed || maxInFlight.CompareAndSwap(observed, current) {
				break
			}
		}
		time.Sleep(50 * time.Millisecond)
		inFlight.Add(-1)
	}})

	const readers = 20
	var wg sync.WaitGroup
	errs := make([]error, readers)
	for i := 0; i < readers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := store.ListResponsesByConversation(ctx, fmt.Sprintf("conv_lease_novel_%d", i), ListOptions{})
			errs[i] = err
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		assert.NoErrorf(t, err, "reader %d", i)
	}
	assert.Equal(t, int64(readers), store.scanInvocations.Load(), "every distinct novel conversation must still get its own scan eventually")
	assert.EqualValues(t, 1, maxInFlight.Load(), "no two scans may run concurrently, even for different conversation IDs")
}

// TestConversationIndexScanLeaseRenewalExtendsBeyondOriginalTTL covers the
// renewal primitive directly: a lease renewed before its original TTL
// elapses survives past that original deadline with the full TTL restored,
// so a scan that runs longer than one lease TTL is not taken over by
// another waiter partway through, as long as it keeps renewing.
func TestConversationIndexScanLeaseRenewalExtendsBeyondOriginalTTL(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	token, err := randomScanLeaseToken()
	require.NoError(t, err)
	acquired, err := store.acquireConversationIndexScanLease(ctx, token)
	require.NoError(t, err)
	require.True(t, acquired)

	leaseKey := store.conversationIndexScanLeaseKey()
	// Simulate a lease close to expiring, as a long scan nearing its
	// initial TTL would be, without waiting out the real 30s constant.
	require.NoError(t, store.client.PExpire(ctx, leaseKey, 200*time.Millisecond).Err())

	renewed, err := store.renewConversationIndexScanLease(ctx, token)
	require.NoError(t, err)
	require.True(t, renewed)

	ttl, err := store.client.TTL(ctx, leaseKey).Result()
	require.NoError(t, err)
	assert.Greater(t, ttl, 5*time.Second, "renewal must restore the full lease TTL, not merely extend the short forced window")

	// The original (pre-renewal) short window elapses; the lease must
	// still belong to token, not have been taken over.
	time.Sleep(300 * time.Millisecond)
	otherToken, err := randomScanLeaseToken()
	require.NoError(t, err)
	stolen, err := store.acquireConversationIndexScanLease(ctx, otherToken)
	require.NoError(t, err)
	assert.False(t, stolen, "a renewed lease must survive past its pre-renewal TTL window")

	require.NoError(t, store.releaseConversationIndexScanLease(ctx, token))
}

// TestConversationIndexScanLeaseRenewalDetectsLoss covers the other half of
// renewal: once a lease's key no longer holds the caller's token (deleted,
// expired-and-reacquired, or overwritten by another holder), renewal must
// report loss (ok=false, not an error) rather than silently succeeding —
// this is exactly the signal withConversationIndexScanLease's background
// renewer relies on to cancel an in-flight scan.
func TestConversationIndexScanLeaseRenewalDetectsLoss(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	token, err := randomScanLeaseToken()
	require.NoError(t, err)
	acquired, err := store.acquireConversationIndexScanLease(ctx, token)
	require.NoError(t, err)
	require.True(t, acquired)

	// Simulate the lease having been deleted (expired) and, in general,
	// simply no longer present or matching this token — an external actor,
	// or the lease's own TTL, made this happen; either way the caller no
	// longer owns it.
	require.NoError(t, store.client.Del(ctx, store.conversationIndexScanLeaseKey()).Err())

	renewed, err := store.renewConversationIndexScanLease(ctx, token)
	require.NoError(t, err)
	assert.False(t, renewed, "renewal must report loss, not silently succeed, once the lease no longer matches this token")
}

// TestConversationIndexScanLeaseWaiterRespectsCancellation covers a waiter
// blocked on an externally-held lease: cancelling its context must return
// promptly (within the test's short deadline), never sit through the full
// backoff schedule or wait for the external holder's lease to expire.
func TestConversationIndexScanLeaseWaiterRespectsCancellation(t *testing.T) {
	store := newConversationIndexStore(t)

	externalToken, err := randomScanLeaseToken()
	require.NoError(t, err)
	// A long-lived external holder the waiter below can never outlast.
	require.NoError(t, store.client.Set(context.Background(), store.conversationIndexScanLeaseKey(), externalToken, time.Minute).Err())

	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()

	start := time.Now()
	err = store.withConversationIndexScanLease(ctx, func(context.Context) error {
		t.Fatal("fn must never run: the lease is held externally for the whole test")
		return nil
	})
	elapsed := time.Since(start)

	require.Error(t, err)
	assert.ErrorIs(t, err, context.DeadlineExceeded)
	assert.Lessf(t, elapsed, 2*time.Second, "a cancelled waiter must return promptly, not block for a much longer default")
}

func TestConversationIndexScanLeaseWaiterReturnsWhenProofAppears(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	require.NoError(t, store.client.Set(ctx, store.conversationIndexScanLeaseKey(), "external-holder", time.Minute).Err())
	proofWritten := make(chan struct{})
	go func() {
		time.Sleep(100 * time.Millisecond)
		_ = store.markConversationMigrated(ctx, "conv_waiter_resolved", conversationIndexProofEmpty)
		close(proofWritten)
	}()

	start := time.Now()
	err := store.ensureConversationIndex(ctx, "conv_waiter_resolved")
	require.NoError(t, err)
	<-proofWritten
	assert.Less(t, time.Since(start), 2*time.Second)
	assert.Zero(t, store.scanInvocations.Load())
	assert.EqualValues(t, 1, exists(t, store, store.conversationIndexScanLeaseKey()),
		"the waiter must return from the proof check without acquiring the still-held lease")
}

// TestConversationIndexScanLeaseSequentialWaitersEachGetATurn covers "holder
// loss allows exactly one waiter to acquire": an externally-held lease with
// a short natural TTL frees up on its own, and multiple waiters queued
// behind it are each admitted in turn — never two at once — as it keeps
// freeing and being retaken.
func TestConversationIndexScanLeaseSequentialWaitersEachGetATurn(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	externalToken, err := randomScanLeaseToken()
	require.NoError(t, err)
	require.NoError(t, store.client.Set(ctx, store.conversationIndexScanLeaseKey(), externalToken, 200*time.Millisecond).Err())

	var active atomic.Int32
	var maxActive atomic.Int32
	const waiters = 4
	var wg sync.WaitGroup
	for i := 0; i < waiters; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := store.withConversationIndexScanLease(ctx, func(context.Context) error {
				current := active.Add(1)
				for {
					observed := maxActive.Load()
					if current <= observed || maxActive.CompareAndSwap(observed, current) {
						break
					}
				}
				time.Sleep(80 * time.Millisecond)
				active.Add(-1)
				return nil
			})
			assert.NoError(t, err)
		}()
	}
	wg.Wait()

	assert.EqualValues(t, 1, maxActive.Load(), "waiters must be admitted one at a time, never concurrently")
}
