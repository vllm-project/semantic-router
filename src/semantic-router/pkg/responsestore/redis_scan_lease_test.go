package responsestore

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
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

// renewalGateHook holds the first lease-renewal round-trip open, so a test
// can place work's completion *inside* the window where a renewal is
// genuinely in flight. It matches the renewal by conditionalRefreshScript's
// SHA, which is what distinguishes it from the compare-delete script the
// release path runs against the very same lease key.
//
// It releases the round-trip when the command's own context is cancelled —
// which is precisely the wrapper's normal shutdown after fn returns — so the
// gated renewal fails the way the real one does, rather than by a contrived
// error the production path never produces.
type renewalGateHook struct {
	hash     string
	inFlight chan struct{}
	used     atomic.Bool
}

func newRenewalGateHook() *renewalGateHook {
	return &renewalGateHook{
		hash:     conditionalRefreshScript.Hash(),
		inFlight: make(chan struct{}),
	}
}

func (h *renewalGateHook) DialHook(next redis.DialHook) redis.DialHook { return next }
func (h *renewalGateHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}

func (h *renewalGateHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		if matchesScriptSHA(cmd, h.hash) && h.used.CompareAndSwap(false, true) {
			close(h.inFlight)
			select {
			case <-ctx.Done():
			case <-time.After(5 * time.Second):
				// Never wedge the suite if the wrapper stops cancelling.
			}
		}
		return next(ctx, cmd)
	}
}

// matchesScriptSHA identifies one specific Lua script's EVALSHA. The scan
// lease's renewal and its release run different scripts against the very same
// key, so the SHA — not the key — is what tells them apart.
func matchesScriptSHA(cmd redis.Cmder, hash string) bool {
	args := cmd.Args()
	if cmd.Name() != "evalsha" || len(args) < 2 {
		return false
	}
	sha, ok := args[1].(string)
	return ok && sha == hash
}

// warmScanLeaseScripts runs one real renewal so the Lua script is cached
// server-side. Without it the first renewal under test answers NOSCRIPT and
// re-runs the work as a follow-up EVAL, which renewalGateHook's one-shot
// EVALSHA match would have already spent on the failed attempt.
func warmScanLeaseScripts(t *testing.T, store *RedisStore, token string) {
	t.Helper()

	ctx := context.Background()
	acquired, err := store.acquireConversationIndexScanLease(ctx, token)
	require.NoError(t, err)
	require.True(t, acquired)
	renewed, err := store.renewConversationIndexScanLease(ctx, token)
	require.NoError(t, err)
	require.True(t, renewed)
}

// TestConversationIndexScanLeaseCompletionSurvivesRenewalCancellation is the
// regression for the completion/renewal race: fn publishes its irreversible
// completion marker while a renewal is in flight, and the wrapper then
// cancels the lease context as its normal shutdown. That cancellation fails
// the in-flight renewal, and the failure must not be reported as a lost
// lease — the sweep finalized, and callers like FinalizeConversationIndex
// would otherwise return an error (and zero stats) for work that durably
// completed and can never be re-run, since the completion key it just wrote
// short-circuits every later call.
//
// Deterministic by construction: fn cannot return until the gate proves a
// renewal round-trip is open, and the gate cannot return until the wrapper's
// shutdown cancels it.
func TestConversationIndexScanLeaseCompletionSurvivesRenewalCancellation(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	token, err := randomScanLeaseToken()
	require.NoError(t, err)
	warmScanLeaseScripts(t, store, token)

	gate := newRenewalGateHook()
	store.client.AddHook(gate)

	completionKey := store.conversationIndexCompletionKey()
	runErr := store.runWithConversationIndexScanLeaseEvery(ctx, token, 20*time.Millisecond, func(leaseCtx context.Context) error {
		<-gate.inFlight
		return store.client.Set(leaseCtx, completionKey, conversationIndexCompletionValue, 0).Err()
	})

	require.NoError(t, runErr, "a scan that published its completion marker must not be reported as having lost its lease, merely because the wrapper's own shutdown cancelled a renewal that was still in flight")

	value, err := store.client.Get(ctx, completionKey).Result()
	require.NoError(t, err)
	assert.Equal(t, conversationIndexCompletionValue, value, "the completion marker the scan published must still stand")
}

// TestScanLeaseRenewerVerdictHandling pins the rule that decides which late
// renewal verdicts count, and it is not a rule about timing: a verdict Redis
// actually answered is evidence about who holds the lease and is recorded
// however late it lands, while a round-trip that was merely aborted is
// evidence about nothing and is discarded once the work has stopped.
func TestScanLeaseRenewerVerdictHandling(t *testing.T) {
	newRenewer := func() (*scanLeaseRenewer, *atomic.Int64) {
		var cancels atomic.Int64
		renewer := &scanLeaseRenewer{
			stopCh: make(chan struct{}),
			done:   make(chan struct{}),
			cancel: func() { cancels.Add(1) },
		}
		// No renewal goroutine: these cases drive the handle directly.
		close(renewer.done)
		return renewer, &cancels
	}

	t.Run("unanswered renewal during work is a loss", func(t *testing.T) {
		renewer, cancels := newRenewer()

		assert.True(t, renewer.record(scanLeaseUnknown),
			"a renewal that could not be confirmed while the work is still running must fail safe")
		assert.EqualValues(t, 1, cancels.Load(), "recording a loss must cancel the work still running under that lease")
		assert.True(t, renewer.stop(), "a loss recorded during the work must be reported to the caller")
	})

	t.Run("unanswered renewal after stop is discarded", func(t *testing.T) {
		renewer, cancels := newRenewer()

		require.False(t, renewer.stop())
		before := cancels.Load()

		assert.False(t, renewer.record(scanLeaseUnknown),
			"the round-trip the wrapper's own shutdown aborted proves nothing and must be discarded")
		assert.Equal(t, before, cancels.Load(), "a discarded verdict must not cancel anything")
		assert.False(t, renewer.stop(), "the outcome must stay settled once the work is complete")
	})

	t.Run("answered loss after stop is still reported", func(t *testing.T) {
		renewer, cancels := newRenewer()

		require.False(t, renewer.stop())
		before := cancels.Load()

		assert.True(t, renewer.record(scanLeaseReleased),
			"Redis answering that the lease is no longer ours proves exclusivity was broken while fn ran, whenever that answer lands")
		assert.Equal(t, before+1, cancels.Load())
		assert.True(t, renewer.stop(), "an authoritative loss must be reported to the caller even after the work completed")
	})

	t.Run("confirmed renewal records nothing", func(t *testing.T) {
		renewer, cancels := newRenewer()

		assert.False(t, renewer.record(scanLeaseHeld))
		assert.Zero(t, cancels.Load())
		assert.False(t, renewer.stop())
	})
}

// TestClassifyScanLeaseRenewal needs no Redis: it pins the one distinction
// the whole shutdown/loss separation rests on — an error means Redis never
// answered, so it can never be read as proof about who holds the lease.
func TestClassifyScanLeaseRenewal(t *testing.T) {
	assert.Equal(t, scanLeaseHeld, classifyScanLeaseRenewal(true, nil))
	assert.Equal(t, scanLeaseReleased, classifyScanLeaseRenewal(false, nil))
	assert.Equal(t, scanLeaseUnknown, classifyScanLeaseRenewal(false, context.Canceled))
	assert.Equal(t, scanLeaseUnknown, classifyScanLeaseRenewal(false, errors.New("connection reset")),
		"a transport failure is not an answer, however it is reported")
}

// answeredRenewalGateHook holds the first lease-renewal round-trip open until
// the wrapper's own shutdown cancels it, and only then lets it reach Redis —
// on a context detached from that cancellation, so Redis genuinely answers.
//
// That is the interleaving no amount of sleeping can force: a renewal sent
// while fn was still working, whose authoritative "this lease is not yours"
// answer arrives after fn has already published its completion marker.
// renewalGateHook is its counterpart, letting the cancellation abort the
// round-trip so it is never answered at all.
type answeredRenewalGateHook struct {
	hash     string
	inFlight chan struct{}
	used     atomic.Bool
}

func newAnsweredRenewalGateHook() *answeredRenewalGateHook {
	return &answeredRenewalGateHook{
		hash:     conditionalRefreshScript.Hash(),
		inFlight: make(chan struct{}),
	}
}

func (h *answeredRenewalGateHook) DialHook(next redis.DialHook) redis.DialHook { return next }
func (h *answeredRenewalGateHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}

func (h *answeredRenewalGateHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		if !matchesScriptSHA(cmd, h.hash) || !h.used.CompareAndSwap(false, true) {
			return next(ctx, cmd)
		}

		close(h.inFlight)
		select {
		case <-ctx.Done():
		case <-time.After(5 * time.Second):
			// Never wedge the suite if the wrapper stops cancelling.
		}
		// Detached on purpose: the round-trip was already at Redis when the
		// shutdown cancelled it, so Redis still answers.
		return next(context.WithoutCancel(ctx), cmd)
	}
}

// TestConversationIndexScanLeaseReportsAuthoritativeLossAfterCompletion is the
// regression for discarding too much: narrowing "lost" to exclude the
// shutdown's own aborted renewal must not also swallow a renewal Redis
// answered. A lease taken over while fn was working is a real loss of
// exclusivity, and the answer proving it commonly lands after fn's last
// write — the sweep's whole point is that the completion marker is fn's final
// act. Reporting it is what stops FinalizeConversationIndex from returning
// success for a sweep that was not exclusive for its whole duration.
//
// Deterministic by construction: fn cannot return until the gate proves a
// renewal round-trip is open, the gate cannot let that round-trip reach Redis
// until the wrapper's shutdown cancels it, and the lease is stolen before the
// first renewal is ever sent, so the answer is always "not yours".
func TestConversationIndexScanLeaseReportsAuthoritativeLossAfterCompletion(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	token, err := randomScanLeaseToken()
	require.NoError(t, err)
	warmScanLeaseScripts(t, store, token)

	// Another holder owns the lease before the first renewal goes out, so
	// conditionalRefreshScript answers 0 rather than failing.
	require.NoError(t, store.client.Set(ctx, store.conversationIndexScanLeaseKey(), "other-holder", time.Minute).Err())

	gate := newAnsweredRenewalGateHook()
	store.client.AddHook(gate)

	completionKey := store.conversationIndexCompletionKey()
	runErr := store.runWithConversationIndexScanLeaseEvery(ctx, token, 20*time.Millisecond, func(leaseCtx context.Context) error {
		<-gate.inFlight
		return store.client.Set(leaseCtx, completionKey, conversationIndexCompletionValue, 0).Err()
	})

	require.Error(t, runErr, "a lease Redis confirmed was taken over must be reported, even though the answer landed after fn published its marker")
	assert.Contains(t, runErr.Error(), "lost mid-scan")

	// The marker is irreversible, which is exactly why the error matters: it
	// is the only signal an operator gets that this completion was published
	// by a scan that was not exclusive throughout.
	value, err := store.client.Get(ctx, completionKey).Result()
	require.NoError(t, err)
	assert.Equal(t, conversationIndexCompletionValue, value)
}

// TestConversationIndexScanLeaseStillFailsWhenLostDuringWork guards the other
// direction: narrowing the lost-lease rule to "while fn was still running"
// must not soften it. A lease taken over mid-scan still has to cancel the
// scan's context and surface as an error even when fn itself returns nil,
// which is what keeps lazyBackfillConversationIndex and
// FinalizeConversationIndex from publishing a proof built by a scan that was
// not exclusive for its whole duration.
func TestConversationIndexScanLeaseStillFailsWhenLostDuringWork(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	token, err := randomScanLeaseToken()
	require.NoError(t, err)
	acquired, err := store.acquireConversationIndexScanLease(ctx, token)
	require.NoError(t, err)
	require.True(t, acquired)

	runErr := store.runWithConversationIndexScanLeaseEvery(ctx, token, 20*time.Millisecond, func(leaseCtx context.Context) error {
		// Another holder takes the lease over while this scan is still
		// working, exactly as an expiry-then-reacquisition would.
		require.NoError(t, store.client.Set(ctx, store.conversationIndexScanLeaseKey(), "other-holder", time.Minute).Err())

		select {
		case <-leaseCtx.Done():
		case <-time.After(5 * time.Second):
			t.Error("the renewer must cancel a scan whose lease was taken over")
		}
		return nil
	})

	require.Error(t, runErr, "a scan that lost its lease mid-flight must fail even when fn returns nil")
	assert.Contains(t, runErr.Error(), "lost mid-scan")
}
