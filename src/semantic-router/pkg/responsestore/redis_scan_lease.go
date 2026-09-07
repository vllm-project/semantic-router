package responsestore

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	mathrand "math/rand/v2"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// conditionalRefreshScript extends KEYS[1]'s TTL by ARGV[2] milliseconds
// only if its current value is still exactly ARGV[1] — "prove you still own
// this key before touching it," which is what makes lease renewal safe
// against a holder that has already lost the lease to someone else.
// Single-key: KEYS[1] only, so it stays legal in Redis Cluster.
var conditionalRefreshScript = redis.NewScript(`
if redis.call("GET", KEYS[1]) ~= ARGV[1] then
	return 0
end
return redis.call("PEXPIRE", KEYS[1], ARGV[2])
`)

// randomScanLeaseToken generates a cryptographically random lease token, so
// two concurrent acquisition attempts can never collide on a guessable
// value and mistake each other for the same holder.
func randomScanLeaseToken() (string, error) {
	buf := make([]byte, 16)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("failed to generate conversation index scan lease token: %w", err)
	}
	return hex.EncodeToString(buf), nil
}

// acquireConversationIndexScanLease attempts to take the single global scan
// lease with a fresh TTL. Single-key SETNX: Cluster safe.
func (s *RedisStore) acquireConversationIndexScanLease(ctx context.Context, token string) (bool, error) {
	acquired, err := s.client.SetNX(ctx, s.conversationIndexScanLeaseKey(), token, conversationIndexScanLeaseTTL).Result()
	if err != nil {
		return false, fmt.Errorf("failed to acquire conversation index scan lease: %w", err)
	}
	return acquired, nil
}

// renewConversationIndexScanLease extends the lease's TTL, but only if
// token still owns it — never blindly, so a holder whose lease already
// expired and was re-acquired by someone else can never extend the new
// holder's lease out from under it.
func (s *RedisStore) renewConversationIndexScanLease(ctx context.Context, token string) (bool, error) {
	res, err := conditionalRefreshScript.Run(ctx, s.client, []string{s.conversationIndexScanLeaseKey()}, token, conversationIndexScanLeaseTTL.Milliseconds()).Result()
	if err != nil {
		return false, fmt.Errorf("failed to renew conversation index scan lease: %w", err)
	}
	code, ok := res.(int64)
	if !ok {
		return false, fmt.Errorf("unexpected conversation index scan lease renewal result type %T", res)
	}
	return code > 0, nil
}

// releaseConversationIndexScanLease compare-deletes the lease, so a holder
// can never release a lease it doesn't currently own (one that already
// expired and was re-acquired by someone else). Reuses
// compareDeleteResponsePayload's single-key compare-delete primitive.
func (s *RedisStore) releaseConversationIndexScanLease(ctx context.Context, token string) error {
	if _, err := s.compareDeleteResponsePayload(ctx, s.conversationIndexScanLeaseKey(), []byte(token)); err != nil {
		return fmt.Errorf("failed to release conversation index scan lease: %w", err)
	}
	return nil
}

// scanLeaseBackoff is a small jittered exponential backoff for waiters
// retrying scan lease acquisition: starts around
// conversationIndexScanLeaseMinBackoff, doubles on every wait, and caps at
// conversationIndexScanLeaseMaxBackoff.
type scanLeaseBackoff struct {
	next time.Duration
}

func newScanLeaseBackoff() *scanLeaseBackoff {
	return &scanLeaseBackoff{next: conversationIndexScanLeaseMinBackoff}
}

// wait sleeps a jittered fraction of the current backoff step (so many
// waiters woken at once do not all retry in lockstep), advances the step
// for next time, and returns promptly with ctx's error if ctx is
// cancelled first — waiters must respect request cancellation, not sit
// through a full backoff step regardless of it.
func (b *scanLeaseBackoff) wait(ctx context.Context) error {
	delay := mathrand.N(b.next)
	b.next *= 2
	if b.next > conversationIndexScanLeaseMaxBackoff {
		b.next = conversationIndexScanLeaseMaxBackoff
	}

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(delay):
		return nil
	}
}

// withConversationIndexScanLease blocks (respecting ctx cancellation, via
// scanLeaseBackoff) until it acquires the single global conversation index
// scan lease, then runs fn with a context that is cancelled the moment the
// lease is confirmed lost — a background goroutine renews the lease every
// conversationIndexScanRenewInterval and cancels fn's context the instant a
// renewal proves the lease no longer belongs to this call. The lease is
// always released on the way out, and a lost lease is reported as an error
// even if fn itself returned nil (a scan that lost its lease must not be
// trusted, whatever it managed to do before losing it — see
// lazyBackfillConversationIndex and FinalizeConversationIndex, both of which
// rely on this to avoid publishing a proof/completion built from a scan that
// was not exclusive for its whole duration).
//
// What counts as "lost" is decided by whether Redis answered the renewal, not
// by when the answer arrived. An answer of "this key no longer holds your
// token" proves exclusivity was already broken when the renewal was sent, and
// is reported however late it lands — including after fn has published its
// completion marker, since no shutdown ordering can make a broken lease
// whole again. A renewal that was merely *aborted* — this call cancels the
// lease context as its own normal shutdown once fn returns — proves nothing
// about who holds the lease, and is discarded rather than mistaken for a
// loss; treating it as one would fail a scan that in fact finalized. See
// scanLeaseRenewalVerdict and scanLeaseRenewer.
//
// Shared by ensureConversationIndex's per-conversation lazy backfill
// (Phase 3) and FinalizeConversationIndex's whole-keyspace sweep (Phase 6)
// — both need "at most one full-keyspace-touching scan running at a time,"
// which is exactly what one global lease (rather than the superseded
// per-conversation lock) provides.
func (s *RedisStore) withConversationIndexScanLease(ctx context.Context, fn func(context.Context) error) error {
	return s.withConversationIndexScanLeaseUntil(ctx, nil, fn)
}

// withConversationIndexScanLeaseUntil is withConversationIndexScanLease with
// an optional completion predicate. Waiters recheck completed before every
// acquisition attempt, so readers waiting for another reader to resolve the
// same conversation can return as soon as its proof appears instead of
// serially acquiring and releasing the global lease afterward.
func (s *RedisStore) withConversationIndexScanLeaseUntil(
	ctx context.Context,
	completed func(context.Context) (bool, error),
	fn func(context.Context) error,
) error {
	token, err := randomScanLeaseToken()
	if err != nil {
		return err
	}

	completedWhileWaiting, err := s.waitForConversationIndexScanLease(ctx, token, completed)
	if err != nil || completedWhileWaiting {
		return err
	}

	return s.runWithConversationIndexScanLease(ctx, token, fn)
}

func (s *RedisStore) waitForConversationIndexScanLease(
	ctx context.Context,
	token string,
	completed func(context.Context) (bool, error),
) (bool, error) {
	backoff := newScanLeaseBackoff()
	for {
		if completed != nil {
			done, completeErr := completed(ctx)
			if completeErr != nil {
				return false, completeErr
			}
			if done {
				return true, nil
			}
		}

		acquired, acquireErr := s.acquireConversationIndexScanLease(ctx, token)
		if acquireErr != nil {
			return false, acquireErr
		}
		if acquired {
			return false, nil
		}
		if waitErr := backoff.wait(ctx); waitErr != nil {
			return false, waitErr
		}
	}
}

func (s *RedisStore) runWithConversationIndexScanLease(ctx context.Context, token string, fn func(context.Context) error) error {
	return s.runWithConversationIndexScanLeaseEvery(ctx, token, conversationIndexScanRenewInterval, fn)
}

// runWithConversationIndexScanLeaseEvery is runWithConversationIndexScanLease
// with an explicit renewal interval, so tests can drive the
// completion/renewal race at a millisecond cadence instead of waiting out
// conversationIndexScanRenewInterval. Every production caller passes that
// constant.
func (s *RedisStore) runWithConversationIndexScanLeaseEvery(
	ctx context.Context,
	token string,
	renewInterval time.Duration,
	fn func(context.Context) error,
) error {
	leaseCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	renewer := s.startScanLeaseRenewer(leaseCtx, token, renewInterval, cancel)

	fnErr := fn(leaseCtx)

	// Stopping settles the lease's verdict before it is read: it waits out
	// any renewal still in flight when fn returned, so an answer proving the
	// lease was already gone is counted, and the shutdown's own aborted
	// round-trip is not mistaken for one.
	lost := renewer.stop()

	if releaseErr := s.releaseConversationIndexScanLease(context.WithoutCancel(ctx), token); releaseErr != nil {
		logging.Debugf("RedisStore: failed to release conversation index scan lease: %v", releaseErr)
	}

	if !lost {
		return fnErr
	}
	if fnErr != nil {
		return fmt.Errorf("conversation index scan lease was lost mid-scan: %w", fnErr)
	}
	return errors.New("conversation index scan lease was lost mid-scan")
}

// scanLeaseRenewalVerdict is what one renewal round-trip actually proved.
//
// The distinction that matters is whether Redis answered at all, not when the
// answer landed: an answered renewal is evidence about who holds the lease,
// and an aborted one is evidence about nothing.
type scanLeaseRenewalVerdict int

const (
	// scanLeaseHeld: Redis answered, and the lease key still holds this
	// call's token.
	scanLeaseHeld scanLeaseRenewalVerdict = iota
	// scanLeaseReleased: Redis answered that the lease key no longer holds
	// this call's token — the lease expired and was re-acquired, or another
	// holder took it. Authoritative, and permanently so: it proves
	// exclusivity was already broken when the renewal was sent.
	scanLeaseReleased
	// scanLeaseUnknown: the round-trip produced no answer at all — a
	// cancelled context, a transport failure, a timeout, or a reply this
	// code cannot parse. It says nothing about who holds the lease.
	scanLeaseUnknown
)

// classifyScanLeaseRenewal reads one renewal's outcome.
// renewConversationIndexScanLease already returns a non-nil error for every
// case where Redis did not give a valid answer (including an unexpected
// result type), so "Redis answered" is exactly err == nil.
func classifyScanLeaseRenewal(held bool, err error) scanLeaseRenewalVerdict {
	switch {
	case err != nil:
		return scanLeaseUnknown
	case !held:
		return scanLeaseReleased
	default:
		return scanLeaseHeld
	}
}

// scanLeaseRenewer keeps one held scan lease alive in the background and
// records whether that lease was lost.
//
// Stopping it is a signal of its own, deliberately distinct from cancelling
// the lease context. The wrapper cancels that context as part of its normal
// shutdown once fn returns, and a renewal round-trip in flight at that moment
// is aborted by the cancellation — which, on its own, is indistinguishable
// from a lease that genuinely went away. Without a separate stop signal that
// expected shutdown is reported as a lost lease, failing a scan whose
// irreversible completion marker was already published.
//
// That discard is narrow on purpose, and applies only to verdicts Redis never
// answered (scanLeaseUnknown). A scanLeaseReleased answer is recorded whether
// or not the work has stopped: see record.
type scanLeaseRenewer struct {
	// mu makes stopping and recording a verdict mutually exclusive, so the
	// two can never interleave into "an unanswered renewal recorded a loss
	// after the work completed."
	mu      sync.Mutex
	stopped bool
	lost    bool

	stopCh chan struct{}
	done   chan struct{}
	cancel context.CancelFunc
}

// startScanLeaseRenewer launches the background renewal loop for token.
// cancel must be leaseCtx's own cancel func: the renewer calls it to abort
// work still running under a lease this call has been shown to no longer
// hold.
func (s *RedisStore) startScanLeaseRenewer(
	leaseCtx context.Context,
	token string,
	interval time.Duration,
	cancel context.CancelFunc,
) *scanLeaseRenewer {
	renewer := &scanLeaseRenewer{
		stopCh: make(chan struct{}),
		done:   make(chan struct{}),
		cancel: cancel,
	}
	go s.renewConversationIndexScanLeaseUntilStopped(leaseCtx, token, interval, renewer)
	return renewer
}

// stop ends the renewal loop and reports whether the lease was lost.
//
// Waiting on done is what settles the outcome: the loop closes it only after
// any verdict it produced has already been through record, so this can never
// read lost while a verdict is still in flight — including the authoritative
// one an in-flight renewal is about to deliver.
//
// The ordering carries the other half of the guarantee: stopCh is closed,
// under mu, *before* the lease context is cancelled, so a renewal aborted by
// that cancellation can only reach record with stopped already set, and is
// discarded. Cancelling first would reintroduce exactly the race this exists
// to close.
func (r *scanLeaseRenewer) stop() bool {
	r.mu.Lock()
	if !r.stopped {
		r.stopped = true
		close(r.stopCh)
	}
	r.mu.Unlock()

	r.cancel()
	<-r.done

	r.mu.Lock()
	defer r.mu.Unlock()
	return r.lost
}

// record applies one renewal verdict, cancelling the work still running under
// the lease if this is the verdict that loses it, and reports whether it was
// the one that did so (so the loop logs exactly once).
//
// scanLeaseReleased is recorded whether or not stop has already run. Redis
// answered that this call's token is no longer in the lease key, which proves
// exclusivity was already broken when the renewal was sent — while fn was
// still working. Nothing that happens afterwards, this call's own shutdown
// included, can make that untrue, so the scan's proof/completion must not be
// trusted even though fn has already published it.
//
// scanLeaseUnknown is recorded only while the work is still running. Once
// stop has run it is, by construction, the round-trip the wrapper's own
// cancellation aborted, and an unanswered renewal is not evidence of
// anything.
func (r *scanLeaseRenewer) record(verdict scanLeaseRenewalVerdict) bool {
	if verdict == scanLeaseHeld {
		return false
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if verdict == scanLeaseUnknown && r.stopped {
		return false
	}
	if r.lost {
		return false
	}
	r.lost = true
	r.cancel()
	return true
}

// renewConversationIndexScanLeaseUntilStopped renews token's lease every
// interval until the renewer is stopped or ctx is done. A renewal Redis
// answers with "the lease is no longer yours" (expired and reacquired, or
// renewed by another holder) records the loss and cancels the lease context,
// so an in-flight scan aborts promptly rather than keep working under a lease
// it no longer holds — and is recorded even if it lands after fn returned,
// because the loss it reports happened while fn was still running.
//
// A renewal that produced no answer is handled separately. If the lease
// context is gone, the round-trip was aborted rather than answered: either
// this call's own shutdown or the caller cancelling, and in the latter case
// fn's own error already carries the real reason, so there is nothing to
// report. Otherwise — a transport failure while the work is still running —
// the lease cannot be confirmed, and this fails safe by treating it as a
// loss.
func (s *RedisStore) renewConversationIndexScanLeaseUntilStopped(
	ctx context.Context,
	token string,
	interval time.Duration,
	renewer *scanLeaseRenewer,
) {
	defer close(renewer.done)

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-renewer.stopCh:
			return
		case <-ctx.Done():
			return
		case <-ticker.C:
		}

		held, err := s.renewConversationIndexScanLease(ctx, token)
		verdict := classifyScanLeaseRenewal(held, err)
		if verdict == scanLeaseUnknown && ctx.Err() != nil {
			// Aborted by a cancellation rather than answered. Note the
			// ordering: only an unanswered round-trip is dismissed this way,
			// so an answer that arrived just before the shutdown still counts.
			return
		}

		switch verdict {
		case scanLeaseHeld:
			continue
		case scanLeaseReleased:
			if renewer.record(verdict) {
				logging.Warnf("RedisStore: conversation index scan lease was lost (expired and reacquired, or renewed by another holder)")
			}
			return
		case scanLeaseUnknown:
			if renewer.record(verdict) {
				logging.Warnf("RedisStore: failed to renew conversation index scan lease: %v", err)
			}
			return
		}
	}
}
