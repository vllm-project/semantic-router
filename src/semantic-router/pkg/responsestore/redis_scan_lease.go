package responsestore

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	mathrand "math/rand/v2"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// conditionalRefreshScript extends KEYS[1]'s TTL by ARGV[2] milliseconds
// only if its current value is still exactly ARGV[1] — the shared shape
// behind both refreshPopulatedConversationProof (Phase 2) and this file's
// scan-lease renewal: "prove you still own this key before touching it."
// Single-key: KEYS[1] only.
var conditionalRefreshScript = refreshPopulatedProofScript

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
// renewal fails or reports the lease no longer belongs to this call. The
// lease is always released on the way out, and a lease lost mid-fn is
// reported as an error even if fn itself returned nil (a scan that lost its
// lease partway through must not be trusted, whatever it managed to do
// before losing it — see lazyBackfillConversationIndex and
// FinalizeConversationIndex, both of which rely on this to avoid
// publishing a proof/completion built from a scan that was not exclusive
// for its whole duration).
//
// Shared by ensureConversationIndex's per-conversation lazy backfill
// (Phase 3) and FinalizeConversationIndex's whole-keyspace sweep (Phase 6)
// — both need "at most one full-keyspace-touching scan running at a time,"
// which is exactly what one global lease (rather than the superseded
// per-conversation lock) provides.
func (s *RedisStore) withConversationIndexScanLease(ctx context.Context, fn func(context.Context) error) error {
	token, err := randomScanLeaseToken()
	if err != nil {
		return err
	}

	backoff := newScanLeaseBackoff()
	for {
		acquired, acquireErr := s.acquireConversationIndexScanLease(ctx, token)
		if acquireErr != nil {
			return acquireErr
		}
		if acquired {
			break
		}
		if waitErr := backoff.wait(ctx); waitErr != nil {
			return waitErr
		}
	}

	leaseCtx, cancel := context.WithCancel(ctx)
	lost := make(chan struct{})
	renewerDone := make(chan struct{})
	go s.renewConversationIndexScanLeaseUntilDone(leaseCtx, token, cancel, lost, renewerDone)

	fnErr := fn(leaseCtx)

	cancel()
	<-renewerDone

	if releaseErr := s.releaseConversationIndexScanLease(context.WithoutCancel(ctx), token); releaseErr != nil {
		logging.Debugf("RedisStore: failed to release conversation index scan lease: %v", releaseErr)
	}

	select {
	case <-lost:
		if fnErr != nil {
			return fmt.Errorf("conversation index scan lease was lost mid-scan: %w", fnErr)
		}
		return errors.New("conversation index scan lease was lost mid-scan")
	default:
		return fnErr
	}
}

// renewConversationIndexScanLeaseUntilDone renews token's lease every
// conversationIndexScanRenewInterval until ctx is done. If a renewal ever
// fails outright or reports the lease is no longer held (renewed by
// someone else, or expired), it closes lost and cancels cancel so the
// in-flight scan using leaseCtx aborts promptly rather than keep working
// under a lease it no longer holds.
func (s *RedisStore) renewConversationIndexScanLeaseUntilDone(ctx context.Context, token string, cancel context.CancelFunc, lost, done chan struct{}) {
	defer close(done)

	ticker := time.NewTicker(conversationIndexScanRenewInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			ok, err := s.renewConversationIndexScanLease(context.WithoutCancel(ctx), token)
			if err != nil {
				logging.Warnf("RedisStore: failed to renew conversation index scan lease: %v", err)
				close(lost)
				cancel()
				return
			}
			if !ok {
				logging.Warnf("RedisStore: conversation index scan lease was lost (expired and reacquired, or renewed by another holder)")
				close(lost)
				cancel()
				return
			}
		}
	}
}
