package responsestore

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// ConversationIndexFinalizationStats reports what one FinalizeConversationIndex
// call actually did: how many response payloads it strictly scanned and
// decoded, and how many of those (the ones with a non-empty ConversationID)
// it indexed. ResponsesScanned >= ResponsesIndexed always; the difference is
// payloads with no ConversationID, which have nothing to index.
type ConversationIndexFinalizationStats struct {
	ResponsesScanned int64
	ResponsesIndexed int64
}

// conversationIndexCompletionKey is the single global (not per-conversation)
// key recording that FinalizeConversationIndex has completed a strict,
// cluster-wide sweep — see ConversationIndexCompletionKeySuffix.
func (s *RedisStore) conversationIndexCompletionKey() string {
	return s.buildKey(ConversationIndexCompletionKeySuffix)
}

// conversationIndexFinalized reports whether FinalizeConversationIndex has
// completed. Backed by conversationIndexFinalizedCache: once this process
// has observed the completion key with its expected value, every later call
// returns true without another round trip — safe because the key, once set
// to conversationIndexCompletionValue, is never unset or changed by
// anything in this package. A cache miss falls through to a GET, checked
// against the exact expected value (not existence alone), so an
// unrecognized value — a future incompatible format, or corruption — is
// treated the same as absent rather than silently trusted.
func (s *RedisStore) conversationIndexFinalized(ctx context.Context) (bool, error) {
	if s.conversationIndexFinalizedCache.Load() {
		return true, nil
	}

	value, err := s.client.Get(ctx, s.conversationIndexCompletionKey()).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return false, nil
		}
		return false, fmt.Errorf("failed to check conversation index completion: %w", err)
	}
	if value != conversationIndexCompletionValue {
		return false, nil
	}

	s.conversationIndexFinalizedCache.Store(true)
	return true, nil
}

// FinalizeConversationIndex performs a one-time, strict, cluster-wide sweep
// of every response payload, indexing each by its ConversationID, and then
// marks the sweep complete so every future read and cascade delete trusts
// the secondary index unconditionally without ever scanning again — see
// ConversationIndexCompletionKeySuffix for exactly what that guarantees and
// what it risks if run too early.
//
// This is an explicit, operator-triggered step: nothing in this package's
// request paths calls it; deployment tooling must invoke it explicitly.
// Operational prerequisites, in order:
//  1. Deploy index-aware code (this package) to every pod first.
//  2. Drain every index-unaware writer — no pod running pre-index code may
//     still be able to write a response payload once this sweep starts, or
//     a write landing after the sweep's cursor has already passed its shard
//     can be permanently missed (the per-conversation lazy fallback that
//     would otherwise self-heal a missed write is bypassed entirely, for
//     every conversation, forever, once completion is set).
//  3. If running against Redis Cluster, pause any in-progress resharding —
//     a slot migrating mid-sweep can let this scan miss keys that move
//     across nodes during the walk.
//  4. Run this one-shot finalizer.
//  5. Never roll back to an index-unaware binary after completion: it would
//     resume writing without maintaining the index the completion record
//     now unconditionally promises is exhaustive.
//
// Idempotent: a no-op, returning a zero-value stats, if already complete.
// Safe to run concurrently with ordinary index-aware traffic — the sweep
// only ZADDs, so a response indexed by its own StoreResponse call mid-sweep
// is simply rediscovered and harmlessly re-added with the same score.
//
// Strict, unlike the per-conversation lazy backfill this reuses the scan
// lease from: a missing payload (raced with an ordinary TTL expiry between
// SCAN and GET) is benign and skipped, but any other GET failure, a
// malformed payload, a scan error, or an index (ZADD) error is fatal and
// aborts the whole sweep without setting completion — a completion record
// this function sets is a permanent, whole-keyspace guarantee, never a
// best-effort snapshot, so partial success must never be mistaken for
// success.
func (s *RedisStore) FinalizeConversationIndex(ctx context.Context) (ConversationIndexFinalizationStats, error) {
	if complete, err := s.conversationIndexFinalized(ctx); err != nil {
		return ConversationIndexFinalizationStats{}, err
	} else if complete {
		return ConversationIndexFinalizationStats{}, nil
	}

	var stats ConversationIndexFinalizationStats
	err := s.withConversationIndexScanLeaseUntil(ctx, s.conversationIndexFinalized, func(leaseCtx context.Context) error {
		// Recheck under the lease: a previous run may have completed
		// between this call's first check above and acquiring the lease.
		if complete, err := s.conversationIndexFinalized(leaseCtx); err != nil {
			return err
		} else if complete {
			return nil
		}

		swept, err := s.sweepAndIndexAllConversations(leaseCtx)
		if err != nil {
			return err
		}
		stats = swept

		// Persistent: this record, unlike the per-conversation migrated
		// marker, is never meant to expire or be re-derived — it is the
		// operator's durable record that the sweep ran to completion.
		if err := s.client.Set(leaseCtx, s.conversationIndexCompletionKey(), conversationIndexCompletionValue, 0).Err(); err != nil {
			return fmt.Errorf("failed to mark conversation index finalized: %w", err)
		}
		s.conversationIndexFinalizedCache.Store(true)
		return nil
	})
	if err != nil {
		return ConversationIndexFinalizationStats{}, err
	}

	return stats, nil
}

// sweepAndIndexAllConversations strictly scans every response payload
// (scanResponsePayloads) and indexes each scanned batch's members
// grouped by conversation, one bounded pipeline of independent ZADDs per
// batch (pipelineIndexBatch) — never accumulating the whole sweep's
// findings in memory first. Each batch's own
// grouping map is local to that batch's callback invocation, so this has no
// shared mutable state to race on even when scanResponsePayloads'
// ForEachMaster invokes it concurrently across Cluster masters — only the
// two atomic counters are shared.
func (s *RedisStore) sweepAndIndexAllConversations(ctx context.Context) (ConversationIndexFinalizationStats, error) {
	var scanned, indexed atomic.Int64

	err := s.scanResponsePayloads(ctx, func(batch []*responseapi.StoredResponse) error {
		scanned.Add(int64(len(batch)))

		byConversation := make(map[string][]redis.Z, len(batch))
		for _, response := range batch {
			if response.ConversationID == "" {
				continue
			}
			byConversation[response.ConversationID] = append(byConversation[response.ConversationID],
				redis.Z{Score: float64(response.CreatedAt), Member: response.ID})
		}
		if len(byConversation) == 0 {
			return nil
		}

		n, err := s.pipelineIndexBatch(ctx, byConversation)
		if err != nil {
			return err
		}
		indexed.Add(n)
		return nil
	})
	if err != nil {
		return ConversationIndexFinalizationStats{}, fmt.Errorf("failed to sweep response payloads for index finalization: %w", err)
	}

	return ConversationIndexFinalizationStats{ResponsesScanned: scanned.Load(), ResponsesIndexed: indexed.Load()}, nil
}

// pipelineIndexBatch ZADDs every conversation's members from one scanned
// batch through a single pipeline of independent single-key commands — one
// ZADD per conversation found in the batch, never a multi-key command, so
// this stays Cluster safe regardless of how many different conversations
// (and therefore slots) one batch happens to span. Each conversation's own
// member slice is already bounded by the batch size
// (scanResponsePayloads never delivers more than redisBackfillBatchSize
// responses per batch), so no further chunking is needed here.
func (s *RedisStore) pipelineIndexBatch(ctx context.Context, byConversation map[string][]redis.Z) (int64, error) {
	pipe := s.client.Pipeline()
	var total int64
	for conversationID, members := range byConversation {
		indexKey := s.conversationIndexKey(conversationID)
		pipe.ZAdd(ctx, indexKey, members...)
		if s.ttl > 0 {
			pipe.Expire(ctx, indexKey, s.ttl)
		}
		total += int64(len(members))
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return 0, fmt.Errorf("failed to index conversation batch during index finalization: %w", err)
	}
	return total, nil
}
