package responsestore

import (
	"context"
	"errors"
	"fmt"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// scanResponsePayloads walks every response payload key exactly once and
// delivers strictly decoded records in bounded batches. A key expiring
// between SCAN and GET is benign; any other GET, decode, or key-identity
// failure aborts the scan so no completeness proof is published from an
// incomplete observation. legacyPolicy additionally makes the finalization
// sweep's authority to promote explicit; lazy request-path scans pass the
// preserving policy and perform no payload writes.
//
// Shared by the per-conversation lazy legacy backfill
// (lazyBackfillConversationIndex) and the whole-keyspace finalization sweep
// (sweepAndIndexAllConversations) — this is the O(N) operation the index
// exists to avoid on the hot read path. See scanResponseKeys for the
// Cluster-aware key walking.
func (s *RedisStore) scanResponsePayloads(
	ctx context.Context,
	legacyPolicy scanLegacyPolicy,
	visit func(batch []scannedResponse) error,
) error {
	return s.scanResponseKeys(ctx, func(ctx context.Context, client redis.UniversalClient, keys []string) ([]scannedResponse, error) {
		return s.getResponsesPipelined(ctx, client, keys, legacyPolicy)
	}, visit)
}

// scanResponseKeys walks every response payload key exactly once via SCAN,
// fetching each bounded batch of keys (redisBackfillBatchSize) through
// fetch and delivering the result to visit.
//
// Cluster-aware: a single Redis Cluster node's keyspace only holds the slots
// assigned to it, so in Cluster mode this scans every master via
// ForEachMaster (which invokes fetch/visit concurrently across masters, so
// callers must not share mutable callback state except through atomics).
// Standalone mode scans the one client directly.
func (s *RedisStore) scanResponseKeys(
	ctx context.Context,
	fetch func(ctx context.Context, client redis.UniversalClient, keys []string) ([]scannedResponse, error),
	visit func(batch []scannedResponse) error,
) error {
	s.scanInvocations.Add(1)

	pattern := s.buildKey(ResponseKeyPrefix + "*")

	if clusterClient, ok := s.client.(*redis.ClusterClient); ok {
		return clusterClient.ForEachMaster(ctx, func(ctx context.Context, master *redis.Client) error {
			return scanResponseNode(ctx, master, pattern, fetch, visit)
		})
	}

	return scanResponseNode(ctx, s.client, pattern, fetch, visit)
}

func scanResponseNode(
	ctx context.Context,
	client redis.UniversalClient,
	pattern string,
	fetch func(context.Context, redis.UniversalClient, []string) ([]scannedResponse, error),
	visit func([]scannedResponse) error,
) error {
	keys := make([]string, 0, redisBackfillBatchSize)
	flush := func() error {
		if len(keys) == 0 {
			return nil
		}
		batch, err := fetch(ctx, client, keys)
		keys = keys[:0]
		if err != nil || len(batch) == 0 {
			return err
		}
		return visit(batch)
	}

	iter := client.Scan(ctx, 0, pattern, redisScanCount).Iterator()
	for iter.Next(ctx) {
		keys = append(keys, iter.Val())
		if len(keys) >= redisBackfillBatchSize {
			if err := flush(); err != nil {
				return err
			}
		}
	}
	if err := iter.Err(); err != nil {
		return fmt.Errorf("failed to scan response keys: %w", err)
	}
	return flush()
}

// getResponsesPipelined decodes the shared raw pipelined results used by both
// lazy backfill and finalization. A key that expired between SCAN and GET is a
// benign TTL race and is skipped; every other read, decode, or key-identity
// failure aborts the whole scan rather than being logged and skipped, so
// neither a per-conversation proof nor the global completion record is ever
// published from an observation known to be incomplete.
//
// Each payload's remaining lifetime is read in the same pipeline, because the
// index these scans build has to outlive the payloads they found — and a
// legacy payload's real lifetime is knowable here and nowhere later. When the
// finalization-only policy is selected, all legacy promotions in this bounded
// batch execute in one additional pipeline rather than one round trip per key.
func (s *RedisStore) getResponsesPipelined(
	ctx context.Context,
	client redis.UniversalClient,
	keys []string,
	legacyPolicy scanLegacyPolicy,
) ([]scannedResponse, error) {
	if len(keys) == 0 {
		return nil, nil
	}

	results := fetchResponsePayloadsAndTTLsPipelined(ctx, client, keys)
	responses := make([]scannedResponse, 0, len(keys))
	promotions := make([]scannedLegacyPromotion, 0, len(keys))
	for i, result := range results {
		record, err := s.decodeScannedResponse(keys[i], result)
		if err != nil {
			return nil, err
		}
		if record == nil {
			continue
		}
		responses = append(responses, scannedResponse{
			response:   record.response,
			generation: record.generation,
			ttlMillis:  result.ttlMillis,
		})
		if legacyPolicy == promoteLegacyForFinalization && record.generation == "" {
			promotions = append(promotions, scannedLegacyPromotion{
				responseIndex: len(responses) - 1,
				key:           keys[i],
				record:        *record,
			})
		}
	}

	s.promoteScannedLegacyPayloads(ctx, client, responses, promotions)
	return responses, nil
}

// scannedLegacyPromotion associates one queued finalization promotion with the
// scannedResponse it updates after the pipeline completes.
type scannedLegacyPromotion struct {
	responseIndex int
	key           string
	record        responseRecord
	prepared      preparedLegacyPromotion
	command       *redis.Cmd
}

// promoteScannedLegacyPayloads upgrades one bounded scan batch in one Redis
// pipeline. This path is reached only by FinalizeConversationIndex after its
// documented operational prerequisite has drained every index-unaware writer;
// lazy backfill deliberately leaves legacy payloads and witnesses blank.
//
// Best-effort by design. A promotion that fails or loses its race must not
// abort the sweep: one stubborn payload would otherwise block finalization
// for the whole store forever. The member is then indexed blank exactly as
// before, and post-finalization cascade deletion can upgrade it on demand — so
// nothing depends on this succeeding, it only makes the common case cheap.
//
// Returns the lifetime to index the member under as well as its generation. A
// successful upgrade reports the PTTL Redis held at the instant of the write,
// not the one this scan's pipeline captured earlier: a byte-identical
// recreation with a longer TTL would otherwise be indexed under the stale,
// shorter bound and the index would retire ahead of the payload it names.
func (s *RedisStore) promoteScannedLegacyPayloads(
	ctx context.Context,
	client redis.UniversalClient,
	responses []scannedResponse,
	promotions []scannedLegacyPromotion,
) {
	if len(promotions) == 0 {
		return
	}

	pipe := client.Pipeline()
	queued := promotions[:0]
	for _, promotion := range promotions {
		prepared, err := prepareLegacyPromotion(promotion.record)
		if err != nil {
			logging.Warnf("RedisStore: failed to prepare legacy response payload %s for finalization upgrade: %v",
				promotion.key, err)
			continue
		}
		promotion.prepared = prepared
		promotion.command = promoteLegacyPayloadScript.Eval(
			ctx, pipe, []string{promotion.key}, promotion.record.raw, prepared.payload,
		)
		queued = append(queued, promotion)
	}
	if len(queued) == 0 {
		return
	}

	// Inspect each command below rather than treating the aggregate Exec error
	// as fatal: promotion is best-effort, and Redis pipelines can contain a mix
	// of successful and failed commands.
	_, _ = pipe.Exec(ctx)
	for _, promotion := range queued {
		result, resultErr := promotion.command.Result()
		ttlMillis, promoted, err := decodeLegacyPromotionResult(promotion.key, result, resultErr)
		if err != nil {
			logging.Warnf("RedisStore: failed to upgrade legacy response payload %s during finalization scan: %v",
				promotion.key, err)
			continue
		}
		if !promoted {
			continue
		}
		responses[promotion.responseIndex].generation = promotion.prepared.generation
		responses[promotion.responseIndex].ttlMillis = ttlMillis
	}
}

func (s *RedisStore) decodeScannedResponse(key string, result responsePayloadResult) (*responseRecord, error) {
	if result.err != nil {
		if errors.Is(result.err, redis.Nil) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to read response at key %s during scan: %w", key, result.err)
	}

	record, err := decodeResponseRecord(result.raw)
	if err != nil {
		return nil, fmt.Errorf("failed to parse response at key %s during scan: %w", key, err)
	}
	if record.response.ID == "" || s.buildKey(ResponseKeyPrefix+record.response.ID) != key {
		return nil, fmt.Errorf("response payload identity does not match key %s", key)
	}
	return &record, nil
}
