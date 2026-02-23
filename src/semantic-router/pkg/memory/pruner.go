package memory

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	// DefaultPruneBatchSize is the number of users to prune per batch when not configured.
	DefaultPruneBatchSize = 50

	// pruneBatchSleep is the pause between batches to avoid overwhelming Milvus.
	pruneBatchSleep = 1 * time.Second
)

// StartPruneSweep starts a background goroutine that periodically prunes stale memories.
// It returns a stop function that should be called during shutdown to terminate the goroutine.
//
// The sweep queries Milvus for memories older than the initial strength period,
// extracts unique user IDs, and calls PruneUser for each in configurable batches.
func StartPruneSweep(cfg config.MemoryQualityScoringConfig, store *MilvusStore) (stop func()) {
	interval, err := time.ParseDuration(cfg.PruneInterval)
	if err != nil || interval <= 0 {
		logging.Warnf("PruneSweep: invalid prune_interval %q, sweep disabled", cfg.PruneInterval)
		return func() {}
	}

	batchSize := cfg.PruneBatchSize
	if batchSize <= 0 {
		batchSize = DefaultPruneBatchSize
	}

	initialStrength := cfg.InitialStrengthDays
	if initialStrength <= 0 {
		initialStrength = DefaultInitialStrengthDays
	}

	stopCh := make(chan struct{})
	ticker := time.NewTicker(interval)

	go func() {
		logging.Infof("PruneSweep: started (interval=%v, batch_size=%d, initial_strength=%dd)",
			interval, batchSize, initialStrength)
		for {
			select {
			case <-ticker.C:
				runSweep(store, initialStrength, batchSize)
			case <-stopCh:
				ticker.Stop()
				logging.Infof("PruneSweep: stopped")
				return
			}
		}
	}()

	return func() {
		close(stopCh)
	}
}

// runSweep executes a single sweep cycle: find stale users, prune in batches.
func runSweep(store *MilvusStore, initialStrengthDays int, batchSize int) {
	start := time.Now()
	ctx := context.Background()

	cutoff := time.Now().Add(-time.Duration(initialStrengthDays) * 24 * time.Hour).Unix()
	userIDs, err := store.ListStaleUserIDs(ctx, cutoff)
	if err != nil {
		logging.Warnf("PruneSweep: ListStaleUserIDs failed: %v", err)
		PruneSweepErrorsTotal.Inc()
		return
	}

	if len(userIDs) == 0 {
		logging.Debugf("PruneSweep: no stale users found, skipping")
		PruneSweepRunsTotal.Inc()
		PruneSweepDuration.Observe(time.Since(start).Seconds())
		return
	}

	logging.Infof("PruneSweep: found %d users with stale memories, pruning in batches of %d", len(userIDs), batchSize)

	var totalDeleted int
	for i := 0; i < len(userIDs); i += batchSize {
		end := i + batchSize
		if end > len(userIDs) {
			end = len(userIDs)
		}
		batch := userIDs[i:end]

		for _, uid := range batch {
			deleted, pruneErr := store.PruneUser(ctx, uid)
			if pruneErr != nil {
				logging.Warnf("PruneSweep: PruneUser failed for user_id=%s: %v", uid, pruneErr)
				PruneSweepErrorsTotal.Inc()
				continue
			}
			totalDeleted += deleted
			PruneSweepUsersProcessedTotal.Inc()
		}

		// Sleep between batches to avoid Milvus pressure (skip after last batch)
		if end < len(userIDs) {
			time.Sleep(pruneBatchSleep)
		}
	}

	duration := time.Since(start).Seconds()
	if totalDeleted > 0 {
		PruneDeletedTotal.WithLabelValues("sweep").Add(float64(totalDeleted))
	}
	PruneSweepRunsTotal.Inc()
	PruneSweepDuration.Observe(duration)

	logging.Infof("PruneSweep: completed in %.2fs — %d users processed, %d memories pruned",
		duration, len(userIDs), totalDeleted)
}
