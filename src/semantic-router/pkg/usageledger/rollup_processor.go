package usageledger

import (
	"context"
	"database/sql"
	"fmt"
	"sort"
	"sync"
	"time"
)

const defaultDirtyBucketLimit = 1000

// RollupResult reports the work completed by one bounded reconciliation pass.
type RollupResult struct {
	RefreshedMinutes int
	RefreshedHours   int
	RefreshedDays    int
	More             bool
}

// RollupProcessor incrementally reconciles immutable ledger events into all
// query grains. A process reports More when another immediate pass is useful.
type RollupProcessor interface {
	ProcessDirty(context.Context, string) (RollupResult, error)
}

// PostgresRollupProcessorOptions bounds one historical reconciliation pass.
type PostgresRollupProcessorOptions struct {
	DirtyBucketLimit int
}

// PostgresRollupProcessor materializes exact UTC usage grains in PostgreSQL.
// Historical reconciliation runs once per namespace/process; committed stream
// batches then drive targeted refreshes before acknowledgement.
type PostgresRollupProcessor struct {
	db          *sql.DB
	rollups     PostgresRollups
	bucketLimit int

	stateMu sync.Mutex
	locks   map[string]*sync.Mutex
}

func NewPostgresRollupProcessor(
	db *sql.DB,
	options PostgresRollupProcessorOptions,
) (*PostgresRollupProcessor, error) {
	if db == nil {
		return nil, fmt.Errorf("usage rollup database is required")
	}
	if options.DirtyBucketLimit == 0 {
		options.DirtyBucketLimit = defaultDirtyBucketLimit
	}
	if options.DirtyBucketLimit < 1 || options.DirtyBucketLimit > 10000 {
		return nil, fmt.Errorf("usage dirty bucket limit must be between 1 and 10000")
	}
	return &PostgresRollupProcessor{
		db: db, rollups: PostgresRollups{DB: db}, bucketLimit: options.DirtyBucketLimit,
		locks: make(map[string]*sync.Mutex),
	}, nil
}

// ProcessDirty finds dirty buckets by comparing the immutable ledger watermark
// with the target rollup watermark. Unlike a fixed lookback, this remains exact
// after an arbitrarily long writer outage or a crash between bucket refreshes.
func (processor *PostgresRollupProcessor) ProcessDirty(
	ctx context.Context,
	namespaceID string,
) (RollupResult, error) {
	if processor == nil || processor.db == nil {
		return RollupResult{}, fmt.Errorf("usage rollup processor is unavailable")
	}
	if err := requireUUID("namespace ID", namespaceID, false); err != nil {
		return RollupResult{}, err
	}
	lock := processor.namespaceLock(namespaceID)
	lock.Lock()
	defer lock.Unlock()
	result, err := processor.processDirty(ctx, namespaceID)
	if err != nil {
		return RollupResult{}, err
	}
	return result, nil
}

func (processor *PostgresRollupProcessor) processDirty(
	ctx context.Context,
	namespaceID string,
) (RollupResult, error) {
	result := RollupResult{}

	minutes, more, processDirtyErr := processor.dirtyMinutes(ctx, namespaceID)
	if processDirtyErr != nil {
		return RollupResult{}, processDirtyErr
	}
	for _, interval := range contiguousIntervals(minutes, time.Minute, 24*time.Hour) {
		if err := processor.rollups.Refresh1m(ctx, namespaceID, interval.start, interval.end); err != nil {
			return RollupResult{}, err
		}
	}
	result.RefreshedMinutes = len(minutes)
	result.More = more

	hours, more, processDirtyErr := processor.dirtyCoarse(ctx, namespaceID, "usage_rollup_1m", "usage_rollup_1h", time.Hour)
	if processDirtyErr != nil {
		return RollupResult{}, processDirtyErr
	}
	for _, interval := range contiguousIntervals(hours, time.Hour, 31*24*time.Hour) {
		if err := processor.rollups.Refresh1h(ctx, namespaceID, interval.start, interval.end); err != nil {
			return RollupResult{}, err
		}
	}
	result.RefreshedHours = len(hours)
	result.More = result.More || more

	days, more, processDirtyErr := processor.dirtyCoarse(ctx, namespaceID, "usage_rollup_1h", "usage_rollup_1d", 24*time.Hour)
	if processDirtyErr != nil {
		return RollupResult{}, processDirtyErr
	}
	for _, interval := range contiguousIntervals(days, 24*time.Hour, 366*24*time.Hour) {
		if err := processor.rollups.Refresh1d(ctx, namespaceID, interval.start, interval.end); err != nil {
			return RollupResult{}, err
		}
	}
	result.RefreshedDays = len(days)
	result.More = result.More || more
	return result, nil
}

// AfterCommit refreshes exactly the buckets touched by one committed stream
// batch. It is intentionally executed before XACK, making the Redis pending
// list the durable recovery queue for both ledger ingestion and rollups.
func (processor *PostgresRollupProcessor) AfterCommit(ctx context.Context, events []TerminalEvent) error {
	if processor == nil || processor.db == nil {
		return fmt.Errorf("usage rollup processor is unavailable")
	}
	if len(events) == 0 {
		return nil
	}
	namespaceID := events[0].NamespaceID
	if err := requireUUID("namespace ID", namespaceID, false); err != nil {
		return err
	}
	minutes := make(map[time.Time]struct{})
	for _, event := range events {
		if event.NamespaceID != namespaceID {
			return fmt.Errorf("committed usage batch spans multiple namespaces")
		}
		minutes[event.OccurredAt.UTC().Truncate(time.Minute)] = struct{}{}
		for _, dispatch := range event.Dispatches {
			minutes[dispatch.StartedAt.UTC().Truncate(time.Minute)] = struct{}{}
		}
	}
	lock := processor.namespaceLock(namespaceID)
	lock.Lock()
	defer lock.Unlock()
	minuteBuckets := sortedUsageBuckets(minutes)
	for _, interval := range contiguousIntervals(minuteBuckets, time.Minute, 24*time.Hour) {
		if err := processor.rollups.Refresh1m(ctx, namespaceID, interval.start, interval.end); err != nil {
			return err
		}
	}
	hourBuckets := parentUsageBuckets(minuteBuckets, time.Hour)
	for _, interval := range contiguousIntervals(hourBuckets, time.Hour, 31*24*time.Hour) {
		if err := processor.rollups.Refresh1h(ctx, namespaceID, interval.start, interval.end); err != nil {
			return err
		}
	}
	dayBuckets := parentUsageBuckets(hourBuckets, 24*time.Hour)
	for _, interval := range contiguousIntervals(dayBuckets, 24*time.Hour, 366*24*time.Hour) {
		if err := processor.rollups.Refresh1d(ctx, namespaceID, interval.start, interval.end); err != nil {
			return err
		}
	}
	return nil
}

func (processor *PostgresRollupProcessor) namespaceLock(namespaceID string) *sync.Mutex {
	processor.stateMu.Lock()
	defer processor.stateMu.Unlock()
	lock := processor.locks[namespaceID]
	if lock == nil {
		lock = &sync.Mutex{}
		processor.locks[namespaceID] = lock
	}
	return lock
}

func sortedUsageBuckets(values map[time.Time]struct{}) []time.Time {
	result := make([]time.Time, 0, len(values))
	for value := range values {
		result = append(result, value.UTC())
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Before(result[j]) })
	return result
}

func parentUsageBuckets(values []time.Time, grain time.Duration) []time.Time {
	unique := make(map[time.Time]struct{}, len(values))
	for _, value := range values {
		unique[value.UTC().Truncate(grain)] = struct{}{}
	}
	return sortedUsageBuckets(unique)
}

func (processor *PostgresRollupProcessor) dirtyMinutes(
	ctx context.Context,
	namespaceID string,
) ([]time.Time, bool, error) {
	buckets, more, err := listDirtyRollupBuckets(
		ctx, processor.db, "usage_rollup_dirty_minutes", namespaceID, processor.bucketLimit,
	)
	if err != nil {
		return nil, false, err
	}
	return dirtyBucketStarts(buckets), more, nil
}

func (processor *PostgresRollupProcessor) dirtyCoarse(
	ctx context.Context,
	namespaceID, sourceTable, targetTable string,
	grain time.Duration,
) ([]time.Time, bool, error) {
	queue := "usage_rollup_dirty_hours"
	if grain == 24*time.Hour && sourceTable == "usage_rollup_1h" && targetTable == "usage_rollup_1d" {
		queue = "usage_rollup_dirty_days"
	} else if grain != time.Hour || sourceTable != "usage_rollup_1m" || targetTable != "usage_rollup_1h" {
		return nil, false, fmt.Errorf("unsupported coarse usage rollup transition")
	}
	buckets, more, err := listDirtyRollupBuckets(
		ctx, processor.db, queue, namespaceID, processor.bucketLimit,
	)
	if err != nil {
		return nil, false, err
	}
	return dirtyBucketStarts(buckets), more, nil
}

type rollupInterval struct {
	start time.Time
	end   time.Time
}

func contiguousIntervals(buckets []time.Time, grain, maximum time.Duration) []rollupInterval {
	if len(buckets) == 0 {
		return nil
	}
	result := make([]rollupInterval, 0, len(buckets))
	start := buckets[0].UTC()
	previous := start
	for _, raw := range buckets[1:] {
		bucket := raw.UTC()
		end := bucket.Add(grain)
		if !bucket.Equal(previous.Add(grain)) || end.Sub(start) > maximum {
			result = append(result, rollupInterval{start: start, end: previous.Add(grain)})
			start = bucket
		}
		previous = bucket
	}
	return append(result, rollupInterval{start: start, end: previous.Add(grain)})
}
