package usageledger

import (
	"context"
	"database/sql"
	"fmt"
	"regexp"
	"sort"
	"sync"
	"time"

	"github.com/lib/pq"
)

const usageStorageLockName = "vllm-sr/usage-storage/v1"

const usageRetentionCandidateLimit = 32

var usagePartitionNamePattern = regexp.MustCompile(
	`^usage_(?:events|dispatches|dispatch_attempts)_[0-9]{6}$`,
)

// StorageLifecycleOptions controls monthly usage partition maintenance. A zero
// RawRetention means that raw usage is retained indefinitely.
type StorageLifecycleOptions struct {
	CreateAheadMonths   int
	MaintenanceInterval time.Duration
	RawRetention        time.Duration
	Now                 func() time.Time
}

// StorageMaintenance reports one bounded partition lifecycle pass.
type StorageMaintenance struct {
	CreatedThrough time.Time
	ScannedMonths  int
	RetiredMonths  int
	BlockedMonths  int
	MoreCandidates bool
	Skipped        bool
}

// StorageLifecycle is shared by ingestion and background maintenance. Writers
// hold a transaction-scoped shared lock; retirement holds the matching
// exclusive lock, so a month cannot disappear between routing and commit.
type StorageLifecycle interface {
	LockWriterTx(context.Context, *sql.Tx) error
	EnsureTx(context.Context, *sql.Tx, []time.Time) error
	Reconcile(context.Context) (StorageMaintenance, error)
}

// PostgresStorageLifecycle owns the physical monthly hierarchy. It stores no
// tenant state in process memory and is safe to run from every Router replica.
type PostgresStorageLifecycle struct {
	db                  *sql.DB
	createAheadMonths   int
	maintenanceInterval time.Duration
	rawRetention        time.Duration
	now                 func() time.Time

	mu          sync.Mutex
	lastSuccess time.Time
	lastResult  StorageMaintenance
}

func NewPostgresStorageLifecycle(
	db *sql.DB,
	options StorageLifecycleOptions,
) (*PostgresStorageLifecycle, error) {
	if db == nil {
		return nil, fmt.Errorf("usage storage database is required")
	}
	if options.CreateAheadMonths < 1 || options.CreateAheadMonths > 24 {
		return nil, fmt.Errorf("usage create-ahead months must be between 1 and 24")
	}
	if options.MaintenanceInterval < time.Minute || options.MaintenanceInterval > 24*time.Hour {
		return nil, fmt.Errorf("usage maintenance interval must be between one minute and 24 hours")
	}
	if options.RawRetention < 0 ||
		(options.RawRetention > 0 && options.RawRetention < time.Hour) ||
		options.RawRetention > 10*365*24*time.Hour {
		return nil, fmt.Errorf("usage raw retention must be disabled or between one hour and ten years")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &PostgresStorageLifecycle{
		db: db, createAheadMonths: options.CreateAheadMonths,
		maintenanceInterval: options.MaintenanceInterval,
		rawRetention:        options.RawRetention, now: now,
	}, nil
}

// LockWriterTx serializes a writer with retirement without creating a
// partition. This lets digest-idempotent delivery recognize a retired
// settlement tombstone without recreating its raw month.
func (lifecycle *PostgresStorageLifecycle) LockWriterTx(
	ctx context.Context,
	tx *sql.Tx,
) error {
	if lifecycle == nil || lifecycle.db == nil || tx == nil {
		return fmt.Errorf("usage storage lifecycle is unavailable")
	}
	if _, err := tx.ExecContext(ctx,
		`SELECT pg_advisory_xact_lock_shared(hashtextextended($1, 0))`, usageStorageLockName,
	); err != nil {
		return fmt.Errorf("lock usage storage writer: %w", err)
	}
	return nil
}

// EnsureTx serializes a writer with retirement and creates every event month
// before the caller inserts facts. The lock is held until caller commit.
func (lifecycle *PostgresStorageLifecycle) EnsureTx(
	ctx context.Context,
	tx *sql.Tx,
	dates []time.Time,
) error {
	if len(dates) == 0 || len(dates) > 1000 {
		return fmt.Errorf("usage partition date batch must contain between 1 and 1000 values")
	}
	if err := lifecycle.LockWriterTx(ctx, tx); err != nil {
		return err
	}
	months := uniqueUsageMonths(dates)
	for _, month := range months {
		if _, err := tx.ExecContext(ctx, `SELECT ensure_usage_month_partition($1::date)`, month); err != nil {
			return fmt.Errorf("ensure usage partition for %s: %w", month.Format("2006-01"), err)
		}
	}
	return nil
}

// Reconcile creates the current and future partitions, then retires only full
// months that are older than the explicit retention window and pass every
// durable safety gate.
func (lifecycle *PostgresStorageLifecycle) Reconcile(
	ctx context.Context,
) (StorageMaintenance, error) {
	if lifecycle == nil || lifecycle.db == nil {
		return StorageMaintenance{}, fmt.Errorf("usage storage lifecycle is unavailable")
	}
	lifecycle.mu.Lock()
	defer lifecycle.mu.Unlock()
	now := lifecycle.now().UTC()
	if !lifecycle.lastSuccess.IsZero() && now.Sub(lifecycle.lastSuccess) < lifecycle.maintenanceInterval {
		result := lifecycle.lastResult
		result.Skipped = true
		return result, nil
	}

	tx, err := lifecycle.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelReadCommitted})
	if err != nil {
		return StorageMaintenance{}, fmt.Errorf("begin usage storage maintenance: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.ExecContext(ctx,
		`SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`, usageStorageLockName,
	); err != nil {
		return StorageMaintenance{}, fmt.Errorf("lock usage storage maintenance: %w", err)
	}

	current := usageMonth(now)
	result := StorageMaintenance{}
	for offset := 0; offset <= lifecycle.createAheadMonths; offset++ {
		month := current.AddDate(0, offset, 0)
		if _, err := tx.ExecContext(ctx, `SELECT ensure_usage_month_partition($1::date)`, month); err != nil {
			return StorageMaintenance{}, fmt.Errorf("create usage partition %s: %w", month.Format("2006-01"), err)
		}
		result.CreatedThrough = month
	}
	if lifecycle.rawRetention > 0 {
		retirement, err := retireUsageMonths(ctx, tx, now.Add(-lifecycle.rawRetention), now)
		if err != nil {
			return StorageMaintenance{}, err
		}
		result.ScannedMonths = retirement.scanned
		result.RetiredMonths = retirement.retired
		result.BlockedMonths = retirement.blocked
		result.MoreCandidates = retirement.more
	}
	if err := tx.Commit(); err != nil {
		return StorageMaintenance{}, fmt.Errorf("commit usage storage maintenance: %w", err)
	}
	lifecycle.lastSuccess = now
	lifecycle.lastResult = result
	return result, nil
}

func uniqueUsageMonths(values []time.Time) []time.Time {
	unique := make(map[time.Time]struct{}, len(values))
	for _, value := range values {
		unique[usageMonth(value)] = struct{}{}
	}
	months := make([]time.Time, 0, len(unique))
	for month := range unique {
		months = append(months, month)
	}
	sort.Slice(months, func(left, right int) bool { return months[left].Before(months[right]) })
	return months
}

func usageMonth(value time.Time) time.Time {
	utc := value.UTC()
	return time.Date(utc.Year(), utc.Month(), 1, 0, 0, 0, 0, time.UTC)
}

type usagePartitionMonth struct {
	start, end                   time.Time
	events, dispatches, attempts string
}

type usageRetirementResult struct {
	scanned int
	retired int
	blocked int
	more    bool
}

func retireUsageMonths(
	ctx context.Context,
	tx *sql.Tx,
	cutoff, now time.Time,
) (usageRetirementResult, error) {
	rows, err := tx.QueryContext(ctx, `SELECT month_start, month_end,
  event_partition, dispatch_partition, attempt_partition
FROM usage_partition_months
WHERE state='active' AND month_end <= $1::date
ORDER BY last_checked_at ASC NULLS FIRST, month_start
LIMIT $2
FOR UPDATE`, cutoff.UTC(), usageRetentionCandidateLimit+1)
	if err != nil {
		return usageRetirementResult{}, fmt.Errorf("list retained usage partitions: %w", err)
	}
	months := make([]usagePartitionMonth, 0, usageRetentionCandidateLimit+1)
	for rows.Next() {
		var month usagePartitionMonth
		if err := rows.Scan(&month.start, &month.end, &month.events, &month.dispatches, &month.attempts); err != nil {
			rows.Close()
			return usageRetirementResult{}, fmt.Errorf("scan retained usage partition: %w", err)
		}
		months = append(months, month)
	}
	if err := rows.Close(); err != nil {
		return usageRetirementResult{}, fmt.Errorf("close retained usage partition rows: %w", err)
	}

	result := usageRetirementResult{more: len(months) > usageRetentionCandidateLimit}
	if len(months) > usageRetentionCandidateLimit {
		months = months[:usageRetentionCandidateLimit]
	}
	for index, month := range months {
		result.scanned++
		safe, err := usageMonthRetirementSafe(ctx, tx, month.start, month.end)
		if err != nil {
			return usageRetirementResult{}, err
		}
		if !safe {
			result.blocked++
			if _, err := tx.ExecContext(ctx, `UPDATE usage_partition_months
SET last_checked_at=clock_timestamp()
WHERE month_start=$1::date AND state='active'`, month.start.UTC()); err != nil {
				return usageRetirementResult{}, fmt.Errorf(
					"mark blocked usage partition %s checked: %w", month.start.Format("2006-01"), err,
				)
			}
			continue
		}
		if err := retireUsageMonth(ctx, tx, month, now); err != nil {
			return usageRetirementResult{}, err
		}
		result.retired = 1
		result.more = result.more || index+1 < len(months)
		break
	}
	return result, nil
}

func usageMonthRetirementSafe(
	ctx context.Context,
	tx *sql.Tx,
	start, end time.Time,
) (bool, error) {
	var blocked bool
	err := tx.QueryRowContext(ctx, `SELECT
  EXISTS (
    SELECT 1 FROM usage_rollup_dirty_minutes
    WHERE event_partition_date >= $1::date AND event_partition_date < $2::date
    UNION ALL
    SELECT 1 FROM usage_rollup_dirty_hours
    WHERE event_partition_date >= $1::date AND event_partition_date < $2::date
    UNION ALL
    SELECT 1 FROM usage_rollup_dirty_days
    WHERE event_partition_date >= $1::date AND event_partition_date < $2::date
  )
  OR EXISTS (
    SELECT 1
    FROM usage_settlements settlement
    JOIN unknown_usage_fences fence
      ON fence.namespace_id=settlement.namespace_id
     AND fence.admission_id=settlement.admission_id
    LEFT JOIN unknown_usage_reconciliation_plans plan ON plan.fence_id=fence.id
    WHERE settlement.event_partition_date >= $1::date
      AND settlement.event_partition_date < $2::date
      AND (fence.state <> 'resolved' OR plan.phase IS DISTINCT FROM 'completed')
  )
  OR EXISTS (
    SELECT 1 FROM inference_replays
    WHERE event_date >= $1::date AND event_date < $2::date
  )`, start.UTC(), end.UTC()).Scan(&blocked)
	if err != nil {
		return false, fmt.Errorf("check usage partition retirement safety: %w", err)
	}
	return !blocked, nil
}

func retireUsageMonth(
	ctx context.Context,
	tx *sql.Tx,
	month usagePartitionMonth,
	now time.Time,
) error {
	for _, name := range []string{month.events, month.dispatches, month.attempts} {
		if !usagePartitionNamePattern.MatchString(name) {
			return fmt.Errorf("usage partition registry contains unsafe identifier %q", name)
		}
	}
	if _, err := tx.ExecContext(ctx, `UPDATE usage_settlements
SET event_retained=FALSE, raw_retired_at=$3
WHERE event_partition_date >= $1::date AND event_partition_date < $2::date
  AND event_retained=TRUE`, month.start.UTC(), month.end.UTC(), now.UTC()); err != nil {
		return fmt.Errorf("mark retired usage settlements for %s: %w", month.start.Format("2006-01"), err)
	}
	for _, partition := range []struct {
		parent string
		child  string
	}{
		{parent: "usage_dispatch_attempts", child: month.attempts},
		{parent: "usage_dispatches", child: month.dispatches},
		{parent: "usage_events", child: month.events},
	} {
		statement := "ALTER TABLE " + pq.QuoteIdentifier(partition.parent) +
			" DETACH PARTITION " + pq.QuoteIdentifier(partition.child)
		if _, err := tx.ExecContext(ctx, statement); err != nil {
			return fmt.Errorf("detach usage partition %q: %w", partition.child, err)
		}
		// Drop each detached child before detaching its parent. PostgreSQL
		// materializes the parent foreign-key contract on child partitions;
		// keeping a detached child would therefore retain a dependency on the
		// next parent partition in the hierarchy.
		if _, err := tx.ExecContext(ctx, "DROP TABLE "+pq.QuoteIdentifier(partition.child)); err != nil {
			return fmt.Errorf("drop usage partition %q: %w", partition.child, err)
		}
	}
	if _, err := tx.ExecContext(ctx, `UPDATE usage_partition_months
SET state='retired', retired_at=$2
WHERE month_start=$1::date AND state='active'`, month.start.UTC(), now.UTC()); err != nil {
		return fmt.Errorf("mark usage partition %s retired: %w", month.start.Format("2006-01"), err)
	}
	return nil
}
