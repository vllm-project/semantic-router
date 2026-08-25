package usageledger

import (
	"context"
	"database/sql"
	"fmt"
	"time"
)

type dirtyRollupBucket struct {
	start     time.Time
	watermark time.Time
}

func listDirtyRollupBuckets(
	ctx context.Context,
	db *sql.DB,
	table, namespaceID string,
	limit int,
) ([]dirtyRollupBucket, bool, error) {
	if !validDirtyRollupTable(table) {
		return nil, false, fmt.Errorf("unsupported dirty usage rollup table %q", table)
	}
	// #nosec G201 -- validDirtyRollupTable restricts table to the fixed dirty-rollup catalog.
	statement := fmt.Sprintf(`SELECT bucket_start, max(ledger_watermark)
FROM %s
WHERE namespace_id=$1
GROUP BY bucket_start
ORDER BY bucket_start
LIMIT $2`, table)
	rows, err := db.QueryContext(ctx, statement, namespaceID, limit+1)
	if err != nil {
		return nil, false, fmt.Errorf("list dirty usage rollups from %s: %w", table, err)
	}
	defer rows.Close()
	result := make([]dirtyRollupBucket, 0, limit)
	more := false
	for rows.Next() {
		var bucket dirtyRollupBucket
		if err := rows.Scan(&bucket.start, &bucket.watermark); err != nil {
			return nil, false, fmt.Errorf("scan dirty usage rollup: %w", err)
		}
		if len(result) == limit {
			more = true
			continue
		}
		bucket.start = bucket.start.UTC()
		bucket.watermark = bucket.watermark.UTC()
		result = append(result, bucket)
	}
	if err := rows.Err(); err != nil {
		return nil, false, fmt.Errorf("iterate dirty usage rollups: %w", err)
	}
	return result, more, nil
}

func dirtyBucketStarts(values []dirtyRollupBucket) []time.Time {
	result := make([]time.Time, 0, len(values))
	for _, value := range values {
		result = append(result, value.start)
	}
	return result
}

func advanceDirtyRollups(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	start, end time.Time,
	sourceQueue, targetQueue, parentUnit string,
) error {
	if !validDirtyRollupTable(sourceQueue) ||
		(targetQueue != "" && !validDirtyRollupTable(targetQueue)) ||
		(parentUnit != "" && parentUnit != "hour" && parentUnit != "day") {
		return fmt.Errorf("invalid usage rollup queue transition")
	}
	if targetQueue != "" {
		// #nosec G201 -- queues and parentUnit are checked against closed catalogs above.
		statement := fmt.Sprintf(`INSERT INTO %s(
  namespace_id,event_partition_date,bucket_start,ledger_watermark
)
SELECT $1, event_partition_date,
  date_trunc('%s', bucket_start AT TIME ZONE 'UTC') AT TIME ZONE 'UTC',
  max(ledger_watermark)
FROM %s source
WHERE namespace_id=$1 AND bucket_start >= $2 AND bucket_start < $3
GROUP BY event_partition_date,3
ON CONFLICT(namespace_id,event_partition_date,bucket_start) DO UPDATE
SET ledger_watermark=GREATEST(%s.ledger_watermark,EXCLUDED.ledger_watermark)`,
			targetQueue, parentUnit, sourceQueue, targetQueue)
		if _, err := tx.ExecContext(ctx, statement, namespaceID, start, end); err != nil {
			return fmt.Errorf("enqueue coarse usage rollups: %w", err)
		}
	}
	// The source queue is read and cleared in the same repeatable-read
	// transaction that replaces the target rollup. A concurrent upsert either
	// remains outside this snapshot or forces a serialization retry, so no
	// committed ledger watermark can be cleared before it is projected.
	// #nosec G201 -- sourceQueue is checked against the fixed dirty-rollup catalog above.
	statement := fmt.Sprintf(`DELETE FROM %s
WHERE namespace_id=$1 AND bucket_start >= $2 AND bucket_start < $3`, sourceQueue)
	if _, err := tx.ExecContext(ctx, statement, namespaceID, start, end); err != nil {
		return fmt.Errorf("clear projected usage rollups: %w", err)
	}
	return nil
}

func validDirtyRollupTable(table string) bool {
	switch table {
	case "usage_rollup_dirty_minutes", "usage_rollup_dirty_hours", "usage_rollup_dirty_days":
		return true
	default:
		return false
	}
}
