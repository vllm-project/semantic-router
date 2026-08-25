package usageledger

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

func loadRequestTimingHistograms(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	start, end time.Time,
	rows []RollupRow,
	index map[string]int,
) error {
	base := `SELECT date_trunc('minute', e.occurred_at),
  COALESCE(e.api_key_id::text,''), COALESCE(e.user_id::text,''), COALESCE(e.team_id::text,''),
  COALESCE(e.entrypoint_id::text,''), COALESCE(e.recipe_id::text,''), e.protocol,
  e.status_code, COALESCE(e.error_code,''), %s AS bucket_index, count(*)::text
FROM usage_events e
WHERE e.namespace_id = $1 AND e.occurred_at >= $2 AND e.occurred_at < $3
  AND e.event_kind IN ('actual','unknown') %s
GROUP BY 1,2,3,4,5,6,7,8,9,10
ORDER BY 1,2,3,4,5,6,7,8,9,10`
	for _, metric := range []struct {
		name      string
		column    string
		condition string
	}{
		{name: "latency", column: "e.latency_ms"},
		{name: "ttft", column: "e.ttft_ms", condition: "AND e.ttft_ms IS NOT NULL"},
	} {
		statement := fmt.Sprintf(base, timingBucketCase(metric.column), metric.condition)
		queryRows, err := tx.QueryContext(ctx, statement, namespaceID, start, end)
		if err != nil {
			return fmt.Errorf("aggregate request %s histogram: %w", metric.name, err)
		}
		for queryRows.Next() {
			var bucket time.Time
			var dims Dimensions
			var bucketIndex int
			var count string
			if err := queryRows.Scan(&bucket, &dims.APIKeyID, &dims.UserID, &dims.TeamID,
				&dims.EntrypointID, &dims.RecipeID, &dims.Protocol, &dims.StatusCode,
				&dims.ErrorCode, &bucketIndex, &count); err != nil {
				queryRows.Close()
				return fmt.Errorf("scan request %s histogram: %w", metric.name, err)
			}
			key, _ := rollupKey(RollupRequest, bucket, dims)
			position, exists := index[key]
			if !exists {
				queryRows.Close()
				return fmt.Errorf("%w: request timing has no numeric rollup", ErrLedgerCorrupt)
			}
			if err := setTimingBucket(&rows[position], metric.name, bucketIndex, count); err != nil {
				queryRows.Close()
				return err
			}
		}
		if err := queryRows.Err(); err != nil {
			queryRows.Close()
			return err
		}
		if err := queryRows.Close(); err != nil {
			return err
		}
	}
	return nil
}

func loadDispatchTimingHistograms(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	start, end time.Time,
	rows []RollupRow,
	index map[string]int,
) error {
	duration := "floor(extract(epoch FROM (d.completed_at-d.started_at))*1000)::bigint"
	// #nosec G201 -- duration is the fixed dispatch-latency expression declared above.
	statement := fmt.Sprintf(`SELECT date_trunc('minute', d.started_at),
  COALESCE(e.api_key_id::text,''), COALESCE(e.user_id::text,''), COALESCE(e.team_id::text,''),
  COALESCE(e.entrypoint_id::text,''), COALESCE(e.recipe_id::text,''), e.protocol,
  e.status_code, COALESCE(e.error_code,''), COALESCE(d.logical_model_id::text,''),
  COALESCE(d.backend_id::text,''), COALESCE(d.provider_id,''), d.dispatch_type,
  %s AS bucket_index, count(*)::text
FROM usage_dispatches d
JOIN usage_events e USING (namespace_id, event_date, event_id)
WHERE d.namespace_id = $1 AND d.started_at >= $2 AND d.started_at < $3
  AND d.corrects_dispatch_id IS NULL AND d.completed_at IS NOT NULL
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14
ORDER BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14`, timingBucketCase(duration))
	queryRows, err := tx.QueryContext(ctx, statement, namespaceID, start, end)
	if err != nil {
		return fmt.Errorf("aggregate dispatch latency histogram: %w", err)
	}
	defer queryRows.Close()
	for queryRows.Next() {
		var bucket time.Time
		var dims Dimensions
		var bucketIndex int
		var count string
		if err := queryRows.Scan(&bucket, &dims.APIKeyID, &dims.UserID, &dims.TeamID,
			&dims.EntrypointID, &dims.RecipeID, &dims.Protocol, &dims.StatusCode,
			&dims.ErrorCode, &dims.LogicalModelID, &dims.BackendID, &dims.ProviderID,
			&dims.DispatchType, &bucketIndex, &count); err != nil {
			return fmt.Errorf("scan dispatch latency histogram: %w", err)
		}
		key, _ := rollupKey(RollupDispatch, bucket, dims)
		position, exists := index[key]
		if !exists {
			return fmt.Errorf("%w: dispatch timing has no numeric rollup", ErrLedgerCorrupt)
		}
		if err := setTimingBucket(&rows[position], "latency", bucketIndex, count); err != nil {
			return err
		}
	}
	return queryRows.Err()
}

func loadCoarseTimingHistograms(
	ctx context.Context,
	tx *sql.Tx,
	table, unit, namespaceID string,
	start, end time.Time,
	rows []RollupRow,
	index map[string]int,
) error {
	for _, metric := range []struct {
		name   string
		column string
	}{
		{name: "latency", column: "latency_histogram"},
		{name: "ttft", column: "ttft_histogram"},
	} {
		if err := loadCoarseTimingMetric(ctx, tx, table, unit, namespaceID, start, end, rows, index, metric.name, metric.column); err != nil {
			return err
		}
	}
	return nil
}

func loadCoarseTimingMetric(
	ctx context.Context,
	tx *sql.Tx,
	table, unit, namespaceID string,
	start, end time.Time,
	rows []RollupRow,
	index map[string]int,
	name, column string,
) error {
	// #nosec G201 -- callers supply only cataloged rollup table/unit/metric columns.
	statement := fmt.Sprintf(`SELECT date_trunc('%s', r.bucket_start), r.view, r.dimensions,
  histogram.ordinality::integer - 1, sum(histogram.value::numeric)::text
FROM %s r
CROSS JOIN LATERAL jsonb_array_elements_text(
  CASE WHEN jsonb_typeof(r.%s) = 'object' THEN r.%s->'counts' ELSE '[]'::jsonb END
) WITH ORDINALITY AS histogram(value, ordinality)
WHERE r.namespace_id = $1 AND r.bucket_start >= $2 AND r.bucket_start < $3
GROUP BY 1,2,3,4 ORDER BY 1,2,3,4`, unit, table, column, column)
	queryRows, err := tx.QueryContext(ctx, statement, namespaceID, start, end)
	if err != nil {
		return fmt.Errorf("aggregate coarse %s histogram: %w", name, err)
	}
	defer queryRows.Close()
	for queryRows.Next() {
		var bucket time.Time
		var view string
		var dimensions []byte
		var bucketIndex int
		var count string
		if err := queryRows.Scan(&bucket, &view, &dimensions, &bucketIndex, &count); err != nil {
			return fmt.Errorf("scan coarse %s histogram: %w", name, err)
		}
		var dims Dimensions
		if err := json.Unmarshal(dimensions, &dims); err != nil {
			return fmt.Errorf("decode coarse timing dimensions: %w", err)
		}
		key, _ := rollupKey(RollupView(view), bucket, dims)
		position, exists := index[key]
		if !exists {
			return fmt.Errorf("%w: coarse timing has no numeric rollup", ErrLedgerCorrupt)
		}
		if err := setTimingBucket(&rows[position], name, bucketIndex, count); err != nil {
			return err
		}
	}
	return queryRows.Err()
}

func timingBucketCase(column string) string {
	clauses := make([]string, 0, len(timingBucketUpperBoundsMilliseconds))
	for index, upper := range timingBucketUpperBoundsMilliseconds[:len(timingBucketUpperBoundsMilliseconds)-1] {
		clauses = append(clauses, fmt.Sprintf("WHEN %s <= %d THEN %d", column, upper, index))
	}
	return "CASE " + strings.Join(clauses, " ") + fmt.Sprintf(" ELSE %d END", len(timingBucketUpperBoundsMilliseconds)-1)
}

func setTimingBucket(row *RollupRow, metric string, bucketIndex int, count string) error {
	if bucketIndex < 0 || bucketIndex >= len(timingBucketUpperBoundsMilliseconds) {
		return fmt.Errorf("%w: timing bucket index %d is outside the contract", ErrLedgerCorrupt, bucketIndex)
	}
	parsed, err := quota.ParseQuotaInteger(count)
	if err != nil {
		return fmt.Errorf("%w: invalid timing bucket count", ErrLedgerCorrupt)
	}
	timing := &row.Latency
	if metric == "ttft" {
		timing = &row.TTFT
	} else if metric != "latency" {
		return fmt.Errorf("%w: unsupported timing metric %q", ErrLedgerCorrupt, metric)
	}
	if len(timing.Histogram) == 0 {
		timing.Histogram = emptyTimingHistogram()
	}
	timing.Histogram[bucketIndex] = parsed
	return nil
}
