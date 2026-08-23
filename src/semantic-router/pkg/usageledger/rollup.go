package usageledger

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"database/sql/driver"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

type RollupView string

const (
	RollupRequest  RollupView = "request"
	RollupDispatch RollupView = "dispatch"
)

type Dimensions struct {
	APIKeyID       string `json:"apiKeyId,omitempty"`
	UserID         string `json:"userId,omitempty"`
	TeamID         string `json:"teamId,omitempty"`
	EntrypointID   string `json:"entrypointId,omitempty"`
	RecipeID       string `json:"recipeId,omitempty"`
	Protocol       string `json:"protocol,omitempty"`
	StatusCode     int    `json:"statusCode,omitempty"`
	ErrorCode      string `json:"errorCode,omitempty"`
	LogicalModelID string `json:"logicalModelId,omitempty"`
	BackendID      string `json:"backendId,omitempty"`
	ProviderID     string `json:"providerId,omitempty"`
	DispatchType   string `json:"dispatchType,omitempty"`
}

type RollupRow struct {
	View                 RollupView
	BucketStart          time.Time
	Dimensions           Dimensions
	Requests             quota.QuotaInteger
	SuccessfulRequests   quota.QuotaInteger
	InputTokens          quota.QuotaInteger
	OutputTokens         quota.QuotaInteger
	Costs                []CostAggregate
	IncompleteDispatches quota.QuotaInteger
	Latency              timingAggregate
	TTFT                 timingAggregate
	LedgerWatermark      time.Time
}

type PostgresRollups struct {
	DB *sql.DB
}

func (r PostgresRollups) Refresh1m(ctx context.Context, namespaceID string, start, end time.Time) error {
	if r.DB == nil {
		return fmt.Errorf("usage rollup database is required")
	}
	if err := validateRollupRange(namespaceID, start, end, time.Minute, 24*time.Hour); err != nil {
		return err
	}
	connection, release, refresh1mErr := acquireRollupConnection(ctx, r.DB, namespaceID, "1m")
	if refresh1mErr != nil {
		return refresh1mErr
	}
	defer func() { _ = release() }()
	tx, refresh1mErr := connection.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead})
	if refresh1mErr != nil {
		return fmt.Errorf("begin one-minute rollup: %w", refresh1mErr)
	}
	defer func() { _ = tx.Rollback() }()
	if err := prepareRollupTransaction(ctx, tx); err != nil {
		return err
	}
	requestRows, refresh1mErr := loadRequestRollups(ctx, tx, namespaceID, start, end)
	if refresh1mErr != nil {
		return refresh1mErr
	}
	dispatchRows, refresh1mErr := loadDispatchRollups(ctx, tx, namespaceID, start, end)
	if refresh1mErr != nil {
		return refresh1mErr
	}
	rows := make([]RollupRow, 0, len(requestRows)+len(dispatchRows))
	rows = append(rows, requestRows...)
	rows = append(rows, dispatchRows...)
	if err := replaceRollupRows(ctx, tx, "usage_rollup_1m", namespaceID, start, end, rows); err != nil {
		return err
	}
	if err := advanceDirtyRollups(ctx, tx, namespaceID, start, end,
		"usage_rollup_dirty_minutes", "usage_rollup_dirty_hours", "hour"); err != nil {
		return err
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit one-minute rollup: %w", err)
	}
	if err := release(); err != nil {
		return fmt.Errorf("release one-minute rollup lock: %w", err)
	}
	return nil
}

func (r PostgresRollups) Refresh1h(ctx context.Context, namespaceID string, start, end time.Time) error {
	return r.refreshCoarser(ctx, namespaceID, start, end, time.Hour, 31*24*time.Hour, "usage_rollup_1m", "usage_rollup_1h")
}

func (r PostgresRollups) Refresh1d(ctx context.Context, namespaceID string, start, end time.Time) error {
	return r.refreshCoarser(ctx, namespaceID, start, end, 24*time.Hour, 366*24*time.Hour, "usage_rollup_1h", "usage_rollup_1d")
}

func (r PostgresRollups) refreshCoarser(
	ctx context.Context,
	namespaceID string,
	start, end time.Time,
	grain, maximum time.Duration,
	source, target string,
) error {
	if r.DB == nil {
		return fmt.Errorf("usage rollup database is required")
	}
	if err := validateRollupRange(namespaceID, start, end, grain, maximum); err != nil {
		return err
	}
	connection, release, refreshCoarserErr := acquireRollupConnection(ctx, r.DB, namespaceID, target)
	if refreshCoarserErr != nil {
		return refreshCoarserErr
	}
	defer func() { _ = release() }()
	tx, refreshCoarserErr := connection.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead})
	if refreshCoarserErr != nil {
		return fmt.Errorf("begin coarse rollup: %w", refreshCoarserErr)
	}
	defer func() { _ = tx.Rollback() }()
	if err := prepareRollupTransaction(ctx, tx); err != nil {
		return err
	}
	rows, refreshCoarserErr := loadCoarseRollups(ctx, tx, namespaceID, start, end, grain, source)
	if refreshCoarserErr != nil {
		return refreshCoarserErr
	}
	if err := replaceRollupRows(ctx, tx, target, namespaceID, start, end, rows); err != nil {
		return err
	}
	sourceQueue, targetQueue, parentUnit := "usage_rollup_dirty_hours", "usage_rollup_dirty_days", "day"
	if target == "usage_rollup_1d" {
		sourceQueue, targetQueue, parentUnit = "usage_rollup_dirty_days", "", ""
	}
	if err := advanceDirtyRollups(ctx, tx, namespaceID, start, end,
		sourceQueue, targetQueue, parentUnit); err != nil {
		return err
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit coarse rollup: %w", err)
	}
	if err := release(); err != nil {
		return fmt.Errorf("release coarse rollup lock: %w", err)
	}
	return nil
}

// acquireRollupConnection serializes one namespace/grain before beginning the
// repeatable-read transaction. Acquiring an xact-scoped advisory lock inside
// that transaction would establish its MVCC snapshot before a waiting replica
// obtains the lock, allowing the replica to replace rows from a stale snapshot.
func acquireRollupConnection(
	ctx context.Context,
	db *sql.DB,
	namespaceID, grain string,
) (*sql.Conn, func() error, error) {
	connection, err := db.Conn(ctx)
	if err != nil {
		return nil, nil, fmt.Errorf("reserve usage rollup connection: %w", err)
	}
	lockName := "usage-rollup:" + grain + ":" + namespaceID
	if _, err := connection.ExecContext(ctx, `SELECT pg_advisory_lock(hashtextextended($1, 0))`, lockName); err != nil {
		_ = connection.Close()
		return nil, nil, fmt.Errorf("lock usage rollup refresh: %w", err)
	}
	released := false
	release := func() error {
		if released {
			return nil
		}
		released = true
		releaseContext, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		var unlocked bool
		unlockErr := connection.QueryRowContext(releaseContext,
			`SELECT pg_advisory_unlock(hashtextextended($1, 0))`, lockName).Scan(&unlocked)
		if unlockErr == nil && !unlocked {
			unlockErr = fmt.Errorf("usage rollup session did not own its advisory lock")
		}
		if unlockErr != nil {
			// A session whose lock state is unknown must never return to the pool.
			_ = connection.Raw(func(any) error { return driver.ErrBadConn })
		}
		closeErr := connection.Close()
		return errors.Join(unlockErr, closeErr)
	}
	return connection, release, nil
}

// prepareRollupTransaction makes bucket boundaries independent of database
// session settings. Cross-replica serialization is acquired before this
// repeatable-read transaction begins.
func prepareRollupTransaction(ctx context.Context, tx *sql.Tx) error {
	if _, err := tx.ExecContext(ctx, `SET LOCAL TIME ZONE 'UTC'`); err != nil {
		return fmt.Errorf("set usage rollup time zone: %w", err)
	}
	return nil
}

func loadRequestRollups(ctx context.Context, tx *sql.Tx, namespaceID string, start, end time.Time) ([]RollupRow, error) {
	rows, err := tx.QueryContext(ctx, `WITH unknown_dispatches AS (
  SELECT d.namespace_id, d.event_date, d.event_id,
    count(*) FILTER (WHERE d.usage_state = 'unknown')::numeric AS incomplete
  FROM usage_dispatches d
  JOIN usage_events e USING (namespace_id, event_date, event_id)
  WHERE e.namespace_id = $1 AND e.occurred_at >= $2 AND e.occurred_at < $3
  GROUP BY d.namespace_id, d.event_date, d.event_id
)
SELECT date_trunc('minute', e.occurred_at),
  COALESCE(e.api_key_id::text,''), COALESCE(e.user_id::text,''), COALESCE(e.team_id::text,''),
  COALESCE(e.entrypoint_id::text,''), COALESCE(e.recipe_id::text,''), e.protocol,
  e.status_code, COALESCE(e.error_code,''),
  count(*) FILTER (WHERE e.event_kind IN ('actual','unknown'))::text,
  count(*) FILTER (WHERE e.event_kind IN ('actual','unknown') AND e.status_code < 400)::text,
  sum(e.input_tokens)::text, sum(e.output_tokens)::text,
  COALESCE(sum(CASE WHEN e.event_kind IN ('actual','unknown')
    THEN COALESCE(u.incomplete,0) ELSE e.incomplete_dispatch_delta END),0)::text,
  count(*) FILTER (WHERE e.event_kind IN ('actual','unknown'))::text,
  COALESCE(sum(e.latency_ms) FILTER (WHERE e.event_kind IN ('actual','unknown')),0)::text,
  count(e.ttft_ms) FILTER (WHERE e.event_kind IN ('actual','unknown'))::text,
  COALESCE(sum(e.ttft_ms) FILTER (WHERE e.event_kind IN ('actual','unknown')),0)::text,
  max(e.ingested_at)
FROM usage_events e
LEFT JOIN unknown_dispatches u USING (namespace_id, event_date, event_id)
WHERE e.namespace_id = $1 AND e.occurred_at >= $2 AND e.occurred_at < $3
GROUP BY 1,2,3,4,5,6,7,8,9
ORDER BY 1,2,3,4,5,6,7,8,9`, namespaceID, start, end)
	if err != nil {
		return nil, fmt.Errorf("aggregate request rollups: %w", err)
	}
	defer rows.Close()
	result := make([]RollupRow, 0)
	index := make(map[string]int)
	for rows.Next() {
		var row RollupRow
		var requests, successful, input, output, incomplete string
		var latencyCount, latencySum, ttftCount, ttftSum string
		row.View = RollupRequest
		if err := rows.Scan(&row.BucketStart, &row.Dimensions.APIKeyID, &row.Dimensions.UserID,
			&row.Dimensions.TeamID, &row.Dimensions.EntrypointID, &row.Dimensions.RecipeID,
			&row.Dimensions.Protocol, &row.Dimensions.StatusCode, &row.Dimensions.ErrorCode,
			&requests, &successful, &input, &output, &incomplete,
			&latencyCount, &latencySum, &ttftCount, &ttftSum, &row.LedgerWatermark); err != nil {
			return nil, fmt.Errorf("scan request rollup: %w", err)
		}
		if err := parseRollupQuantities(&row, requests, successful, input, output, incomplete); err != nil {
			return nil, err
		}
		if err := parseRollupTimings(&row, latencyCount, latencySum, ttftCount, ttftSum); err != nil {
			return nil, err
		}
		key, _ := rollupKey(row.View, row.BucketStart, row.Dimensions)
		index[key] = len(result)
		result = append(result, row)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate request rollups: %w", err)
	}
	if err := loadRequestCosts(ctx, tx, namespaceID, start, end, result, index); err != nil {
		return nil, err
	}
	if err := loadRequestTimingHistograms(ctx, tx, namespaceID, start, end, result, index); err != nil {
		return nil, err
	}
	return result, nil
}

func loadRequestCosts(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	start, end time.Time,
	rows []RollupRow,
	index map[string]int,
) error {
	costRows, err := tx.QueryContext(ctx, `SELECT date_trunc('minute', e.occurred_at),
  COALESCE(e.api_key_id::text,''), COALESCE(e.user_id::text,''), COALESCE(e.team_id::text,''),
  COALESCE(e.entrypoint_id::text,''), COALESCE(e.recipe_id::text,''), e.protocol,
  e.status_code, COALESCE(e.error_code,''), c->>'currency',
  sum((c->>'knownNumerator')::numeric)::text,
  sum((c->>'knownDispatches')::numeric)::text,
  sum((c->>'incompleteDispatches')::numeric)::text
FROM usage_events e
CROSS JOIN LATERAL jsonb_array_elements(e.costs) c
WHERE e.namespace_id = $1 AND e.occurred_at >= $2 AND e.occurred_at < $3
GROUP BY 1,2,3,4,5,6,7,8,9,10
ORDER BY 1,2,3,4,5,6,7,8,9,10`, namespaceID, start, end)
	if err != nil {
		return fmt.Errorf("aggregate request costs: %w", err)
	}
	defer costRows.Close()
	for costRows.Next() {
		var bucket time.Time
		var dims Dimensions
		var currency, numerator, known, incomplete string
		if err := costRows.Scan(&bucket, &dims.APIKeyID, &dims.UserID, &dims.TeamID,
			&dims.EntrypointID, &dims.RecipeID, &dims.Protocol, &dims.StatusCode, &dims.ErrorCode,
			&currency, &numerator, &known, &incomplete); err != nil {
			return fmt.Errorf("scan request cost rollup: %w", err)
		}
		key, _ := rollupKey(RollupRequest, bucket, dims)
		position, exists := index[key]
		if !exists {
			return fmt.Errorf("%w: request cost has no numeric rollup", ErrLedgerCorrupt)
		}
		cost, err := parseCostAggregate(currency, numerator, known, incomplete)
		if err != nil {
			return err
		}
		rows[position].Costs = append(rows[position].Costs, cost)
	}
	return costRows.Err()
}

func loadDispatchRollups(ctx context.Context, tx *sql.Tx, namespaceID string, start, end time.Time) ([]RollupRow, error) {
	rows, err := tx.QueryContext(ctx, `SELECT date_trunc('minute', d.started_at),
  COALESCE(e.api_key_id::text,''), COALESCE(e.user_id::text,''), COALESCE(e.team_id::text,''),
  COALESCE(e.entrypoint_id::text,''), COALESCE(e.recipe_id::text,''), e.protocol,
  e.status_code, COALESCE(e.error_code,''), COALESCE(d.logical_model_id::text,''),
  COALESCE(d.backend_id::text,''), COALESCE(d.provider_id,''), d.dispatch_type,
  count(DISTINCT e.admission_id)::text,
  count(DISTINCT e.admission_id) FILTER (WHERE e.status_code < 400)::text,
  sum(d.input_tokens)::text, sum(d.output_tokens)::text,
  sum(CASE WHEN d.corrects_dispatch_id IS NULL AND d.usage_state='unknown' THEN 1
           WHEN d.corrects_dispatch_id IS NOT NULL THEN -1 ELSE 0 END)::text,
  count(*) FILTER (WHERE d.corrects_dispatch_id IS NULL AND d.completed_at IS NOT NULL)::text,
  COALESCE(sum(floor(extract(epoch FROM (d.completed_at-d.started_at))*1000)::bigint)
    FILTER (WHERE d.corrects_dispatch_id IS NULL AND d.completed_at IS NOT NULL),0)::text,
  '0', '0',
  max(e.ingested_at),
  d.currency, COALESCE(sum(d.cost_numerator),0)::text,
  count(*) FILTER (WHERE d.cost_numerator IS NOT NULL
    AND (d.corrects_dispatch_id IS NULL OR e.reconciliation_strategy='actual'))::text,
  sum(CASE WHEN d.corrects_dispatch_id IS NULL AND d.cost_numerator IS NULL THEN 1
           WHEN d.corrects_dispatch_id IS NOT NULL THEN -1 ELSE 0 END)::text
FROM usage_dispatches d
JOIN usage_events e USING (namespace_id, event_date, event_id)
WHERE d.namespace_id = $1 AND d.started_at >= $2 AND d.started_at < $3
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,d.currency
ORDER BY 1,2,3,4,5,6,7,8,9,10,11,12,13,d.currency`, namespaceID, start, end)
	if err != nil {
		return nil, fmt.Errorf("aggregate dispatch rollups: %w", err)
	}
	defer rows.Close()
	result := make([]RollupRow, 0)
	for rows.Next() {
		var row RollupRow
		var requests, successful, input, output, incomplete string
		var currency, numerator, knownCost, incompleteCost string
		var latencyCount, latencySum, ttftCount, ttftSum string
		row.View = RollupDispatch
		if err := rows.Scan(&row.BucketStart, &row.Dimensions.APIKeyID, &row.Dimensions.UserID,
			&row.Dimensions.TeamID, &row.Dimensions.EntrypointID, &row.Dimensions.RecipeID,
			&row.Dimensions.Protocol, &row.Dimensions.StatusCode, &row.Dimensions.ErrorCode,
			&row.Dimensions.LogicalModelID, &row.Dimensions.BackendID, &row.Dimensions.ProviderID,
			&row.Dimensions.DispatchType, &requests, &successful, &input, &output, &incomplete,
			&latencyCount, &latencySum, &ttftCount, &ttftSum,
			&row.LedgerWatermark, &currency, &numerator, &knownCost, &incompleteCost); err != nil {
			return nil, fmt.Errorf("scan dispatch rollup: %w", err)
		}
		if err := parseRollupQuantities(&row, requests, successful, input, output, incomplete); err != nil {
			return nil, err
		}
		if err := parseRollupTimings(&row, latencyCount, latencySum, ttftCount, ttftSum); err != nil {
			return nil, err
		}
		cost, err := parseCostAggregate(currency, numerator, knownCost, incompleteCost)
		if err != nil {
			return nil, err
		}
		row.Costs = []CostAggregate{cost}
		result = append(result, row)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate dispatch rollups: %w", err)
	}
	index := make(map[string]int, len(result))
	for position, row := range result {
		key, _ := rollupKey(row.View, row.BucketStart, row.Dimensions)
		index[key] = position
	}
	if err := loadDispatchTimingHistograms(ctx, tx, namespaceID, start, end, result, index); err != nil {
		return nil, err
	}
	return result, nil
}

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
		statement := fmt.Sprintf(`SELECT date_trunc('%s', r.bucket_start), r.view, r.dimensions,
  histogram.ordinality::integer - 1, sum(histogram.value::numeric)::text
FROM %s r
CROSS JOIN LATERAL jsonb_array_elements_text(
  CASE WHEN jsonb_typeof(r.%s) = 'object' THEN r.%s->'counts' ELSE '[]'::jsonb END
) WITH ORDINALITY AS histogram(value, ordinality)
WHERE r.namespace_id = $1 AND r.bucket_start >= $2 AND r.bucket_start < $3
GROUP BY 1,2,3,4 ORDER BY 1,2,3,4`, unit, table, metric.column, metric.column)
		queryRows, err := tx.QueryContext(ctx, statement, namespaceID, start, end)
		if err != nil {
			return fmt.Errorf("aggregate coarse %s histogram: %w", metric.name, err)
		}
		for queryRows.Next() {
			var bucket time.Time
			var view string
			var dimensions []byte
			var bucketIndex int
			var count string
			if err := queryRows.Scan(&bucket, &view, &dimensions, &bucketIndex, &count); err != nil {
				queryRows.Close()
				return fmt.Errorf("scan coarse %s histogram: %w", metric.name, err)
			}
			var dims Dimensions
			if err := json.Unmarshal(dimensions, &dims); err != nil {
				queryRows.Close()
				return fmt.Errorf("decode coarse timing dimensions: %w", err)
			}
			key, _ := rollupKey(RollupView(view), bucket, dims)
			position, exists := index[key]
			if !exists {
				queryRows.Close()
				return fmt.Errorf("%w: coarse timing has no numeric rollup", ErrLedgerCorrupt)
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

func loadCoarseRollups(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	start, end time.Time,
	grain time.Duration,
	table string,
) ([]RollupRow, error) {
	unit := "hour"
	if grain == 24*time.Hour {
		unit = "day"
	}
	query := fmt.Sprintf(`SELECT date_trunc('%s', bucket_start), view, dimensions,
  sum(requests)::text, sum(successful_requests)::text, sum(input_tokens)::text,
  sum(output_tokens)::text, sum(incomplete_dispatches)::text,
  sum(latency_count)::text, sum(latency_sum_ms)::text,
  sum(ttft_count)::text, sum(ttft_sum_ms)::text, max(ledger_watermark)
FROM %s
WHERE namespace_id = $1 AND bucket_start >= $2 AND bucket_start < $3
GROUP BY 1,2,3
ORDER BY 1,2,3`, unit, table)
	baseRows, queryContextErr := tx.QueryContext(ctx, query, namespaceID, start, end)
	if queryContextErr != nil {
		return nil, fmt.Errorf("aggregate %s coarse rollups: %w", table, queryContextErr)
	}
	result := make([]RollupRow, 0)
	index := make(map[string]int)
	for baseRows.Next() {
		var row RollupRow
		var view string
		var dimensions []byte
		var requests, successful, input, output, incomplete string
		var latencyCount, latencySum, ttftCount, ttftSum string
		if err := baseRows.Scan(&row.BucketStart, &view, &dimensions, &requests, &successful,
			&input, &output, &incomplete, &latencyCount, &latencySum, &ttftCount, &ttftSum,
			&row.LedgerWatermark); err != nil {
			baseRows.Close()
			return nil, fmt.Errorf("scan coarse rollup: %w", err)
		}
		row.View = RollupView(view)
		if err := json.Unmarshal(dimensions, &row.Dimensions); err != nil {
			baseRows.Close()
			return nil, fmt.Errorf("decode coarse rollup dimensions: %w", err)
		}
		if err := parseRollupQuantities(&row, requests, successful, input, output, incomplete); err != nil {
			baseRows.Close()
			return nil, err
		}
		if err := parseRollupTimings(&row, latencyCount, latencySum, ttftCount, ttftSum); err != nil {
			baseRows.Close()
			return nil, err
		}
		key, _ := rollupKey(row.View, row.BucketStart, row.Dimensions)
		index[key] = len(result)
		result = append(result, row)
	}
	if err := baseRows.Close(); err != nil {
		return nil, err
	}
	costQuery := fmt.Sprintf(`SELECT date_trunc('%s', r.bucket_start), r.view, r.dimensions,
  c->>'currency', sum((c->>'knownNumerator')::numeric)::text,
  sum((c->>'knownDispatches')::numeric)::text,
  sum((c->>'incompleteDispatches')::numeric)::text
FROM %s r CROSS JOIN LATERAL jsonb_array_elements(r.costs) c
WHERE r.namespace_id = $1 AND r.bucket_start >= $2 AND r.bucket_start < $3
GROUP BY 1,2,3,4 ORDER BY 1,2,3,4`, unit, table)
	costRows, queryContextErr := tx.QueryContext(ctx, costQuery, namespaceID, start, end)
	if queryContextErr != nil {
		return nil, fmt.Errorf("aggregate coarse costs: %w", queryContextErr)
	}
	defer costRows.Close()
	for costRows.Next() {
		var bucket time.Time
		var view string
		var dimensions []byte
		var currency, numerator, known, incomplete string
		if err := costRows.Scan(&bucket, &view, &dimensions, &currency, &numerator, &known, &incomplete); err != nil {
			return nil, fmt.Errorf("scan coarse cost: %w", err)
		}
		var dims Dimensions
		if err := json.Unmarshal(dimensions, &dims); err != nil {
			return nil, err
		}
		key, _ := rollupKey(RollupView(view), bucket, dims)
		position, exists := index[key]
		if !exists {
			return nil, fmt.Errorf("%w: coarse cost has no numeric rollup", ErrLedgerCorrupt)
		}
		cost, err := parseCostAggregate(currency, numerator, known, incomplete)
		if err != nil {
			return nil, err
		}
		result[position].Costs = append(result[position].Costs, cost)
	}
	if err := costRows.Err(); err != nil {
		return nil, err
	}
	if err := costRows.Close(); err != nil {
		return nil, err
	}
	if err := loadCoarseTimingHistograms(ctx, tx, table, unit, namespaceID, start, end, result, index); err != nil {
		return nil, err
	}
	return result, nil
}

func replaceRollupRows(
	ctx context.Context,
	tx *sql.Tx,
	table, namespaceID string,
	start, end time.Time,
	rows []RollupRow,
) error {
	if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM %s
WHERE namespace_id = $1 AND bucket_start >= $2 AND bucket_start < $3`, table), namespaceID, start, end); err != nil {
		return fmt.Errorf("clear %s interval: %w", table, err)
	}
	query := fmt.Sprintf(`INSERT INTO %s (
  namespace_id, bucket_start, view, dimensions, dimensions_digest, requests,
  successful_requests, input_tokens, output_tokens, costs, incomplete_dispatches,
  latency_count, latency_sum_ms, latency_histogram,
  ttft_count, ttft_sum_ms, ttft_histogram, ledger_watermark
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)`, table)
	for _, row := range rows {
		dimensions, err := json.Marshal(row.Dimensions)
		if err != nil {
			return fmt.Errorf("encode rollup dimensions: %w", err)
		}
		digest := sha256.Sum256(dimensions)
		costs, err := json.Marshal(internalCostRows(row.Costs))
		if err != nil {
			return fmt.Errorf("encode rollup costs: %w", err)
		}
		latencyHistogram, err := encodeTimingHistogram(row.Latency.Histogram)
		if err != nil {
			return err
		}
		ttftHistogram, err := encodeTimingHistogram(row.TTFT.Histogram)
		if err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, query, namespaceID, row.BucketStart, string(row.View),
			dimensions, digest[:], row.Requests.String(), row.SuccessfulRequests.String(),
			row.InputTokens.String(), row.OutputTokens.String(), costs,
			row.IncompleteDispatches.String(), row.Latency.Count.String(),
			row.Latency.SumMilliseconds.String(), latencyHistogram, row.TTFT.Count.String(),
			row.TTFT.SumMilliseconds.String(), ttftHistogram, row.LedgerWatermark); err != nil {
			return fmt.Errorf("insert %s row: %w", table, err)
		}
	}
	return nil
}

func parseRollupTimings(row *RollupRow, latencyCount, latencySum, ttftCount, ttftSum string) error {
	values := []struct {
		label string
		text  string
		out   *quota.QuotaInteger
	}{
		{"latency count", latencyCount, &row.Latency.Count},
		{"latency sum", latencySum, &row.Latency.SumMilliseconds},
		{"TTFT count", ttftCount, &row.TTFT.Count},
		{"TTFT sum", ttftSum, &row.TTFT.SumMilliseconds},
	}
	for _, value := range values {
		parsed, err := quota.ParseQuotaInteger(value.text)
		if err != nil {
			return fmt.Errorf("%w: invalid rollup %s", ErrLedgerCorrupt, value.label)
		}
		*value.out = parsed
	}
	row.Latency.Histogram = emptyTimingHistogram()
	row.TTFT.Histogram = emptyTimingHistogram()
	return nil
}

func parseRollupQuantities(row *RollupRow, requests, successful, input, output, incomplete string) error {
	values := []struct {
		label string
		text  string
		out   *quota.QuotaInteger
	}{
		{"requests", requests, &row.Requests},
		{"successful requests", successful, &row.SuccessfulRequests},
		{"input tokens", input, &row.InputTokens},
		{"output tokens", output, &row.OutputTokens},
		{"incomplete dispatches", incomplete, &row.IncompleteDispatches},
	}
	for _, value := range values {
		parsed, err := quota.ParseQuotaInteger(value.text)
		if err != nil {
			return fmt.Errorf("%w: invalid rollup %s: %w", ErrLedgerCorrupt, value.label, err)
		}
		*value.out = parsed
	}
	return nil
}

func parseCostAggregate(currency, numerator, known, incomplete string) (CostAggregate, error) {
	result := CostAggregate{Currency: currency}
	var err error
	if result.KnownNumerator, err = quota.ParseQuotaInteger(numerator); err != nil {
		return CostAggregate{}, fmt.Errorf("%w: invalid cost numerator", ErrLedgerCorrupt)
	}
	if result.KnownDispatches, err = quota.ParseQuotaInteger(known); err != nil {
		return CostAggregate{}, fmt.Errorf("%w: invalid known cost count", ErrLedgerCorrupt)
	}
	if result.IncompleteDispatches, err = quota.ParseQuotaInteger(incomplete); err != nil {
		return CostAggregate{}, fmt.Errorf("%w: invalid incomplete cost count", ErrLedgerCorrupt)
	}
	return result, nil
}

func rollupKey(view RollupView, bucket time.Time, dimensions Dimensions) (string, error) {
	payload, err := json.Marshal(dimensions)
	if err != nil {
		return "", err
	}
	return string(view) + "\x00" + bucket.UTC().Format(time.RFC3339Nano) + "\x00" + string(payload), nil
}

func validateRollupRange(namespaceID string, start, end time.Time, alignment, maximum time.Duration) error {
	if err := requireUUID("namespace ID", namespaceID, false); err != nil {
		return err
	}
	if start.IsZero() || end.IsZero() || !start.Before(end) || end.Sub(start) > maximum {
		return fmt.Errorf("rollup range is empty, reversed, or exceeds %s", maximum)
	}
	if !start.Equal(start.UTC().Truncate(alignment)) || !end.Equal(end.UTC().Truncate(alignment)) {
		return fmt.Errorf("rollup range must be UTC-aligned to %s", alignment)
	}
	return nil
}
