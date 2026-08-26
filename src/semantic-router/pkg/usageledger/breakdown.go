package usageledger

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"
)

type BreakdownDimension string

const (
	BreakdownAPIKey       BreakdownDimension = "api_key"
	BreakdownUser         BreakdownDimension = "user"
	BreakdownTeam         BreakdownDimension = "team"
	BreakdownEntrypoint   BreakdownDimension = "entrypoint"
	BreakdownRecipe       BreakdownDimension = "recipe"
	BreakdownDecision     BreakdownDimension = "decision"
	BreakdownLogicalModel BreakdownDimension = "logical_model"
	BreakdownBackend      BreakdownDimension = "backend"
	BreakdownProvider     BreakdownDimension = "provider"
	BreakdownStatus       BreakdownDimension = "status"
	BreakdownDispatchType BreakdownDimension = "dispatch_type"
)

type BreakdownQuery struct {
	UsageQuery
	Dimension BreakdownDimension
	Limit     int
}

type BreakdownRow struct {
	Value  string      `json:"value"`
	Totals UsageTotals `json:"totals"`
}

type UsageBreakdown struct {
	Dimension       BreakdownDimension `json:"dimension"`
	Rows            []BreakdownRow     `json:"rows"`
	Grain           Grain              `json:"grain"`
	AsOf            *time.Time         `json:"asOf,omitempty"`
	LedgerWatermark *time.Time         `json:"ledgerWatermark,omitempty"`
	IngestionLag    *time.Duration     `json:"ingestionLag,omitempty"`
	Final           bool               `json:"final"`
}

type rawBreakdownTotals struct {
	requests, successful, input, output, incomplete string
	timings                                         timingPair
}

func (q PostgresQueries) Breakdown(ctx context.Context, query BreakdownQuery) (UsageBreakdown, error) {
	if q.DB == nil {
		return UsageBreakdown{}, fmt.Errorf("usage query database is required")
	}
	grain, table, view, breakdownErr := prepareQuery(query.UsageQuery)
	if breakdownErr != nil {
		return UsageBreakdown{}, breakdownErr
	}
	key, internal, ok := breakdownKey(query.Dimension)
	if !ok {
		return UsageBreakdown{}, invalidQueryf("usage breakdown dimension is unsupported")
	}
	if internal {
		view = RollupDispatch
	}
	if query.Limit == 0 {
		query.Limit = 50
	}
	if query.Limit < 1 || query.Limit > 200 {
		return UsageBreakdown{}, invalidQueryf("usage breakdown limit must be between 1 and 200")
	}
	where, args := rollupWhere(query.UsageQuery, view)
	args = append(args, query.Limit)
	expression := fmt.Sprintf("COALESCE(dimensions->>'%s','')", key)
	statement := fmt.Sprintf(`SELECT %s,
  sum(requests)::text, sum(successful_requests)::text, sum(input_tokens)::text,
  sum(output_tokens)::text, sum(incomplete_dispatches)::text,
  sum(latency_count)::text, sum(latency_sum_ms)::text,
  sum(ttft_count)::text, sum(ttft_sum_ms)::text
FROM %s WHERE %s
GROUP BY 1 ORDER BY sum(input_tokens + output_tokens) DESC, 1 LIMIT $%d`, expression, table, where, len(args))
	result, rawByValue, breakdownErr := q.loadBreakdownRows(ctx, query.Dimension, grain, statement, args)
	if breakdownErr != nil {
		return UsageBreakdown{}, breakdownErr
	}
	groupedCosts, groupedTimings, breakdownErr := q.loadBreakdownDetails(
		ctx, table, where, expression, args[:len(args)-1], result.Rows,
	)
	if breakdownErr != nil {
		return UsageBreakdown{}, breakdownErr
	}
	for position := range result.Rows {
		value := result.Rows[position].Value
		raw := rawByValue[value]
		histograms := groupedTimings[value]
		ensureTimingPairHistograms(&histograms)
		raw.timings.Latency.Histogram = histograms.Latency.Histogram
		raw.timings.TTFT.Histogram = histograms.TTFT.Histogram
		result.Rows[position].Totals, breakdownErr = publicTotals(
			raw.requests, raw.successful, raw.input, raw.output, raw.incomplete,
			groupedCosts[value], raw.timings,
		)
		if breakdownErr != nil {
			return UsageBreakdown{}, breakdownErr
		}
	}
	if err := q.setBreakdownFreshness(ctx, &result, table, where, args[:len(args)-1]); err != nil {
		return UsageBreakdown{}, err
	}
	return result, nil
}

func (q PostgresQueries) loadBreakdownRows(
	ctx context.Context,
	dimension BreakdownDimension,
	grain Grain,
	statement string,
	args []any,
) (UsageBreakdown, map[string]rawBreakdownTotals, error) {
	rows, err := q.DB.QueryContext(ctx, statement, args...)
	if err != nil {
		return UsageBreakdown{}, nil, fmt.Errorf("query usage breakdown: %w", err)
	}
	defer rows.Close()
	result := UsageBreakdown{Dimension: dimension, Grain: grain, Rows: make([]BreakdownRow, 0)}
	rawByValue := make(map[string]rawBreakdownTotals)
	for rows.Next() {
		var row BreakdownRow
		var raw rawBreakdownTotals
		var latencyCount, latencySum, ttftCount, ttftSum string
		if scanErr := rows.Scan(&row.Value, &raw.requests, &raw.successful, &raw.input,
			&raw.output, &raw.incomplete, &latencyCount, &latencySum, &ttftCount, &ttftSum); scanErr != nil {
			return UsageBreakdown{}, nil, fmt.Errorf("scan usage breakdown: %w", scanErr)
		}
		raw.timings.Latency, err = parseTimingAggregate(latencyCount, latencySum)
		if err != nil {
			return UsageBreakdown{}, nil, err
		}
		raw.timings.TTFT, err = parseTimingAggregate(ttftCount, ttftSum)
		if err != nil {
			return UsageBreakdown{}, nil, err
		}
		rawByValue[row.Value] = raw
		result.Rows = append(result.Rows, row)
	}
	if err := rows.Err(); err != nil {
		return UsageBreakdown{}, nil, err
	}
	return result, rawByValue, nil
}

func (q PostgresQueries) loadBreakdownDetails(
	ctx context.Context,
	table, where, expression string,
	baseArgs []any,
	rows []BreakdownRow,
) (map[string][]CostAggregate, map[string]timingPair, error) {
	groupedCosts := make(map[string][]CostAggregate)
	groupedTimings := make(map[string]timingPair)
	if len(rows) == 0 {
		return groupedCosts, groupedTimings, nil
	}
	selectedClauses := make([]string, 0, len(rows))
	costArgs := append([]any(nil), baseArgs...)
	for _, row := range rows {
		costArgs = append(costArgs, row.Value)
		selectedClauses = append(selectedClauses, fmt.Sprintf("$%d", len(costArgs)))
	}
	costWhere := where + " AND " + expression + " IN (" + strings.Join(selectedClauses, ",") + ")"
	// #nosec G201 -- table, predicate, and dimension expression come from the closed rollup/dimension catalogs.
	costStatement := fmt.Sprintf(`SELECT %s, c->>'currency',
  sum((c->>'knownNumerator')::numeric)::text,
  sum((c->>'knownDispatches')::numeric)::text,
  sum((c->>'incompleteDispatches')::numeric)::text
FROM %s r CROSS JOIN LATERAL jsonb_array_elements(r.costs) c
WHERE %s GROUP BY 1,2`, expression, table, costWhere)
	costRows, err := q.DB.QueryContext(ctx, costStatement, costArgs...)
	if err != nil {
		return nil, nil, fmt.Errorf("query usage breakdown costs: %w", err)
	}
	defer costRows.Close()
	for costRows.Next() {
		var value, currency, numerator, known, incomplete string
		if scanErr := costRows.Scan(&value, &currency, &numerator, &known, &incomplete); scanErr != nil {
			return nil, nil, scanErr
		}
		cost, costErr := parseCostAggregate(currency, numerator, known, incomplete)
		if costErr != nil {
			return nil, nil, costErr
		}
		groupedCosts[value] = append(groupedCosts[value], cost)
	}
	if rowsErr := costRows.Err(); rowsErr != nil {
		return nil, nil, rowsErr
	}
	groupedTimings, err = q.timingHistogramsByValue(ctx, table, costWhere, costArgs, expression)
	return groupedCosts, groupedTimings, err
}

func (q PostgresQueries) setBreakdownFreshness(
	ctx context.Context,
	result *UsageBreakdown,
	table, where string,
	args []any,
) error {
	var watermark sql.NullTime
	var asOf time.Time
	if err := q.DB.QueryRowContext(ctx, fmt.Sprintf(
		"SELECT max(ledger_watermark), clock_timestamp() FROM %s WHERE %s", table, where,
	), args...).Scan(&watermark, &asOf); err != nil {
		return fmt.Errorf("query usage breakdown freshness: %w", err)
	}
	result.AsOf = &asOf
	if watermark.Valid {
		result.LedgerWatermark = &watermark.Time
		lag := asOf.Sub(watermark.Time)
		if lag < 0 {
			lag = 0
		}
		result.IngestionLag = &lag
	}
	// A producer event-time watermark is required before a result can be final.
	result.Final = false
	return nil
}

func breakdownKey(dimension BreakdownDimension) (string, bool, bool) {
	values := map[BreakdownDimension]struct {
		key      string
		internal bool
	}{
		BreakdownAPIKey: {"apiKeyId", false}, BreakdownUser: {"userId", false},
		BreakdownTeam: {"teamId", false}, BreakdownEntrypoint: {"entrypointId", false},
		BreakdownRecipe: {"recipeId", false}, BreakdownDecision: {"decisionId", false},
		BreakdownLogicalModel: {"logicalModelId", true},
		BreakdownBackend:      {"backendId", true}, BreakdownProvider: {"providerId", true},
		BreakdownStatus: {"statusCode", false}, BreakdownDispatchType: {"dispatchType", true},
	}
	value, ok := values[dimension]
	return value.key, value.internal, ok
}
