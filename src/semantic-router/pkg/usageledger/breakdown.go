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
	AsOf            *time.Time         `json:"asOf"`
	LedgerWatermark *time.Time         `json:"ledgerWatermark"`
	IngestionLag    *time.Duration     `json:"ingestionLag"`
	Final           bool               `json:"final"`
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
	rows, breakdownErr := q.DB.QueryContext(ctx, statement, args...)
	if breakdownErr != nil {
		return UsageBreakdown{}, fmt.Errorf("query usage breakdown: %w", breakdownErr)
	}
	result := UsageBreakdown{Dimension: query.Dimension, Grain: grain, Rows: make([]BreakdownRow, 0)}
	type rawTotals struct {
		requests, successful, input, output, incomplete string
		timings                                         timingPair
	}
	rawByValue := make(map[string]rawTotals)
	for rows.Next() {
		var row BreakdownRow
		var raw rawTotals
		var latencyCount, latencySum, ttftCount, ttftSum string
		if err := rows.Scan(&row.Value, &raw.requests, &raw.successful, &raw.input,
			&raw.output, &raw.incomplete, &latencyCount, &latencySum, &ttftCount, &ttftSum); err != nil {
			return UsageBreakdown{}, fmt.Errorf("scan usage breakdown: %w", err)
		}
		raw.timings.Latency, breakdownErr = parseTimingAggregate(latencyCount, latencySum)
		if breakdownErr != nil {
			return UsageBreakdown{}, breakdownErr
		}
		raw.timings.TTFT, breakdownErr = parseTimingAggregate(ttftCount, ttftSum)
		if breakdownErr != nil {
			return UsageBreakdown{}, breakdownErr
		}
		rawByValue[row.Value] = raw
		result.Rows = append(result.Rows, row)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return UsageBreakdown{}, err
	}
	if err := rows.Close(); err != nil {
		return UsageBreakdown{}, fmt.Errorf("close usage breakdown rows: %w", err)
	}
	groupedCosts := make(map[string][]CostAggregate)
	groupedTimings := make(map[string]timingPair)
	if len(result.Rows) != 0 {
		selectedClauses := make([]string, 0, len(result.Rows))
		costArgs := append([]any(nil), args[:len(args)-1]...)
		for _, row := range result.Rows {
			costArgs = append(costArgs, row.Value)
			selectedClauses = append(selectedClauses, fmt.Sprintf("$%d", len(costArgs)))
		}
		costWhere := where + " AND " + expression + " IN (" + strings.Join(selectedClauses, ",") + ")"
		costStatement := fmt.Sprintf(`SELECT %s, c->>'currency',
  sum((c->>'knownNumerator')::numeric)::text,
  sum((c->>'knownDispatches')::numeric)::text,
  sum((c->>'incompleteDispatches')::numeric)::text
FROM %s r CROSS JOIN LATERAL jsonb_array_elements(r.costs) c
WHERE %s GROUP BY 1,2`, expression, table, costWhere)
		costRows, breakdownErr2 := q.DB.QueryContext(ctx, costStatement, costArgs...)
		if breakdownErr2 != nil {
			return UsageBreakdown{}, fmt.Errorf("query usage breakdown costs: %w", breakdownErr2)
		}
		for costRows.Next() {
			var value, currency, numerator, known, incomplete string
			if err := costRows.Scan(&value, &currency, &numerator, &known, &incomplete); err != nil {
				return UsageBreakdown{}, err
			}
			cost, err := parseCostAggregate(currency, numerator, known, incomplete)
			if err != nil {
				return UsageBreakdown{}, err
			}
			groupedCosts[value] = append(groupedCosts[value], cost)
		}
		if err := costRows.Err(); err != nil {
			costRows.Close()
			return UsageBreakdown{}, err
		}
		if err := costRows.Close(); err != nil {
			return UsageBreakdown{}, fmt.Errorf("close usage breakdown costs: %w", err)
		}
		groupedTimings, breakdownErr2 = q.timingHistogramsByValue(ctx, table, costWhere, costArgs, expression)
		if breakdownErr2 != nil {
			return UsageBreakdown{}, breakdownErr2
		}
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
	var watermark sql.NullTime
	var asOf time.Time
	if err := q.DB.QueryRowContext(ctx, fmt.Sprintf(
		"SELECT max(ledger_watermark), clock_timestamp() FROM %s WHERE %s", table, where,
	), args[:len(args)-1]...).Scan(&watermark, &asOf); err != nil {
		return UsageBreakdown{}, fmt.Errorf("query usage breakdown freshness: %w", err)
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
	return result, nil
}

func breakdownKey(dimension BreakdownDimension) (string, bool, bool) {
	values := map[BreakdownDimension]struct {
		key      string
		internal bool
	}{
		BreakdownAPIKey: {"apiKeyId", false}, BreakdownUser: {"userId", false},
		BreakdownTeam: {"teamId", false}, BreakdownEntrypoint: {"entrypointId", false},
		BreakdownRecipe: {"recipeId", false}, BreakdownLogicalModel: {"logicalModelId", true},
		BreakdownBackend: {"backendId", true}, BreakdownProvider: {"providerId", true},
		BreakdownStatus: {"statusCode", false}, BreakdownDispatchType: {"dispatchType", true},
	}
	value, ok := values[dimension]
	return value.key, value.internal, ok
}
