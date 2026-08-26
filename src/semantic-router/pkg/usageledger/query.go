package usageledger

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

type Grain string

const (
	GrainAuto   Grain = "auto"
	GrainMinute Grain = "minute"
	GrainHour   Grain = "hour"
	GrainDay    Grain = "day"
)

type UsageFilters struct {
	APIKeyID       string `json:"apiKeyId,omitempty"`
	UserID         string `json:"userId,omitempty"`
	TeamID         string `json:"teamId,omitempty"`
	EntrypointID   string `json:"entrypointId,omitempty"`
	RecipeID       string `json:"recipeId,omitempty"`
	LogicalModelID string `json:"logicalModelId,omitempty"`
	BackendID      string `json:"backendId,omitempty"`
	ProviderID     string `json:"providerId,omitempty"`
	DispatchType   string `json:"dispatchType,omitempty"`
	Protocol       string `json:"protocol,omitempty"`
	StatusCode     int    `json:"statusCode,omitempty"`
	ErrorCode      string `json:"errorCode,omitempty"`
}

type UsageQuery struct {
	NamespaceID string
	Start       time.Time
	End         time.Time
	Grain       Grain
	TimeZone    string
	Filters     UsageFilters
	Visibility  QueryVisibility
}

var ErrInvalidQuery = errors.New("invalid usage query")

// QueryVisibility is an already-authorized result-set envelope. It must be
// supplied by the Management authorization runtime, never decoded from query
// parameters. Restricted dimensions are a union, then caller filters narrow
// that union further.
type QueryVisibility struct {
	All       bool
	TeamIDs   []string
	UserIDs   []string
	APIKeyIDs []string
}

type Completeness string

const (
	CompletenessComplete Completeness = "complete"
	CompletenessPartial  Completeness = "partial"
	CompletenessUnknown  Completeness = "unknown"
)

type CostSummary struct {
	Currency             string       `json:"currency"`
	KnownAmount          string       `json:"knownAmount"`
	Completeness         Completeness `json:"completeness"`
	KnownDispatches      string       `json:"knownDispatches"`
	IncompleteDispatches string       `json:"incompleteDispatches"`
}

type UsageTotals struct {
	Requests             string        `json:"requests"`
	SuccessfulRequests   string        `json:"successfulRequests"`
	InputTokens          string        `json:"inputTokens"`
	OutputTokens         string        `json:"outputTokens"`
	TotalTokens          string        `json:"totalTokens"`
	IncompleteDispatches string        `json:"incompleteDispatches"`
	Completeness         Completeness  `json:"completeness"`
	Costs                []CostSummary `json:"costs"`
	Latency              TimingSummary `json:"latency"`
	TTFT                 TimingSummary `json:"ttft"`
}

type UsageSummary struct {
	Totals          UsageTotals    `json:"totals"`
	Grain           Grain          `json:"grain"`
	AsOf            *time.Time     `json:"asOf,omitempty"`
	LedgerWatermark *time.Time     `json:"ledgerWatermark,omitempty"`
	IngestionLag    *time.Duration `json:"ingestionLag,omitempty"`
	Final           bool           `json:"final"`
}

type SeriesPoint struct {
	BucketStart time.Time   `json:"bucketStart"`
	Totals      UsageTotals `json:"totals"`
}

type UsageSeries struct {
	Points          []SeriesPoint  `json:"points"`
	Grain           Grain          `json:"grain"`
	AsOf            *time.Time     `json:"asOf,omitempty"`
	LedgerWatermark *time.Time     `json:"ledgerWatermark,omitempty"`
	IngestionLag    *time.Duration `json:"ingestionLag,omitempty"`
	Final           bool           `json:"final"`
}

type pendingSeriesPoint struct {
	point                                           SeriesPoint
	watermark                                       time.Time
	requests, successful, input, output, incomplete string
	latency, ttft                                   timingAggregate
}

type PostgresQueries struct {
	DB *sql.DB
}

func (q PostgresQueries) Summary(ctx context.Context, query UsageQuery) (UsageSummary, error) {
	if q.DB == nil {
		return UsageSummary{}, fmt.Errorf("usage query database is required")
	}
	grain, table, view, summaryErr := prepareQuery(query)
	if summaryErr != nil {
		return UsageSummary{}, summaryErr
	}
	where, args := rollupWhere(query, view)
	// #nosec G201 -- prepareQuery returns only cataloged rollup tables; predicates use bind parameters.
	statement := fmt.Sprintf(`SELECT
  COALESCE(sum(requests),0)::text, COALESCE(sum(successful_requests),0)::text,
  COALESCE(sum(input_tokens),0)::text, COALESCE(sum(output_tokens),0)::text,
  COALESCE(sum(incomplete_dispatches),0)::text,
  COALESCE(sum(latency_count),0)::text, COALESCE(sum(latency_sum_ms),0)::text,
  COALESCE(sum(ttft_count),0)::text, COALESCE(sum(ttft_sum_ms),0)::text,
  max(ledger_watermark), clock_timestamp()
FROM %s WHERE %s`, table, where)
	var requests, successful, input, output, incomplete string
	var latencyCount, latencySum, ttftCount, ttftSum string
	var watermark sql.NullTime
	var serverTime time.Time
	if err := q.DB.QueryRowContext(ctx, statement, args...).Scan(
		&requests, &successful, &input, &output, &incomplete,
		&latencyCount, &latencySum, &ttftCount, &ttftSum, &watermark, &serverTime,
	); err != nil {
		return UsageSummary{}, fmt.Errorf("query usage summary: %w", err)
	}
	costs, summaryErr := q.queryCosts(ctx, table, where, args, "")
	if summaryErr != nil {
		return UsageSummary{}, summaryErr
	}
	timings, summaryErr := q.timingHistograms(ctx, table, where, args)
	if summaryErr != nil {
		return UsageSummary{}, summaryErr
	}
	timings.Latency, summaryErr = timingWithHistogram(latencyCount, latencySum, timings.Latency.Histogram)
	if summaryErr != nil {
		return UsageSummary{}, summaryErr
	}
	timings.TTFT, summaryErr = timingWithHistogram(ttftCount, ttftSum, timings.TTFT.Histogram)
	if summaryErr != nil {
		return UsageSummary{}, summaryErr
	}
	totals, summaryErr := publicTotals(requests, successful, input, output, incomplete, costs, timings)
	if summaryErr != nil {
		return UsageSummary{}, summaryErr
	}
	result := UsageSummary{Totals: totals, Grain: grain, AsOf: &serverTime}
	if watermark.Valid {
		result.LedgerWatermark = &watermark.Time
		lag := serverTime.Sub(watermark.Time)
		if lag < 0 {
			lag = 0
		}
		result.IngestionLag = &lag
	}
	// max(ingested_at) is a freshness observation, not a contiguous producer
	// watermark. Keep final conservative until the quota-runtime publishes an
	// event-time watermark that proves no earlier terminal item can arrive.
	result.Final = false
	return result, nil
}

func (q PostgresQueries) Series(ctx context.Context, query UsageQuery) (UsageSeries, error) {
	if q.DB == nil {
		return UsageSeries{}, fmt.Errorf("usage query database is required")
	}
	grain, _, view, seriesErr := prepareQuery(query)
	if seriesErr != nil {
		return UsageSeries{}, seriesErr
	}
	table, bucketUnit := seriesSource(grain, query.TimeZone)
	where, args := rollupWhere(query, view)
	bucketExpression := "bucket_start"
	if bucketUnit != "" {
		zone := query.TimeZone
		if zone == "" {
			zone = "UTC"
		}
		args = append(args, zone)
		bucketExpression = fmt.Sprintf(
			"date_trunc('%s', bucket_start AT TIME ZONE $%d) AT TIME ZONE $%d",
			bucketUnit, len(args), len(args),
		)
	}
	// #nosec G201 -- grain/table/predicate fragments are selected from closed catalogs before this query is built.
	statement := fmt.Sprintf(`SELECT %s AS bucket_start,
  sum(requests)::text, sum(successful_requests)::text, sum(input_tokens)::text,
  sum(output_tokens)::text, sum(incomplete_dispatches)::text,
  sum(latency_count)::text, sum(latency_sum_ms)::text,
  sum(ttft_count)::text, sum(ttft_sum_ms)::text, max(ledger_watermark)
FROM %s WHERE %s
GROUP BY %s ORDER BY %s`, bucketExpression, table, where, bucketExpression, bucketExpression)
	rows, seriesErr := q.DB.QueryContext(ctx, statement, args...)
	if seriesErr != nil {
		return UsageSeries{}, fmt.Errorf("query usage series: %w", seriesErr)
	}
	points, latest, seriesErr := readSeriesPoints(rows)
	if seriesErr != nil {
		rows.Close()
		return UsageSeries{}, seriesErr
	}
	if err := rows.Close(); err != nil {
		return UsageSeries{}, fmt.Errorf("close usage series rows: %w", err)
	}
	costsByBucket, seriesErr := q.queryCostsByBucket(ctx, table, where, args, bucketExpression)
	if seriesErr != nil {
		return UsageSeries{}, seriesErr
	}
	timingsByBucket, seriesErr := q.timingHistogramsByBucket(ctx, table, where, args, bucketExpression)
	if seriesErr != nil {
		return UsageSeries{}, seriesErr
	}
	serverTime, seriesErr := q.databaseTime(ctx)
	if seriesErr != nil {
		return UsageSeries{}, seriesErr
	}
	result := UsageSeries{Grain: grain, Points: make([]SeriesPoint, 0, len(points)), AsOf: &serverTime}
	for _, value := range points {
		key := timingBucketKey(value.point.BucketStart)
		histograms := timingsByBucket[key]
		ensureTimingPairHistograms(&histograms)
		value.latency.Histogram = histograms.Latency.Histogram
		value.ttft.Histogram = histograms.TTFT.Histogram
		value.point.Totals, seriesErr = publicTotals(
			value.requests, value.successful, value.input, value.output, value.incomplete,
			costsByBucket[value.point.BucketStart], timingPair{Latency: value.latency, TTFT: value.ttft},
		)
		if seriesErr != nil {
			return UsageSeries{}, seriesErr
		}
		result.Points = append(result.Points, value.point)
	}
	if !latest.IsZero() {
		result.LedgerWatermark = &latest
		lag := serverTime.Sub(latest)
		if lag < 0 {
			lag = 0
		}
		result.IngestionLag = &lag
	}
	result.Final = false
	return result, nil
}

func readSeriesPoints(rows *sql.Rows) ([]pendingSeriesPoint, time.Time, error) {
	points := make([]pendingSeriesPoint, 0)
	var latest time.Time
	for rows.Next() {
		var value pendingSeriesPoint
		var latencyCount, latencySum, ttftCount, ttftSum string
		if err := rows.Scan(&value.point.BucketStart, &value.requests, &value.successful,
			&value.input, &value.output, &value.incomplete, &latencyCount, &latencySum,
			&ttftCount, &ttftSum, &value.watermark); err != nil {
			return nil, time.Time{}, fmt.Errorf("scan usage series: %w", err)
		}
		var err error
		value.latency, err = parseTimingAggregate(latencyCount, latencySum)
		if err != nil {
			return nil, time.Time{}, err
		}
		value.ttft, err = parseTimingAggregate(ttftCount, ttftSum)
		if err != nil {
			return nil, time.Time{}, err
		}
		if value.watermark.After(latest) {
			latest = value.watermark
		}
		points = append(points, value)
	}
	if err := rows.Err(); err != nil {
		return nil, time.Time{}, fmt.Errorf("iterate usage series: %w", err)
	}
	return points, latest, nil
}

func (q PostgresQueries) databaseTime(ctx context.Context) (time.Time, error) {
	var now time.Time
	if err := q.DB.QueryRowContext(ctx, `SELECT clock_timestamp()`).Scan(&now); err != nil {
		return time.Time{}, fmt.Errorf("read usage query time: %w", err)
	}
	return now, nil
}

func (q PostgresQueries) queryCosts(
	ctx context.Context,
	table, where string,
	args []any,
	group string,
) ([]CostAggregate, error) {
	groupSelect := ""
	groupBy := ""
	if group != "" {
		groupSelect = group + ","
		groupBy = "1,"
	}
	// #nosec G201 -- the prefix, table, and predicate are internal catalog fragments; filter values are bound.
	statement := fmt.Sprintf(`SELECT %s c->>'currency',
  sum((c->>'knownNumerator')::numeric)::text,
  sum((c->>'knownDispatches')::numeric)::text,
  sum((c->>'incompleteDispatches')::numeric)::text
FROM %s r CROSS JOIN LATERAL jsonb_array_elements(r.costs) c
WHERE %s GROUP BY %s c->>'currency' ORDER BY %s c->>'currency'`, groupSelect, table, where, groupBy, groupBy)
	rows, err := q.DB.QueryContext(ctx, statement, args...)
	if err != nil {
		return nil, fmt.Errorf("query usage costs: %w", err)
	}
	defer rows.Close()
	result := make([]CostAggregate, 0)
	for rows.Next() {
		var currency, numerator, known, incomplete string
		if err := rows.Scan(&currency, &numerator, &known, &incomplete); err != nil {
			return nil, err
		}
		cost, err := parseCostAggregate(currency, numerator, known, incomplete)
		if err != nil {
			return nil, err
		}
		result = append(result, cost)
	}
	return result, rows.Err()
}

func (q PostgresQueries) queryCostsByBucket(
	ctx context.Context,
	table, where string,
	args []any,
	bucketExpression string,
) (map[time.Time][]CostAggregate, error) {
	// #nosec G201 -- the bucket/table/predicate fragments are validated catalog values; filter values are bound.
	statement := fmt.Sprintf(`SELECT %s AS bucket_start, c->>'currency',
  sum((c->>'knownNumerator')::numeric)::text,
  sum((c->>'knownDispatches')::numeric)::text,
  sum((c->>'incompleteDispatches')::numeric)::text
FROM %s r CROSS JOIN LATERAL jsonb_array_elements(r.costs) c
WHERE %s GROUP BY %s, c->>'currency' ORDER BY %s, c->>'currency'`, bucketExpression, table, where, bucketExpression, bucketExpression)
	rows, err := q.DB.QueryContext(ctx, statement, args...)
	if err != nil {
		return nil, fmt.Errorf("query usage series costs: %w", err)
	}
	defer rows.Close()
	result := make(map[time.Time][]CostAggregate)
	for rows.Next() {
		var bucket time.Time
		var currency, numerator, known, incomplete string
		if err := rows.Scan(&bucket, &currency, &numerator, &known, &incomplete); err != nil {
			return nil, err
		}
		cost, err := parseCostAggregate(currency, numerator, known, incomplete)
		if err != nil {
			return nil, err
		}
		result[bucket] = append(result[bucket], cost)
	}
	return result, rows.Err()
}

func prepareQuery(query UsageQuery) (Grain, string, RollupView, error) {
	if err := validateUsageQuery(query); err != nil {
		return "", "", "", err
	}
	grain := selectGrain(query.Start, query.End, query.Grain)
	table := map[Grain]string{GrainMinute: "usage_rollup_1m", GrainHour: "usage_rollup_1h", GrainDay: "usage_rollup_1d"}[grain]
	view := RollupRequest
	if query.Filters.LogicalModelID != "" || query.Filters.BackendID != "" ||
		query.Filters.ProviderID != "" || query.Filters.DispatchType != "" {
		view = RollupDispatch
	}
	return grain, table, view, nil
}

func seriesSource(grain Grain, zone string) (string, string) {
	if zone == "" {
		zone = "UTC"
	}
	if zone == "UTC" || grain == GrainMinute {
		return map[Grain]string{
			GrainMinute: "usage_rollup_1m",
			GrainHour:   "usage_rollup_1h",
			GrainDay:    "usage_rollup_1d",
		}[grain], ""
	}
	table := "usage_rollup_1m"
	unit := "hour"
	if grain == GrainDay {
		table = "usage_rollup_1h"
		unit = "day"
	}
	return table, unit
}

func validateUsageQuery(query UsageQuery) error {
	if err := requireUUID("namespace ID", query.NamespaceID, false); err != nil {
		return invalidQuery(err)
	}
	if query.Start.IsZero() || query.End.IsZero() || !query.Start.Before(query.End) || query.End.Sub(query.Start) > 5*366*24*time.Hour {
		return invalidQueryf("usage query range is empty, reversed, or exceeds five years")
	}
	if query.Grain != "" && query.Grain != GrainAuto && query.Grain != GrainMinute && query.Grain != GrainHour && query.Grain != GrainDay {
		return invalidQueryf("usage query grain is unsupported")
	}
	zone := query.TimeZone
	if zone == "" {
		zone = "UTC"
	}
	if _, err := time.LoadLocation(zone); err != nil {
		return invalidQueryf("usage query time zone is unsupported")
	}
	grain := selectGrain(query.Start, query.End, query.Grain)
	steps := query.End.Sub(query.Start) / grainDuration(grain)
	if steps > 10000 {
		return invalidQueryf("usage query would return more than 10000 points")
	}
	for label, value := range map[string]string{
		"API key ID": query.Filters.APIKeyID, "user ID": query.Filters.UserID,
		"team ID": query.Filters.TeamID, "backend ID": query.Filters.BackendID,
	} {
		if err := requireUUID(label, value, true); err != nil {
			return invalidQuery(err)
		}
	}
	for label, value := range map[string]string{
		"entrypoint ID":    query.Filters.EntrypointID,
		"recipe ID":        query.Filters.RecipeID,
		"logical Model ID": query.Filters.LogicalModelID,
		"provider ID":      query.Filters.ProviderID, "dispatch type": query.Filters.DispatchType,
		"protocol": query.Filters.Protocol, "error code": query.Filters.ErrorCode,
	} {
		if err := boundedCode(label, value, false); err != nil {
			return invalidQuery(err)
		}
	}
	if query.Filters.StatusCode != 0 && (query.Filters.StatusCode < 100 || query.Filters.StatusCode > 599) {
		return invalidQueryf("usage status filter is outside HTTP range")
	}
	if err := validateQueryVisibility(query.Visibility); err != nil {
		return invalidQuery(err)
	}
	return nil
}

func invalidQuery(err error) error {
	if err == nil || errors.Is(err, ErrInvalidQuery) {
		return err
	}
	return fmt.Errorf("%w: %w", ErrInvalidQuery, err)
}

func invalidQueryf(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalidQuery, fmt.Sprintf(format, args...))
}

func validateQueryVisibility(visibility QueryVisibility) error {
	if visibility.All {
		if len(visibility.TeamIDs) != 0 || len(visibility.UserIDs) != 0 || len(visibility.APIKeyIDs) != 0 {
			return invalidQueryf("unrestricted usage visibility cannot include restricted dimensions")
		}
		return nil
	}
	if len(visibility.TeamIDs)+len(visibility.UserIDs)+len(visibility.APIKeyIDs) == 0 {
		return invalidQueryf("usage visibility is required")
	}
	if len(visibility.TeamIDs)+len(visibility.UserIDs)+len(visibility.APIKeyIDs) > 10000 {
		return invalidQueryf("usage visibility exceeds 10000 authorized subjects")
	}
	seen := make(map[string]struct{})
	for label, values := range map[string][]string{
		"Team": visibility.TeamIDs, "User": visibility.UserIDs, "API key": visibility.APIKeyIDs,
	} {
		for _, value := range values {
			if _, err := uuid.Parse(value); err != nil {
				return invalidQueryf("usage visibility contains an invalid %s ID", label)
			}
			key := label + ":" + value
			if _, duplicate := seen[key]; duplicate {
				return invalidQueryf("usage visibility contains a duplicate %s ID", label)
			}
			seen[key] = struct{}{}
		}
	}
	return nil
}

func selectGrain(start, end time.Time, requested Grain) Grain {
	if requested != "" && requested != GrainAuto {
		return requested
	}
	rangeSize := end.Sub(start)
	if rangeSize <= 12*time.Hour {
		return GrainMinute
	}
	if rangeSize <= 31*24*time.Hour {
		return GrainHour
	}
	return GrainDay
}

func grainDuration(grain Grain) time.Duration {
	switch grain {
	case GrainMinute:
		return time.Minute
	case GrainHour:
		return time.Hour
	default:
		return 24 * time.Hour
	}
}

func rollupWhere(query UsageQuery, view RollupView) (string, []any) {
	args := []any{query.NamespaceID, query.Start, query.End, string(view)}
	clauses := []string{"namespace_id = $1", "bucket_start >= $2", "bucket_start < $3", "view = $4"}
	filters := []struct {
		key   string
		value string
	}{
		{"apiKeyId", query.Filters.APIKeyID},
		{"userId", query.Filters.UserID},
		{"teamId", query.Filters.TeamID},
		{"entrypointId", query.Filters.EntrypointID},
		{"recipeId", query.Filters.RecipeID},
		{"logicalModelId", query.Filters.LogicalModelID},
		{"backendId", query.Filters.BackendID},
		{"providerId", query.Filters.ProviderID},
		{"dispatchType", query.Filters.DispatchType},
		{"protocol", query.Filters.Protocol},
		{"errorCode", query.Filters.ErrorCode},
	}
	for _, filter := range filters {
		if filter.value == "" {
			continue
		}
		args = append(args, filter.value)
		clauses = append(clauses, fmt.Sprintf("dimensions->>'%s' = $%d", filter.key, len(args)))
	}
	if query.Filters.StatusCode != 0 {
		args = append(args, strconv.Itoa(query.Filters.StatusCode))
		clauses = append(clauses, fmt.Sprintf("dimensions->>'statusCode' = $%d", len(args)))
	}
	appendRollupVisibility(&clauses, &args, query.Visibility)
	return strings.Join(clauses, " AND "), args
}

func appendRollupVisibility(clauses *[]string, args *[]any, visibility QueryVisibility) {
	if visibility.All {
		return
	}
	parts := make([]string, 0, 3)
	for _, dimension := range []struct {
		key    string
		values []string
	}{
		{"teamId", visibility.TeamIDs},
		{"userId", visibility.UserIDs},
		{"apiKeyId", visibility.APIKeyIDs},
	} {
		if len(dimension.values) == 0 {
			continue
		}
		*args = append(*args, pq.Array(dimension.values))
		parts = append(parts, fmt.Sprintf("dimensions->>'%s' = ANY($%d::text[])", dimension.key, len(*args)))
	}
	*clauses = append(*clauses, "("+strings.Join(parts, " OR ")+")")
}

func publicTotals(
	requests, successful, input, output, incomplete string,
	costs []CostAggregate,
	timings timingPair,
) (UsageTotals, error) {
	inputValue, err := quota.ParseQuotaInteger(input)
	if err != nil {
		return UsageTotals{}, fmt.Errorf("%w: invalid query input total", ErrLedgerCorrupt)
	}
	outputValue, err := quota.ParseQuotaInteger(output)
	if err != nil {
		return UsageTotals{}, fmt.Errorf("%w: invalid query output total", ErrLedgerCorrupt)
	}
	total, err := inputValue.Add(outputValue)
	if err != nil {
		return UsageTotals{}, fmt.Errorf("%w: query token total overflows", ErrLedgerCorrupt)
	}
	incompleteValue, err := quota.ParseQuotaInteger(incomplete)
	if err != nil {
		return UsageTotals{}, fmt.Errorf("%w: invalid incomplete count", ErrLedgerCorrupt)
	}
	completeness := CompletenessComplete
	if !incompleteValue.IsZero() {
		if inputValue.IsZero() && outputValue.IsZero() {
			completeness = CompletenessUnknown
		} else {
			completeness = CompletenessPartial
		}
	}
	latency, err := timingSummary(timings.Latency)
	if err != nil {
		return UsageTotals{}, err
	}
	ttft, err := timingSummary(timings.TTFT)
	if err != nil {
		return UsageTotals{}, err
	}
	return UsageTotals{
		Requests: requests, SuccessfulRequests: successful, InputTokens: input,
		OutputTokens: output, TotalTokens: total.String(), IncompleteDispatches: incomplete,
		Completeness: completeness, Costs: publicCosts(costs), Latency: latency, TTFT: ttft,
	}, nil
}

func timingWithHistogram(count, sum string, histogram []quota.QuotaInteger) (timingAggregate, error) {
	value, err := parseTimingAggregate(count, sum)
	if err != nil {
		return timingAggregate{}, err
	}
	value.Histogram = histogram
	if len(value.Histogram) == 0 {
		value.Histogram = emptyTimingHistogram()
	}
	return value, nil
}

func publicCosts(costs []CostAggregate) []CostSummary {
	result := make([]CostSummary, 0, len(costs))
	for _, cost := range costs {
		completeness := CompletenessComplete
		if !cost.IncompleteDispatches.IsZero() {
			if cost.KnownDispatches.IsZero() {
				completeness = CompletenessUnknown
			} else {
				completeness = CompletenessPartial
			}
		}
		result = append(result, CostSummary{
			Currency: cost.Currency, KnownAmount: numeratorToDecimal(cost.KnownNumerator),
			Completeness: completeness, KnownDispatches: cost.KnownDispatches.String(),
			IncompleteDispatches: cost.IncompleteDispatches.String(),
		})
	}
	return result
}
