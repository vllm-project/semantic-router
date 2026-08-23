package usageledger

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
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
	AsOf            *time.Time     `json:"asOf"`
	LedgerWatermark *time.Time     `json:"ledgerWatermark"`
	IngestionLag    *time.Duration `json:"ingestionLag"`
	Final           bool           `json:"final"`
}

type SeriesPoint struct {
	BucketStart time.Time   `json:"bucketStart"`
	Totals      UsageTotals `json:"totals"`
}

type UsageSeries struct {
	Points          []SeriesPoint  `json:"points"`
	Grain           Grain          `json:"grain"`
	AsOf            *time.Time     `json:"asOf"`
	LedgerWatermark *time.Time     `json:"ledgerWatermark"`
	IngestionLag    *time.Duration `json:"ingestionLag"`
	Final           bool           `json:"final"`
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
	type pendingPoint struct {
		point                                           SeriesPoint
		watermark                                       time.Time
		requests, successful, input, output, incomplete string
		latency, ttft                                   timingAggregate
	}
	points := make([]pendingPoint, 0)
	var latest time.Time
	for rows.Next() {
		var value pendingPoint
		var latencyCount, latencySum, ttftCount, ttftSum string
		if err := rows.Scan(&value.point.BucketStart, &value.requests, &value.successful,
			&value.input, &value.output, &value.incomplete, &latencyCount, &latencySum,
			&ttftCount, &ttftSum, &value.watermark); err != nil {
			return UsageSeries{}, fmt.Errorf("scan usage series: %w", err)
		}
		value.latency, seriesErr = parseTimingAggregate(latencyCount, latencySum)
		if seriesErr != nil {
			return UsageSeries{}, seriesErr
		}
		value.ttft, seriesErr = parseTimingAggregate(ttftCount, ttftSum)
		if seriesErr != nil {
			return UsageSeries{}, seriesErr
		}
		if value.watermark.After(latest) {
			latest = value.watermark
		}
		points = append(points, value)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return UsageSeries{}, fmt.Errorf("iterate usage series: %w", err)
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

type LogCursorCodec struct {
	key []byte
}

func NewLogCursorCodec(key []byte) (*LogCursorCodec, error) {
	if len(key) < 32 {
		return nil, fmt.Errorf("request-log cursor HMAC key must contain at least 32 bytes")
	}
	return &LogCursorCodec{key: append([]byte(nil), key...)}, nil
}

// Close erases the process-owned cursor signing material. A codec must not be
// reused after Close; Management composition closes it before releasing its
// borrowed deployment keyrings.
func (codec *LogCursorCodec) Close() {
	if codec == nil {
		return
	}
	for index := range codec.key {
		codec.key[index] = 0
	}
	codec.key = nil
}

type LogQuery struct {
	NamespaceID string
	Start       time.Time
	End         time.Time
	Filters     UsageFilters
	Visibility  QueryVisibility
	PageSize    int
	Cursor      string
}

type RequestLog struct {
	AdmissionID         string            `json:"admissionId"`
	EventID             string            `json:"eventId"`
	OccurredAt          time.Time         `json:"occurredAt"`
	CompletedAt         time.Time         `json:"completedAt"`
	Protocol            string            `json:"protocol"`
	Path                string            `json:"path"`
	StatusCode          int               `json:"statusCode"`
	ErrorCode           string            `json:"errorCode,omitempty"`
	UsageState          UsageState        `json:"usageState"`
	InputTokens         string            `json:"inputTokens"`
	OutputTokens        string            `json:"outputTokens"`
	LatencyMilliseconds int64             `json:"latencyMilliseconds"`
	TTFTMilliseconds    *int64            `json:"ttftMilliseconds,omitempty"`
	Stream              bool              `json:"stream"`
	ToolCall            bool              `json:"toolCall"`
	APIKeyID            string            `json:"apiKeyId,omitempty"`
	UserID              string            `json:"userId,omitempty"`
	TeamID              string            `json:"teamId,omitempty"`
	EntrypointID        string            `json:"entrypointId,omitempty"`
	RecipeID            string            `json:"recipeId,omitempty"`
	Metadata            map[string]string `json:"metadata,omitempty"`
	Costs               []CostSummary     `json:"costs"`
}

type LogPage struct {
	Items      []RequestLog `json:"items"`
	NextCursor string       `json:"nextCursor,omitempty"`
}

type logCursor struct {
	Version     int    `json:"v"`
	NamespaceID string `json:"n"`
	QueryDigest string `json:"q"`
	OccurredAt  int64  `json:"t"`
	EventID     string `json:"e"`
}

func (q PostgresQueries) ListLogs(ctx context.Context, query LogQuery, codec *LogCursorCodec) (LogPage, error) {
	if q.DB == nil || codec == nil {
		return LogPage{}, fmt.Errorf("request-log database and cursor codec are required")
	}
	usage := UsageQuery{
		NamespaceID: query.NamespaceID, Start: query.Start, End: query.End,
		Grain: GrainDay, Filters: query.Filters, Visibility: query.Visibility,
	}
	if err := validateUsageQuery(usage); err != nil {
		return LogPage{}, err
	}
	if query.Filters.LogicalModelID != "" || query.Filters.BackendID != "" || query.Filters.ProviderID != "" || query.Filters.DispatchType != "" {
		return LogPage{}, invalidQueryf("raw request-log list does not accept internal dispatch filters")
	}
	if query.PageSize == 0 {
		query.PageSize = 50
	}
	if query.PageSize < 1 || query.PageSize > 200 {
		return LogPage{}, invalidQueryf("request-log page size must be between 1 and 200")
	}
	filterDigest := logFilterDigest(query)
	var cursor *logCursor
	if query.Cursor != "" {
		decoded, err := codec.decode(query.Cursor)
		if err != nil {
			return LogPage{}, invalidQuery(err)
		}
		if decoded.NamespaceID != query.NamespaceID || decoded.QueryDigest != filterDigest {
			return LogPage{}, invalidQueryf("request-log cursor does not belong to this query")
		}
		cursor = &decoded
	}
	statement, args := rawLogPageQuery(query, cursor)
	rows, listLogsErr := q.DB.QueryContext(ctx, statement, args...)
	if listLogsErr != nil {
		return LogPage{}, fmt.Errorf("list request logs: %w", listLogsErr)
	}
	defer rows.Close()
	items := make([]RequestLog, 0, query.PageSize+1)
	for rows.Next() {
		item, err := scanRequestLog(rows)
		if err != nil {
			return LogPage{}, err
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return LogPage{}, err
	}
	page := LogPage{Items: items}
	if len(items) > query.PageSize {
		last := items[query.PageSize-1]
		page.Items = items[:query.PageSize]
		page.NextCursor, listLogsErr = codec.encode(logCursor{
			Version: 1, NamespaceID: query.NamespaceID, QueryDigest: filterDigest,
			OccurredAt: last.OccurredAt.UnixNano(), EventID: last.EventID,
		})
		if listLogsErr != nil {
			return LogPage{}, listLogsErr
		}
	}
	return page, nil
}

func rawLogPageQuery(query LogQuery, cursor *logCursor) (string, []any) {
	where, args := rawLogWhere(query)
	if cursor != nil {
		args = append(args, time.Unix(0, cursor.OccurredAt).UTC(), cursor.EventID)
		where += fmt.Sprintf(" AND (occurred_at, event_id) < ($%d, $%d::uuid)", len(args)-1, len(args))
	}
	args = append(args, query.PageSize+1)
	return fmt.Sprintf(`SELECT admission_id, event_id::text, occurred_at, protocol, path,
  status_code, COALESCE(error_code,''), usage_state, input_tokens::text, output_tokens::text,
  latency_ms, ttft_ms, COALESCE(api_key_id::text,''), COALESCE(user_id::text,''), COALESCE(team_id::text,''),
  COALESCE(entrypoint_id::text,''), COALESCE(recipe_id::text,''), costs, request_metadata
FROM usage_events WHERE %s
ORDER BY occurred_at DESC, event_id DESC LIMIT $%d`, where, len(args)), args
}

type rowScanner interface {
	Scan(...any) error
}

func scanRequestLog(row rowScanner) (RequestLog, error) {
	item, _, err := scanRequestLogWithMetadata(row)
	return item, err
}

func scanRequestLogWithMetadata(row rowScanner) (RequestLog, safeEventMetadata, error) {
	var item RequestLog
	var costsJSON, metadataJSON []byte
	if err := row.Scan(&item.AdmissionID, &item.EventID, &item.OccurredAt, &item.Protocol,
		&item.Path, &item.StatusCode, &item.ErrorCode, &item.UsageState, &item.InputTokens,
		&item.OutputTokens, &item.LatencyMilliseconds, &item.TTFTMilliseconds, &item.APIKeyID, &item.UserID,
		&item.TeamID, &item.EntrypointID, &item.RecipeID, &costsJSON, &metadataJSON); err != nil {
		return RequestLog{}, safeEventMetadata{}, fmt.Errorf("scan request log: %w", err)
	}
	var costs []storedCost
	if err := json.Unmarshal(costsJSON, &costs); err != nil {
		return RequestLog{}, safeEventMetadata{}, fmt.Errorf("%w: decode request-log costs", ErrLedgerCorrupt)
	}
	aggregates := make([]CostAggregate, 0, len(costs))
	for _, value := range costs {
		cost, err := parseCostAggregate(value.Currency, value.KnownNumerator, value.KnownDispatches, value.IncompleteDispatches)
		if err != nil {
			return RequestLog{}, safeEventMetadata{}, err
		}
		aggregates = append(aggregates, cost)
	}
	item.Costs = publicCosts(aggregates)
	var metadata safeEventMetadata
	if err := json.Unmarshal(metadataJSON, &metadata); err != nil {
		return RequestLog{}, safeEventMetadata{}, fmt.Errorf("%w: decode request-log metadata", ErrLedgerCorrupt)
	}
	item.CompletedAt = metadata.CompletedAt
	item.Stream = metadata.Stream
	item.ToolCall = metadata.ToolCall
	item.Metadata = metadata.Metadata
	return item, metadata, nil
}

func rawLogWhere(query LogQuery) (string, []any) {
	partitionStart := query.Start.UTC().Truncate(24 * time.Hour)
	partitionEnd := query.End.UTC().Truncate(24 * time.Hour)
	if !query.End.UTC().Equal(partitionEnd) {
		partitionEnd = partitionEnd.Add(24 * time.Hour)
	}
	args := []any{query.NamespaceID, query.Start, query.End, partitionStart, partitionEnd}
	clauses := []string{
		"namespace_id = $1", "occurred_at >= $2", "occurred_at < $3",
		"event_date >= $4::date", "event_date < $5::date",
		"event_kind IN ('actual','unknown')",
	}
	filters := []struct {
		column string
		value  string
		cast   string
	}{
		{"api_key_id", query.Filters.APIKeyID, "uuid"},
		{"user_id", query.Filters.UserID, "uuid"},
		{"team_id", query.Filters.TeamID, "uuid"},
		{"entrypoint_id", query.Filters.EntrypointID, "text"},
		{"recipe_id", query.Filters.RecipeID, "text"},
		{"protocol", query.Filters.Protocol, "text"},
		{"error_code", query.Filters.ErrorCode, "text"},
	}
	for _, filter := range filters {
		if filter.value == "" {
			continue
		}
		args = append(args, filter.value)
		clauses = append(clauses, fmt.Sprintf("%s = $%d::%s", filter.column, len(args), filter.cast))
	}
	if query.Filters.StatusCode != 0 {
		args = append(args, query.Filters.StatusCode)
		clauses = append(clauses, fmt.Sprintf("status_code = $%d", len(args)))
	}
	appendRawLogVisibility(&clauses, &args, query.Visibility)
	return strings.Join(clauses, " AND "), args
}

func appendRawLogVisibility(clauses *[]string, args *[]any, visibility QueryVisibility) {
	if visibility.All {
		return
	}
	parts := make([]string, 0, 3)
	for _, dimension := range []struct {
		column string
		values []string
	}{
		{"team_id", visibility.TeamIDs},
		{"user_id", visibility.UserIDs},
		{"api_key_id", visibility.APIKeyIDs},
	} {
		if len(dimension.values) == 0 {
			continue
		}
		*args = append(*args, pq.Array(dimension.values))
		parts = append(parts, fmt.Sprintf("%s = ANY($%d::uuid[])", dimension.column, len(*args)))
	}
	*clauses = append(*clauses, "("+strings.Join(parts, " OR ")+")")
}

func logFilterDigest(query LogQuery) string {
	payload, _ := json.Marshal(struct {
		Start      int64           `json:"start"`
		End        int64           `json:"end"`
		Filters    UsageFilters    `json:"filters"`
		Visibility QueryVisibility `json:"visibility"`
	}{Start: query.Start.UnixNano(), End: query.End.UnixNano(), Filters: query.Filters, Visibility: query.Visibility})
	digest := sha256.Sum256(payload)
	return hex.EncodeToString(digest[:])
}

func (c *LogCursorCodec) encode(value logCursor) (string, error) {
	if c == nil || len(c.key) < sha256.Size {
		return "", fmt.Errorf("request-log cursor codec is closed")
	}
	payload, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	mac := hmac.New(sha256.New, c.key)
	_, _ = mac.Write(payload)
	return base64.RawURLEncoding.EncodeToString(payload) + "." + base64.RawURLEncoding.EncodeToString(mac.Sum(nil)), nil
}

func (c *LogCursorCodec) decode(encoded string) (logCursor, error) {
	if c == nil || len(c.key) < sha256.Size {
		return logCursor{}, fmt.Errorf("request-log cursor codec is closed")
	}
	payloadPart, signaturePart, ok := strings.Cut(encoded, ".")
	if !ok || len(encoded) > 2048 {
		return logCursor{}, fmt.Errorf("request-log cursor is malformed")
	}
	payload, err := base64.RawURLEncoding.DecodeString(payloadPart)
	if err != nil || base64.RawURLEncoding.EncodeToString(payload) != payloadPart {
		return logCursor{}, fmt.Errorf("request-log cursor is malformed")
	}
	signature, err := base64.RawURLEncoding.DecodeString(signaturePart)
	if err != nil || len(signature) != sha256.Size ||
		base64.RawURLEncoding.EncodeToString(signature) != signaturePart {
		return logCursor{}, fmt.Errorf("request-log cursor is malformed")
	}
	mac := hmac.New(sha256.New, c.key)
	_, _ = mac.Write(payload)
	if !hmac.Equal(signature, mac.Sum(nil)) {
		return logCursor{}, fmt.Errorf("request-log cursor signature is invalid")
	}
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.DisallowUnknownFields()
	var value logCursor
	if err := decoder.Decode(&value); err != nil || value.Version != 1 || value.OccurredAt <= 0 {
		return logCursor{}, fmt.Errorf("request-log cursor is invalid")
	}
	if _, err := uuid.Parse(value.NamespaceID); err != nil {
		return logCursor{}, fmt.Errorf("request-log cursor is invalid")
	}
	if _, err := uuid.Parse(value.EventID); err != nil {
		return logCursor{}, fmt.Errorf("request-log cursor is invalid")
	}
	if !isHexDigest(value.QueryDigest) {
		return logCursor{}, fmt.Errorf("request-log cursor is invalid")
	}
	return value, nil
}
