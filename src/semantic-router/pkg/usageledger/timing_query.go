package usageledger

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

func (q PostgresQueries) timingHistograms(
	ctx context.Context,
	table, where string,
	args []any,
) (timingPair, error) {
	result := emptyTimingPair()
	for _, metric := range timingMetrics() {
		statement := fmt.Sprintf(`SELECT histogram.ordinality::integer - 1,
  sum(histogram.value::numeric)::text
FROM %s r
CROSS JOIN LATERAL jsonb_array_elements_text(
  CASE WHEN jsonb_typeof(r.%s) = 'object' THEN r.%s->'counts' ELSE '[]'::jsonb END
) WITH ORDINALITY AS histogram(value, ordinality)
WHERE %s GROUP BY 1 ORDER BY 1`, table, metric.column, metric.column, where)
		rows, err := q.DB.QueryContext(ctx, statement, args...)
		if err != nil {
			return timingPair{}, fmt.Errorf("query %s histogram: %w", metric.name, err)
		}
		for rows.Next() {
			var index int
			var count string
			if err := rows.Scan(&index, &count); err != nil {
				rows.Close()
				return timingPair{}, fmt.Errorf("scan %s histogram: %w", metric.name, err)
			}
			if err := setTimingHistogramBucket(metric.selectHistogram(&result), index, count); err != nil {
				rows.Close()
				return timingPair{}, err
			}
		}
		if err := closeTimingRows(rows); err != nil {
			return timingPair{}, err
		}
	}
	return result, nil
}

func (q PostgresQueries) timingHistogramsByBucket(
	ctx context.Context,
	table, where string,
	args []any,
	bucketExpression string,
) (map[string]timingPair, error) {
	result := make(map[string]timingPair)
	for _, metric := range timingMetrics() {
		statement := fmt.Sprintf(`SELECT %s AS bucket_start, histogram.ordinality::integer - 1,
  sum(histogram.value::numeric)::text
FROM %s r
CROSS JOIN LATERAL jsonb_array_elements_text(
  CASE WHEN jsonb_typeof(r.%s) = 'object' THEN r.%s->'counts' ELSE '[]'::jsonb END
) WITH ORDINALITY AS histogram(value, ordinality)
WHERE %s GROUP BY %s, histogram.ordinality ORDER BY %s, histogram.ordinality`,
			bucketExpression, table, metric.column, metric.column, where, bucketExpression, bucketExpression)
		rows, err := q.DB.QueryContext(ctx, statement, args...)
		if err != nil {
			return nil, fmt.Errorf("query bucketed %s histogram: %w", metric.name, err)
		}
		for rows.Next() {
			var bucket time.Time
			var index int
			var count string
			if err := rows.Scan(&bucket, &index, &count); err != nil {
				rows.Close()
				return nil, fmt.Errorf("scan bucketed %s histogram: %w", metric.name, err)
			}
			key := timingBucketKey(bucket)
			pair := result[key]
			ensureTimingPairHistograms(&pair)
			if err := setTimingHistogramBucket(metric.selectHistogram(&pair), index, count); err != nil {
				rows.Close()
				return nil, err
			}
			result[key] = pair
		}
		if err := closeTimingRows(rows); err != nil {
			return nil, err
		}
	}
	return result, nil
}

func (q PostgresQueries) timingHistogramsByValue(
	ctx context.Context,
	table, where string,
	args []any,
	valueExpression string,
) (map[string]timingPair, error) {
	result := make(map[string]timingPair)
	for _, metric := range timingMetrics() {
		statement := fmt.Sprintf(`SELECT %s AS dimension_value, histogram.ordinality::integer - 1,
  sum(histogram.value::numeric)::text
FROM %s r
CROSS JOIN LATERAL jsonb_array_elements_text(
  CASE WHEN jsonb_typeof(r.%s) = 'object' THEN r.%s->'counts' ELSE '[]'::jsonb END
) WITH ORDINALITY AS histogram(value, ordinality)
WHERE %s GROUP BY %s, histogram.ordinality ORDER BY %s, histogram.ordinality`,
			valueExpression, table, metric.column, metric.column, where, valueExpression, valueExpression)
		rows, err := q.DB.QueryContext(ctx, statement, args...)
		if err != nil {
			return nil, fmt.Errorf("query grouped %s histogram: %w", metric.name, err)
		}
		for rows.Next() {
			var value string
			var index int
			var count string
			if err := rows.Scan(&value, &index, &count); err != nil {
				rows.Close()
				return nil, fmt.Errorf("scan grouped %s histogram: %w", metric.name, err)
			}
			pair := result[value]
			ensureTimingPairHistograms(&pair)
			if err := setTimingHistogramBucket(metric.selectHistogram(&pair), index, count); err != nil {
				rows.Close()
				return nil, err
			}
			result[value] = pair
		}
		if err := closeTimingRows(rows); err != nil {
			return nil, err
		}
	}
	return result, nil
}

type timingMetric struct {
	name   string
	column string
	ttft   bool
}

func timingMetrics() []timingMetric {
	return []timingMetric{
		{name: "latency", column: "latency_histogram"},
		{name: "TTFT", column: "ttft_histogram", ttft: true},
	}
}

func (metric timingMetric) selectHistogram(pair *timingPair) []quota.QuotaInteger {
	if metric.ttft {
		return pair.TTFT.Histogram
	}
	return pair.Latency.Histogram
}

func emptyTimingPair() timingPair {
	return timingPair{
		Latency: timingAggregate{Histogram: emptyTimingHistogram()},
		TTFT:    timingAggregate{Histogram: emptyTimingHistogram()},
	}
}

func ensureTimingPairHistograms(pair *timingPair) {
	if len(pair.Latency.Histogram) == 0 {
		pair.Latency.Histogram = emptyTimingHistogram()
	}
	if len(pair.TTFT.Histogram) == 0 {
		pair.TTFT.Histogram = emptyTimingHistogram()
	}
}

func timingBucketKey(value time.Time) string {
	return value.UTC().Format(time.RFC3339Nano)
}

func closeTimingRows(rows *sql.Rows) error {
	if err := rows.Err(); err != nil {
		rows.Close()
		return err
	}
	if err := rows.Close(); err != nil {
		return err
	}
	return nil
}
