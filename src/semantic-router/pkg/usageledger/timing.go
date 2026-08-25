package usageledger

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

// timingBucketUpperBoundsMilliseconds is a stable, mergeable histogram
// contract. The final bucket covers every valid int64 duration. Bounds are
// intentionally denser around interactive response times while remaining
// compact enough to store for each dimension tuple.
var timingBucketUpperBoundsMilliseconds = [...]int64{
	0, 1, 2, 4, 8, 16, 32, 64, 125, 250, 500, 750,
	1_000, 1_500, 2_000, 3_000, 5_000, 7_500, 10_000,
	15_000, 20_000, 30_000, 45_000, 60_000, 90_000, 120_000,
	180_000, 300_000, 600_000, 1_200_000, 3_600_000, math.MaxInt64,
}

type timingAggregate struct {
	Count           quota.QuotaInteger
	SumMilliseconds quota.QuotaInteger
	Histogram       []quota.QuotaInteger
}

type timingPair struct {
	Latency timingAggregate
	TTFT    timingAggregate
}

type TimingSummary struct {
	SampleCount             string  `json:"sampleCount"`
	TotalMilliseconds       string  `json:"totalMilliseconds"`
	AverageMilliseconds     float64 `json:"averageMilliseconds"`
	P50Milliseconds         int64   `json:"p50Milliseconds"`
	P95Milliseconds         int64   `json:"p95Milliseconds"`
	P99Milliseconds         int64   `json:"p99Milliseconds"`
	PercentilesAreEstimated bool    `json:"percentilesAreEstimated"`
}

type storedHistogram struct {
	Counts []string `json:"counts"`
}

func emptyTimingHistogram() []quota.QuotaInteger {
	return make([]quota.QuotaInteger, len(timingBucketUpperBoundsMilliseconds))
}

func parseTimingAggregate(count, sum string) (timingAggregate, error) {
	countValue, err := quota.ParseQuotaInteger(count)
	if err != nil {
		return timingAggregate{}, fmt.Errorf("%w: invalid timing sample count", ErrLedgerCorrupt)
	}
	sumValue, err := quota.ParseQuotaInteger(sum)
	if err != nil {
		return timingAggregate{}, fmt.Errorf("%w: invalid timing sum", ErrLedgerCorrupt)
	}
	return timingAggregate{
		Count: countValue, SumMilliseconds: sumValue, Histogram: emptyTimingHistogram(),
	}, nil
}

func setTimingHistogramBucket(histogram []quota.QuotaInteger, index int, count string) error {
	if len(histogram) != len(timingBucketUpperBoundsMilliseconds) || index < 0 || index >= len(histogram) {
		return fmt.Errorf("%w: timing histogram bucket is outside the contract", ErrLedgerCorrupt)
	}
	parsed, err := quota.ParseQuotaInteger(count)
	if err != nil {
		return fmt.Errorf("%w: invalid timing histogram count", ErrLedgerCorrupt)
	}
	histogram[index] = parsed
	return nil
}

func encodeTimingHistogram(values []quota.QuotaInteger) ([]byte, error) {
	if len(values) == 0 {
		values = emptyTimingHistogram()
	}
	if len(values) != len(timingBucketUpperBoundsMilliseconds) {
		return nil, fmt.Errorf("%w: timing histogram has %d buckets, want %d",
			ErrLedgerCorrupt, len(values), len(timingBucketUpperBoundsMilliseconds))
	}
	counts := make([]string, len(values))
	for index, value := range values {
		counts[index] = value.String()
	}
	return json.Marshal(storedHistogram{Counts: counts})
}

func decodeTimingHistogram(payload []byte) ([]quota.QuotaInteger, error) {
	if len(payload) == 0 || string(payload) == "[]" {
		return emptyTimingHistogram(), nil
	}
	var stored storedHistogram
	if err := json.Unmarshal(payload, &stored); err != nil {
		return nil, fmt.Errorf("%w: decode timing histogram", ErrLedgerCorrupt)
	}
	if len(stored.Counts) != len(timingBucketUpperBoundsMilliseconds) {
		return nil, fmt.Errorf("%w: timing histogram has %d buckets, want %d",
			ErrLedgerCorrupt, len(stored.Counts), len(timingBucketUpperBoundsMilliseconds))
	}
	result := make([]quota.QuotaInteger, len(stored.Counts))
	for index, value := range stored.Counts {
		parsed, err := quota.ParseQuotaInteger(value)
		if err != nil {
			return nil, fmt.Errorf("%w: invalid timing bucket %d", ErrLedgerCorrupt, index)
		}
		result[index] = parsed
	}
	return result, nil
}

func timingSummary(value timingAggregate) (TimingSummary, error) {
	result := TimingSummary{
		SampleCount: value.Count.String(), TotalMilliseconds: value.SumMilliseconds.String(),
		PercentilesAreEstimated: true,
	}
	if value.Count.IsZero() {
		return result, nil
	}
	count, ok := new(big.Int).SetString(value.Count.String(), 10)
	if !ok {
		return TimingSummary{}, fmt.Errorf("%w: invalid timing sample count", ErrLedgerCorrupt)
	}
	sum, ok := new(big.Int).SetString(value.SumMilliseconds.String(), 10)
	if !ok {
		return TimingSummary{}, fmt.Errorf("%w: invalid timing sum", ErrLedgerCorrupt)
	}
	average, _ := new(big.Rat).SetFrac(sum, count).Float64()
	result.AverageMilliseconds = average
	if len(value.Histogram) != len(timingBucketUpperBoundsMilliseconds) {
		return TimingSummary{}, fmt.Errorf("%w: timing histogram shape does not match contract", ErrLedgerCorrupt)
	}
	var err error
	result.P50Milliseconds, err = timingPercentile(value.Histogram, value.Count, 50)
	if err != nil {
		return TimingSummary{}, err
	}
	result.P95Milliseconds, err = timingPercentile(value.Histogram, value.Count, 95)
	if err != nil {
		return TimingSummary{}, err
	}
	result.P99Milliseconds, err = timingPercentile(value.Histogram, value.Count, 99)
	return result, err
}

func timingPercentile(histogram []quota.QuotaInteger, count quota.QuotaInteger, percentile uint64) (int64, error) {
	// ceil(count*percentile/100) expressed with the exact decimal integer so
	// the percentile boundary stays correct even for very large aggregates.
	countValue, ok := new(big.Int).SetString(count.String(), 10)
	if !ok {
		return 0, fmt.Errorf("%w: invalid timing sample count", ErrLedgerCorrupt)
	}
	numerator := new(big.Int).Mul(countValue, new(big.Int).SetUint64(percentile))
	numerator.Add(numerator, big.NewInt(99))
	numerator.Quo(numerator, big.NewInt(100))
	target, err := quota.ParseQuotaInteger(numerator.String())
	if err != nil {
		return 0, fmt.Errorf("%w: timing percentile rank is invalid", ErrLedgerCorrupt)
	}
	cumulative := quota.QuotaInteger{}
	for index, bucket := range histogram {
		cumulative, err = cumulative.Add(bucket)
		if err != nil {
			return 0, fmt.Errorf("%w: timing histogram overflows", ErrLedgerCorrupt)
		}
		if cumulative.Compare(target) >= 0 {
			return timingBucketUpperBoundsMilliseconds[index], nil
		}
	}
	return 0, fmt.Errorf("%w: timing histogram contains fewer samples than its count", ErrLedgerCorrupt)
}
