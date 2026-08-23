package usageledger

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
)

func TestTimingSummaryUsesMergeableHistogram(t *testing.T) {
	count, _ := quota.ParseQuotaInteger("100")
	sum, _ := quota.ParseQuotaInteger("123400")
	histogram := emptyTimingHistogram()
	histogram[timingBucketIndex(t, 500)] = mustTimingQuantity(t, "94")
	histogram[timingBucketIndex(t, 1000)] = mustTimingQuantity(t, "6")

	summary, err := timingSummary(timingAggregate{
		Count: count, SumMilliseconds: sum, Histogram: histogram,
	})
	if err != nil {
		t.Fatal(err)
	}
	if summary.AverageMilliseconds != 1234 || summary.P50Milliseconds != 500 ||
		summary.P95Milliseconds != 1000 || summary.P99Milliseconds != 1000 {
		t.Fatalf("timing summary = %+v", summary)
	}
	if !summary.PercentilesAreEstimated {
		t.Fatal("histogram percentiles must be identified as estimates")
	}
}

func TestTimingHistogramStorageRoundTripsExactly(t *testing.T) {
	histogram := emptyTimingHistogram()
	histogram[0] = mustTimingQuantity(t, "7")
	histogram[len(histogram)-1] = mustTimingQuantity(t, "999999999999999999999999999999999999999999")
	payload, err := encodeTimingHistogram(histogram)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := decodeTimingHistogram(payload)
	if err != nil {
		t.Fatal(err)
	}
	for index := range histogram {
		if histogram[index] != decoded[index] {
			t.Fatalf("bucket %d = %s, want %s", index, decoded[index].String(), histogram[index].String())
		}
	}
}

func TestTimingPercentileUsesCeilingRank(t *testing.T) {
	histogram := emptyTimingHistogram()
	histogram[timingBucketIndex(t, 64)] = mustTimingQuantity(t, "1")
	value, err := timingPercentile(histogram, mustTimingQuantity(t, "1"), 95)
	if err != nil {
		t.Fatal(err)
	}
	if value != 64 {
		t.Fatalf("single-sample P95 = %d, want 64", value)
	}
}

func timingBucketIndex(t *testing.T, upper int64) int {
	t.Helper()
	for index, value := range timingBucketUpperBoundsMilliseconds {
		if value == upper {
			return index
		}
	}
	t.Fatalf("timing bucket %d does not exist", upper)
	return 0
}

func mustTimingQuantity(t *testing.T, value string) quota.QuotaInteger {
	t.Helper()
	parsed, err := quota.ParseQuotaInteger(value)
	if err != nil {
		t.Fatal(err)
	}
	return parsed
}
