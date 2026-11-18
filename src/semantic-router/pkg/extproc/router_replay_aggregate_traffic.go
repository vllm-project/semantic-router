package extproc

import (
	"fmt"
	"math"
	"math/rand/v2"
	"net/url"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const (
	// maxRouterReplayTrafficBuckets bounds a traffic series, and with it the
	// accumulator's memory.
	maxRouterReplayTrafficBuckets = 1000
	// replayTrafficLatencySamples bounds the durations one bucket keeps for its
	// percentiles, and replayTrafficWindowLatencySamples those the whole window
	// keeps; past either a uniform reservoir sample stands in for the rest.
	replayTrafficLatencySamples       = 256
	replayTrafficWindowLatencySamples = 1024
)

// routerReplayTrafficQuery asks an aggregate for its records in contiguous
// intervals from since to until.
type routerReplayTrafficQuery struct {
	since    time.Time
	interval time.Duration
	until    time.Time
}

// parseRouterReplayTrafficQuery reads the aggregate's interval. Without one
// the aggregate has no traffic series; with one it needs since, so the series
// covers a bounded window ending now.
func parseRouterReplayTrafficQuery(values url.Values, since, now time.Time) (*routerReplayTrafficQuery, error) {
	raw := strings.TrimSpace(values.Get("interval"))
	if raw == "" {
		return nil, nil
	}
	interval, err := time.ParseDuration(raw)
	if err != nil || interval < time.Minute {
		return nil, fmt.Errorf("interval must be a duration of at least 1m, such as 5m or 1h")
	}
	if since.IsZero() {
		return nil, fmt.Errorf("interval needs since: the series runs from since to now")
	}
	if buckets := routerReplayTrafficBucketCount(since, now, interval); buckets > maxRouterReplayTrafficBuckets {
		return nil, fmt.Errorf("interval %s splits the window into %d intervals; use one that gives at most %d", raw, buckets, maxRouterReplayTrafficBuckets)
	}
	return &routerReplayTrafficQuery{since: since.UTC(), interval: interval, until: now.UTC()}, nil
}

// routerReplayTrafficBucketCount is the number of intervals from since to
// until, at least one. Time.Sub saturates at the largest Duration for a
// distant since, so the ceiling divides before it adds, and the count is
// clamped to what an int holds on every platform.
func routerReplayTrafficBucketCount(since, until time.Time, interval time.Duration) int {
	if !until.After(since) || interval <= 0 {
		return 1
	}
	span := until.Sub(since)
	count := int64(span / interval)
	if span%interval != 0 {
		count++
	}
	return int(min(count, math.MaxInt32))
}

// cacheKey extends the filters' key, so a series and a plain aggregate of the
// same records are cached apart.
func (query *routerReplayTrafficQuery) cacheKey(filtersKey string) string {
	if query == nil {
		return filtersKey
	}
	values, _ := url.ParseQuery(filtersKey)
	values.Set("interval", query.interval.String())
	return values.Encode()
}

// replayTrafficAccumulator counts records into intervals as a scan streams
// them, keeping per bucket only counts, a cost accumulator and a bounded
// latency sample, and one more bounded sample for the whole window.
type replayTrafficAccumulator struct {
	query   routerReplayTrafficQuery
	buckets map[int]*replayTrafficBucket
	window  latencySample
	sampler *rand.Rand
}

type replayTrafficBucket struct {
	counts    routerreplay.TrafficBucket
	costs     *replayCostAccumulator
	latencies latencySample
}

// latencySample keeps every duration up to its bound, then replaces kept ones
// with the probability that keeps the sample uniform over all it was offered.
type latencySample struct {
	values  []int64
	offered int
}

func (s *latencySample) add(duration int64, bound int, sampler *rand.Rand) {
	s.offered++
	if len(s.values) < bound {
		s.values = append(s.values, duration)
		return
	}
	if slot := sampler.IntN(s.offered); slot < bound {
		s.values[slot] = duration
	}
}

func newReplayTrafficAccumulator(query *routerReplayTrafficQuery) *replayTrafficAccumulator {
	if query == nil {
		return nil
	}
	return &replayTrafficAccumulator{
		query:   *query,
		buckets: make(map[int]*replayTrafficBucket),
		// A fixed seed keeps a sampled series reproducible for the same records.
		sampler: rand.New(rand.NewPCG(1, 2)),
	}
}

func (a *replayTrafficAccumulator) add(record routerreplay.RoutingRecord) {
	if a == nil || record.Timestamp.Before(a.query.since) {
		return
	}
	index := int(record.Timestamp.Sub(a.query.since) / a.query.interval)
	if index >= maxRouterReplayTrafficBuckets {
		return
	}
	bucket := a.buckets[index]
	if bucket == nil {
		bucket = &replayTrafficBucket{costs: newReplayCostAccumulator()}
		a.buckets[index] = bucket
	}
	bucket.counts.Requests++
	if record.FromCache {
		bucket.counts.Cached++
	}
	if record.LifecycleState == routerreplay.LifecycleFailed || record.LifecycleState == routerreplay.LifecycleAborted {
		bucket.counts.Failed++
	}
	bucket.costs.add(record)
	if record.LifecycleState == routerreplay.LifecycleCompleted && record.DurationMS > 0 {
		bucket.latencies.add(record.DurationMS, replayTrafficLatencySamples, a.sampler)
		a.window.add(record.DurationMS, replayTrafficWindowLatencySamples, a.sampler)
	}
}

func (a *replayTrafficAccumulator) result() *routerreplay.TrafficSeries {
	if a == nil {
		return nil
	}
	count := routerReplayTrafficBucketCount(a.query.since, a.query.until, a.query.interval)
	for index := range a.buckets {
		count = max(count, index+1)
	}
	buckets := make([]routerreplay.TrafficBucket, count)
	for index := range buckets {
		bucket := routerreplay.TrafficBucket{Cost: []routerreplay.CurrencyCostSummary{}}
		if kept := a.buckets[index]; kept != nil {
			bucket = kept.counts
			bucket.LatencyP50MS = latencyPercentile(kept.latencies.values, 0.50)
			bucket.LatencyP95MS = latencyPercentile(kept.latencies.values, 0.95)
			bucket.Cost = kept.costs.result().ByCurrency
			if bucket.Cost == nil {
				bucket.Cost = []routerreplay.CurrencyCostSummary{}
			}
		}
		bucket.Start = a.query.since.Add(time.Duration(index) * a.query.interval)
		buckets[index] = bucket
	}
	return &routerreplay.TrafficSeries{
		Since:           a.query.since,
		IntervalSeconds: int64(a.query.interval / time.Second),
		LatencyP50MS:    latencyPercentile(a.window.values, 0.50),
		LatencyP95MS:    latencyPercentile(a.window.values, 0.95),
		Buckets:         buckets,
	}
}
