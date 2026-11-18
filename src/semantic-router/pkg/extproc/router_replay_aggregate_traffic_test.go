package extproc

import (
	"context"
	"encoding/json"
	"math"
	"net/url"
	"strings"
	"testing"
	"time"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestParseRouterReplayTrafficQueryBoundsTheSeries(t *testing.T) {
	now := time.Date(2026, 11, 18, 12, 0, 0, 0, time.UTC)
	since := now.Add(-24 * time.Hour)
	for _, tc := range []struct {
		name, interval string
		since          time.Time
		wantErr        string
	}{
		{name: "absent", since: since},
		{name: "hourly", interval: "1h", since: since},
		{name: "unparsable", interval: "hourly", since: since, wantErr: "at least 1m"},
		{name: "too fine", interval: "30s", since: since, wantErr: "at least 1m"},
		{name: "unbounded", interval: "1h", wantErr: "needs since"},
		{name: "too many", interval: "1m", since: now.Add(-30 * 24 * time.Hour), wantErr: "at most 1000"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			values := url.Values{}
			if tc.interval != "" {
				values.Set("interval", tc.interval)
			}
			query, err := parseRouterReplayTrafficQuery(values, tc.since, now)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("err = %v, want one naming %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.interval == "" && query != nil {
				t.Fatalf("no interval asked for a series: %+v", query)
			}
			if tc.interval == "1h" && (query == nil || query.interval != time.Hour || !query.until.Equal(now)) {
				t.Fatalf("query = %+v", query)
			}
		})
	}
}

func TestRouterReplayTrafficBucketCountStaysPositiveAtTheBounds(t *testing.T) {
	now := time.Date(2026, 11, 18, 12, 0, 0, 0, time.UTC)
	distant := time.Time{}.Add(time.Nanosecond)
	for _, tc := range []struct {
		name     string
		since    time.Time
		interval time.Duration
		want     int
	}{
		{name: "exact multiple", since: now.Add(-1000 * time.Minute), interval: time.Minute, want: 1000},
		{name: "one nanosecond over", since: now.Add(-1000*time.Minute - time.Nanosecond), interval: time.Minute, want: 1001},
		{name: "since in the future", since: now.Add(time.Hour), interval: time.Minute, want: 1},
		{name: "saturated span", since: distant, interval: time.Minute, want: int(time.Duration(math.MaxInt64)/time.Minute) + 1},
		{name: "clamped count", since: distant, interval: time.Nanosecond, want: math.MaxInt32},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := routerReplayTrafficBucketCount(tc.since, now, tc.interval); got != tc.want {
				t.Fatalf("count = %d, want %d", got, tc.want)
			}
		})
	}
	// A distant since is refused before any scan allocates its buckets.
	if _, err := parseRouterReplayTrafficQuery(url.Values{"interval": {"1m"}}, distant, now); err == nil || !strings.Contains(err.Error(), "at most 1000") {
		t.Fatalf("err = %v, want the series bound", err)
	}
}

func trafficTestRecord(id string, at time.Time, state string, durationMS int64, cost float64) routerreplay.RoutingRecord {
	record := routerreplay.RoutingRecord{
		ID: id, Timestamp: at, LifecycleState: state, DurationMS: durationMS,
		Recipe: "default", Decision: "fast", SelectedModel: "qwen3-8b",
	}
	if cost > 0 {
		currency := "USD"
		record.ActualCost, record.Currency = &cost, &currency
	}
	return record
}

func TestReplayTrafficAccumulatorCountsContiguousIntervals(t *testing.T) {
	since := time.Date(2026, 11, 18, 8, 0, 0, 0, time.UTC)
	until := since.Add(4*time.Hour - time.Minute)
	accumulator := newReplayTrafficAccumulator(&routerReplayTrafficQuery{since: since, interval: time.Hour, until: until})

	cached := trafficTestRecord("cached", since.Add(5*time.Minute), routerreplay.LifecycleCompleted, 40, 0)
	cached.FromCache = true
	for _, record := range []routerreplay.RoutingRecord{
		trafficTestRecord("before", since.Add(-time.Minute), routerreplay.LifecycleCompleted, 900, 0.5),
		cached,
		trafficTestRecord("a", since.Add(10*time.Minute), routerreplay.LifecycleCompleted, 100, 0.01),
		trafficTestRecord("b", since.Add(20*time.Minute), routerreplay.LifecycleCompleted, 300, 0.02),
		trafficTestRecord("failed", since.Add(30*time.Minute), routerreplay.LifecycleFailed, 5000, 0),
		trafficTestRecord("late", since.Add(3*time.Hour+30*time.Minute), routerreplay.LifecycleInProgress, 0, 0),
	} {
		accumulator.add(record)
	}

	series := accumulator.result()
	if series.IntervalSeconds != 3600 || !series.Since.Equal(since) || len(series.Buckets) != 4 {
		t.Fatalf("series = %+v", series)
	}
	if series.LatencyP50MS == nil || *series.LatencyP50MS != 100 || series.LatencyP95MS == nil || *series.LatencyP95MS != 300 {
		t.Fatalf("window latency = %v / %v, want 100 / 300 from the completed records in it", series.LatencyP50MS, series.LatencyP95MS)
	}
	first := series.Buckets[0]
	if first.Requests != 4 || first.Cached != 1 || first.Failed != 1 {
		t.Fatalf("first bucket counts = %+v", first)
	}
	if first.LatencyP50MS == nil || *first.LatencyP50MS != 100 || first.LatencyP95MS == nil || *first.LatencyP95MS != 300 {
		t.Fatalf("first bucket latency = %v / %v, want 100 / 300 from the completed records", first.LatencyP50MS, first.LatencyP95MS)
	}
	if len(first.Cost) != 1 || first.Cost[0].Currency != "USD" || first.Cost[0].TotalSpend < 0.0299 || first.Cost[0].TotalSpend > 0.0301 {
		t.Fatalf("first bucket cost = %+v", first.Cost)
	}
	empty := series.Buckets[1]
	if !empty.Start.Equal(since.Add(time.Hour)) || empty.Requests != 0 || empty.LatencyP95MS != nil || empty.Cost == nil || len(empty.Cost) != 0 {
		t.Fatalf("empty bucket = %+v", empty)
	}
	if last := series.Buckets[3]; last.Requests != 1 || last.LatencyP50MS != nil {
		t.Fatalf("last bucket = %+v", last)
	}
}

func TestReplayTrafficLatencySampleStaysBounded(t *testing.T) {
	since := time.Date(2026, 11, 18, 8, 0, 0, 0, time.UTC)
	accumulator := newReplayTrafficAccumulator(&routerReplayTrafficQuery{since: since, interval: time.Hour, until: since.Add(time.Hour)})
	for index := range 5000 {
		accumulator.add(trafficTestRecord("r", since.Add(time.Second), routerreplay.LifecycleCompleted, int64(index%100+1), 0))
	}
	bucket := accumulator.buckets[0]
	if len(bucket.latencies.values) != replayTrafficLatencySamples || bucket.latencies.offered != 5000 {
		t.Fatalf("bucket kept %d of %d durations", len(bucket.latencies.values), bucket.latencies.offered)
	}
	if len(accumulator.window.values) != replayTrafficWindowLatencySamples {
		t.Fatalf("window kept %d durations", len(accumulator.window.values))
	}
	series := accumulator.result()
	for name, p95 := range map[string]int64{"bucket": *series.Buckets[0].LatencyP95MS, "window": *series.LatencyP95MS} {
		if p95 < 85 || p95 > 100 {
			t.Fatalf("sampled %s p95 = %d, want about 95", name, p95)
		}
	}
}

func TestRouterReplayAggregateAPIServesTrafficOnRequest(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	now := time.Now().UTC()
	for _, record := range []routerreplay.RoutingRecord{
		trafficTestRecord("recent", now.Add(-30*time.Minute), routerreplay.LifecycleCompleted, 250, 0.01),
		trafficTestRecord("older", now.Add(-90*time.Minute), routerreplay.LifecycleCompleted, 750, 0.02),
	} {
		if _, err := recorder.AddRecord(record); err != nil {
			t.Fatalf("add record: %v", err)
		}
	}
	router := &OpenAIRouter{ReplayRecorder: recorder, ReplayStoreShared: true}
	// Just under two hours, so the window splits into exactly two intervals.
	since := url.QueryEscape(now.Add(-119 * time.Minute).Truncate(time.Minute).Format(time.RFC3339))

	plain := router.handleRouterReplayAPI(context.Background(), "GET", routerReplayAggregatePath+"?since="+since)
	if _, found := decodeJSONBody(t, plain.GetImmediateResponse().Body)["traffic"]; found {
		t.Fatal("an aggregate without interval must keep its earlier shape")
	}

	response := router.handleRouterReplayAPI(context.Background(), "GET", routerReplayAggregatePath+"?since="+since+"&interval=1h")
	var payload routerreplay.AggregateResponse
	if err := json.Unmarshal(response.GetImmediateResponse().Body, &payload); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if payload.RecordCount != 2 || payload.Traffic == nil || len(payload.Traffic.Buckets) != 2 {
		t.Fatalf("traffic = %+v", payload.Traffic)
	}
	if payload.Traffic.Buckets[0].Requests+payload.Traffic.Buckets[1].Requests != 2 {
		t.Fatalf("buckets = %+v", payload.Traffic.Buckets)
	}

	invalid := router.handleRouterReplayAPI(context.Background(), "GET", routerReplayAggregatePath+"?interval=1h")
	if code := invalid.GetImmediateResponse().GetStatus().GetCode(); code != typev3.StatusCode_BadRequest {
		t.Fatalf("interval without since answered %v, want 400", code)
	}
}
