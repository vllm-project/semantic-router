package extproc

import (
	"fmt"
	"math"
	"net/url"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// routerReplayReviewFilters select approval reviews. Outcome and risk exist
// only on a completed review, so filtering by them leaves out the rest.
type routerReplayReviewFilters struct {
	requestKind, status, outcome, risk string
}

var routerReplayReviewStatuses = []string{
	store.ReviewStatusPending, store.ReviewStatusCompleted, store.ReviewStatusToolCall,
	store.ReviewStatusInvalid, store.ReviewStatusFailed, store.ReviewStatusCancelled,
}

func parseRouterReplayReviewFilters(values url.Values) (routerReplayReviewFilters, error) {
	var filters routerReplayReviewFilters
	for _, field := range []struct {
		key     string
		target  *string
		allowed []string
	}{
		{"request_kind", &filters.requestKind, []string{store.RequestKindCodexApprovalReview}},
		{"review_status", &filters.status, routerReplayReviewStatuses},
		{"review_outcome", &filters.outcome, []string{store.ReviewOutcomeAllow, store.ReviewOutcomeDeny}},
		{"review_risk", &filters.risk, []string{"low", "medium", "high", "critical"}},
	} {
		value := strings.TrimSpace(values.Get(field.key))
		if value == "" || value == "all" {
			continue
		}
		if !slices.Contains(field.allowed, value) {
			return filters, fmt.Errorf("%s must be one of all, %s", field.key, strings.Join(field.allowed, ", "))
		}
		*field.target = value
	}
	return filters, nil
}

func (filters routerReplayReviewFilters) matches(record routerreplay.RoutingRecord) bool {
	status := ""
	if record.Review != nil {
		status = record.Review.Status
	}
	return matchesOptionalValue(record.RequestKind, filters.requestKind) &&
		matchesOptionalValue(status, filters.status) &&
		matchesOptionalValue(record.Review.Outcome(), filters.outcome) &&
		matchesOptionalValue(record.Review.RiskLevel(), filters.risk)
}

// replayReviewAccumulator builds the review summary over a streamed scan. It
// keeps per-minute counts and verdict latencies, never the records.
type replayReviewAccumulator struct {
	summary   routerreplay.ReviewAggregate
	latencies []int64
	minutes   map[int64]*routerreplay.ReviewTimeBucket
	models    map[string]*routerreplay.ReviewModelEntry
	costs     map[string]*routerreplay.ReviewCost
	seen      bool
}

func newReplayReviewAccumulator() *replayReviewAccumulator {
	return &replayReviewAccumulator{
		minutes: make(map[int64]*routerreplay.ReviewTimeBucket),
		models:  make(map[string]*routerreplay.ReviewModelEntry),
		costs:   make(map[string]*routerreplay.ReviewCost),
	}
}

func (a *replayReviewAccumulator) add(record routerreplay.RoutingRecord) {
	review := record.Review
	if record.RequestKind != store.RequestKindCodexApprovalReview || review == nil {
		return
	}
	a.seen = true
	model := a.model(routerReplayFallbackName(record.SelectedModel))
	model.Requests++
	a.addCost(record, model)
	switch review.Status {
	case store.ReviewStatusPending:
		a.summary.Pending++
		return
	case store.ReviewStatusToolCall:
		a.summary.ToolCalls++
		return
	}
	a.summary.Reviews++
	a.countSettled(review, a.minuteBucket(record.Timestamp))
	if review.Outcome() != "" {
		latency := review.LatencyMS
		if latency <= 0 {
			latency = record.DurationMS
		}
		a.latencies = append(a.latencies, latency)
	}
}

func (a *replayReviewAccumulator) countSettled(review *routerreplay.Review, bucket *routerreplay.ReviewTimeBucket) {
	switch {
	case review.Outcome() == store.ReviewOutcomeAllow:
		a.summary.Allowed++
		bucket.Allowed++
	case review.Outcome() == store.ReviewOutcomeDeny:
		a.summary.Denied++
		bucket.Denied++
	case review.Status == store.ReviewStatusFailed:
		a.summary.Failed++
		bucket.Failed++
	case review.Status == store.ReviewStatusCancelled:
		a.summary.Cancelled++
		bucket.Cancelled++
	default:
		a.summary.Invalid++
		bucket.Invalid++
	}
}

func (a *replayReviewAccumulator) model(name string) *routerreplay.ReviewModelEntry {
	entry := a.models[name]
	if entry == nil {
		entry = &routerreplay.ReviewModelEntry{Model: name}
		a.models[name] = entry
	}
	return entry
}

func (a *replayReviewAccumulator) minuteBucket(timestamp time.Time) *routerreplay.ReviewTimeBucket {
	minute := timestamp.UTC().Truncate(time.Minute)
	bucket := a.minutes[minute.Unix()]
	if bucket == nil {
		bucket = &routerreplay.ReviewTimeBucket{Start: minute}
		a.minutes[minute.Unix()] = bucket
	}
	return bucket
}

// addCost adds a request's recorded cost. A request without one is counted
// as unpriced, never as free.
func (a *replayReviewAccumulator) addCost(record routerreplay.RoutingRecord, model *routerreplay.ReviewModelEntry) {
	currency := ""
	if record.Currency != nil {
		currency = normalizeReplayCurrency(*record.Currency)
	}
	if !finiteReplayCost(record.ActualCost) || currency == "" {
		a.summary.UnpricedRequests++
		return
	}
	total := a.costs[currency]
	if total == nil {
		total = &routerreplay.ReviewCost{Currency: currency}
		a.costs[currency] = total
	}
	total.Amount += *record.ActualCost
	total.Requests++
	for index := range model.Cost {
		if model.Cost[index].Currency == currency {
			model.Cost[index].Amount += *record.ActualCost
			model.Cost[index].Requests++
			return
		}
	}
	model.Cost = append(model.Cost, routerreplay.ReviewCost{Currency: currency, Amount: *record.ActualCost, Requests: 1})
}

func (a *replayReviewAccumulator) result() *routerreplay.ReviewAggregate {
	if !a.seen {
		return nil
	}
	summary := a.summary
	if verdicts := summary.Allowed + summary.Denied; verdicts > 0 {
		rate := float64(summary.Allowed) / float64(verdicts)
		summary.ApprovalRate = &rate
	}
	summary.LatencyP50MS, summary.LatencyP95MS = latencyPercentile(a.latencies, 0.50), latencyPercentile(a.latencies, 0.95)
	summary.Cost = make([]routerreplay.ReviewCost, 0, len(a.costs))
	for _, cost := range a.costs {
		summary.Cost = append(summary.Cost, *cost)
	}
	sort.Slice(summary.Cost, func(i, j int) bool { return summary.Cost[i].Currency < summary.Cost[j].Currency })
	summary.ByModel = make([]routerreplay.ReviewModelEntry, 0, len(a.models))
	for _, entry := range a.models {
		if entry.Cost == nil {
			// An unpriced model is an empty list, as the totals are.
			entry.Cost = []routerreplay.ReviewCost{}
		}
		sort.Slice(entry.Cost, func(i, j int) bool { return entry.Cost[i].Currency < entry.Cost[j].Currency })
		summary.ByModel = append(summary.ByModel, *entry)
	}
	sort.Slice(summary.ByModel, func(i, j int) bool {
		if summary.ByModel[i].Requests == summary.ByModel[j].Requests {
			return summary.ByModel[i].Model < summary.ByModel[j].Model
		}
		return summary.ByModel[i].Requests > summary.ByModel[j].Requests
	})
	summary.Bucket, summary.OverTime = reviewTimeBuckets(a.minutes)
	return &summary
}

// latencyPercentile is the nearest-rank percentile, or nil with no samples.
func latencyPercentile(values []int64, quantile float64) *int64 {
	if len(values) == 0 {
		return nil
	}
	sorted := slices.Clone(values)
	slices.Sort(sorted)
	rank := int(math.Ceil(quantile*float64(len(sorted)))) - 1
	value := sorted[max(rank, 0)]
	return &value
}

// reviewTimeBuckets rolls per-minute counts into contiguous buckets whose
// width suits the span: minutes up to two hours, hours up to a week, then days.
func reviewTimeBuckets(minutes map[int64]*routerreplay.ReviewTimeBucket) (string, []routerreplay.ReviewTimeBucket) {
	if len(minutes) == 0 {
		return "", []routerreplay.ReviewTimeBucket{}
	}
	keys := make([]int64, 0, len(minutes))
	for key := range minutes {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	first, last := time.Unix(keys[0], 0).UTC(), time.Unix(keys[len(keys)-1], 0).UTC()
	name, width := "day", 24*time.Hour
	switch span := last.Sub(first); {
	case span <= 2*time.Hour:
		name, width = "minute", time.Minute
	case span <= 7*24*time.Hour:
		name, width = "hour", time.Hour
	}
	start := first.Truncate(width)
	buckets := make([]routerreplay.ReviewTimeBucket, int(last.Truncate(width).Sub(start)/width)+1)
	for index := range buckets {
		buckets[index].Start = start.Add(time.Duration(index) * width)
	}
	for _, key := range keys {
		minute := minutes[key]
		bucket := &buckets[int(minute.Start.Truncate(width).Sub(start)/width)]
		bucket.Allowed += minute.Allowed
		bucket.Denied += minute.Denied
		bucket.Invalid += minute.Invalid
		bucket.Failed += minute.Failed
		bucket.Cancelled += minute.Cancelled
	}
	return name, buckets
}
