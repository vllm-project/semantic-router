package routerreplay

import "time"

// TrafficSeries counts an aggregate's filtered records in contiguous
// intervals of IntervalSeconds from Since to the time of the query, oldest
// first. A caller asks for it with the aggregate's since and interval. The
// latency percentiles read the completed records' durations over the whole
// window, and are absent without one.
type TrafficSeries struct {
	Since           time.Time       `json:"since"`
	IntervalSeconds int64           `json:"interval_seconds"`
	LatencyP50MS    *int64          `json:"latency_p50_ms,omitempty"`
	LatencyP95MS    *int64          `json:"latency_p95_ms,omitempty"`
	Buckets         []TrafficBucket `json:"buckets"`
}

// TrafficBucket covers the records that started in one interval. Failed
// counts the failed and aborted ones. The latency percentiles read the
// completed records' durations, and are absent without one. Cost totals the
// interval's priced records per currency as the aggregate summary does, and is
// an empty list when none was priced.
type TrafficBucket struct {
	Start        time.Time             `json:"start"`
	Requests     int                   `json:"requests"`
	Cached       int                   `json:"cached"`
	Failed       int                   `json:"failed"`
	LatencyP50MS *int64                `json:"latency_p50_ms,omitempty"`
	LatencyP95MS *int64                `json:"latency_p95_ms,omitempty"`
	Cost         []CurrencyCostSummary `json:"cost"`
}
