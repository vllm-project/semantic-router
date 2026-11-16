package routerreplay

import "time"

// ReviewAggregate summarizes the approval reviews among an aggregate's
// filtered records. A review that asked for a tool check continues in a
// later request, so tool-check requests and reviews still in flight are
// counted apart from Reviews.
type ReviewAggregate struct {
	// Reviews counts settled reviews: verdicts, invalid answers, failed and
	// cancelled calls.
	Reviews   int `json:"reviews"`
	Allowed   int `json:"allowed"`
	Denied    int `json:"denied"`
	Invalid   int `json:"invalid"`
	Failed    int `json:"failed"`
	Cancelled int `json:"cancelled"`
	Pending   int `json:"pending"`
	ToolCalls int `json:"tool_calls"`
	// ApprovalRate is allowed / (allowed + denied). Failed, invalid and
	// cancelled reviews are not verdicts and stay out of it. Absent with no
	// verdict.
	ApprovalRate *float64 `json:"approval_rate,omitempty"`
	// Latency percentiles of reviews that reached a verdict.
	LatencyP50MS *int64 `json:"latency_p50_ms,omitempty"`
	LatencyP95MS *int64 `json:"latency_p95_ms,omitempty"`
	// Cost sums the recorded cost of every review request, tool checks
	// included, per currency. UnpricedRequests had no recorded cost.
	Cost             []ReviewCost `json:"cost"`
	UnpricedRequests int          `json:"unpriced_requests"`
	// Bucket is the width of each OverTime entry: minute, hour or day.
	Bucket   string             `json:"bucket,omitempty"`
	OverTime []ReviewTimeBucket `json:"over_time"`
	ByModel  []ReviewModelEntry `json:"by_model"`
}

// ReviewTimeBucket counts the settled reviews that started in one bucket.
type ReviewTimeBucket struct {
	Start     time.Time `json:"start"`
	Allowed   int       `json:"allowed"`
	Denied    int       `json:"denied"`
	Invalid   int       `json:"invalid"`
	Failed    int       `json:"failed"`
	Cancelled int       `json:"cancelled"`
}

// ReviewModelEntry is one review model's requests, tool checks included, and
// their recorded cost.
type ReviewModelEntry struct {
	Model    string       `json:"model"`
	Requests int          `json:"requests"`
	Cost     []ReviewCost `json:"cost"`
}

// ReviewCost is the recorded cost of review requests in one currency.
type ReviewCost struct {
	Currency string  `json:"currency"`
	Amount   float64 `json:"amount"`
	Requests int     `json:"requests"`
}
