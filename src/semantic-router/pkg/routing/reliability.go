package routing

import "time"

// Reliability overrides the upstream timeouts and retries for one call, as a
// decision's reliability block does. A nil or empty field keeps the provider
// model's setting.
//
// An override merges the way Envoy merges its per-request headers, so both
// gateway modes apply it alike: timeouts and the retry count replace the
// provider model's, while retry conditions and retriable status codes add to
// them.
type Reliability struct {
	// TotalTimeout bounds the whole call, retries included. Zero disables it.
	TotalTimeout *time.Duration
	// PerTryTimeout bounds each attempt until its response starts. Zero
	// disables it.
	PerTryTimeout *time.Duration
	// IdleTimeout bounds each wait for more of the response. Zero disables it.
	IdleTimeout *time.Duration
	// FirstByteTimeout bounds the wait for the first response body byte. Zero
	// disables it.
	FirstByteTimeout *time.Duration
	// RetryCount replaces the number of retries.
	RetryCount *int
	// RetryOn adds Envoy retry_on conditions.
	RetryOn []string
	// RetriableStatusCodes adds statuses that retriable-status-codes retries.
	RetriableStatusCodes []int
	// RetryBackOffBase, RetryBackOffMax and RetryAfterMax replace the retry
	// back-off and the bound on a Retry-After wait.
	RetryBackOffBase *time.Duration
	RetryBackOffMax  *time.Duration
	RetryAfterMax    *time.Duration
}

// ReliabilitySession is a Session whose planned call carries a reliability
// override, such as the matched decision's; the engine attaches it to
// Call.Reliability.
type ReliabilitySession interface {
	Session
	Reliability() *Reliability
}
