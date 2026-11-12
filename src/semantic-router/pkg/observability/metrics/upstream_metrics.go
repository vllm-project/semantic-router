package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Upstream metric labels are bounded by configuration: cluster is a provider
// model alias and endpoint one of its backend names.
var (
	// UpstreamAttemptsTotal counts upstream attempts by how each ended: the
	// response status class (2xx, 4xx, 5xx) or the failure kind
	// (connect_failure, reset, timeout, overflow, ...).
	UpstreamAttemptsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_upstream_attempts_total",
			Help: "Upstream attempts by cluster, endpoint and outcome (status class or failure kind)",
		},
		[]string{"cluster", "endpoint", "outcome"},
	)

	// UpstreamAttemptDuration is the time from the start of an attempt until
	// its response was ready to commit, or until it failed.
	UpstreamAttemptDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_upstream_attempt_duration_seconds",
			Help:    "Time from the start of an upstream attempt until its response was ready or it failed",
			Buckets: prometheus.ExponentialBuckets(0.005, 2.5, 13),
		},
		[]string{"cluster", "endpoint"},
	)

	// UpstreamActiveRequests is the number of requests an endpoint is serving,
	// streamed bodies included. Least-request balancing reads the same count.
	UpstreamActiveRequests = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_upstream_active_requests",
			Help: "Upstream requests in flight by cluster and endpoint, streamed bodies included",
		},
		[]string{"cluster", "endpoint"},
	)

	// UpstreamStreamsTotal counts response bodies by how they ended: complete,
	// closed early by the caller, or the failure kind that cut them short.
	UpstreamStreamsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_upstream_streams_total",
			Help: "Upstream response bodies by cluster, endpoint and outcome (complete, closed, or failure kind)",
		},
		[]string{"cluster", "endpoint", "outcome"},
	)
)

var (
	// UpstreamRequestsTotal counts upstream calls by how they ended: the
	// served response's status class, or the failure kind of a local reply.
	UpstreamRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_upstream_requests_total",
			Help: "Upstream calls by cluster and outcome (status class served, or the failure kind answered locally)",
		},
		[]string{"cluster", "outcome"},
	)

	// UpstreamRetriesTotal counts retries by what the retried attempt got: a
	// status class or a failure kind.
	UpstreamRetriesTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_upstream_retries_total",
			Help: "Upstream retries by cluster and the retried attempt's outcome (status class or failure kind)",
		},
		[]string{"cluster", "outcome"},
	)

	// UpstreamEndpointHealthy is 1 while neither active health checks nor
	// outlier detection hold an endpoint out of rotation.
	UpstreamEndpointHealthy = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_upstream_endpoint_healthy",
			Help: "1 while an upstream endpoint is in rotation, 0 while a health check or outlier ejection holds it out",
		},
		[]string{"cluster", "endpoint"},
	)

	// UpstreamClusterPanic is 1 while fewer than half of a cluster's
	// endpoints are healthy, so it balances over all of them.
	UpstreamClusterPanic = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_upstream_cluster_panic",
			Help: "1 while fewer than half of a cluster's endpoints are healthy and it balances over all of them",
		},
		[]string{"cluster"},
	)

	// UpstreamEjectionsTotal counts outlier ejections by reason
	// (consecutive_5xx, success_rate).
	UpstreamEjectionsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_upstream_ejections_total",
			Help: "Outlier ejections of upstream endpoints by cluster, endpoint and reason (consecutive_5xx, success_rate)",
		},
		[]string{"cluster", "endpoint", "reason"},
	)

	// UpstreamHealthChecksTotal counts active health checks by outcome
	// (success, status, network).
	UpstreamHealthChecksTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_upstream_health_checks_total",
			Help: "Active upstream health checks by cluster, endpoint and outcome (success, status, network)",
		},
		[]string{"cluster", "endpoint", "outcome"},
	)

	// UpstreamOverflowTotal counts requests a circuit breaker rejected, by
	// the limit reached (max_requests, max_pending_requests, max_retries).
	UpstreamOverflowTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_upstream_overflow_total",
			Help: "Upstream requests rejected by a circuit breaker, by cluster and limit",
		},
		[]string{"cluster", "limit"},
	)
)

// RecordUpstreamAttempt records one finished attempt.
func RecordUpstreamAttempt(cluster, endpoint, outcome string, seconds float64) {
	cluster, endpoint = labelOrUnknown(cluster), labelOrUnknown(endpoint)
	UpstreamAttemptsTotal.WithLabelValues(cluster, endpoint, labelOrUnknown(outcome)).Inc()
	UpstreamAttemptDuration.WithLabelValues(cluster, endpoint).Observe(seconds)
}

// AddUpstreamActiveRequests moves an endpoint's in-flight count by delta.
func AddUpstreamActiveRequests(cluster, endpoint string, delta float64) {
	UpstreamActiveRequests.WithLabelValues(labelOrUnknown(cluster), labelOrUnknown(endpoint)).Add(delta)
}

// RecordUpstreamStream records how one response body ended.
func RecordUpstreamStream(cluster, endpoint, outcome string) {
	UpstreamStreamsTotal.WithLabelValues(
		labelOrUnknown(cluster), labelOrUnknown(endpoint), labelOrUnknown(outcome),
	).Inc()
}

// RecordUpstreamRequest records how one upstream call ended.
func RecordUpstreamRequest(cluster, outcome string) {
	UpstreamRequestsTotal.WithLabelValues(labelOrUnknown(cluster), labelOrUnknown(outcome)).Inc()
}

// RecordUpstreamRetry records one retry.
func RecordUpstreamRetry(cluster, outcome string) {
	UpstreamRetriesTotal.WithLabelValues(labelOrUnknown(cluster), labelOrUnknown(outcome)).Inc()
}

// SetUpstreamEndpointHealthy publishes whether an endpoint is in rotation.
func SetUpstreamEndpointHealthy(cluster, endpoint string, healthy bool) {
	UpstreamEndpointHealthy.WithLabelValues(labelOrUnknown(cluster), labelOrUnknown(endpoint)).Set(boolGauge(healthy))
}

// SetUpstreamClusterPanic publishes whether a cluster is below its healthy
// panic threshold.
func SetUpstreamClusterPanic(cluster string, panicking bool) {
	UpstreamClusterPanic.WithLabelValues(labelOrUnknown(cluster)).Set(boolGauge(panicking))
}

// RecordUpstreamEjection records one outlier ejection.
func RecordUpstreamEjection(cluster, endpoint, reason string) {
	UpstreamEjectionsTotal.WithLabelValues(
		labelOrUnknown(cluster), labelOrUnknown(endpoint), labelOrUnknown(reason),
	).Inc()
}

// RecordUpstreamHealthCheck records one active health check.
func RecordUpstreamHealthCheck(cluster, endpoint, outcome string) {
	UpstreamHealthChecksTotal.WithLabelValues(
		labelOrUnknown(cluster), labelOrUnknown(endpoint), labelOrUnknown(outcome),
	).Inc()
}

// RecordUpstreamOverflow records one request a circuit breaker rejected.
func RecordUpstreamOverflow(cluster, limit string) {
	UpstreamOverflowTotal.WithLabelValues(labelOrUnknown(cluster), labelOrUnknown(limit)).Inc()
}

func boolGauge(value bool) float64 {
	if value {
		return 1
	}
	return 0
}

// DeleteUpstreamCluster drops every upstream series of a cluster that no
// longer exists, so removed aliases do not linger on dashboards.
func DeleteUpstreamCluster(cluster string) {
	labels := prometheus.Labels{"cluster": labelOrUnknown(cluster)}
	for _, vec := range []interface {
		DeletePartialMatch(prometheus.Labels) int
	}{
		UpstreamRequestsTotal, UpstreamRetriesTotal, UpstreamAttemptsTotal, UpstreamAttemptDuration,
		UpstreamActiveRequests, UpstreamStreamsTotal,
		UpstreamEndpointHealthy, UpstreamClusterPanic, UpstreamEjectionsTotal, UpstreamHealthChecksTotal,
		UpstreamOverflowTotal,
	} {
		vec.DeletePartialMatch(labels)
	}
}
