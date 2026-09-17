package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// RemoteConnectorOutcomeSuccess is the outcome label of a call that returned
// a response. Failures carry the connector's error kind instead.
const RemoteConnectorOutcomeSuccess = "success"

// RemoteConnectorOutcomeUnclassified is the fallback for an error the
// connector did not tag with a kind. It should never appear; it exists so a
// metrics label can never be the reason a call fails.
const RemoteConnectorOutcomeUnclassified = "unclassified"

var (
	// RemoteConnectorRequestDuration is the wall time of one call through the
	// shared remote-classifier connector, retries included, so it reflects
	// what the signal actually waited for.
	RemoteConnectorRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_remote_connector_request_duration_seconds",
			Help:    "Wall time of one remote classifier call through the shared connector, retries included, by operation",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"operation"},
	)

	// RemoteConnectorRequestsTotal counts calls by operation and outcome. The
	// outcome is "success" or the connector's error kind, so a scorer that is
	// down (transport), overloaded (status) or misconfigured (authorization)
	// can be told apart on a dashboard.
	RemoteConnectorRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_remote_connector_requests_total",
			Help: "Remote classifier calls through the shared connector by operation and outcome (success, request, authorization, transport, status, response, unclassified)",
		},
		[]string{"operation", "outcome"},
	)

	// RemoteConnectorRetriesTotal counts attempts that were retried. A rising
	// rate with a flat error rate means the remote is flapping under the
	// retry budget rather than healthy.
	RemoteConnectorRetriesTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_remote_connector_retries_total",
			Help: "Retried attempts of remote classifier calls through the shared connector, by operation",
		},
		[]string{"operation"},
	)
)

// RecordRemoteConnectorRequest records one completed call, however it ended.
func RecordRemoteConnectorRequest(operation, outcome string, latencySeconds float64) {
	operation = labelOrUnknown(operation)
	RemoteConnectorRequestsTotal.WithLabelValues(operation, labelOrUnknown(outcome)).Inc()
	RemoteConnectorRequestDuration.WithLabelValues(operation).Observe(latencySeconds)
}

// RecordRemoteConnectorRetry records one retried attempt.
func RecordRemoteConnectorRetry(operation string) {
	RemoteConnectorRetriesTotal.WithLabelValues(labelOrUnknown(operation)).Inc()
}
