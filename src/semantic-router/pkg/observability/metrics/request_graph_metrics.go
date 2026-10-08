package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// RequestGraphNodeDuration labels are bounded by configuration: node_type is
// a registered request-graph node type and template the name of the graph
// that ran, such as a built-in Looper template.
var RequestGraphNodeDuration = promauto.NewHistogramVec(
	prometheus.HistogramOpts{
		Name:    "llm_request_graph_node_duration_seconds",
		Help:    "Time a request-graph step ran, nested steps included, by node type and template",
		Buckets: prometheus.ExponentialBuckets(0.001, 2.5, 15),
	},
	[]string{"node_type", "template"},
)

// RecordRequestGraphNode records how long one step of a request graph ran.
func RecordRequestGraphNode(nodeType, template string, seconds float64) {
	RequestGraphNodeDuration.WithLabelValues(labelOrUnknown(nodeType), labelOrUnknown(template)).Observe(seconds)
}
