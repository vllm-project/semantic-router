package metrics

import (
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Configuration lifecycle metrics. Labels are bounded: the update source,
// how the attempt ended, and the lifecycle stage it reached.
var (
	// ConfigUpdatesTotal counts configuration updates by source and outcome:
	// active (ACK), failed (NACK) or superseded.
	ConfigUpdatesTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_config_updates_total",
			Help: "Configuration updates by source, result (active, failed, superseded) and the lifecycle stage reached",
		},
		[]string{"source", "result", "stage"},
	)

	// ConfigUpdateDuration is how long an update took from receipt to its
	// result, warming included.
	ConfigUpdateDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_config_update_duration_seconds",
			Help:    "Time from receiving a configuration update until it activated, failed or was superseded",
			Buckets: prometheus.ExponentialBuckets(0.01, 3, 12),
		},
		[]string{"result"},
	)

	// ConfigActiveVersion is the version of the configuration snapshot that
	// serves.
	ConfigActiveVersion = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "llm_config_active_version",
		Help: "Version of the configuration snapshot that serves",
	})

	// ConfigActiveInfo is 1 for the active snapshot's version and document
	// hash; it holds one series at a time.
	ConfigActiveInfo = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_config_active_info",
			Help: "1 for the active configuration snapshot, labeled with its version and document hash",
		},
		[]string{"version", "hash"},
	)

	// ConfigLastRejection is when the most recent configuration update was
	// rejected (NACK), as a Unix timestamp.
	ConfigLastRejection = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "llm_config_last_rejection_timestamp_seconds",
		Help: "Unix time of the most recent rejected configuration update",
	})
)

// RecordConfigUpdate records how one configuration update ended.
func RecordConfigUpdate(source, result, stage string, duration time.Duration) {
	ConfigUpdatesTotal.WithLabelValues(source, result, stage).Inc()
	ConfigUpdateDuration.WithLabelValues(result).Observe(duration.Seconds())
}

// SetConfigActive records the snapshot that serves.
func SetConfigActive(version uint64, hash string) {
	ConfigActiveVersion.Set(float64(version))
	ConfigActiveInfo.Reset()
	ConfigActiveInfo.WithLabelValues(strconv.FormatUint(version, 10), hash).Set(1)
}

// SetConfigLastRejection records when an update was last rejected.
func SetConfigLastRejection(at time.Time) {
	ConfigLastRejection.Set(float64(at.UnixNano()) / float64(time.Second))
}
