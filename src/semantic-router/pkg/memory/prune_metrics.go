package memory

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// PruneDeletedTotal tracks the total number of memories pruned, labeled by trigger source.
	PruneDeletedTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "memory_prune_deleted_total",
			Help: "Total number of memories deleted by pruning",
		},
		[]string{"trigger"}, // "cap" or "sweep"
	)

	// PruneCapTriggeredTotal tracks how many times a Store() call triggered cap enforcement.
	PruneCapTriggeredTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "memory_prune_cap_triggered_total",
			Help: "Total number of times cap enforcement was triggered on Store()",
		},
	)

	// PruneSweepRunsTotal tracks how many background sweep cycles have completed.
	PruneSweepRunsTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "memory_prune_sweep_runs_total",
			Help: "Total number of background prune sweep cycles completed",
		},
	)

	// PruneSweepDuration tracks how long each background sweep cycle takes.
	PruneSweepDuration = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "memory_prune_sweep_duration_seconds",
			Help:    "Duration of background prune sweep cycles in seconds",
			Buckets: prometheus.DefBuckets,
		},
	)

	// PruneSweepUsersProcessedTotal tracks how many users were evaluated during sweeps.
	PruneSweepUsersProcessedTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "memory_prune_sweep_users_processed_total",
			Help: "Total number of users processed during background prune sweeps",
		},
	)

	// PruneSweepErrorsTotal tracks errors encountered during background sweeps.
	PruneSweepErrorsTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "memory_prune_sweep_errors_total",
			Help: "Total number of errors encountered during background prune sweeps",
		},
	)
)
