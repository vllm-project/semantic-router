package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// GuardVerdictStoreOpsTotal separates local reuse, Redis lookups, publication
// and renewal scheduling. Labels never contain recipes, window keys or errors.
var GuardVerdictStoreOpsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "vsr_guard_verdict_store_ops_total",
	Help: "Guard verdict cache operations by operation and bounded outcome",
}, []string{"op", "outcome"})

// RecordGuardVerdictStoreOp records a cache operation using fixed labels.
func RecordGuardVerdictStoreOp(op, outcome string) {
	GuardVerdictStoreOpsTotal.WithLabelValues(op, outcome).Inc()
}
