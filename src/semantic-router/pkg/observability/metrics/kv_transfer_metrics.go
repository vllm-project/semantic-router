package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var KVTransferOutcomes = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "llm_kv_transfer_outcomes_total",
	Help: "Completed KV handoff attempts by status and bounded reason",
}, []string{"status", "reason"})
