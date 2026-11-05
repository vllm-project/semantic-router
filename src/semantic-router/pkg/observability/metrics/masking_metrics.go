package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
)

var maskedEntitiesTotal = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "llm_masking_entities_total",
		Help: "PII spans masked before provider dispatch, by decision and entity type.",
	},
	[]string{"decision", "entity_type"},
)

// RecordMaskedEntities counts masked spans by entity type and decision.
// Never pass a value or a placeholder as a label: entityType is a class name
// like EMAIL_ADDRESS, never the text that was masked (#3566).
func RecordMaskedEntities(decisionKey, entityType string, count int) {
	if count <= 0 {
		return
	}
	if decisionKey == "" {
		decisionKey = consts.UnknownLabel
	}
	if entityType == "" {
		entityType = consts.UnknownLabel
	}
	maskedEntitiesTotal.WithLabelValues(decisionKey, entityType).Add(float64(count))
}
