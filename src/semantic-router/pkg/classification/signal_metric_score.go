package classification

import "encoding/json"

type (
	signalMetricsJSONFields SignalMetrics
	signalMetricsJSON       struct {
		signalMetricsJSONFields
		Confidence *float64 `json:"confidence"`
	}
)

// JSONWire preserves older metric producers while allowing a producer to
// explicitly report that no model score exists for a categorical/policy match.
// Serialization and schema discovery share this exact representation.
func (m SignalMetrics) JSONWire() any {
	var confidence *float64
	if m.ConfidenceAvailable == nil || *m.ConfidenceAvailable {
		confidence = &m.Confidence
	}
	return signalMetricsJSON{signalMetricsJSONFields: signalMetricsJSONFields(m), Confidence: confidence}
}

func (m SignalMetrics) MarshalJSON() ([]byte, error) { return json.Marshal(m.JSONWire()) }
