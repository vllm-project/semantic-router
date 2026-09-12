package classification

import "encoding/json"

// MarshalJSON preserves older metric producers while allowing a producer to
// explicitly report that no model score exists for a categorical/policy match.
func (m SignalMetrics) MarshalJSON() ([]byte, error) {
	type alias SignalMetrics
	var confidence *float64
	if m.ConfidenceAvailable == nil || *m.ConfidenceAvailable {
		confidence = &m.Confidence
	}
	return json.Marshal(struct {
		alias
		Confidence *float64 `json:"confidence"`
	}{alias: alias(m), Confidence: confidence})
}
