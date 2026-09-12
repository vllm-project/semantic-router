package handlers

import "encoding/json"

// Preserve categorical/error-policy matches without inventing a score.
func (s MatchedSignal) MarshalJSON() ([]byte, error) {
	type alias MatchedSignal
	var confidence *float64
	if s.ConfidenceAvailable == nil || *s.ConfidenceAvailable {
		value := s.Confidence
		confidence = &value
	}
	return json.Marshal(struct {
		alias
		Confidence *float64 `json:"confidence"`
	}{alias(s), confidence})
}
