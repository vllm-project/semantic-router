package store

import "encoding/json"

func cloneBoolMap(values map[string]bool) map[string]bool {
	if values == nil {
		return nil
	}
	cloned := make(map[string]bool, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}

func availableFloat32(value float32, available bool) *float32 {
	if !available {
		return nil
	}
	return &value
}

// Unmarked historical records remain unknown. Neither a structural decision
// constant nor an unavailable guard verdict is serialized as a model score.
func (r Record) MarshalJSON() ([]byte, error) {
	type alias Record
	var score *float64
	if r.ConfidenceScoreAvailable {
		score = &r.ConfidenceScore
	}
	return json.Marshal(struct {
		alias
		ConfidenceScore             *float64 `json:"confidence_score"`
		JailbreakConfidence         *float32 `json:"jailbreak_confidence,omitempty"`
		ResponseJailbreakConfidence *float32 `json:"response_jailbreak_confidence,omitempty"`
	}{
		alias: alias(r), ConfidenceScore: score,
		JailbreakConfidence:         availableFloat32(r.JailbreakConfidence, r.JailbreakScoreAvailable),
		ResponseJailbreakConfidence: availableFloat32(r.ResponseJailbreakConfidence, r.ResponseJailbreakScoreAvailable),
	})
}
