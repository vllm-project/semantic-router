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

type recordJSONFields Record

type recordJSON struct {
	recordJSONFields
	ConfidenceScore             *float64 `json:"confidence_score"`
	JailbreakConfidence         *float32 `json:"jailbreak_confidence,omitempty"`
	ResponseJailbreakConfidence *float32 `json:"response_jailbreak_confidence,omitempty"`
}

// JSONWire is the shared representation for serialization and schema discovery.
// Unmarked historical records remain unknown; unavailable guard verdicts are
// never serialized as model scores.
func (r Record) JSONWire() any {
	var score *float64
	if r.ConfidenceScoreAvailable {
		score = &r.ConfidenceScore
	}
	return recordJSON{
		recordJSONFields: recordJSONFields(r), ConfidenceScore: score,
		JailbreakConfidence:         availableFloat32(r.JailbreakConfidence, r.JailbreakScoreAvailable),
		ResponseJailbreakConfidence: availableFloat32(r.ResponseJailbreakConfidence, r.ResponseJailbreakScoreAvailable),
	}
}

func (r Record) MarshalJSON() ([]byte, error) { return json.Marshal(r.JSONWire()) }
