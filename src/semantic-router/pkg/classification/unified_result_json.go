package classification

import "encoding/json"

// An absent score remains null on the wire rather than becoming a probability
// through the compatibility struct's numeric zero value.
func (r PIIResult) MarshalJSON() ([]byte, error) {
	type fields PIIResult
	var confidence *float32
	if r.ScoresAvailable != nil && *r.ScoresAvailable {
		confidence = &r.Confidence
	}
	return json.Marshal(struct {
		fields
		Confidence *float32 `json:"confidence"`
	}{fields(r), confidence})
}

func (r SecurityResult) MarshalJSON() ([]byte, error) {
	type fields SecurityResult
	var confidence *float32
	if r.ScoresAvailable != nil && *r.ScoresAvailable {
		confidence = &r.Confidence
	}
	return json.Marshal(struct {
		fields
		Confidence *float32 `json:"confidence"`
	}{fields(r), confidence})
}
