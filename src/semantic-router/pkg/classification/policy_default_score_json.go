package classification

import "encoding/json"

func classificationConfidence(value float32, available bool) *float32 {
	if !available {
		return nil
	}
	return &value
}

func (r FactCheckResult) MarshalJSON() ([]byte, error) {
	type alias FactCheckResult
	return json.Marshal(struct {
		alias
		Confidence *float32 `json:"confidence"`
	}{alias: alias(r), Confidence: classificationConfidence(r.Confidence, r.ConfidenceAvailable)})
}

func (r FeedbackResult) MarshalJSON() ([]byte, error) {
	type alias FeedbackResult
	return json.Marshal(struct {
		alias
		Confidence *float32 `json:"confidence"`
	}{alias: alias(r), Confidence: classificationConfidence(r.Confidence, r.ConfidenceAvailable)})
}
