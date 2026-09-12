package services

import "encoding/json"

func confidenceAvailability(available bool) *bool { return &available }

func reportedConfidence(value float64, available *bool) *float64 {
	if available != nil && !*available {
		return nil
	}
	return &value
}

// The internal numeric field remains available to existing routing callers;
// JSON consumers receive null when the producer explicitly reports no score.
func (c Classification) MarshalJSON() ([]byte, error) {
	type alias Classification
	return json.Marshal(struct {
		alias
		Confidence *float64 `json:"confidence"`
	}{alias: alias(c), Confidence: reportedConfidence(c.Confidence, c.ConfidenceAvailable)})
}

func (d DecisionResult) MarshalJSON() ([]byte, error) {
	type alias DecisionResult
	return json.Marshal(struct {
		alias
		Confidence *float64 `json:"confidence"`
	}{alias: alias(d), Confidence: reportedConfidence(d.Confidence, d.ConfidenceAvailable)})
}

func (r FactCheckResponse) MarshalJSON() ([]byte, error) {
	type alias FactCheckResponse
	return json.Marshal(struct {
		alias
		Confidence *float64 `json:"confidence"`
	}{alias: alias(r), Confidence: reportedConfidence(r.Confidence, &r.ConfidenceAvailable)})
}

func (r UserFeedbackResponse) MarshalJSON() ([]byte, error) {
	type alias UserFeedbackResponse
	return json.Marshal(struct {
		alias
		Confidence *float64 `json:"confidence"`
	}{alias: alias(r), Confidence: reportedConfidence(r.Confidence, &r.ConfidenceAvailable)})
}
