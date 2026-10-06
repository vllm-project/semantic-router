package services

import "encoding/json"

func confidenceAvailability(available bool) *bool { return &available }

func reportedConfidence(value float64, available *bool) *float64 {
	if available != nil && !*available {
		return nil
	}
	return &value
}

// These representations preserve internal numeric fields for routing callers
// while JSON and its schema report null when the producer has no model score.
type (
	classificationJSONFields Classification
	classificationJSON       struct {
		classificationJSONFields
		Confidence *float64 `json:"confidence"`
	}
)

// JSONWire is the representation used by both serialization and schema discovery.
func (c Classification) JSONWire() any {
	return classificationJSON{classificationJSONFields: classificationJSONFields(c), Confidence: reportedConfidence(c.Confidence, c.ConfidenceAvailable)}
}

func (c Classification) MarshalJSON() ([]byte, error) { return json.Marshal(c.JSONWire()) }

type (
	decisionResultJSONFields DecisionResult
	decisionResultJSON       struct {
		decisionResultJSONFields
		Confidence *float64 `json:"confidence"`
	}
)

// JSONWire is the representation used by both serialization and schema discovery.
func (d DecisionResult) JSONWire() any {
	return decisionResultJSON{decisionResultJSONFields: decisionResultJSONFields(d), Confidence: reportedConfidence(d.Confidence, d.ConfidenceAvailable)}
}

func (d DecisionResult) MarshalJSON() ([]byte, error) { return json.Marshal(d.JSONWire()) }

type (
	factCheckResponseJSONFields FactCheckResponse
	factCheckResponseJSON       struct {
		factCheckResponseJSONFields
		Confidence *float64 `json:"confidence"`
	}
)

// JSONWire is the representation used by both serialization and schema discovery.
func (r FactCheckResponse) JSONWire() any {
	return factCheckResponseJSON{factCheckResponseJSONFields: factCheckResponseJSONFields(r), Confidence: reportedConfidence(r.Confidence, &r.ConfidenceAvailable)}
}

func (r FactCheckResponse) MarshalJSON() ([]byte, error) { return json.Marshal(r.JSONWire()) }

type (
	userFeedbackResponseJSONFields UserFeedbackResponse
	userFeedbackResponseJSON       struct {
		userFeedbackResponseJSONFields
		Confidence *float64 `json:"confidence"`
	}
)

// JSONWire is the representation used by both serialization and schema discovery.
func (r UserFeedbackResponse) JSONWire() any {
	return userFeedbackResponseJSON{userFeedbackResponseJSONFields: userFeedbackResponseJSONFields(r), Confidence: reportedConfidence(r.Confidence, &r.ConfidenceAvailable)}
}

func (r UserFeedbackResponse) MarshalJSON() ([]byte, error) { return json.Marshal(r.JSONWire()) }
