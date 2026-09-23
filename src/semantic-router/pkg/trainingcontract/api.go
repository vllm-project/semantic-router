package trainingcontract

type SubmitRunRequest struct {
	SchemaVersion  string  `json:"schema_version" jsonschema:"enum=semantic-router.training/v1"`
	IdempotencyKey string  `json:"idempotency_key" jsonschema:"minLength=1"`
	Spec           RunSpec `json:"spec"`
}
type ComparisonRequest struct {
	RunIDs []string `json:"run_ids" jsonschema:"minItems=2,uniqueItems=true"`
}
type ComparisonEntry struct {
	Run         TrainingRun  `json:"run"`
	Evaluations []Evaluation `json:"evaluations"`
}

// APIError is the JSON body of every non-success management response.
// HTTP status carries the category; message explains the rejected operation.
type APIError struct {
	Code    string `json:"code" jsonschema:"enum=invalid_request,enum=unauthenticated,enum=forbidden,enum=not_found,enum=conflict,enum=internal_error"`
	Message string `json:"message"`
}

// ValidationRequest validates a proposed run without submitting it.
type ValidationRequest struct {
	SchemaVersion string  `json:"schema_version" jsonschema:"enum=semantic-router.training/v1"`
	Spec          RunSpec `json:"spec"`
}

type ValidationResponse struct {
	Valid bool `json:"valid"`
}

type EventPage struct {
	Events    []Event `json:"events"`
	NextAfter int64   `json:"next_after"`
}

type ComparisonResponse struct {
	Entries []ComparisonEntry `json:"entries"`
}
