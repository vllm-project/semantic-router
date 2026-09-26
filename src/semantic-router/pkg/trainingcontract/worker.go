package trainingcontract

// Workers receive frozen inputs and owned handles, never server filesystem paths.
type WorkerRequest struct {
	BaseModel     *ModelRef         `json:"base_model,omitempty"`
	RunID         string            `json:"run_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	SchemaVersion string            `json:"schema_version" jsonschema:"enum=semantic-router.training/v1"`
	AttemptID     string            `json:"attempt_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	TaskID        string            `json:"task_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	Executor      Component         `json:"executor"`
	Trainer       Component         `json:"trainer"`
	Parameters    map[string]any    `json:"parameters,omitempty"`
	Snapshot      DataSnapshot      `json:"snapshot"`
	Inputs        []ArtifactVariant `json:"inputs,omitempty"`
}

// ArtifactResult groups formats under one logical output. Management assigns
// the artifact ID and the ID of each variant when accepting the result.
type ArtifactResult struct {
	Profile        Profile               `json:"profile"`
	Variants       []ArtifactVariantSpec `json:"variants" jsonschema:"minItems=1"`
	ManifestBundle *File                 `json:"manifest_bundle,omitempty"`
}
type WorkerResult struct {
	SchemaVersion  string              `json:"schema_version" jsonschema:"enum=semantic-router.training/v1"`
	Status         Status              `json:"status"`
	Diagnostic     string              `json:"diagnostic,omitempty"`
	Artifacts      []ArtifactResult    `json:"artifacts,omitempty"`
	Evaluations    []EvaluationSpec    `json:"evaluations,omitempty"`
	Qualifications []QualificationSpec `json:"qualifications,omitempty"`
}

// WorkerSubmission acknowledges an idempotent submission by attempt_id.
type WorkerSubmission struct {
	SchemaVersion string `json:"schema_version" jsonschema:"enum=semantic-router.training/v1"`
	WorkerHandle  string `json:"worker_handle" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
}
