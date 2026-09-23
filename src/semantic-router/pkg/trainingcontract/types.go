// Package trainingcontract defines the versioned, transport-independent training control plane.
package trainingcontract

import "time"

const Version = "semantic-router.training/v1"

type Target string

const (
	Selector    Target = "selector.model-choice/v1"
	LabelScores Target = "signal.label-scores/v1"
	Spans       Target = "signal.spans/v1"
)

type Status string

const (
	Pending    Status = "pending"
	Running    Status = "running"
	Cancelling Status = "cancelling"
	Succeeded  Status = "succeeded"
	Failed     Status = "failed"
	Cancelled  Status = "cancelled"
	Skipped    Status = "skipped"
)

type Metadata struct {
	SchemaVersion string    `json:"schema_version" jsonschema:"enum=semantic-router.training/v1"`
	ID            string    `json:"id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	CreatedAt     time.Time `json:"created_at"`
}

type Component struct {
	Name    string `json:"name" jsonschema:"minLength=1"`
	Version string `json:"version" jsonschema:"minLength=1"`
}

type ModelRef struct {
	Repository string `json:"repository" jsonschema:"pattern=^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"`
	Revision   string `json:"revision" jsonschema:"pattern=^[0-9a-f]{40}$"`
}

type SelectorProfile struct {
	CandidateModels   []string `json:"candidate_models" jsonschema:"minItems=1,uniqueItems=true"`
	ObservationFields []string `json:"observation_fields" jsonschema:"minItems=1,uniqueItems=true"`
}

type ClassifierProfile struct {
	LabelMapping map[string]int `json:"label_mapping" jsonschema:"minProperties=1"`
}

type SpanProfile struct {
	Labels     []string `json:"labels" jsonschema:"minItems=1,uniqueItems=true"`
	OffsetUnit string   `json:"offset_unit" jsonschema:"enum=unicode-codepoint"`
}

// Profile is a tagged union. Dataset and artifact profiles must agree with the target.
type Profile struct {
	TargetContract Target             `json:"target_contract"`
	Selector       *SelectorProfile   `json:"selector,omitempty"`
	Classifier     *ClassifierProfile `json:"classifier,omitempty"`
	Spans          *SpanProfile       `json:"spans,omitempty"`
}

type DataAssetSpec struct {
	Name           string `json:"name" jsonschema:"minLength=1"`
	TargetContract Target `json:"target_contract"`
}
type DataAsset struct {
	Metadata
	DataAssetSpec
}

type File struct {
	Handle    string `json:"handle" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	Digest    string `json:"digest" jsonschema:"pattern=^sha256:[0-9a-f]{64}$"`
	SizeBytes int64  `json:"size_bytes" jsonschema:"minimum=0"`
}
type Upload struct {
	Metadata
	File
}

type SnapshotSpec struct {
	AssetID       string      `json:"asset_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	UploadHandle  string      `json:"upload_handle" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	Source        *ModelRef   `json:"source,omitempty"`
	Preprocessing []Component `json:"preprocessing,omitempty"`
	Profile       Profile     `json:"profile"`
}

// DataSnapshot has no update operation. Its identity includes the bytes and preprocessing.
type DataSnapshot struct {
	Metadata
	SnapshotSpec
	Content File `json:"content"`
}

type ExperimentSpec struct {
	Name           string `json:"name" jsonschema:"minLength=1"`
	TargetContract Target `json:"target_contract"`
}
type Experiment struct {
	Metadata
	ExperimentSpec
}

type TaskSpec struct {
	Key       string    `json:"key" jsonschema:"minLength=1"`
	DependsOn []string  `json:"depends_on,omitempty" jsonschema:"uniqueItems=true"`
	Executor  Component `json:"executor"`
}

// RunSpec freezes dataset identity, trainer version and JSON parameters at submission.
// Trainer-specific parameter validation belongs to the capability planner.
type RunSpec struct {
	BaseModel      *ModelRef      `json:"base_model,omitempty"`
	ExperimentID   string         `json:"experiment_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	SnapshotID     string         `json:"snapshot_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	TargetContract Target         `json:"target_contract"`
	Trainer        Component      `json:"trainer"`
	Parameters     map[string]any `json:"parameters,omitempty"`
	Tasks          []TaskSpec     `json:"tasks" jsonschema:"minItems=1"`
}
type TrainingRun struct {
	Metadata
	Spec      RunSpec   `json:"spec"`
	Status    Status    `json:"status"`
	UpdatedAt time.Time `json:"updated_at"`
}
type Attempt struct {
	Metadata
	Number       int        `json:"number" jsonschema:"minimum=1"`
	WorkerHandle string     `json:"worker_handle,omitempty" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	Status       Status     `json:"status"`
	StartedAt    *time.Time `json:"started_at,omitempty"`
	FinishedAt   *time.Time `json:"finished_at,omitempty"`
	Diagnostic   string     `json:"diagnostic,omitempty"`
}
type RunTask struct {
	Metadata
	RunID    string    `json:"run_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	Spec     TaskSpec  `json:"spec"`
	Status   Status    `json:"status"`
	Attempts []Attempt `json:"attempts"`
}
type RunGraph struct {
	Run     TrainingRun `json:"run"`
	Tasks   []RunTask   `json:"tasks"`
	Outputs RunOutputs  `json:"outputs"`
}

// RunOutputs indexes published resources so consumers can discover results from
// a run ID alone. Resource provenance identifies the producing task and attempt.
type RunOutputs struct {
	ArtifactIDs      []string `json:"artifact_ids" jsonschema:"uniqueItems=true"`
	EvaluationIDs    []string `json:"evaluation_ids" jsonschema:"uniqueItems=true"`
	QualificationIDs []string `json:"qualification_ids" jsonschema:"uniqueItems=true"`
}

// Provenance links produced resources to their immutable inputs and execution.
// ManifestBundle preserves the existing, richer Router Model provenance manifests.
type Provenance struct {
	RunID          string    `json:"run_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	TaskID         string    `json:"task_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	AttemptID      string    `json:"attempt_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	SnapshotID     string    `json:"snapshot_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	Trainer        Component `json:"trainer"`
	ManifestBundle *File     `json:"manifest_bundle,omitempty"`
}
type Artifact struct {
	Metadata
	Provenance Provenance `json:"provenance"`
	Profile    Profile    `json:"profile"`
}
type ArtifactVariant struct {
	Metadata
	ArtifactID string `json:"artifact_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	ArtifactVariantSpec
}

// ArtifactVariantSpec describes one representation of the same logical artifact.
type ArtifactVariantSpec struct {
	Format Component `json:"format"`
	// Files maps logical relative names (e.g. config.json) to owned bytes.
	// Names describe the artifact layout, never server filesystem locations.
	Files map[string]File `json:"files"`
}
type Evaluation struct {
	Metadata
	Provenance Provenance `json:"provenance"`
	EvaluationSpec
}

// EvaluationSpec references an existing variant; evaluation does not mint new artifacts.
type EvaluationSpec struct {
	VariantID  string             `json:"variant_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	SnapshotID string             `json:"snapshot_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	Method     Component          `json:"method"`
	Metrics    map[string]float64 `json:"metrics"`
}
type Qualification struct {
	Metadata
	Provenance Provenance `json:"provenance"`
	QualificationSpec
}

type QualificationSpec struct {
	VariantID  string    `json:"variant_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	Runtime    Component `json:"runtime"`
	Compatible bool      `json:"compatible"`
	Receipt    File      `json:"receipt"`
}
type BindingProposalSpec struct {
	VariantID       string `json:"variant_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	QualificationID string `json:"qualification_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	Name            string `json:"name" jsonschema:"minLength=1"`
}
type BindingProposal struct {
	Metadata
	BindingProposalSpec
}
type Event struct {
	Sequence   int64     `json:"sequence"`
	RunID      string    `json:"run_id" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	TaskID     string    `json:"task_id,omitempty" jsonschema:"pattern=^[a-z]+_[a-zA-Z0-9-]+$"`
	Status     Status    `json:"status"`
	Diagnostic string    `json:"diagnostic,omitempty"`
	RecordedAt time.Time `json:"recorded_at"`
}
