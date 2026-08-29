// Package evaluationplane owns durable evaluation run bundles and execution.
// Scientific evidence lives in the bundle files under the configured data
// directory; the Dashboard does not project those records into a database.
package evaluationplane

import (
	"errors"
	"time"
)

const (
	SchemaVersion       = "evaluation.v1"
	GateContractVersion = "evaluation-release-gates.v1"
)

var (
	ErrNotFound = errors.New("evaluation resource not found")
	ErrConflict = errors.New("evaluation resource conflict")
	ErrInvalid  = errors.New("invalid evaluation request")
)

type (
	TrackID       string
	Mode          string
	EvidenceLevel string
	RunStatus     string
	GateVerdict   string
	ChangeProfile string
)

const (
	ModeReplay Mode = "replay"
	ModeLive   Mode = "live"

	StatusPending   RunStatus = "pending"
	StatusRunning   RunStatus = "running"
	StatusCompleted RunStatus = "completed"
	StatusFailed    RunStatus = "failed"
	StatusCancelled RunStatus = "cancelled"
)

type CatalogTrack struct {
	ID             TrackID         `json:"id"`
	Name           string          `json:"name"`
	Description    string          `json:"description"`
	Modes          []Mode          `json:"modes"`
	Metrics        []string        `json:"metrics"`
	EvidenceLevels []EvidenceLevel `json:"evidence_levels,omitempty"`
}

type CatalogSuite struct {
	ID            string        `json:"id"`
	Name          string        `json:"name"`
	Description   string        `json:"description"`
	TrackIDs      []TrackID     `json:"track_ids"`
	Modes         []Mode        `json:"modes"`
	EvidenceLevel EvidenceLevel `json:"evidence_level"`
	CaseCount     int           `json:"case_count,omitempty"`
	Revision      string        `json:"revision,omitempty"`
	Tags          []string      `json:"tags,omitempty"`
}

// CatalogTarget is intentionally safe for browser disclosure. Endpoint URLs
// are retained only in the server-owned target registry and staged manifest.
type CatalogTarget struct {
	ID            string            `json:"id"`
	Name          string            `json:"name"`
	Description   string            `json:"description"`
	Kind          string            `json:"kind"`
	TrackIDs      []TrackID         `json:"track_ids"`
	Modes         []Mode            `json:"modes"`
	EvidenceLevel EvidenceLevel     `json:"evidence_level,omitempty"`
	Healthy       *bool             `json:"healthy,omitempty"`
	Labels        map[string]string `json:"labels,omitempty"`
}

type CatalogChangeProfile struct {
	ID          ChangeProfile `json:"id"`
	Name        string        `json:"name"`
	Description string        `json:"description"`
}

type Catalog struct {
	SchemaVersion       string                 `json:"schema_version"`
	GateContractVersion string                 `json:"gate_contract_version"`
	GeneratedAt         time.Time              `json:"generated_at"`
	ChangeProfiles      []CatalogChangeProfile `json:"change_profiles"`
	Tracks              []CatalogTrack         `json:"tracks"`
	Suites              []CatalogSuite         `json:"suites"`
	Targets             []CatalogTarget        `json:"targets"`
}

type CreateRunRequest struct {
	Name          string        `json:"name"`
	Description   string        `json:"description"`
	SuiteIDs      []string      `json:"suite_ids"`
	TrackIDs      []TrackID     `json:"track_ids"`
	Mode          Mode          `json:"mode"`
	TargetID      string        `json:"target_id"`
	ChangeProfile ChangeProfile `json:"change_profile"`
	SampleLimit   int           `json:"sample_limit"`
	Concurrency   int           `json:"concurrency"`
	Seed          int64         `json:"seed"`
	BaselineRunID string        `json:"baseline_run_id,omitempty"`
	AutoStart     bool          `json:"auto_start"`
}

type RunProgress struct {
	Percent        float64 `json:"percent"`
	Completed      int     `json:"completed"`
	Total          int     `json:"total"`
	CurrentTrackID TrackID `json:"current_track_id,omitempty"`
	Message        string  `json:"message,omitempty"`
}

type Run struct {
	SchemaVersion string        `json:"schema_version"`
	ID            string        `json:"id"`
	Name          string        `json:"name"`
	Description   string        `json:"description"`
	Status        RunStatus     `json:"status"`
	Mode          Mode          `json:"mode"`
	EvidenceLevel EvidenceLevel `json:"evidence_level"`
	TargetID      string        `json:"target_id"`
	ChangeProfile ChangeProfile `json:"change_profile"`
	SuiteIDs      []string      `json:"suite_ids"`
	TrackIDs      []TrackID     `json:"track_ids"`
	SampleLimit   int           `json:"sample_limit"`
	Concurrency   int           `json:"concurrency"`
	Seed          int64         `json:"seed"`
	BaselineRunID string        `json:"baseline_run_id,omitempty"`
	Progress      RunProgress   `json:"progress"`
	CreatedAt     time.Time     `json:"created_at"`
	StartedAt     *time.Time    `json:"started_at,omitempty"`
	CompletedAt   *time.Time    `json:"completed_at,omitempty"`
	Error         string        `json:"error,omitempty"`
}

type ManifestTarget struct {
	SchemaVersion         string     `json:"schema_version"`
	ID                    string     `json:"id"`
	Kind                  string     `json:"kind"`
	RouterAPIURL          string     `json:"router_api_url,omitempty"`
	EnvoyURL              string     `json:"envoy_url,omitempty"`
	RouterAPIKey          *SecretRef `json:"router_api_key,omitempty"`
	EnvoyAPIKey           *SecretRef `json:"envoy_api_key,omitempty"`
	ModelArms             []ModelArm `json:"model_arms,omitempty"`
	BackendTopologyDigest string     `json:"backend_topology_digest,omitempty"`
}

// SecretRef names a server-owned environment variable made available only to
// the fixed evaluation worker. Literal credentials are not part of the
// manifest contract.
type SecretRef struct {
	SchemaVersion string `json:"schema_version"`
	Env           string `json:"env"`
}

// ModelArm is a server-owned logical model identity. Connectivity and literal
// provider identities remain outside the evidence bundle; the latter is
// represented only by a one-way digest.
type ModelArm struct {
	ID                            string   `json:"id"`
	Model                         string   `json:"model"`
	ProviderModelIDDigest         string   `json:"provider_model_id_digest"`
	InputCostPerMillionTokensUSD  float64  `json:"input_cost_per_million_tokens_usd"`
	OutputCostPerMillionTokensUSD float64  `json:"output_cost_per_million_tokens_usd"`
	Capabilities                  []string `json:"capabilities,omitempty"`
	Modalities                    []string `json:"modalities,omitempty"`
	ContextWindowTokens           *int     `json:"context_window_tokens,omitempty"`
	ParameterSize                 *string  `json:"parameter_size,omitempty"`
	RuntimeRevision               *string  `json:"runtime_revision,omitempty"`
	ConfigDigest                  *string  `json:"config_digest,omitempty"`
}

type RunManifest struct {
	SchemaVersion string `json:"schema_version"`
	// ManifestDigest is computed by the Dashboard over the semantic manifest
	// value with this field omitted. The Python worker treats it as an opaque,
	// server-owned identity and echoes it into lineage evidence.
	ManifestDigest       string            `json:"manifest_digest"`
	RunID                string            `json:"run_id"`
	Mode                 Mode              `json:"mode"`
	Target               ManifestTarget    `json:"target"`
	ChangeProfile        ChangeProfile     `json:"change_profile"`
	GateContractVersion  string            `json:"gate_contract_version"`
	SuiteIDs             []string          `json:"suite_ids"`
	SuiteRevisions       map[string]string `json:"suite_revisions"`
	TrackIDs             []TrackID         `json:"track_ids"`
	SampleLimit          int               `json:"sample_limit"`
	Concurrency          int               `json:"concurrency"`
	Seed                 int64             `json:"seed"`
	BaselineRunID        string            `json:"baseline_run_id,omitempty"`
	CreatedAt            time.Time         `json:"created_at"`
	CodeRevision         string            `json:"code_revision"`
	ConfigDigest         string            `json:"config_digest"`
	PolicySnapshotDigest string            `json:"policy_snapshot_digest"`
	RedactionPolicy      string            `json:"redaction_policy"`
}

type Event struct {
	ID        string              `json:"id,omitempty"`
	RunID     string              `json:"run_id"`
	Type      string              `json:"type"`
	Timestamp time.Time           `json:"timestamp"`
	Message   string              `json:"message"`
	TrackID   TrackID             `json:"track_id,omitempty"`
	Progress  *RunProgress        `json:"progress,omitempty"`
	Payload   *WorkerEventPayload `json:"payload,omitempty"`
}

// WorkerEventPayload is deliberately scalar-only. Worker stdout is an
// untrusted control protocol and must never project prompts, provider
// diagnostics, credentials, URLs, or arbitrary JSON into durable SSE events.
type WorkerEventPayload struct {
	RecordCount *int        `json:"record_count,omitempty"`
	Verdict     GateVerdict `json:"verdict,omitempty"`
}

// WorkerEvent is the only stdout protocol accepted from the fixed worker.
// Run identity, durable event identity, and timestamps are server-owned.
type WorkerEvent struct {
	Type     string              `json:"type"`
	Message  string              `json:"message"`
	TrackID  TrackID             `json:"track_id,omitempty"`
	Progress *RunProgress        `json:"progress,omitempty"`
	Payload  *WorkerEventPayload `json:"payload,omitempty"`
}

type Coverage struct {
	Evaluated          int       `json:"evaluated"`
	Total              int       `json:"total"`
	Fraction           float64   `json:"fraction"`
	Unavailable        int       `json:"unavailable,omitempty"`
	ConfidenceLevel    float64   `json:"confidence_level,omitempty"`
	ConfidenceInterval []float64 `json:"confidence_interval,omitempty"`
}

type Metric struct {
	ID                 string    `json:"id"`
	Name               string    `json:"name"`
	TrackID            TrackID   `json:"track_id,omitempty"`
	Value              *float64  `json:"value"`
	Unit               string    `json:"unit"`
	Direction          string    `json:"direction,omitempty"`
	BaselineValue      *float64  `json:"baseline_value,omitempty"`
	Delta              *float64  `json:"delta,omitempty"`
	ConfidenceInterval []float64 `json:"confidence_interval,omitempty"`
	SampleCount        int       `json:"sample_count,omitempty"`
}

type GateThreshold struct {
	Operator string  `json:"operator"`
	Value    float64 `json:"value"`
	Unit     string  `json:"unit,omitempty"`
}

type Gate struct {
	ID              string         `json:"id"`
	Name            string         `json:"name"`
	Description     string         `json:"description,omitempty"`
	TrackID         TrackID        `json:"track_id,omitempty"`
	Disposition     string         `json:"disposition"`
	Verdict         GateVerdict    `json:"verdict"`
	ChangeProfile   ChangeProfile  `json:"change_profile"`
	ContractVersion string         `json:"contract_version"`
	EvidenceRefs    []string       `json:"evidence_refs"`
	EvidenceLevel   EvidenceLevel  `json:"evidence_level,omitempty"`
	Observed        *float64       `json:"observed,omitempty"`
	Threshold       *GateThreshold `json:"threshold,omitempty"`
	SampleCount     *int           `json:"sample_count,omitempty"`
	Coverage        *Coverage      `json:"coverage,omitempty"`
	Owner           string         `json:"owner,omitempty"`
	EvaluatedAt     *time.Time     `json:"evaluated_at,omitempty"`
	Rationale       string         `json:"rationale,omitempty"`
}

type Artifact struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Kind      string `json:"kind"`
	URI       string `json:"uri,omitempty"`
	Digest    string `json:"digest,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	SizeBytes int64  `json:"size_bytes,omitempty"`
}

type Provenance struct {
	SchemaVersion             string            `json:"schema_version"`
	GeneratedAt               time.Time         `json:"generated_at"`
	CodeRevision              string            `json:"code_revision,omitempty"`
	BenchmarkRevisions        map[string]string `json:"benchmark_revisions,omitempty"`
	PolicySnapshotDigest      string            `json:"policy_snapshot_digest,omitempty"`
	BindingSnapshotDigest     string            `json:"binding_snapshot_digest,omitempty"`
	PoolSnapshotDigest        string            `json:"pool_snapshot_digest,omitempty"`
	WorkloadSnapshotDigest    string            `json:"workload_snapshot_digest,omitempty"`
	EnvironmentSnapshotDigest string            `json:"environment_snapshot_digest,omitempty"`
	TargetID                  string            `json:"target_id"`
	Seed                      int64             `json:"seed"`
	RedactionPolicy           string            `json:"redaction_policy,omitempty"`
}

type CostAmount struct {
	Amount       *float64 `json:"amount"`
	Currency     string   `json:"currency"`
	InputTokens  int64    `json:"input_tokens,omitempty"`
	OutputTokens int64    `json:"output_tokens,omitempty"`
	GPUSeconds   float64  `json:"gpu_seconds,omitempty"`
	EnergyKWh    float64  `json:"energy_kwh,omitempty"`
}

type CostLedgers struct {
	Runtime            CostAmount `json:"runtime"`
	EvaluationOverhead CostAmount `json:"evaluation_overhead"`
	CapacityTCO        CostAmount `json:"capacity_tco"`
}

type TrackReport struct {
	TrackID       TrackID       `json:"track_id"`
	Status        string        `json:"status"`
	EvidenceLevel EvidenceLevel `json:"evidence_level"`
	Summary       string        `json:"summary"`
	Coverage      Coverage      `json:"coverage"`
	Metrics       []Metric      `json:"metrics"`
	Gates         []Gate        `json:"gates"`
	Artifacts     []Artifact    `json:"artifacts,omitempty"`
	Error         string        `json:"error,omitempty"`
}

type ReportSummary struct {
	Verdict          GateVerdict `json:"verdict"`
	QualityScore     *float64    `json:"quality_score"`
	LatencyP95MS     *float64    `json:"latency_p95_ms"`
	RuntimeCost      *float64    `json:"runtime_cost"`
	CapacityTCO      *float64    `json:"capacity_tco"`
	Coverage         Coverage    `json:"coverage"`
	PassedGates      int         `json:"passed_gates"`
	FailedGates      int         `json:"failed_gates"`
	UnavailableGates int         `json:"unavailable_gates"`
}

type Report struct {
	SchemaVersion   string        `json:"schema_version"`
	Run             Run           `json:"run"`
	Summary         ReportSummary `json:"summary"`
	Tracks          []TrackReport `json:"tracks"`
	Metrics         []Metric      `json:"metrics"`
	Gates           []Gate        `json:"gates"`
	Costs           CostLedgers   `json:"costs"`
	Recommendations []string      `json:"recommendations"`
	Provenance      Provenance    `json:"provenance"`
	Artifacts       []Artifact    `json:"artifacts"`
}

type Comparison struct {
	SchemaVersion   string      `json:"schema_version"`
	BaselineRunID   string      `json:"baseline_run_id"`
	CandidateRunID  string      `json:"candidate_run_id"`
	Verdict         GateVerdict `json:"verdict"`
	Summary         string      `json:"summary"`
	Metrics         []Metric    `json:"metrics"`
	Gates           []Gate      `json:"gates"`
	Recommendations []string    `json:"recommendations"`
	CreatedAt       time.Time   `json:"created_at,omitempty"`
}
