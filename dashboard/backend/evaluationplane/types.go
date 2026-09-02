// Package evaluationplane owns durable evaluation run bundles and execution.
// Scientific evidence lives in the bundle files under the configured data
// directory; the Dashboard does not project those records into a database.
package evaluationplane

import (
	"errors"
	"time"
)

const (
	SchemaVersion             = "evaluation.v1"
	GateContractVersion       = "evaluation-release-gates.v2"
	ServerAttestationRevision = "evaluation-server-attestation.v2"
)

var (
	ErrNotFound = errors.New("evaluation resource not found")
	ErrConflict = errors.New("evaluation resource conflict")
	ErrInvalid  = errors.New("invalid evaluation request")
)

type (
	TrackID         string
	Mode            string
	EvidenceLevel   string
	RunStatus       string
	GateVerdict     string
	DecisionVerdict string
	GateDisposition string
	ChangeProfile   string
)

const (
	GateVerdictPass          GateVerdict = "pass"
	GateVerdictFail          GateVerdict = "fail"
	GateVerdictUnavailable   GateVerdict = "unavailable"
	GateVerdictNotApplicable GateVerdict = "not_applicable"

	DecisionVerdictPass        DecisionVerdict = "pass"
	DecisionVerdictFail        DecisionVerdict = "fail"
	DecisionVerdictUnavailable DecisionVerdict = "unavailable"
)

func validGateVerdict(verdict GateVerdict) bool {
	return verdict == GateVerdictPass || verdict == GateVerdictFail ||
		verdict == GateVerdictUnavailable || verdict == GateVerdictNotApplicable
}

func validDecisionVerdict(verdict DecisionVerdict) bool {
	return verdict == DecisionVerdictPass || verdict == DecisionVerdictFail ||
		verdict == DecisionVerdictUnavailable
}

const (
	GateDispositionRequired      GateDisposition = "required"
	GateDispositionAdvisory      GateDisposition = "advisory"
	GateDispositionNotApplicable GateDisposition = "not_applicable"
)

func validGateDisposition(disposition GateDisposition) bool {
	return disposition == GateDispositionRequired || disposition == GateDispositionAdvisory ||
		disposition == GateDispositionNotApplicable
}

const (
	ModeReplay Mode = "replay"
	ModeLive   Mode = "live"

	StatusPending   RunStatus = "pending"
	StatusRunning   RunStatus = "running"
	StatusSealing   RunStatus = "sealing"
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
	EvidenceLevels []EvidenceLevel `json:"evidence_levels"`
}

type CatalogSuite struct {
	ID               string            `json:"id"`
	Name             string            `json:"name"`
	Description      string            `json:"description"`
	Executors        map[Mode]string   `json:"executors"`
	TrackIDs         []TrackID         `json:"track_ids"`
	Modes            []Mode            `json:"modes"`
	EvidenceLevel    EvidenceLevel     `json:"evidence_level"`
	CaseCount        int               `json:"case_count,omitempty"`
	CampaignProtocol *CampaignProtocol `json:"campaign_protocol,omitempty"`
	Revision         string            `json:"revision"`
	Tags             []string          `json:"tags"`
	Methods          []CatalogMethod   `json:"methods"`
}

type CatalogMethodEvidenceSource string

const (
	CatalogMethodEvidenceSourceDiagnosticFixture  CatalogMethodEvidenceSource = "diagnostic_fixture"
	CatalogMethodEvidenceSourceLiveRuntime        CatalogMethodEvidenceSource = "live_runtime"
	CatalogMethodEvidenceSourceNormalizedImport   CatalogMethodEvidenceSource = "normalized_import"
	CatalogMethodEvidenceSourceServerBrokeredLive CatalogMethodEvidenceSource = "server_brokered_live"
	CatalogMethodEvidenceSourceLiveProduction     CatalogMethodEvidenceSource = "live_production"
)

var catalogMethodEvidenceSourceInventory = [...]CatalogMethodEvidenceSource{
	CatalogMethodEvidenceSourceDiagnosticFixture,
	CatalogMethodEvidenceSourceLiveRuntime,
	CatalogMethodEvidenceSourceNormalizedImport,
	CatalogMethodEvidenceSourceServerBrokeredLive,
	CatalogMethodEvidenceSourceLiveProduction,
}

func validCatalogMethodEvidenceSource(source CatalogMethodEvidenceSource) bool {
	for _, known := range catalogMethodEvidenceSourceInventory {
		if source == known {
			return true
		}
	}
	return false
}

type CatalogMethod struct {
	ID               string                      `json:"id"`
	TrackID          TrackID                     `json:"track_id"`
	QualifiedGateIDs []string                    `json:"qualified_gate_ids"`
	EvidenceSource   CatalogMethodEvidenceSource `json:"evidence_source"`
	Status           string                      `json:"status"`
	Reason           string                      `json:"reason,omitempty"`
}

// CatalogTarget is intentionally safe for browser disclosure. Endpoint URLs
// are retained only in the server-owned target registry and staged manifest.
type CatalogTarget struct {
	ID                string            `json:"id"`
	Name              string            `json:"name"`
	Description       string            `json:"description"`
	Kind              string            `json:"kind"`
	TrackIDs          []TrackID         `json:"track_ids"`
	Modes             []Mode            `json:"modes"`
	AcceptedExecutors map[Mode][]string `json:"accepted_executors"`
	EvidenceLevel     EvidenceLevel     `json:"evidence_level,omitempty"`
	Healthy           *bool             `json:"healthy,omitempty"`
	Labels            map[string]string `json:"labels,omitempty"`
	Mixture           *CatalogMixture   `json:"mixture,omitempty"`
}

// CatalogMixture is the connectivity-free, immutable identity of one
// request-facing Mixture-of-Models. It is safe to persist in run status and
// disclose to authenticated evaluation readers.
type CatalogMixture struct {
	ID                   string                   `json:"id"`
	EntrypointModel      string                   `json:"entrypoint_model"`
	Aliases              []string                 `json:"aliases"`
	RecipeName           string                   `json:"recipe_name"`
	RecipeDescription    string                   `json:"recipe_description"`
	RecipeDigest         string                   `json:"recipe_digest"`
	PoolDigest           string                   `json:"pool_digest"`
	SelectorPolicyDigest string                   `json:"selector_policy_digest"`
	SelectorDigest       string                   `json:"selector_digest"`
	AdaptationDigest     string                   `json:"adaptation_digest"`
	BindingDigest        string                   `json:"binding_digest"`
	ModelArms            []ModelArm               `json:"model_arms"`
	SupportModels        []SupportModel           `json:"support_models"`
	FallbackArmID        string                   `json:"fallback_arm_id,omitempty"`
	Decisions            []MixtureDecisionBinding `json:"decisions"`
	RoutingRecipePlan    RoutingRecipePlan        `json:"routing_recipe_plan"`
}

// SupportModel freezes the executable identity of a model used by a Recipe's
// selector but not eligible as a routed pool arm. Its backend topology is part
// of the selector binding, not the candidate-pool runtime environment.
type SupportModel struct {
	Model                 string  `json:"model"`
	ProviderModelIDDigest string  `json:"provider_model_id_digest"`
	ConfigDigest          string  `json:"config_digest"`
	RuntimeRevision       *string `json:"runtime_revision,omitempty"`
	BackendTopologyDigest string  `json:"backend_topology_digest"`
}

// MixtureDecisionBinding freezes the candidate arm boundary and selection
// algorithm for one decision in the selected recipe.
type MixtureDecisionBinding struct {
	Name      string   `json:"name"`
	Algorithm string   `json:"algorithm"`
	ArmIDs    []string `json:"arm_ids"`
}

type CatalogChangeProfile struct {
	ID            ChangeProfile         `json:"id"`
	Name          string                `json:"name"`
	Description   string                `json:"description"`
	CampaignSlots []CatalogCampaignSlot `json:"campaign_slots"`
}

type CampaignBindingKind string

const (
	CampaignBindingRun            CampaignBindingKind = "run"
	CampaignBindingControlledPair CampaignBindingKind = "controlled_pair"
	CampaignBindingFidelityPair   CampaignBindingKind = "fidelity_pair"
)

// CatalogCampaignSlot is the server-owned evidence composition contract for one
// release gate. Campaign validation and reduction consume this catalog value
// directly; there is no second campaign disposition or role matrix.
type CatalogCampaignSlot struct {
	GateID               string              `json:"gate_id"`
	Name                 string              `json:"name"`
	Description          string              `json:"description"`
	Disposition          GateDisposition     `json:"disposition"`
	BindingKind          CampaignBindingKind `json:"binding_kind"`
	TrackID              TrackID             `json:"track_id,omitempty"`
	Mode                 Mode                `json:"mode,omitempty"`
	MinimumEvidenceLevel EvidenceLevel       `json:"minimum_evidence_level"`
	AcceptedExecutorIDs  []string            `json:"accepted_executor_ids"`
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
	ClientRequestID      string                `json:"client_request_id"`
	Name                 string                `json:"name"`
	Description          string                `json:"description"`
	SuiteIDs             []string              `json:"suite_ids"`
	TrackIDs             []TrackID             `json:"track_ids"`
	Mode                 Mode                  `json:"mode"`
	TargetID             string                `json:"target_id"`
	ChangeProfile        ChangeProfile         `json:"change_profile"`
	SampleLimit          int                   `json:"sample_limit"`
	Concurrency          int                   `json:"concurrency"`
	CapacitySLO          *CapacitySLO          `json:"capacity_slo,omitempty"`
	CapacityLoadProtocol *CapacityLoadProtocol `json:"capacity_load_protocol,omitempty"`
	Seed                 int64                 `json:"seed"`
	BaselineRunID        string                `json:"baseline_run_id,omitempty"`
}

// CapacitySLO is the immutable operating objective for one live capacity
// sweep. Throughput applies at and above RequiredConcurrency. Scaling
// efficiency is throughput growth divided by concurrency growth between
// adjacent measured levels.
type CapacitySLO struct {
	SchemaVersion                  string  `json:"schema_version"`
	RequiredConcurrency            int64   `json:"required_concurrency"`
	MaxLatencyP95MS                float64 `json:"max_latency_p95_ms"`
	MaxErrorRate                   float64 `json:"max_error_rate"`
	MinThroughputRPS               float64 `json:"min_throughput_rps"`
	MinThroughputScalingEfficiency float64 `json:"min_throughput_scaling_efficiency"`
}

// CapacityLoadProtocol freezes the only admitted live load process. The
// geometric ladder, warmup, independent repetitions, confidence level and
// stability bounds are evidence, not worker-selected tuning knobs.
type CapacityLoadProtocol struct {
	SchemaVersion                      string  `json:"schema_version"`
	Kind                               string  `json:"kind"`
	ConcurrencyLevels                  []int64 `json:"concurrency_levels"`
	WarmupRequestMultiplier            int64   `json:"warmup_request_multiplier"`
	MeasurementRequestsPerRepetition   int64   `json:"measurement_requests_per_repetition"`
	RepetitionsPerLevel                int64   `json:"repetitions_per_level"`
	MinimumMeasurementClustersPerLevel int64   `json:"minimum_measurement_clusters_per_level"`
	ConfidenceLevel                    float64 `json:"confidence_level"`
	MaxErrorRateClusterRange           float64 `json:"max_error_rate_cluster_range"`
	MaxThroughputCV                    float64 `json:"max_throughput_cv"`
	MaxLatencyP95CV                    float64 `json:"max_latency_p95_cv"`
}

type RunProgress struct {
	Percent        float64 `json:"percent"`
	Completed      int     `json:"completed"`
	Total          int     `json:"total"`
	CurrentTrackID TrackID `json:"current_track_id,omitempty"`
	Message        string  `json:"message,omitempty"`
}

type Run struct {
	SchemaVersion        string                       `json:"schema_version"`
	ID                   string                       `json:"id"`
	ClientRequestID      string                       `json:"client_request_id"`
	Name                 string                       `json:"name"`
	Description          string                       `json:"description"`
	Status               RunStatus                    `json:"status"`
	Mode                 Mode                         `json:"mode"`
	EvidenceLevel        EvidenceLevel                `json:"evidence_level"`
	TrackEvidenceLevels  map[TrackID]EvidenceLevel    `json:"track_evidence_levels"`
	TargetID             string                       `json:"target_id"`
	Mixture              *CatalogMixture              `json:"mixture,omitempty"`
	ChangeProfile        ChangeProfile                `json:"change_profile"`
	SuiteIDs             []string                     `json:"suite_ids"`
	TrackIDs             []TrackID                    `json:"track_ids"`
	SampleLimit          int                          `json:"sample_limit"`
	Concurrency          int                          `json:"concurrency"`
	CapacitySLO          *CapacitySLO                 `json:"capacity_slo,omitempty"`
	CapacityLoadProtocol *CapacityLoadProtocol        `json:"capacity_load_protocol,omitempty"`
	Seed                 int64                        `json:"seed"`
	BaselineRunID        string                       `json:"baseline_run_id,omitempty"`
	ControlledPair       *ControlledPairRunMembership `json:"controlled_pair,omitempty"`
	Progress             RunProgress                  `json:"progress"`
	CreatedAt            time.Time                    `json:"created_at"`
	StartedAt            *time.Time                   `json:"started_at,omitempty"`
	CompletedAt          *time.Time                   `json:"completed_at,omitempty"`
	Error                string                       `json:"error,omitempty"`
}

type ControlledPairRunMembership struct {
	PairID string `json:"pair_id"`
	Role   string `json:"role"`
}

type ManifestTarget struct {
	SchemaVersion              string           `json:"schema_version"`
	ID                         string           `json:"id"`
	Kind                       string           `json:"kind"`
	RouterAPIURL               string           `json:"router_api_url,omitempty"`
	EnvoyURL                   string           `json:"envoy_url,omitempty"`
	RouterAPIKey               *SecretRef       `json:"router_api_key,omitempty"`
	EnvoyAPIKey                *SecretRef       `json:"envoy_api_key,omitempty"`
	AgentTaskLedger            *ServiceEndpoint `json:"agent_task_ledger,omitempty"`
	FaultRecoveryLedger        *ServiceEndpoint `json:"fault_recovery_ledger,omitempty"`
	HardPolicyLedger           *ServiceEndpoint `json:"hard_policy_ledger,omitempty"`
	ProductionExperimentLedger *ServiceEndpoint `json:"production_experiment_ledger,omitempty"`
	Mixture                    *ManifestMixture `json:"mixture,omitempty"`
	BackendTopologyDigest      string           `json:"backend_topology_digest,omitempty"`
}

// ManifestMixture is the server-sealed Mixture-of-Models execution subject.
// It contains logical identities and digests only; physical connectivity stays
// on ManifestTarget and is supplied exclusively by the fixed worker broker.
type ManifestMixture struct {
	SchemaVersion        string                   `json:"schema_version"`
	ID                   string                   `json:"id"`
	EntrypointModel      string                   `json:"entrypoint_model"`
	Aliases              []string                 `json:"aliases"`
	RecipeName           string                   `json:"recipe_name"`
	RecipeDescription    string                   `json:"recipe_description"`
	RecipeDigest         string                   `json:"recipe_digest"`
	PoolDigest           string                   `json:"pool_digest"`
	SelectorPolicyDigest string                   `json:"selector_policy_digest"`
	SelectorDigest       string                   `json:"selector_digest"`
	AdaptationDigest     string                   `json:"adaptation_digest"`
	BindingDigest        string                   `json:"binding_digest"`
	ModelArms            []ModelArm               `json:"model_arms"`
	SupportModels        []SupportModel           `json:"support_models"`
	FallbackArmID        string                   `json:"fallback_arm_id,omitempty"`
	Decisions            []MixtureDecisionBinding `json:"decisions"`
	RoutingRecipePlan    RoutingRecipePlan        `json:"routing_recipe_plan"`
}

// SecretRef names a server-owned environment variable. The fixed evaluation
// worker receives only this identity; the Go broker resolves the value and
// applies it to an admitted request. Literal credentials are not part of the
// manifest contract or worker environment.
type SecretRef struct {
	SchemaVersion string `json:"schema_version"`
	Env           string `json:"env"`
}

// ServiceEndpoint freezes one server-owned broker destination. Workers receive
// this declarative identity but never direct network access or literal secrets.
type ServiceEndpoint struct {
	SchemaVersion  string     `json:"schema_version"`
	URL            string     `json:"url"`
	APIKey         *SecretRef `json:"api_key,omitempty"`
	TimeoutSeconds float64    `json:"timeout_seconds"`
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
	ManifestDigest       string                `json:"manifest_digest"`
	RunID                string                `json:"run_id"`
	Name                 string                `json:"name"`
	Description          string                `json:"description"`
	Mode                 Mode                  `json:"mode"`
	Target               ManifestTarget        `json:"target"`
	ChangeProfile        ChangeProfile         `json:"change_profile"`
	GateContractVersion  string                `json:"gate_contract_version"`
	SuiteIDs             []string              `json:"suite_ids"`
	SuiteRevisions       map[string]string     `json:"suite_revisions"`
	SuiteExecutors       map[string]string     `json:"suite_executors"`
	TrackIDs             []TrackID             `json:"track_ids"`
	SampleLimit          int                   `json:"sample_limit"`
	Concurrency          int                   `json:"concurrency"`
	CapacitySLO          *CapacitySLO          `json:"capacity_slo,omitempty"`
	CapacityLoadProtocol *CapacityLoadProtocol `json:"capacity_load_protocol,omitempty"`
	Seed                 int64                 `json:"seed"`
	BaselineRunID        string                `json:"baseline_run_id,omitempty"`
	CreatedAt            time.Time             `json:"created_at"`
	CodeRevision         string                `json:"code_revision"`
	ConfigDigest         string                `json:"config_digest"`
	PolicySnapshotDigest string                `json:"policy_snapshot_digest"`
	RedactionPolicy      string                `json:"redaction_policy"`
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
	RecordCount *int            `json:"record_count,omitempty"`
	Verdict     DecisionVerdict `json:"verdict,omitempty"`
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
