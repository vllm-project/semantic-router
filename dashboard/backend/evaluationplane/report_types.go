package evaluationplane

import "time"

type Coverage struct {
	Evaluated          int       `json:"evaluated"`
	Total              int       `json:"total"`
	Fraction           float64   `json:"fraction"`
	Unavailable        int       `json:"unavailable,omitempty"`
	ConfidenceLevel    float64   `json:"confidence_level,omitempty"`
	ConfidenceInterval []float64 `json:"confidence_interval,omitempty"`
}

type Metric struct {
	ID                 string                   `json:"id"`
	Name               string                   `json:"name"`
	TrackID            TrackID                  `json:"track_id,omitempty"`
	Value              *float64                 `json:"value"`
	Unit               string                   `json:"unit"`
	Direction          string                   `json:"direction,omitempty"`
	BaselineValue      *float64                 `json:"baseline_value,omitempty"`
	Delta              *float64                 `json:"delta,omitempty"`
	ConfidenceInterval []float64                `json:"confidence_interval,omitempty"`
	SampleCount        int                      `json:"sample_count,omitempty"`
	AnalysisProvenance MetricAnalysisProvenance `json:"analysis_provenance"`
}

// MetricAnalysisProvenance describes the estimator that produced one published
// metric. It is mandatory evidence, not a display hint: reports without this
// versioned plan are rejected before server attestation.
type MetricAnalysisProvenance struct {
	ContractVersion    string `json:"contract_version"`
	EstimatorID        string `json:"estimator_id"`
	EstimatorVersion   string `json:"estimator_version"`
	AnalysisUnit       string `json:"analysis_unit"`
	ClusterUnit        string `json:"cluster_unit"`
	Weighting          string `json:"weighting"`
	Missingness        string `json:"missingness"`
	ExclusionPolicy    string `json:"exclusion_policy"`
	ObservedExclusions *int   `json:"observed_exclusions"`
}

type GateThreshold struct {
	Operator string  `json:"operator"`
	Value    float64 `json:"value"`
	Unit     string  `json:"unit,omitempty"`
}

type Gate struct {
	ID              string          `json:"id"`
	Name            string          `json:"name"`
	Description     string          `json:"description,omitempty"`
	TrackID         TrackID         `json:"track_id,omitempty"`
	Disposition     GateDisposition `json:"disposition"`
	Verdict         GateVerdict     `json:"verdict"`
	ChangeProfile   ChangeProfile   `json:"change_profile"`
	ContractVersion string          `json:"contract_version"`
	EvidenceRefs    []string        `json:"evidence_refs"`
	EvidenceLevel   EvidenceLevel   `json:"evidence_level,omitempty"`
	Observed        *float64        `json:"observed,omitempty"`
	Threshold       *GateThreshold  `json:"threshold,omitempty"`
	SampleCount     *int            `json:"sample_count,omitempty"`
	Coverage        *Coverage       `json:"coverage,omitempty"`
	Owner           string          `json:"owner,omitempty"`
	EvaluatedAt     *time.Time      `json:"evaluated_at,omitempty"`
	Rationale       string          `json:"rationale,omitempty"`
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
	Verdict          DecisionVerdict `json:"verdict"`
	QualityScore     *float64        `json:"quality_score"`
	LatencyP95MS     *float64        `json:"latency_p95_ms"`
	RuntimeCost      *float64        `json:"runtime_cost"`
	CapacityTCO      *float64        `json:"capacity_tco"`
	Coverage         Coverage        `json:"coverage"`
	PassedGates      int             `json:"passed_gates"`
	FailedGates      int             `json:"failed_gates"`
	UnavailableGates int             `json:"unavailable_gates"`
}

type Report struct {
	SchemaVersion       string                         `json:"schema_version"`
	AttestationRevision string                         `json:"attestation_revision"`
	Run                 Run                            `json:"run"`
	Summary             ReportSummary                  `json:"summary"`
	Tracks              []TrackReport                  `json:"tracks"`
	Metrics             []Metric                       `json:"metrics"`
	Gates               []Gate                         `json:"gates"`
	Costs               CostLedgers                    `json:"costs"`
	Recommendations     []string                       `json:"recommendations"`
	Provenance          Provenance                     `json:"provenance"`
	Artifacts           []Artifact                     `json:"artifacts"`
	MethodReports       []CompoundModelBudgetReport    `json:"method_reports"`
	RoutingRecipeReport *RoutingRecipeEvaluationReport `json:"routing_recipe_report"`
}

type Comparison struct {
	SchemaVersion       string                `json:"schema_version"`
	AttestationRevision string                `json:"attestation_revision"`
	BaselineRunID       string                `json:"baseline_run_id"`
	CandidateRunID      string                `json:"candidate_run_id"`
	Verdict             DecisionVerdict       `json:"verdict"`
	Summary             string                `json:"summary"`
	Metrics             []Metric              `json:"metrics"`
	Statistics          []ComparisonStatistic `json:"statistics"`
	Gates               []Gate                `json:"gates"`
	Recommendations     []string              `json:"recommendations"`
	CreatedAt           time.Time             `json:"created_at"`
}

// ComparisonStatistic is a server reduction over independent, case-clustered
// baseline/candidate analysis units. The worker cannot emit or attest this
// contract. A statistic is promotion-conclusive only when its frozen minimum
// cohort and two-sided 95% intervals are present.
type ComparisonStatistic struct {
	ID                          string      `json:"id"`
	TrackID                     TrackID     `json:"track_id"`
	EstimatorID                 string      `json:"estimator_id"`
	EstimatorVersion            string      `json:"estimator_version"`
	AnalysisUnit                string      `json:"analysis_unit"`
	Direction                   string      `json:"direction"`
	NonInferiorityMargin        float64     `json:"non_inferiority_margin"`
	BaselineValue               float64     `json:"baseline_value"`
	CandidateValue              float64     `json:"candidate_value"`
	Delta                       float64     `json:"delta"`
	ConfidenceLevel             float64     `json:"confidence_level"`
	DeltaConfidenceInterval     []float64   `json:"delta_confidence_interval"`
	CandidateConfidenceInterval []float64   `json:"candidate_confidence_interval"`
	SampleCount                 int         `json:"sample_count"`
	Verdict                     GateVerdict `json:"verdict"`
}
