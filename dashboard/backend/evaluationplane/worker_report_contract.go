package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"time"
)

// workerReport is the untrusted worker-to-server evidence envelope. It has no
// attestation field: only the server can publish the externally readable
// Report contract after every evidence validator succeeds.
type workerReport struct {
	SchemaVersion string         `json:"schema_version"`
	Run           workerRunState `json:"run"`
	Summary       ReportSummary  `json:"summary"`
	Tracks        []TrackReport  `json:"tracks"`
	Metrics       []Metric       `json:"metrics"`
	Gates         []Gate         `json:"gates"`
	Costs         CostLedgers    `json:"costs"`
	Provenance    Provenance     `json:"provenance"`
	Artifacts     []Artifact     `json:"artifacts"`
}

// workerRunState is the exact nested run echo accepted from the untrusted
// worker. Server-derived per-track evidence levels and controlled-pair
// membership are absent and therefore fail strict JSON decoding; publication
// canonicalizes every lifecycle echo against the durable Run.
type workerRunState struct {
	SchemaVersion        string                `json:"schema_version"`
	ID                   string                `json:"id"`
	ClientRequestID      string                `json:"client_request_id"`
	Name                 string                `json:"name"`
	Description          string                `json:"description"`
	Status               RunStatus             `json:"status"`
	Mode                 Mode                  `json:"mode"`
	EvidenceLevel        EvidenceLevel         `json:"evidence_level"`
	TargetID             string                `json:"target_id"`
	Mixture              *CatalogMixture       `json:"mixture,omitempty"`
	ChangeProfile        ChangeProfile         `json:"change_profile"`
	SuiteIDs             []string              `json:"suite_ids"`
	TrackIDs             []TrackID             `json:"track_ids"`
	SampleLimit          int                   `json:"sample_limit"`
	Concurrency          int                   `json:"concurrency"`
	CapacitySLO          *CapacitySLO          `json:"capacity_slo,omitempty"`
	CapacityLoadProtocol *CapacityLoadProtocol `json:"capacity_load_protocol,omitempty"`
	Seed                 int64                 `json:"seed"`
	BaselineRunID        string                `json:"baseline_run_id,omitempty"`
	Progress             RunProgress           `json:"progress"`
	CreatedAt            time.Time             `json:"created_at"`
	StartedAt            *time.Time            `json:"started_at,omitempty"`
	CompletedAt          *time.Time            `json:"completed_at,omitempty"`
	Error                string                `json:"error,omitempty"`
}

func (state workerRunState) reportRun() Run {
	return Run{
		SchemaVersion: state.SchemaVersion, ID: state.ID, ClientRequestID: state.ClientRequestID,
		Name: state.Name, Description: state.Description, Status: state.Status, Mode: state.Mode,
		EvidenceLevel: state.EvidenceLevel, TargetID: state.TargetID, Mixture: state.Mixture,
		ChangeProfile: state.ChangeProfile, SuiteIDs: state.SuiteIDs, TrackIDs: state.TrackIDs,
		SampleLimit: state.SampleLimit, Concurrency: state.Concurrency, CapacitySLO: state.CapacitySLO,
		CapacityLoadProtocol: state.CapacityLoadProtocol, Seed: state.Seed,
		BaselineRunID: state.BaselineRunID, Progress: state.Progress, CreatedAt: state.CreatedAt,
		StartedAt: state.StartedAt, CompletedAt: state.CompletedAt, Error: state.Error,
	}
}

func decodeWorkerReportStrict(runID string, data []byte) (Report, error) {
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return Report{}, fmt.Errorf("%w: decode evaluation worker report: %w", ErrInvalid, err)
	}
	var draft workerReport
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&draft); err != nil {
		return Report{}, fmt.Errorf("%w: decode evaluation worker report: %w", ErrInvalid, err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return Report{}, fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	report := Report{
		SchemaVersion: draft.SchemaVersion,
		Run:           draft.Run.reportRun(), Summary: draft.Summary, Tracks: draft.Tracks,
		Metrics: draft.Metrics, Gates: draft.Gates, Costs: draft.Costs,
		Recommendations: []string{}, Provenance: draft.Provenance,
		Artifacts: draft.Artifacts,
		// Method reports are server-owned reductions. Workers never submit
		// aggregates that could be mistaken for independently attested curves.
		MethodReports: []CompoundModelBudgetReport{},
	}
	if err := validateReportShape(runID, report); err != nil {
		return Report{}, fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	return report, nil
}
