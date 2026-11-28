// Package selectiontrace contains data-only explanations of model selection.
package selectiontrace

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

type StageAction string

const (
	StageApplied StageAction = "applied"
	StageSkipped StageAction = "skipped"
)

type StageReason string

const (
	ToleranceBand             StageReason = "tolerance_band"
	NoAvailableValues         StageReason = "no_available_values"
	IncompleteLatencyCoverage StageReason = "incomplete_latency_coverage"
)

type EliminationReason string

const (
	MissingMeasurement EliminationReason = "missing_measurement"
	OutsideTolerance   EliminationReason = "outside_tolerance"
)

// MultiFactorObjective describes the base lexicographic objective after hard
// eligibility filters. Its size is bounded by configured priorities and their
// surviving candidates. Weighted objectives omit the stages and survivors, and
// record a trace only for candidate estimates. Stages not reached after a
// singleton survives are omitted, rather than reported as evaluated.
type MultiFactorObjective struct {
	Stages []ObjectiveStage `json:"stages,omitempty"`
	// FinalSurvivors are objective-eligible candidates, not a claim about the
	// final dispatched model after learning or session policy.
	FinalSurvivors []config.ModelRef `json:"final_survivors,omitempty"`
	// Estimates are the per-request candidate estimates, when the decision
	// turns them on.
	Estimates *CandidateEstimates `json:"estimates,omitempty"`
}

// CandidateEstimates are the per-request estimates of the candidates within
// the SLO ceilings, one per candidate in candidate order. In observe
// mode they are recorded only: the objective above and the selected model
// are what they would be without them.
type CandidateEstimates struct {
	Mode       string              `json:"mode"`
	Candidates []CandidateEstimate `json:"candidates"`
}

type EstimateStatus string

const (
	EstimateAvailable        EstimateStatus = "available"
	EstimateUnavailable      EstimateStatus = "unavailable"
	EstimateDeadlineExceeded EstimateStatus = "deadline_exceeded"
)

// EstimateSourceSystemOne is an estimate the System One model answered zero
// shot.
const EstimateSourceSystemOne = "systemone"

// CandidateEstimate is one candidate's estimate, keyed by the exact
// candidate (model and reasoning setting). Its values are present only when
// Status is available.
type CandidateEstimate struct {
	Candidate config.ModelRef `json:"candidate"`
	Status    EstimateStatus  `json:"status"`
	// Reason is why an estimate is unavailable: the model's or the call's
	// error code.
	Reason string `json:"reason,omitempty"`
	// SuccessProbability is the model's probability that the candidate
	// answers correctly. It is not calibrated: Calibrated stays false until
	// an accepted calibration artifact covers the candidate.
	SuccessProbability *float64 `json:"success_probability,omitempty"`
	Calibrated         bool     `json:"calibrated"`
	// ExpectedOutputTokens is the output length the model expects, bounded
	// by the candidate's output capacity.
	ExpectedOutputTokens *int `json:"expected_output_tokens,omitempty"`
	// EstimatedCost prices the request's input tokens and the expected
	// output tokens at the candidate's rates; it is absent without rates.
	EstimatedCost *float64 `json:"estimated_cost,omitempty"`
	Currency      string   `json:"currency,omitempty"`
	Source        string   `json:"source"`
	Deployment    string   `json:"deployment,omitempty"`
}

type ObjectiveStage struct {
	Factor     string           `json:"factor"`
	Metric     string           `json:"metric,omitempty"`
	Percentile int              `json:"percentile,omitempty"`
	Action     StageAction      `json:"action"`
	Reason     StageReason      `json:"reason"`
	Tolerance  float64          `json:"tolerance"`
	Available  int              `json:"available"`
	Total      int              `json:"total"`
	Candidates []CandidateValue `json:"candidates"`
}

type CandidateValue struct {
	Candidate config.ModelRef `json:"candidate"`
	// Value is nil for unavailable evidence, preserving genuine numeric zero.
	// Latency uses seconds; cost is the configured request-cost forecast in
	// USD, not an output-token limit or a provider invoice.
	Value             *float64          `json:"value,omitempty"`
	Metric            string            `json:"metric,omitempty"`
	EliminationReason EliminationReason `json:"elimination_reason,omitempty"`
}

// Clone isolates a recorded objective from subsequent caller or replay changes.
func (trace *MultiFactorObjective) Clone() *MultiFactorObjective {
	if trace == nil {
		return nil
	}
	cloned := *trace
	if trace.Stages != nil {
		cloned.Stages = make([]ObjectiveStage, len(trace.Stages))
		for i, stage := range trace.Stages {
			cloned.Stages[i] = stage
			if stage.Candidates == nil {
				continue
			}
			cloned.Stages[i].Candidates = make([]CandidateValue, len(stage.Candidates))
			for j, candidate := range stage.Candidates {
				row := candidate
				row.Candidate = cloneCandidate(candidate.Candidate)
				if candidate.Value != nil {
					value := *candidate.Value
					row.Value = &value
				}
				cloned.Stages[i].Candidates[j] = row
			}
		}
	}
	if trace.FinalSurvivors != nil {
		cloned.FinalSurvivors = make([]config.ModelRef, len(trace.FinalSurvivors))
		for i, ref := range trace.FinalSurvivors {
			cloned.FinalSurvivors[i] = cloneCandidate(ref)
		}
	}
	cloned.Estimates = trace.Estimates.clone()
	return &cloned
}

func (estimates *CandidateEstimates) clone() *CandidateEstimates {
	if estimates == nil {
		return nil
	}
	cloned := *estimates
	if estimates.Candidates != nil {
		cloned.Candidates = make([]CandidateEstimate, len(estimates.Candidates))
		for i, estimate := range estimates.Candidates {
			row := estimate
			row.Candidate = cloneCandidate(estimate.Candidate)
			row.SuccessProbability = clonePointer(estimate.SuccessProbability)
			row.ExpectedOutputTokens = clonePointer(estimate.ExpectedOutputTokens)
			row.EstimatedCost = clonePointer(estimate.EstimatedCost)
			cloned.Candidates[i] = row
		}
	}
	return &cloned
}

func clonePointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	copied := *value
	return &copied
}

func cloneCandidate(ref config.ModelRef) config.ModelRef {
	if ref.UseReasoning != nil {
		value := *ref.UseReasoning
		ref.UseReasoning = &value
	}
	return ref
}
