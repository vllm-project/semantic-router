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
// surviving candidates. Weighted objectives omit this trace. Stages not reached
// after a singleton survives are omitted, rather than reported as evaluated.
type MultiFactorObjective struct {
	Stages []ObjectiveStage `json:"stages"`
	// FinalSurvivors are objective-eligible candidates, not a claim about the
	// final dispatched model after learning or session policy.
	FinalSurvivors []config.ModelRef `json:"final_survivors"`
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
	return &cloned
}

func cloneCandidate(ref config.ModelRef) config.ModelRef {
	if ref.UseReasoning != nil {
		value := *ref.UseReasoning
		ref.UseReasoning = &value
	}
	return ref
}
