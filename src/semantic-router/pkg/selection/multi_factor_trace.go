package selection

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selectiontrace"
)

func (s *MultiFactorSelector) objectiveStageTrace(
	priority MultiFactorPriority, signals []signalSet, active []int,
) selectiontrace.ObjectiveStage {
	stage := selectiontrace.ObjectiveStage{
		Factor: priority.Factor, Tolerance: priority.Tolerance,
		Action: selectiontrace.StageApplied, Reason: selectiontrace.ToleranceBand,
		Total: len(active), Candidates: make([]selectiontrace.CandidateValue, 0, len(active)),
	}
	if priority.Factor == config.MultiFactorFactorLatency {
		stage.Metric = s.config.LatencyMetric
		if stage.Metric == "" {
			stage.Metric = "tpot_then_ttft"
		}
		stage.Percentile = s.config.LatencyPercentile
	}
	for _, index := range active {
		signal := signals[index]
		row := selectiontrace.CandidateValue{Candidate: traceCandidate(signal.candidate)}
		if value, available, _ := factorValue(signal, priority.Factor); available {
			row.Value = &value
			stage.Available++
		}
		if priority.Factor == config.MultiFactorFactorLatency {
			row.Metric = signal.latencyMetric
		}
		stage.Candidates = append(stage.Candidates, row)
	}
	if priority.Factor == config.MultiFactorFactorLatency && stage.Available < stage.Total {
		stage.Action, stage.Reason = selectiontrace.StageSkipped, selectiontrace.IncompleteLatencyCoverage
	} else if stage.Available == 0 {
		stage.Action, stage.Reason = selectiontrace.StageSkipped, selectiontrace.NoAvailableValues
	}
	return stage
}

func traceCandidate(ref config.ModelRef) config.ModelRef {
	if ref.UseReasoning != nil {
		value := *ref.UseReasoning
		ref.UseReasoning = &value
	}
	return ref
}
