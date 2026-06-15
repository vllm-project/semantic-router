package classification

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/projectiontrace"
)

func mergeProjectionTrace(results *SignalResults, p config.Projections) *projectiontrace.Trace {
	var existingPartitions []projectiontrace.PartitionResolution
	if results.ProjectionTrace != nil && len(results.ProjectionTrace.Partitions) > 0 {
		existingPartitions = append([]projectiontrace.PartitionResolution(nil), results.ProjectionTrace.Partitions...)
	}
	tr := &projectiontrace.Trace{SchemaVersion: projectiontrace.SchemaVersion}
	tr.Partitions = existingPartitions
	for _, score := range p.Scores {
		sb := projectiontrace.ScoreBreakdown{Name: score.Name}
		var sum float64
		for _, input := range score.Inputs {
			v := projectionInputValue(input, results)
			contrib := input.Weight * v
			sum += contrib
			sb.Inputs = append(sb.Inputs, projectiontrace.ScoreInputPart{
				Type:         input.Type,
				Name:         input.Name,
				KB:           input.KB,
				Metric:       input.Metric,
				Weight:       input.Weight,
				Value:        v,
				Contribution: contrib,
			})
		}
		sb.Total = sum
		tr.Scores = append(tr.Scores, sb)
	}
	for _, mapping := range p.Mappings {
		scoreValue, ok := results.ProjectionScores[mapping.Source]
		if !ok {
			continue
		}
		md := projectiontrace.MappingDecision{
			MappingName: mapping.Name,
			SourceScore: mapping.Source,
			ScoreValue:  scoreValue,
		}
		for _, output := range mapping.Outputs {
			matched := projectionOutputMatches(output, scoreValue)
			d := projectionBoundaryDistance(output, scoreValue)
			md.Outputs = append(md.Outputs, projectiontrace.OutputEvalStep{
				Name:             output.Name,
				Matched:          matched,
				BoundaryDistance: d,
			})
			if matched && md.SelectedOutput == "" {
				out := output
				md.SelectedOutput = out.Name
				md.Confidence = projectionOutputConfidence(mapping, out, scoreValue)
				md.BoundaryDistance = d
			}
		}
		tr.Mappings = append(tr.Mappings, md)
	}
	return tr
}
