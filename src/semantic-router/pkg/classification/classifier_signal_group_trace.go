package classification

import (
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/projectiontrace"
)

func appendPartitionTraceEntry(results *SignalResults, entry projectiontrace.PartitionResolution) {
	if results == nil {
		return
	}
	if results.ProjectionTrace == nil {
		results.ProjectionTrace = &projectiontrace.Trace{SchemaVersion: projectiontrace.SchemaVersion}
	}
	results.ProjectionTrace.Partitions = append(results.ProjectionTrace.Partitions, entry)
}

func appendPartitionDefaultTrace(results *SignalResults, signalType string, group config.ProjectionPartition) {
	appendPartitionTraceEntry(results, projectiontrace.PartitionResolution{
		GroupName:   group.Name,
		SignalType:  signalType,
		Semantics:   group.Semantics,
		Temperature: group.Temperature,
		Winner:      group.Default,
		DefaultUsed: true,
	})
}

func appendPartitionWinnerTrace(
	results *SignalResults,
	signalType string,
	group config.ProjectionPartition,
	contenders []string,
	confidences map[string]float64,
	winner string,
	winnerScore float64,
) {
	raw := make([]float64, len(contenders))
	for i, name := range contenders {
		raw[i] = confidences[signalConfidenceKey(signalType, name)]
	}
	entry := projectiontrace.PartitionResolution{
		GroupName:      group.Name,
		SignalType:     signalType,
		Semantics:      group.Semantics,
		Temperature:    group.Temperature,
		Winner:         winner,
		WinnerScore:    winnerScore,
		RawWinnerScore: rawScoreForName(winner, contenders, raw),
	}
	softmax := strings.EqualFold(group.Semantics, "softmax_exclusive")
	var norm []float64
	if softmax {
		norm = softmaxScores(raw, group.Temperature)
	}
	for i, name := range contenders {
		pc := projectiontrace.PartitionContender{Name: name, RawScore: raw[i]}
		if softmax && i < len(norm) {
			ns := norm[i]
			pc.NormalizedScore = &ns
		}
		entry.Contenders = append(entry.Contenders, pc)
	}
	if softmax && len(norm) > 0 {
		entry.Margin = topTwoMargin(norm)
	} else {
		entry.Margin = topTwoMargin(raw)
	}
	appendPartitionTraceEntry(results, entry)
}

func rawScoreForName(winner string, contenders []string, raw []float64) float64 {
	for i, name := range contenders {
		if name == winner && i < len(raw) {
			return raw[i]
		}
	}
	return 0
}

func topTwoMargin(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}
	sorted := append([]float64(nil), values...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] > sorted[j] })
	return sorted[0] - sorted[1]
}
