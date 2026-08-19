package selection

import (
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var benchmarkSelectionAdjustments CacheAffinityResult

// BenchmarkSelectionTopology measures candidate-count and context-utilization
// interactions through the actual cache-affinity selection seam.
func BenchmarkSelectionTopology(b *testing.B) {
	for _, scenario := range []struct {
		candidates  int
		context     int
		window      int
		utilization string
	}{
		{candidates: 1, context: 512, window: 32768, utilization: "low"},
		{candidates: 3, context: 16384, window: 32768, utilization: "medium"},
		{candidates: 16, context: 31129, window: 32768, utilization: "boundary"},
	} {
		candidates := make([]config.ModelRef, scenario.candidates)
		windows := make(map[string]int, scenario.candidates)
		scores := make(map[string]float64, scenario.candidates)
		for i := range candidates {
			name := fmt.Sprintf("model-%d", i)
			candidates[i] = config.ModelRef{Model: name}
			windows[name] = scenario.window
			scores[name] = 1 - float64(i)/float64(max(scenario.candidates, 1)+1)
		}
		context := &CacheAffinityContext{
			TurnIndex:           3,
			PreviousModel:       candidates[0].Model,
			HistoryTokens:       scenario.context / 2,
			ContextTokens:       scenario.context,
			ModelContextWindows: windows,
		}
		name := fmt.Sprintf("candidates=%d/utilization=%s", scenario.candidates, scenario.utilization)
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				benchmarkSelectionAdjustments = ComputeCacheAffinityAdjustments(context, candidates, scores)
			}
		})
	}
}
