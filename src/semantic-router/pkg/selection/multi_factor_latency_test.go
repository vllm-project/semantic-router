package selection

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMultiFactorComparesTheRequestedLatencyMetric(t *testing.T) {
	// The streaming model emits tokens quickly after a slow prefill. A third
	// model has only TPOT observations; that is not evidence of a fast TTFT.
	tpot := map[string]float64{"streaming": 0.01, "interactive": 0.04, "unknown": 0.001}
	ttft := map[string]float64{"streaming": 2, "interactive": 0.1}
	for _, tc := range []struct{ metric, want string }{
		{"ttft", "streaming"}, {"tpot", "unknown"}, {"", "unknown"},
	} {
		t.Run("metric="+tc.metric, func(t *testing.T) {
			cfg := DefaultMultiFactorConfig()
			cfg.LatencyMetric = tc.metric
			cfg.Objective = MultiFactorObjective{
				Strategy:   config.MultiFactorObjectiveLexicographic,
				Priorities: []MultiFactorPriority{{Factor: config.MultiFactorFactorLatency}},
			}
			s := buildMFSelector(cfg, nil, func(string) int { return 0 },
				func(m string, _ int) (float64, bool) { v, ok := tpot[m]; return v, ok },
				func(m string, _ int) (float64, bool) { v, ok := ttft[m]; return v, ok },
			)
			result, err := s.Select(context.Background(), &SelectionContext{
				CandidateModels: candidates("streaming", "interactive", "unknown"),
			})
			if err != nil || result.SelectedModel != tc.want {
				t.Fatalf("metric=%q: result=%+v err=%v; want %s", tc.metric, result, err, tc.want)
			}
			if tc.metric == "ttft" {
				if _, known, _ := s.latencyMeasurement("unknown"); known {
					t.Fatal("TPOT was substituted for missing TTFT")
				}
				if stage := result.MultiFactor.Stages[0]; stage.Action != "skipped" || stage.Available != 2 || stage.Total != 3 {
					t.Fatalf("partial TTFT coverage must preserve all candidates: %+v", stage)
				}
			}
		})
	}
}
