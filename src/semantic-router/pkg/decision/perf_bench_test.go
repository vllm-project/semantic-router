package decision

import (
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var benchmarkDecisionResult atomic.Pointer[DecisionResult]

type decisionBenchmarkMatch struct {
	name    string
	signals *SignalMatches
}

// BenchmarkDecisionTopology measures the real decision engine as policy size
// and match position change. The no-match case guards the common worst-case
// scan, while the parallel case exposes request-concurrency allocation changes.
func BenchmarkDecisionTopology(b *testing.B) {
	for _, decisionCount := range []int{1, 16, 128, 512} {
		decisions := benchmarkDecisions(decisionCount)
		engine := NewDecisionEngine(nil, nil, nil, decisions, config.RoutingStrategyPriority)
		for _, match := range benchmarkDecisionMatches(decisionCount) {
			runDecisionScenario(b, engine, decisionCount, match)
		}
	}
}

func benchmarkDecisionMatches(decisionCount int) []decisionBenchmarkMatch {
	return []decisionBenchmarkMatch{
		{name: "first", signals: &SignalMatches{KeywordRules: []string{"route-0"}}},
		{name: "last", signals: &SignalMatches{KeywordRules: []string{fmt.Sprintf("route-%d", decisionCount-1)}}},
		{name: "no-match", signals: &SignalMatches{KeywordRules: []string{"absent"}}},
	}
}

func runDecisionScenario(b *testing.B, engine *DecisionEngine, decisionCount int, match decisionBenchmarkMatch) {
	name := fmt.Sprintf("decisions=%d/position=%s", decisionCount, match.name)
	b.Run(name+"/serial", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			result, err := engine.EvaluateDecisionsWithSignals(match.signals)
			if err != nil {
				b.Fatal(err)
			}
			benchmarkDecisionResult.Store(result)
		}
	})
	b.Run(name+"/parallel", func(b *testing.B) {
		b.ReportAllocs()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				result, err := engine.EvaluateDecisionsWithSignals(match.signals)
				if err != nil {
					b.Error(err)
					return
				}
				benchmarkDecisionResult.Store(result)
			}
		})
	})
}

func benchmarkDecisions(count int) []config.Decision {
	decisions := make([]config.Decision, count)
	for i := range decisions {
		decisions[i] = config.Decision{
			Name:     fmt.Sprintf("decision-%d", i),
			Priority: count - i,
			Rules: config.RuleCombination{
				Type: config.SignalTypeKeyword,
				Name: fmt.Sprintf("route-%d", i),
			},
			ModelRefs: []config.ModelRef{{Model: "benchmark-model"}},
		}
	}
	return decisions
}
