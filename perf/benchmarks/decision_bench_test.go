//go:build !windows && cgo

package benchmarks

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

var (
	testEngine *decision.Engine
)

func setupDecisionEngine(b *testing.B) {
	if testEngine != nil {
		return
	}

	// Load config
	cfg, err := config.LoadConfig("../config/testing/config.e2e.yaml")
	if err != nil {
		b.Fatalf("Failed to load config: %v", err)
	}

	// Initialize decision engine
	engine := decision.NewEngine(cfg)
	testEngine = engine

	b.ResetTimer()
}

// BenchmarkEvaluateDecisions_SingleDomain benchmarks decision evaluation with single domain
func BenchmarkEvaluateDecisions_SingleDomain(b *testing.B) {
	setupDecisionEngine(b)

	domains := map[string]float64{
		"math": 0.95,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := testEngine.EvaluateDecisions(domains, []string{})
		if err != nil {
			b.Fatalf("Decision evaluation failed: %v", err)
		}
	}
}

// BenchmarkEvaluateDecisions_MultipleDomains benchmarks decision evaluation with multiple domains
func BenchmarkEvaluateDecisions_MultipleDomains(b *testing.B) {
	setupDecisionEngine(b)

	domains := map[string]float64{
		"math":     0.60,
		"code":     0.30,
		"business": 0.10,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := testEngine.EvaluateDecisions(domains, []string{})
		if err != nil {
			b.Fatalf("Decision evaluation failed: %v", err)
		}
	}
}

// BenchmarkEvaluateDecisions_WithKeywords benchmarks decision evaluation with keywords
func BenchmarkEvaluateDecisions_WithKeywords(b *testing.B) {
	setupDecisionEngine(b)

	domains := map[string]float64{
		"math": 0.95,
	}
	keywords := []string{"derivative", "calculus"}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := testEngine.EvaluateDecisions(domains, keywords)
		if err != nil {
			b.Fatalf("Decision evaluation failed: %v", err)
		}
	}
}

// BenchmarkEvaluateDecisions_ComplexScenario benchmarks complex decision scenario
func BenchmarkEvaluateDecisions_ComplexScenario(b *testing.B) {
	setupDecisionEngine(b)

	domains := map[string]float64{
		"math":       0.40,
		"code":       0.30,
		"business":   0.15,
		"healthcare": 0.10,
		"legal":      0.05,
	}
	keywords := []string{"api", "integration", "optimization"}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := testEngine.EvaluateDecisions(domains, keywords)
		if err != nil {
			b.Fatalf("Decision evaluation failed: %v", err)
		}
	}
}

// BenchmarkEvaluateDecisions_Parallel benchmarks parallel decision evaluation
func BenchmarkEvaluateDecisions_Parallel(b *testing.B) {
	setupDecisionEngine(b)

	domains := map[string]float64{
		"math": 0.95,
	}

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := testEngine.EvaluateDecisions(domains, []string{})
			if err != nil {
				b.Fatalf("Decision evaluation failed: %v", err)
			}
		}
	})
}

// BenchmarkRuleEvaluation_AND benchmarks AND rule evaluation
func BenchmarkRuleEvaluation_AND(b *testing.B) {
	setupDecisionEngine(b)

	// This benchmarks the rule matching logic
	domains := map[string]float64{
		"math": 0.95,
		"code": 0.85,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := testEngine.EvaluateDecisions(domains, []string{})
		if err != nil {
			b.Fatalf("Rule evaluation failed: %v", err)
		}
	}
}

// BenchmarkRuleEvaluation_OR benchmarks OR rule evaluation
func BenchmarkRuleEvaluation_OR(b *testing.B) {
	setupDecisionEngine(b)

	domains := map[string]float64{
		"business": 0.50,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := testEngine.EvaluateDecisions(domains, []string{})
		if err != nil {
			b.Fatalf("Rule evaluation failed: %v", err)
		}
	}
}

// BenchmarkPrioritySelection benchmarks decision priority selection
func BenchmarkPrioritySelection(b *testing.B) {
	setupDecisionEngine(b)

	// Scenario where multiple decisions could match
	domains := map[string]float64{
		"math":     0.60,
		"code":     0.55,
		"business": 0.50,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := testEngine.EvaluateDecisions(domains, []string{})
		if err != nil {
			b.Fatalf("Priority selection failed: %v", err)
		}
	}
}
