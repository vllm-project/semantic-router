//go:build !windows

package benchmarks

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

var (
	decisionEngineOnce sync.Once
	decisionEngine     *decision.DecisionEngine
	decisionEngineErr  error
)

// initDecisionEngine initializes the decision engine once
func initDecisionEngine(b *testing.B) {
	decisionEngineOnce.Do(func() {
		// Find the project root
		wd, err := os.Getwd()
		if err != nil {
			decisionEngineErr = err
			return
		}

		projectRoot := filepath.Join(wd, "../..")

		// Load config
		configPath := filepath.Join(projectRoot, "config", "config.yaml")
		cfg, err := config.Load(configPath)
		if err != nil {
			decisionEngineErr = err
			return
		}
		recipe, ok := cfg.RecipeByName(config.DefaultRecipeName)
		if !ok {
			decisionEngineErr = fmt.Errorf("%s declares no %s recipe", configPath, config.DefaultRecipeName)
			return
		}

		// Create the default recipe's decision engine
		decisionEngine = decision.NewDecisionEngine(recipe.Profile.Decisions, config.RoutingStrategyPriority)
	})

	if decisionEngineErr != nil {
		b.Fatalf("Failed to initialize decision engine: %v", decisionEngineErr)
	}
}

// BenchmarkEvaluateDecisions_SingleDomain benchmarks decision evaluation with single domain
func BenchmarkEvaluateDecisions_SingleDomain(b *testing.B) {
	initDecisionEngine(b)

	// Single domain match
	signals := &decision.SignalMatches{DomainRules: []string{"math"}}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := decisionEngine.EvaluateDecisionsWithSignals(signals)
		if err != nil {
			// It's okay if no decision matches - some configs may not have all domains
			continue
		}
	}
}

// BenchmarkEvaluateDecisions_MultipleDomains benchmarks decision evaluation with multiple domains
func BenchmarkEvaluateDecisions_MultipleDomains(b *testing.B) {
	initDecisionEngine(b)

	// Multiple domain matches
	signals := &decision.SignalMatches{DomainRules: []string{"math", "code", "business"}}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := decisionEngine.EvaluateDecisionsWithSignals(signals)
		if err != nil {
			// It's okay if no decision matches
			continue
		}
	}
}

// BenchmarkEvaluateDecisions_WithKeywords benchmarks decision evaluation with keywords
func BenchmarkEvaluateDecisions_WithKeywords(b *testing.B) {
	initDecisionEngine(b)

	signals := &decision.SignalMatches{
		DomainRules:  []string{"math"},
		KeywordRules: []string{"derivative", "calculus"},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := decisionEngine.EvaluateDecisionsWithSignals(signals)
		if err != nil {
			// It's okay if no decision matches
			continue
		}
	}
}

// BenchmarkEvaluateDecisions_ComplexScenario benchmarks complex decision scenario
func BenchmarkEvaluateDecisions_ComplexScenario(b *testing.B) {
	initDecisionEngine(b)

	signals := &decision.SignalMatches{
		DomainRules:  []string{"math", "code", "business", "healthcare", "legal"},
		KeywordRules: []string{"api", "integration", "optimization"},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := decisionEngine.EvaluateDecisionsWithSignals(signals)
		if err != nil {
			// It's okay if no decision matches
			continue
		}
	}
}

// BenchmarkEvaluateDecisions_Parallel benchmarks parallel decision evaluation
func BenchmarkEvaluateDecisions_Parallel(b *testing.B) {
	initDecisionEngine(b)

	signals := &decision.SignalMatches{DomainRules: []string{"math"}}

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := decisionEngine.EvaluateDecisionsWithSignals(signals)
			if err != nil {
				// It's okay if no decision matches
				continue
			}
		}
	})
}

// BenchmarkPrioritySelection benchmarks decision priority selection
func BenchmarkPrioritySelection(b *testing.B) {
	initDecisionEngine(b)

	// Scenario where multiple decisions could match
	signals := &decision.SignalMatches{DomainRules: []string{"math", "code", "business"}}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := decisionEngine.EvaluateDecisionsWithSignals(signals)
		if err != nil {
			// It's okay if no decision matches
			continue
		}
	}
}
