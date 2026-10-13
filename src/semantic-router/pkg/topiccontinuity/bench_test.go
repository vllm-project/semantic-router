package topiccontinuity

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Benchmarks are informational; deterministic bounds are enforced by
// counted-limit tests, not wall-clock thresholds.

func benchmarkHistory(turns, bytesPerTurn int) []llmprotocol.Message {
	block := strings.Repeat("parse the config_loader output and validateToken in auth.ts ", bytesPerTurn/60+1)
	var messages []llmprotocol.Message
	for i := 0; i < turns; i++ {
		messages = append(messages, exchange(block[:bytesPerTurn/2], block[:bytesPerTurn/2])...)
	}
	return append(messages, user("Now update validateToken in auth.ts to log failures"))
}

func benchmarkPolicy(b *testing.B, policy HistoryPolicy, messages []llmprotocol.Message) {
	cfg := defaultConfig(policy)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prepared := prepare(context.Background(), messages, true, policy)
		_ = classify(cfg, prepared, extract(context.Background(), prepared))
	}
}

func BenchmarkDefaultLimits(b *testing.B) {
	benchmarkPolicy(b, defaultPolicy, benchmarkHistory(12, 16384))
}

func BenchmarkMaximumLimits(b *testing.B) {
	policy := HistoryPolicy{Limits: Limits{
		MaxPriorTurns: MaxPriorTurns, MaxTurnBytes: 32768,
		MaxInputBytes: MaxInputBytes,
	}, IncludeAssistant: true}
	benchmarkPolicy(b, policy, benchmarkHistory(40, 32768))
}

func BenchmarkAdversarialQuotes(b *testing.B) {
	benchmarkPolicy(b, adversarialPolicy, liveTurnHistory(strings.Repeat(" 'a", MaxTurnBytes/3)))
}

func benchmarkRules(b *testing.B, distinctPolicies int) {
	messages := benchmarkHistory(12, 16384)
	configs := make([]EvalConfig, 8)
	for i := range configs {
		policy := defaultPolicy
		policy.Limits.MaxPriorTurns = 1 + i%distinctPolicies
		configs[i] = EvalConfig{Name: fmt.Sprintf("r%d", i), Policy: policy, Continuation: 0.35, Change: 0.08}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = evaluateGroups(messages, configs)
	}
}

func BenchmarkEightRulesOnePolicy(b *testing.B)     { benchmarkRules(b, 1) }
func BenchmarkEightRulesEightPolicies(b *testing.B) { benchmarkRules(b, 8) }
