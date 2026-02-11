package latency

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func containsRule(rules []string, target string) bool {
	for _, r := range rules {
		if r == target {
			return true
		}
	}
	return false
}

func TestLatencyClassifier(t *testing.T) {
	t.Run("should select best model when both TPOT and TTFT are configured", func(t *testing.T) {
		ResetTPOT()
		ResetTTFT()

		rules := []config.LatencyRule{
			{
				Name:           "low_latency_comprehensive",
				TPOTPercentile: 50,
				TTFTPercentile: 50,
				Description:    "Fast start and fast generation",
			},
		}
		classifier, err := NewLatencyClassifier(rules)
		if err != nil {
			t.Fatalf("NewLatencyClassifier() error = %v", err)
		}

		UpdateTPOT("model-a", 0.05)
		UpdateTPOT("model-a", 0.06)
		UpdateTPOT("model-a", 0.07)
		UpdateTPOT("model-a", 0.05)
		UpdateTPOT("model-a", 0.06)

		UpdateTPOT("model-b", 0.03)
		UpdateTPOT("model-b", 0.04)
		UpdateTPOT("model-b", 0.05)
		UpdateTPOT("model-b", 0.03)
		UpdateTPOT("model-b", 0.04)

		UpdateTPOT("model-c", 0.02)
		UpdateTPOT("model-c", 0.03)
		UpdateTPOT("model-c", 0.04)
		UpdateTPOT("model-c", 0.02)
		UpdateTPOT("model-c", 0.03)

		UpdateTTFT("model-a", 0.30)
		UpdateTTFT("model-a", 0.35)
		UpdateTTFT("model-a", 0.40)
		UpdateTTFT("model-a", 0.30)
		UpdateTTFT("model-a", 0.35)

		UpdateTTFT("model-b", 0.20)
		UpdateTTFT("model-b", 0.25)
		UpdateTTFT("model-b", 0.30)
		UpdateTTFT("model-b", 0.20)
		UpdateTTFT("model-b", 0.25)

		UpdateTTFT("model-c", 0.10)
		UpdateTTFT("model-c", 0.15)
		UpdateTTFT("model-c", 0.20)
		UpdateTTFT("model-c", 0.10)
		UpdateTTFT("model-c", 0.15)

		result, err := classifier.Classify([]string{"model-a", "model-b", "model-c"})
		if err != nil {
			t.Fatalf("Classify() error = %v", err)
		}
		if result == nil {
			t.Fatalf("Classify() returned nil result")
		}
		if !containsRule(result.MatchedRules, "low_latency_comprehensive") {
			t.Errorf("expected matched rule low_latency_comprehensive, got %v", result.MatchedRules)
		}
	})

	t.Run("should handle TPOT-only rule", func(t *testing.T) {
		ResetTPOT()
		ResetTTFT()

		rules := []config.LatencyRule{
			{
				Name:           "batch_processing",
				TPOTPercentile: 50,
				Description:    "Only TPOT matters for batch processing",
			},
		}
		classifier, err := NewLatencyClassifier(rules)
		if err != nil {
			t.Fatalf("NewLatencyClassifier() error = %v", err)
		}

		UpdateTPOT("fast-model", 0.02)
		UpdateTPOT("fast-model", 0.03)
		UpdateTPOT("fast-model", 0.04)
		UpdateTPOT("fast-model", 0.02)
		UpdateTPOT("fast-model", 0.03)

		UpdateTPOT("slow-model", 0.10)
		UpdateTPOT("slow-model", 0.11)
		UpdateTPOT("slow-model", 0.12)
		UpdateTPOT("slow-model", 0.13)
		UpdateTPOT("slow-model", 0.14)

		result, err := classifier.Classify([]string{"fast-model", "slow-model"})
		if err != nil {
			t.Fatalf("Classify() error = %v", err)
		}
		if result == nil {
			t.Fatalf("Classify() returned nil result")
		}
		if !containsRule(result.MatchedRules, "batch_processing") {
			t.Errorf("expected matched rule batch_processing, got %v", result.MatchedRules)
		}
	})

	t.Run("should handle TTFT-only rule", func(t *testing.T) {
		ResetTPOT()
		ResetTTFT()

		rules := []config.LatencyRule{
			{
				Name:           "chat_fast_start",
				TTFTPercentile: 50,
				Description:    "Only TTFT matters for chat apps",
			},
		}
		classifier, err := NewLatencyClassifier(rules)
		if err != nil {
			t.Fatalf("NewLatencyClassifier() error = %v", err)
		}

		UpdateTTFT("fast-start-model", 0.10)
		UpdateTTFT("fast-start-model", 0.15)
		UpdateTTFT("fast-start-model", 0.20)
		UpdateTTFT("fast-start-model", 0.10)
		UpdateTTFT("fast-start-model", 0.15)

		UpdateTTFT("slow-start-model", 0.50)
		UpdateTTFT("slow-start-model", 0.55)
		UpdateTTFT("slow-start-model", 0.60)
		UpdateTTFT("slow-start-model", 0.65)
		UpdateTTFT("slow-start-model", 0.70)

		result, err := classifier.Classify([]string{"fast-start-model", "slow-start-model"})
		if err != nil {
			t.Fatalf("Classify() error = %v", err)
		}
		if result == nil {
			t.Fatalf("Classify() returned nil result")
		}
		if !containsRule(result.MatchedRules, "chat_fast_start") {
			t.Errorf("expected matched rule chat_fast_start, got %v", result.MatchedRules)
		}
	})

	t.Run("should handle models that don't meet strict thresholds", func(t *testing.T) {
		ResetTPOT()
		ResetTTFT()

		rules := []config.LatencyRule{
			{
				Name:           "very_low_latency",
				TPOTPercentile: 10,
				TTFTPercentile: 10,
			},
		}
		classifier, err := NewLatencyClassifier(rules)
		if err != nil {
			t.Fatalf("NewLatencyClassifier() error = %v", err)
		}

		UpdateTPOT("average-model", 0.05)
		UpdateTPOT("average-model", 0.06)
		UpdateTPOT("average-model", 0.07)
		UpdateTPOT("average-model", 0.08)
		UpdateTPOT("average-model", 0.09)

		UpdateTTFT("average-model", 0.30)
		UpdateTTFT("average-model", 0.35)
		UpdateTTFT("average-model", 0.40)
		UpdateTTFT("average-model", 0.45)
		UpdateTTFT("average-model", 0.50)

		result, err := classifier.Classify([]string{"average-model"})
		if err != nil {
			t.Fatalf("Classify() error = %v", err)
		}
		if result == nil {
			t.Fatalf("Classify() returned nil result")
		}
	})

	t.Run("should work with small sample sizes (1-2 observations)", func(t *testing.T) {
		ResetTPOT()
		ResetTTFT()

		rules := []config.LatencyRule{
			{
				Name:           "early_evaluation",
				TPOTPercentile: 50,
				TTFTPercentile: 50,
			},
		}
		classifier, err := NewLatencyClassifier(rules)
		if err != nil {
			t.Fatalf("NewLatencyClassifier() error = %v", err)
		}

		UpdateTPOT("new-model", 0.04)
		UpdateTPOT("new-model", 0.05)
		UpdateTTFT("new-model", 0.20)
		UpdateTTFT("new-model", 0.25)

		result, err := classifier.Classify([]string{"new-model"})
		if err != nil {
			t.Fatalf("Classify() error = %v", err)
		}
		if result == nil {
			t.Fatalf("Classify() returned nil result")
		}
	})

	t.Run("should handle models without TPOT/TTFT data", func(t *testing.T) {
		ResetTPOT()
		ResetTTFT()

		rules := []config.LatencyRule{
			{
				Name:           "low_latency",
				TPOTPercentile: 50,
				TTFTPercentile: 50,
			},
		}
		classifier, err := NewLatencyClassifier(rules)
		if err != nil {
			t.Fatalf("NewLatencyClassifier() error = %v", err)
		}

		result, err := classifier.Classify([]string{"unknown-model"})
		if err != nil {
			t.Fatalf("Classify() error = %v", err)
		}
		if result == nil {
			t.Fatalf("Classify() returned nil result")
		}
		if len(result.MatchedRules) != 0 {
			t.Errorf("expected no matched rules, got %v", result.MatchedRules)
		}
	})

	t.Run("should handle empty model list", func(t *testing.T) {
		ResetTPOT()
		ResetTTFT()

		rules := []config.LatencyRule{
			{
				Name:           "low_latency",
				TPOTPercentile: 50,
				TTFTPercentile: 50,
			},
		}
		classifier, err := NewLatencyClassifier(rules)
		if err != nil {
			t.Fatalf("NewLatencyClassifier() error = %v", err)
		}

		result, err := classifier.Classify([]string{})
		if err != nil {
			t.Fatalf("Classify() error = %v", err)
		}
		if result == nil {
			t.Fatalf("Classify() returned nil result")
		}
		if len(result.MatchedRules) != 0 {
			t.Errorf("expected no matched rules, got %v", result.MatchedRules)
		}
	})

	t.Run("should use exponential moving average for TPOT", func(t *testing.T) {
		ResetTPOT()
		ResetTTFT()

		UpdateTPOT("test-model", 0.10)
		UpdateTPOT("test-model", 0.12)
		UpdateTPOT("test-model", 0.08)

		tpot, exists := GetTPOT("test-model")
		if !exists {
			t.Fatalf("expected TPOT to exist")
		}
		if tpot <= 0.08 {
			t.Errorf("expected TPOT > 0.08, got %v", tpot)
		}
		if tpot >= 0.12 {
			t.Errorf("expected TPOT < 0.12, got %v", tpot)
		}
	})

	t.Run("should handle multiple rules with different percentiles", func(t *testing.T) {
		ResetTPOT()
		ResetTTFT()

		rules := []config.LatencyRule{
			{
				Name:           "p10_latency",
				TPOTPercentile: 10,
				TTFTPercentile: 10,
			},
			{
				Name:           "p50_latency",
				TPOTPercentile: 50,
				TTFTPercentile: 50,
			},
			{
				Name:           "p90_latency",
				TPOTPercentile: 90,
				TTFTPercentile: 90,
			},
		}
		classifier, err := NewLatencyClassifier(rules)
		if err != nil {
			t.Fatalf("NewLatencyClassifier() error = %v", err)
		}

		UpdateTPOT("multi-rule-model", 0.03)
		UpdateTPOT("multi-rule-model", 0.04)
		UpdateTPOT("multi-rule-model", 0.05)
		UpdateTPOT("multi-rule-model", 0.06)
		UpdateTPOT("multi-rule-model", 0.07)

		UpdateTTFT("multi-rule-model", 0.20)
		UpdateTTFT("multi-rule-model", 0.25)
		UpdateTTFT("multi-rule-model", 0.30)
		UpdateTTFT("multi-rule-model", 0.35)
		UpdateTTFT("multi-rule-model", 0.40)

		result, err := classifier.Classify([]string{"multi-rule-model"})
		if err != nil {
			t.Fatalf("Classify() error = %v", err)
		}
		if result == nil {
			t.Fatalf("Classify() returned nil result")
		}
	})

	t.Run("should test AND logic: both TPOT and TTFT must meet thresholds", func(t *testing.T) {
		ResetTPOT()
		ResetTTFT()

		rules := []config.LatencyRule{
			{
				Name:           "both_required",
				TPOTPercentile: 50,
				TTFTPercentile: 50,
			},
		}
		classifier, err := NewLatencyClassifier(rules)
		if err != nil {
			t.Fatalf("NewLatencyClassifier() error = %v", err)
		}

		UpdateTPOT("model-a", 0.02)
		UpdateTPOT("model-a", 0.03)
		UpdateTPOT("model-a", 0.04)
		UpdateTPOT("model-a", 0.05)
		UpdateTPOT("model-a", 0.06)

		UpdateTTFT("model-a", 0.60)
		UpdateTTFT("model-a", 0.65)
		UpdateTTFT("model-a", 0.70)
		UpdateTTFT("model-a", 0.75)
		UpdateTTFT("model-a", 0.80)

		UpdateTPOT("model-b", 0.10)
		UpdateTPOT("model-b", 0.11)
		UpdateTPOT("model-b", 0.12)
		UpdateTPOT("model-b", 0.13)
		UpdateTPOT("model-b", 0.14)

		UpdateTTFT("model-b", 0.10)
		UpdateTTFT("model-b", 0.15)
		UpdateTTFT("model-b", 0.20)
		UpdateTTFT("model-b", 0.25)
		UpdateTTFT("model-b", 0.30)

		UpdateTPOT("model-c", 0.02)
		UpdateTPOT("model-c", 0.03)
		UpdateTPOT("model-c", 0.04)
		UpdateTPOT("model-c", 0.02)
		UpdateTPOT("model-c", 0.03)

		UpdateTTFT("model-c", 0.10)
		UpdateTTFT("model-c", 0.15)
		UpdateTTFT("model-c", 0.20)
		UpdateTTFT("model-c", 0.10)
		UpdateTTFT("model-c", 0.15)

		result, err := classifier.Classify([]string{"model-a", "model-b", "model-c"})
		if err != nil {
			t.Fatalf("Classify() error = %v", err)
		}
		if result == nil {
			t.Fatalf("Classify() returned nil result")
		}
		if !containsRule(result.MatchedRules, "both_required") {
			t.Errorf("expected matched rule both_required, got %v", result.MatchedRules)
		}
	})
}
