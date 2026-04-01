package classification

import (
	"math"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func stubReaskEmbeddings(t *testing.T, embeddings map[string][]float32) {
	t.Helper()

	restore := SetEmbeddingFuncForTests(func(text string, modelType string, targetDim int) (*candle_binding.EmbeddingOutput, error) {
		if embedding, ok := embeddings[text]; ok {
			return &candle_binding.EmbeddingOutput{Embedding: embedding}, nil
		}
		return &candle_binding.EmbeddingOutput{Embedding: makeEmbedding(0, 1)}, nil
	})
	t.Cleanup(restore)
}

func approxEqual(got float64, want float64) bool {
	return math.Abs(got-want) < 1e-6
}

func TestReaskClassifier_ClassifyNoPriorUserTurns(t *testing.T) {
	stubReaskEmbeddings(t, map[string][]float32{
		"current": makeEmbedding(1, 0),
	})

	classifier, err := NewReaskClassifier([]config.ReaskRule{{Name: "likely_dissatisfied"}}, "test-model")
	if err != nil {
		t.Fatalf("NewReaskClassifier() error = %v", err)
	}

	matches, err := classifier.Classify("current", nil)
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if len(matches) != 0 {
		t.Fatalf("expected no matches, got %+v", matches)
	}
}

func TestReaskClassifier_ClassifyOneTurnRepeatMatches(t *testing.T) {
	stubReaskEmbeddings(t, map[string][]float32{
		"current":     makeEmbedding(1, 0),
		"previous":    makeEmbedding(1, 0),
		"unrelated":   makeEmbedding(0, 1),
		"threshold80": makeEmbedding(0.8, 0.6),
	})

	classifier, err := NewReaskClassifier([]config.ReaskRule{{
		Name:          "likely_dissatisfied",
		Threshold:     0.8,
		LookbackTurns: 1,
	}}, "test-model")
	if err != nil {
		t.Fatalf("NewReaskClassifier() error = %v", err)
	}

	matches, err := classifier.Classify("current", []string{"previous"})
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if len(matches) != 1 {
		t.Fatalf("expected 1 match, got %+v", matches)
	}
	if matches[0].RuleName != "likely_dissatisfied" {
		t.Fatalf("expected likely_dissatisfied, got %q", matches[0].RuleName)
	}
	if !approxEqual(matches[0].MinSimilarity, 1) {
		t.Fatalf("expected min similarity 1, got %v", matches[0].MinSimilarity)
	}
	if matches[0].MatchedTurns != 1 {
		t.Fatalf("expected matched turns 1, got %d", matches[0].MatchedTurns)
	}
}

func TestReaskClassifier_ClassifyOneTurnRepeatNonMatch(t *testing.T) {
	stubReaskEmbeddings(t, map[string][]float32{
		"current":  makeEmbedding(1, 0),
		"previous": makeEmbedding(0, 1),
	})

	classifier, err := NewReaskClassifier([]config.ReaskRule{{
		Name:          "likely_dissatisfied",
		Threshold:     0.8,
		LookbackTurns: 1,
	}}, "test-model")
	if err != nil {
		t.Fatalf("NewReaskClassifier() error = %v", err)
	}

	matches, err := classifier.Classify("current", []string{"previous"})
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if len(matches) != 0 {
		t.Fatalf("expected no matches, got %+v", matches)
	}
}

func TestReaskClassifier_ClassifyTwoTurnConsecutiveRepeatMatches(t *testing.T) {
	stubReaskEmbeddings(t, map[string][]float32{
		"current":       makeEmbedding(1, 0),
		"older-repeat":  makeEmbedding(0.8, 0.6),
		"recent-repeat": makeEmbedding(1, 0),
	})

	classifier, err := NewReaskClassifier([]config.ReaskRule{{
		Name:          "persistently_dissatisfied",
		Threshold:     0.8,
		LookbackTurns: 2,
	}}, "test-model")
	if err != nil {
		t.Fatalf("NewReaskClassifier() error = %v", err)
	}

	matches, err := classifier.Classify("current", []string{"older-repeat", "recent-repeat"})
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if len(matches) != 1 {
		t.Fatalf("expected 1 match, got %+v", matches)
	}
	if matches[0].RuleName != "persistently_dissatisfied" {
		t.Fatalf("expected persistently_dissatisfied, got %q", matches[0].RuleName)
	}
	if !approxEqual(matches[0].MinSimilarity, 0.8) {
		t.Fatalf("expected min similarity 0.8, got %v", matches[0].MinSimilarity)
	}
	if matches[0].MatchedTurns != 2 {
		t.Fatalf("expected matched turns 2, got %d", matches[0].MatchedTurns)
	}
	if matches[0].LookbackTurns != 2 {
		t.Fatalf("expected lookback turns 2, got %d", matches[0].LookbackTurns)
	}
}

func TestReaskClassifier_ClassifyOlderMatchBrokenByMostRecentTurn(t *testing.T) {
	stubReaskEmbeddings(t, map[string][]float32{
		"current":       makeEmbedding(1, 0),
		"older-repeat":  makeEmbedding(1, 0),
		"recent-answer": makeEmbedding(0, 1),
	})

	classifier, err := NewReaskClassifier([]config.ReaskRule{{
		Name:          "persistently_dissatisfied",
		Threshold:     0.8,
		LookbackTurns: 2,
	}}, "test-model")
	if err != nil {
		t.Fatalf("NewReaskClassifier() error = %v", err)
	}

	matches, err := classifier.Classify("current", []string{"older-repeat", "recent-answer"})
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if len(matches) != 0 {
		t.Fatalf("expected no matches, got %+v", matches)
	}
}

func TestReaskClassifier_ClassifyRetainsOnlyMaxLookbackMatch(t *testing.T) {
	stubReaskEmbeddings(t, map[string][]float32{
		"current":       makeEmbedding(1, 0),
		"older-repeat":  makeEmbedding(0.8, 0.6),
		"recent-repeat": makeEmbedding(1, 0),
	})

	classifier, err := NewReaskClassifier([]config.ReaskRule{
		{
			Name:          "likely_dissatisfied",
			Threshold:     0.8,
			LookbackTurns: 1,
		},
		{
			Name:          "persistently_dissatisfied",
			Threshold:     0.8,
			LookbackTurns: 2,
		},
	}, "test-model")
	if err != nil {
		t.Fatalf("NewReaskClassifier() error = %v", err)
	}

	matches, err := classifier.Classify("current", []string{"older-repeat", "recent-repeat"})
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if len(matches) != 1 {
		t.Fatalf("expected only max-lookback match, got %+v", matches)
	}
	if matches[0].RuleName != "persistently_dissatisfied" {
		t.Fatalf("expected persistently_dissatisfied, got %q", matches[0].RuleName)
	}
}

func TestClassifierEvaluateAllSignalsWithContext_ReaskRecordsConfidenceAndStreak(t *testing.T) {
	stubReaskEmbeddings(t, map[string][]float32{
		"current":       makeEmbedding(1, 0),
		"older-repeat":  makeEmbedding(0.8, 0.6),
		"recent-repeat": makeEmbedding(1, 0),
	})

	rules := []config.ReaskRule{{
		Name:          "persistently_dissatisfied",
		Threshold:     0.8,
		LookbackTurns: 2,
	}}
	reaskClassifier, err := NewReaskClassifier(rules, "test-model")
	if err != nil {
		t.Fatalf("NewReaskClassifier() error = %v", err)
	}

	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					ReaskRules: rules,
				},
			},
		},
		reaskClassifier: reaskClassifier,
	}

	results := classifier.EvaluateAllSignalsWithContext(
		"current",
		"assistant answer current",
		"current",
		[]string{"older-repeat", "recent-repeat"},
		[]string{"assistant answer"},
		true,
		true,
		"",
		nil,
	)

	if len(results.MatchedReaskRules) != 1 || results.MatchedReaskRules[0] != "persistently_dissatisfied" {
		t.Fatalf("matched reask rules = %v, want [persistently_dissatisfied]", results.MatchedReaskRules)
	}
	if got := results.SignalConfidences["reask:persistently_dissatisfied"]; !approxEqual(got, 0.8) {
		t.Fatalf("signal confidence = %v, want 0.8", got)
	}
	if got := results.SignalValues["reask:persistently_dissatisfied"]; got != 2 {
		t.Fatalf("signal value = %v, want 2", got)
	}
	if got := results.Metrics.Reask.Confidence; !approxEqual(got, 0.8) {
		t.Fatalf("metrics confidence = %v, want 0.8", got)
	}
}

func TestClassifierEvaluateAllSignalsWithContext_ReaskRetainsOnlyPersistentTier(t *testing.T) {
	stubReaskEmbeddings(t, map[string][]float32{
		"current":       makeEmbedding(1, 0),
		"older-repeat":  makeEmbedding(0.8, 0.6),
		"recent-repeat": makeEmbedding(1, 0),
	})

	rules := []config.ReaskRule{
		{
			Name:          "likely_dissatisfied",
			Threshold:     0.8,
			LookbackTurns: 1,
		},
		{
			Name:          "persistently_dissatisfied",
			Threshold:     0.8,
			LookbackTurns: 2,
		},
	}
	reaskClassifier, err := NewReaskClassifier(rules, "test-model")
	if err != nil {
		t.Fatalf("NewReaskClassifier() error = %v", err)
	}

	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					ReaskRules: rules,
				},
			},
		},
		reaskClassifier: reaskClassifier,
	}

	results := classifier.EvaluateAllSignalsWithContext(
		"current",
		"assistant answer current",
		"current",
		[]string{"older-repeat", "recent-repeat"},
		[]string{"assistant answer"},
		true,
		true,
		"",
		nil,
	)

	if len(results.MatchedReaskRules) != 1 || results.MatchedReaskRules[0] != "persistently_dissatisfied" {
		t.Fatalf("matched reask rules = %v, want [persistently_dissatisfied]", results.MatchedReaskRules)
	}
	if _, ok := results.SignalConfidences["reask:likely_dissatisfied"]; ok {
		t.Fatalf("unexpected likely_dissatisfied confidence in %+v", results.SignalConfidences)
	}
	if _, ok := results.SignalValues["reask:likely_dissatisfied"]; ok {
		t.Fatalf("unexpected likely_dissatisfied value in %+v", results.SignalValues)
	}
	if got := results.SignalConfidences["reask:persistently_dissatisfied"]; !approxEqual(got, 0.8) {
		t.Fatalf("signal confidence = %v, want 0.8", got)
	}
	if got := results.SignalValues["reask:persistently_dissatisfied"]; got != 2 {
		t.Fatalf("signal value = %v, want 2", got)
	}
}

func TestClassifierEvaluateAllSignalsWithContext_ReaskIgnoresNonUserMessages(t *testing.T) {
	stubReaskEmbeddings(t, map[string][]float32{
		"current": makeEmbedding(1, 0),
	})

	rules := []config.ReaskRule{{
		Name:          "likely_dissatisfied",
		Threshold:     0.8,
		LookbackTurns: 1,
	}}
	reaskClassifier, err := NewReaskClassifier(rules, "test-model")
	if err != nil {
		t.Fatalf("NewReaskClassifier() error = %v", err)
	}

	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					ReaskRules: rules,
				},
			},
		},
		reaskClassifier: reaskClassifier,
	}

	results := classifier.EvaluateAllSignalsWithContext(
		"current",
		"system current assistant current",
		"current",
		nil,
		[]string{"current", "current"},
		false,
		true,
		"",
		nil,
	)

	if len(results.MatchedReaskRules) != 0 {
		t.Fatalf("expected no reask matches from non-user messages, got %+v", results.MatchedReaskRules)
	}
}
