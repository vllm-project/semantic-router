package classification

import (
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSignalGroupSoftmaxExclusiveChoosesSingleDomainWinner(t *testing.T) {
	probs := make([]float32, 14)
	probs[4] = 0.40
	probs[6] = 0.38
	for i := range probs {
		if i != 4 && i != 6 {
			probs[i] = 0.02
		}
	}

	classifier := buildGroupedDomainClassifier(&MockCategoryInference{
		classifyWithProbsResult: candle_binding.ClassResultWithProbs{
			Class: 4, Confidence: 0.40,
			Probabilities: probs, NumClasses: 14,
		},
	})

	signals := classifier.EvaluateAllSignals("What are the economic impacts of healthcare reform?")
	if len(signals.MatchedDomainRules) != 1 || signals.MatchedDomainRules[0] != "economics" {
		t.Fatalf("expected only economics after group normalization, got %v", signals.MatchedDomainRules)
	}
	if _, ok := signals.SignalConfidences["domain:health"]; ok {
		t.Fatalf("expected health confidence to be removed after group normalization: %+v", signals.SignalConfidences)
	}
	if signals.SignalConfidences["domain:economics"] <= 0.5 {
		t.Fatalf("expected normalized economics confidence > 0.5, got %+v", signals.SignalConfidences)
	}

	result, err := classifier.EvaluateDecisionWithEngine(signals)
	if err != nil {
		t.Fatalf("EvaluateDecisionWithEngine failed: %v", err)
	}
	if result == nil || result.Decision.Name != "economics-route" {
		t.Fatalf("expected economics-route, got %+v", result)
	}
}

func TestSignalGroupSoftmaxExclusiveChoosesSingleEmbeddingWinner(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"TensorFlow pipeline":  makeEmbedding(1.0, 0.0, 0.0),
		"machine learning":     makeEmbedding(0.95, 0.0, 0.0),
		"neural network":       makeEmbedding(0.90, 0.0, 0.0),
		"python code":          makeEmbedding(0.82, 0.08, 0.0),
		"software development": makeEmbedding(0.80, 0.06, 0.0),
		"recipe":               makeEmbedding(0.20, 0.0, 0.0),
		"ingredients":          makeEmbedding(0.18, 0.0, 0.0),
	})

	classifier := buildGroupedEmbeddingClassifier(t)
	signals := classifier.EvaluateAllSignals("TensorFlow pipeline")
	if len(signals.MatchedEmbeddingRules) != 1 || signals.MatchedEmbeddingRules[0] != "ai" {
		t.Fatalf("expected only ai after group normalization, got %v", signals.MatchedEmbeddingRules)
	}
	if _, ok := signals.SignalConfidences["embedding:programming"]; ok {
		t.Fatalf("expected programming confidence to be removed after group normalization: %+v", signals.SignalConfidences)
	}
	if signals.SignalConfidences["embedding:ai"] <= 0.5 {
		t.Fatalf("expected normalized ai confidence > 0.5, got %+v", signals.SignalConfidences)
	}

	result, err := classifier.EvaluateDecisionWithEngine(signals)
	if err != nil {
		t.Fatalf("EvaluateDecisionWithEngine failed: %v", err)
	}
	if result == nil || result.Decision.Name != "ai-route" {
		t.Fatalf("expected ai-route, got %+v", result)
	}
}

func TestSignalGroupDefaultFallbackMatchesDomainRouteWhenNoGroupMemberFires(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					Categories: []config.Category{
						{CategoryMetadata: config.CategoryMetadata{Name: "economics"}},
						{CategoryMetadata: config.CategoryMetadata{Name: "health"}},
						{CategoryMetadata: config.CategoryMetadata{Name: "other"}},
					},
				},
				Projections: config.Projections{
					Partitions: []config.ProjectionPartition{{
						Name:        "finance-vs-health",
						Semantics:   "softmax_exclusive",
						Temperature: 0.1,
						Members:     []string{"economics", "health", "other"},
						Default:     "other",
					}},
				},
				Decisions: []config.Decision{
					{
						Name:     "other-route",
						Priority: 100,
						Rules: config.RuleCombination{
							Operator: "AND",
							Conditions: []config.RuleCondition{
								{Type: config.SignalTypeDomain, Name: "other"},
							},
						},
					},
				},
			},
		},
	}

	signals := classifier.applySignalGroups(&SignalResults{
		SignalConfidences: map[string]float64{},
	})
	if len(signals.MatchedDomainRules) != 1 || signals.MatchedDomainRules[0] != "other" {
		t.Fatalf("expected default domain fallback, got %v", signals.MatchedDomainRules)
	}

	result, err := classifier.EvaluateDecisionWithEngine(signals)
	if err != nil {
		t.Fatalf("EvaluateDecisionWithEngine failed: %v", err)
	}
	if result == nil || result.Decision.Name != "other-route" {
		t.Fatalf("expected other-route, got %+v", result)
	}
}

func TestSignalGroupDefaultFallbackMatchesEmbeddingRouteWhenNoGroupMemberFires(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					EmbeddingRules: []config.EmbeddingRule{
						{Name: "ai"},
						{Name: "programming"},
						{Name: "cooking"},
					},
				},
				Projections: config.Projections{
					Partitions: []config.ProjectionPartition{{
						Name:        "tech-topics",
						Semantics:   "softmax_exclusive",
						Temperature: 0.1,
						Members:     []string{"ai", "programming", "cooking"},
						Default:     "cooking",
					}},
				},
				Decisions: []config.Decision{
					{
						Name:     "cooking-route",
						Priority: 100,
						Rules: config.RuleCombination{
							Operator: "AND",
							Conditions: []config.RuleCondition{
								{Type: config.SignalTypeEmbedding, Name: "cooking"},
							},
						},
					},
				},
			},
		},
	}

	signals := classifier.applySignalGroups(&SignalResults{
		SignalConfidences: map[string]float64{},
	})
	if len(signals.MatchedEmbeddingRules) != 1 || signals.MatchedEmbeddingRules[0] != "cooking" {
		t.Fatalf("expected default embedding fallback, got %v", signals.MatchedEmbeddingRules)
	}

	result, err := classifier.EvaluateDecisionWithEngine(signals)
	if err != nil {
		t.Fatalf("EvaluateDecisionWithEngine failed: %v", err)
	}
	if result == nil || result.Decision.Name != "cooking-route" {
		t.Fatalf("expected cooking-route, got %+v", result)
	}
}

func TestAnalyzeSoftmaxSignalGroupCentroidsWarnsOnSimilarMembers(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"machine learning":     makeEmbedding(1.0, 0.0, 0.0),
		"neural network":       makeEmbedding(0.95, 0.0, 0.0),
		"python code":          makeEmbedding(0.98, 0.05, 0.0),
		"software development": makeEmbedding(0.96, 0.04, 0.0),
		"recipe":               makeEmbedding(0.0, 0.0, 1.0),
		"ingredients":          makeEmbedding(0.0, 0.0, 0.95),
	})

	classifier := &Classifier{
		Config: &config.RouterConfig{
			InlineModels: config.InlineModels{
				EmbeddingModels: config.EmbeddingModels{
					EmbeddingConfig: config.HNSWConfig{PreloadEmbeddings: true},
				},
			},
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					EmbeddingRules: []config.EmbeddingRule{
						{
							Name:                      "ai",
							Candidates:                []string{"machine learning", "neural network"},
							SimilarityThreshold:       0.7,
							AggregationMethodConfiged: config.AggregationMethodMax,
						},
						{
							Name:                      "programming",
							Candidates:                []string{"python code", "software development"},
							SimilarityThreshold:       0.7,
							AggregationMethodConfiged: config.AggregationMethodMax,
						},
						{
							Name:                      "cooking",
							Candidates:                []string{"recipe", "ingredients"},
							SimilarityThreshold:       0.7,
							AggregationMethodConfiged: config.AggregationMethodMax,
						},
					},
				},
				Projections: config.Projections{
					Partitions: []config.ProjectionPartition{{
						Name:        "tech-topics",
						Semantics:   "softmax_exclusive",
						Temperature: 0.1,
						Members:     []string{"ai", "programming", "cooking"},
						Default:     "cooking",
					}},
				},
			},
		},
		keywordEmbeddingClassifier: newTestEmbeddingClassifier(t, []config.EmbeddingRule{
			{
				Name:                      "ai",
				Candidates:                []string{"machine learning", "neural network"},
				SimilarityThreshold:       0.7,
				AggregationMethodConfiged: config.AggregationMethodMax,
			},
			{
				Name:                      "programming",
				Candidates:                []string{"python code", "software development"},
				SimilarityThreshold:       0.7,
				AggregationMethodConfiged: config.AggregationMethodMax,
			},
			{
				Name:                      "cooking",
				Candidates:                []string{"recipe", "ingredients"},
				SimilarityThreshold:       0.7,
				AggregationMethodConfiged: config.AggregationMethodMax,
			},
		}, config.HNSWConfig{PreloadEmbeddings: true}),
	}

	warnings, err := classifier.AnalyzeSoftmaxSignalGroupCentroids(0.7)
	if err != nil {
		t.Fatalf("AnalyzeSoftmaxSignalGroupCentroids failed: %v", err)
	}
	if len(warnings) != 1 {
		t.Fatalf("expected 1 centroid warning, got %+v", warnings)
	}
	if warnings[0].GroupName != "tech-topics" || warnings[0].LeftMember != "ai" || warnings[0].RightMember != "programming" {
		t.Fatalf("unexpected warning payload: %+v", warnings[0])
	}
	if warnings[0].Similarity < 0.7 {
		t.Fatalf("expected similarity >= 0.7, got %+v", warnings[0])
	}
}

func buildGroupedDomainClassifier(mock *MockCategoryInference) *Classifier {
	cfg := domainTestConfig()
	cfg.Decisions = []config.Decision{
		{
			Name:     "health-route",
			Priority: 200,
			Rules: config.RuleCombination{
				Operator: "AND",
				Conditions: []config.RuleCondition{
					{Type: config.SignalTypeDomain, Name: "health"},
				},
			},
		},
		{
			Name:     "economics-route",
			Priority: 100,
			Rules: config.RuleCombination{
				Operator: "AND",
				Conditions: []config.RuleCondition{
					{Type: config.SignalTypeDomain, Name: "economics"},
				},
			},
		},
	}
	cfg.Projections.Partitions = []config.ProjectionPartition{
		{
			Name:        "finance-vs-health",
			Semantics:   "softmax_exclusive",
			Temperature: 0.1,
			Members:     []string{"economics", "health", "other"},
			Default:     "other",
		},
	}
	cfg.Categories = []config.Category{
		{CategoryMetadata: config.CategoryMetadata{Name: "economics"}},
		{CategoryMetadata: config.CategoryMetadata{Name: "health"}},
		{CategoryMetadata: config.CategoryMetadata{Name: "other"}},
	}

	classifier := buildDomainClassifier(mock)
	classifier.Config = cfg
	return classifier
}

func buildGroupedEmbeddingClassifier(t *testing.T) *Classifier {
	t.Helper()

	rules := topicRules()
	embeddingClassifier := newTestEmbeddingClassifier(t, rules, config.HNSWConfig{
		PreloadEmbeddings: true,
		TopK:              intPtr(2),
	})

	return &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					EmbeddingRules: rules,
				},
				Projections: config.Projections{
					Partitions: []config.ProjectionPartition{{
						Name:        "tech-topics",
						Semantics:   "softmax_exclusive",
						Temperature: 0.1,
						Members:     []string{"ai", "programming", "cooking"},
						Default:     "cooking",
					}},
				},
				Decisions: []config.Decision{
					{
						Name:     "programming-route",
						Priority: 200,
						Rules: config.RuleCombination{
							Operator: "AND",
							Conditions: []config.RuleCondition{
								{Type: config.SignalTypeEmbedding, Name: "programming"},
							},
						},
					},
					{
						Name:     "ai-route",
						Priority: 100,
						Rules: config.RuleCombination{
							Operator: "AND",
							Conditions: []config.RuleCondition{
								{Type: config.SignalTypeEmbedding, Name: "ai"},
							},
						},
					},
				},
			},
		},
		keywordEmbeddingClassifier: embeddingClassifier,
	}
}

func TestApplySignalGroupsRecordsPartitionTraceExclusiveWinner(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					Categories: []config.Category{
						{CategoryMetadata: config.CategoryMetadata{Name: "economics"}},
						{CategoryMetadata: config.CategoryMetadata{Name: "health"}},
					},
				},
				Projections: config.Projections{
					Partitions: []config.ProjectionPartition{{
						Name:      "econ-health",
						Semantics: "exclusive",
						Members:   []string{"economics", "health"},
					}},
				},
			},
		},
	}
	results := classifier.applySignalGroups(&SignalResults{
		MatchedDomainRules: []string{"economics", "health"},
		SignalConfidences: map[string]float64{
			"domain:economics": 0.9,
			"domain:health":    0.4,
		},
	})
	if results.ProjectionTrace == nil || len(results.ProjectionTrace.Partitions) != 1 {
		t.Fatalf("expected one partition trace, got %+v", results.ProjectionTrace)
	}
	p := results.ProjectionTrace.Partitions[0]
	if p.Winner != "economics" {
		t.Fatalf("winner = %q", p.Winner)
	}
	if p.Margin < 0.49 {
		t.Fatalf("margin = %v want ~0.5 raw spread", p.Margin)
	}
	if len(p.Contenders) != 2 {
		t.Fatalf("contenders = %+v", p.Contenders)
	}
}

func TestApplySignalGroupsRecordsPartitionTraceDefaultSynthetic(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					Categories: []config.Category{
						{CategoryMetadata: config.CategoryMetadata{Name: "economics"}},
						{CategoryMetadata: config.CategoryMetadata{Name: "health"}},
						{CategoryMetadata: config.CategoryMetadata{Name: "other"}},
					},
				},
				Projections: config.Projections{
					Partitions: []config.ProjectionPartition{{
						Name:      "finance-vs-health",
						Semantics: "softmax_exclusive",
						Members:   []string{"economics", "health", "other"},
						Default:   "other",
					}},
				},
			},
		},
	}
	results := classifier.applySignalGroups(&SignalResults{SignalConfidences: map[string]float64{}})
	if results.ProjectionTrace == nil || len(results.ProjectionTrace.Partitions) != 1 {
		t.Fatalf("expected default partition trace, got %+v", results.ProjectionTrace)
	}
	p := results.ProjectionTrace.Partitions[0]
	if !p.DefaultUsed || p.Winner != "other" {
		t.Fatalf("partition trace = %+v", p)
	}
}
