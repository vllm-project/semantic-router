package classification

import (
	"errors"
	"math"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEmbeddingClassifier_SoftMatchingDisabledWithoutHardMatch(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"query":        makeEmbedding(1.0, 0.0, 0.0),
		"candidate_a1": makeEmbedding(0.60, 0.0, 0.0),
		"candidate_a2": makeEmbedding(0.55, 0.0, 0.0),
		"candidate_b1": makeEmbedding(0.65, 0.0, 0.0),
		"candidate_b2": makeEmbedding(0.60, 0.0, 0.0),
		"candidate_c1": makeEmbedding(0.72, 0.0, 0.0),
		"candidate_c2": makeEmbedding(0.70, 0.0, 0.0),
	})

	softMatchingDisabled := false
	classifier := newTestEmbeddingClassifier(t, softMatchingRules(), config.HNSWConfig{
		PreloadEmbeddings:  true,
		EnableSoftMatching: &softMatchingDisabled,
		MinScoreThreshold:  0.5,
		TopK:               intPtr(1),
	})

	ruleName, score, err := classifier.Classify("query")
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}
	if ruleName != "" {
		t.Errorf("Expected no match, got rule: %s with score: %.2f", ruleName, score)
	}
}

func TestEmbeddingClassifier_SoftMatchingEnabledReturnsBestRule(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"query":        makeEmbedding(1.0, 0.0, 0.0),
		"candidate_a1": makeEmbedding(0.60, 0.0, 0.0),
		"candidate_a2": makeEmbedding(0.55, 0.0, 0.0),
		"candidate_b1": makeEmbedding(0.65, 0.0, 0.0),
		"candidate_b2": makeEmbedding(0.60, 0.0, 0.0),
		"candidate_c1": makeEmbedding(0.72, 0.0, 0.0),
		"candidate_c2": makeEmbedding(0.70, 0.0, 0.0),
	})

	softMatchingEnabled := true
	classifier := newTestEmbeddingClassifier(t, softMatchingRules(), config.HNSWConfig{
		PreloadEmbeddings:  true,
		EnableSoftMatching: &softMatchingEnabled,
		MinScoreThreshold:  0.5,
		TopK:               intPtr(1),
	})

	ruleName, score, err := classifier.Classify("query")
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}
	if ruleName != "rule_c" {
		t.Errorf("Expected rule_c, got: %s", ruleName)
	}
	if score < 0.71 || score > 0.73 {
		t.Errorf("Expected score ~0.72, got: %.2f", score)
	}
}

func TestEmbeddingClassifier_ClassifyAllDefaultTopKReturnsBestHardMatch(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"TensorFlow pipeline":  makeEmbedding(0.90, 0.85, 0.10),
		"machine learning":     makeEmbedding(0.85, 0.0, 0.0),
		"neural network":       makeEmbedding(0.80, 0.0, 0.0),
		"python code":          makeEmbedding(0.0, 0.90, 0.0),
		"software development": makeEmbedding(0.0, 0.85, 0.0),
		"recipe":               makeEmbedding(0.0, 0.0, 0.30),
		"ingredients":          makeEmbedding(0.0, 0.0, 0.25),
	})

	classifier := newTestEmbeddingClassifier(t, topicRules(), config.HNSWConfig{PreloadEmbeddings: true})

	matched, err := classifier.ClassifyAll("TensorFlow pipeline")
	if err != nil {
		t.Fatalf("ClassifyAll failed: %v", err)
	}
	if len(matched) != 1 {
		t.Fatalf("Expected 1 match with default top_k, got %d: %+v", len(matched), matched)
	}
	if matched[0].RuleName != "programming" {
		t.Fatalf("Expected top hard match to be 'programming', got %+v", matched)
	}
	if matched[0].Method != "hard" {
		t.Fatalf("Expected hard match, got %+v", matched[0])
	}
}

func TestEmbeddingClassifier_ClassifyAllExplicitTopKReturnsMultipleHardMatches(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"TensorFlow pipeline":  makeEmbedding(0.90, 0.85, 0.10),
		"machine learning":     makeEmbedding(0.85, 0.0, 0.0),
		"neural network":       makeEmbedding(0.80, 0.0, 0.0),
		"python code":          makeEmbedding(0.0, 0.90, 0.0),
		"software development": makeEmbedding(0.0, 0.85, 0.0),
		"recipe":               makeEmbedding(0.0, 0.0, 0.30),
		"ingredients":          makeEmbedding(0.0, 0.0, 0.25),
	})

	classifier := newTestEmbeddingClassifier(t, topicRules(), config.HNSWConfig{
		PreloadEmbeddings: true,
		TopK:              intPtr(2),
	})

	matched, err := classifier.ClassifyAll("TensorFlow pipeline")
	if err != nil {
		t.Fatalf("ClassifyAll failed: %v", err)
	}
	if len(matched) != 2 {
		t.Fatalf("Expected 2 matches, got %d: %+v", len(matched), matched)
	}

	expectedOrder := []string{"programming", "ai"}
	for i, match := range matched {
		if match.RuleName != expectedOrder[i] {
			t.Fatalf("Expected match %d to be %q, got %+v", i, expectedOrder[i], matched)
		}
		if match.Method != "hard" {
			t.Errorf("Expected hard match for %s, got %s", match.RuleName, match.Method)
		}
		if match.Score < 0.70 {
			t.Errorf("Expected score >= 0.70 for %s, got %.4f", match.RuleName, match.Score)
		}
	}
}

func TestEmbeddingClassifier_ClassifyDetailedReturnsAllAcceptedMatchesBeforeTopK(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"TensorFlow pipeline":  makeEmbedding(0.90, 0.85, 0.10),
		"machine learning":     makeEmbedding(0.85, 0.0, 0.0),
		"neural network":       makeEmbedding(0.80, 0.0, 0.0),
		"python code":          makeEmbedding(0.0, 0.90, 0.0),
		"software development": makeEmbedding(0.0, 0.85, 0.0),
		"recipe":               makeEmbedding(0.0, 0.0, 0.30),
		"ingredients":          makeEmbedding(0.0, 0.0, 0.25),
	})

	classifier := newTestEmbeddingClassifier(t, topicRules(), config.HNSWConfig{
		PreloadEmbeddings: true,
		TopK:              intPtr(1),
	})

	detailed, err := classifier.ClassifyDetailed("TensorFlow pipeline")
	if err != nil {
		t.Fatalf("ClassifyDetailed failed: %v", err)
	}
	if len(detailed.Scores) != 3 {
		t.Fatalf("expected full score distribution across 3 rules, got %+v", detailed.Scores)
	}
	if len(detailed.Matches) != 2 {
		t.Fatalf("expected both hard matches before top_k output shaping, got %+v", detailed.Matches)
	}
	if detailed.Matches[0].RuleName != "programming" || detailed.Matches[1].RuleName != "ai" {
		t.Fatalf("expected programming then ai from detailed matches, got %+v", detailed.Matches)
	}

	limited, err := classifier.ClassifyAll("TensorFlow pipeline")
	if err != nil {
		t.Fatalf("ClassifyAll failed: %v", err)
	}
	if len(limited) != 1 || limited[0].RuleName != "programming" {
		t.Fatalf("expected top_k output shaping to keep only programming, got %+v", limited)
	}
}

func TestEmbeddingClassifier_ClassifyAllMatchesClassify(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"query":                makeEmbedding(1.0, 0.0, 0.0),
		"machine learning":     makeEmbedding(0.85, 0.0, 0.0),
		"neural network":       makeEmbedding(0.80, 0.0, 0.0),
		"python code":          makeEmbedding(0.30, 0.0, 0.0),
		"software development": makeEmbedding(0.25, 0.0, 0.0),
		"recipe":               makeEmbedding(0.10, 0.0, 0.0),
		"ingredients":          makeEmbedding(0.05, 0.0, 0.0),
	})

	classifier := newTestEmbeddingClassifier(t, topicRules(), config.HNSWConfig{PreloadEmbeddings: true})

	matched, err := classifier.ClassifyAll("query")
	if err != nil {
		t.Fatalf("ClassifyAll failed: %v", err)
	}
	if len(matched) != 1 || matched[0].RuleName != "ai" {
		t.Fatalf("Expected single 'ai' match, got: %+v", matched)
	}

	ruleName, score, err := classifier.Classify("query")
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}
	if ruleName != matched[0].RuleName {
		t.Errorf("Classify returned %q but ClassifyAll returned %q", ruleName, matched[0].RuleName)
	}
	if score != matched[0].Score {
		t.Errorf("Classify score %.4f != ClassifyAll score %.4f", score, matched[0].Score)
	}
}

func TestEmbeddingClassifier_ClassifyAllSoftMatchesRespectConfiguredTopK(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"query":       makeEmbedding(1.0, 0.0, 0.0),
		"candidate_a": makeEmbedding(0.71, 0.0, 0.0),
		"candidate_b": makeEmbedding(0.68, 0.0, 0.0),
		"candidate_c": makeEmbedding(0.62, 0.0, 0.0),
	})

	softMatchingEnabled := true
	classifier := newTestEmbeddingClassifier(t, []config.EmbeddingRule{
		{
			Name:                      "rule_a",
			Candidates:                []string{"candidate_a"},
			SimilarityThreshold:       0.80,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
		{
			Name:                      "rule_b",
			Candidates:                []string{"candidate_b"},
			SimilarityThreshold:       0.80,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
		{
			Name:                      "rule_c",
			Candidates:                []string{"candidate_c"},
			SimilarityThreshold:       0.80,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
	}, config.HNSWConfig{
		PreloadEmbeddings:  true,
		EnableSoftMatching: &softMatchingEnabled,
		MinScoreThreshold:  0.5,
		TopK:               intPtr(2),
	})

	matched, err := classifier.ClassifyAll("query")
	if err != nil {
		t.Fatalf("ClassifyAll failed: %v", err)
	}
	if len(matched) != 2 {
		t.Fatalf("Expected top 2 soft matches, got %d: %+v", len(matched), matched)
	}
	if matched[0].RuleName != "rule_a" || matched[1].RuleName != "rule_b" {
		t.Fatalf("Expected rule_a then rule_b, got %+v", matched)
	}
	for _, match := range matched {
		if match.Method != "soft" {
			t.Fatalf("Expected soft match, got %+v", match)
		}
	}
}

func TestEmbeddingClassifier_MaxAggregationUsesPrototypeAwareScoring(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"query":      makeEmbedding(1.0, 0.0, 0.0),
		"candidate1": makeEmbedding(1.0, 0.0, 0.0),
		"candidate2": makeEmbedding(0.4, 0.0, 0.0),
	})

	classifier := newTestEmbeddingClassifier(t, []config.EmbeddingRule{
		{
			Name:                      "rule_a",
			Candidates:                []string{"candidate1", "candidate2"},
			SimilarityThreshold:       0.90,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
	}, config.HNSWConfig{
		PreloadEmbeddings: true,
		PrototypeScoring: config.PrototypeScoringConfig{
			BestWeight: 0.75,
			TopM:       2,
		},
	})

	detailed, err := classifier.ClassifyDetailed("query")
	if err != nil {
		t.Fatalf("ClassifyDetailed failed: %v", err)
	}
	if len(detailed.Scores) != 1 {
		t.Fatalf("expected a single scored rule, got %+v", detailed.Scores)
	}

	score := detailed.Scores[0]
	if score.Best != 1.0 {
		t.Fatalf("expected best similarity to stay at 1.0, got %.4f", score.Best)
	}
	if math.Abs(float64(score.Support)-0.7) > 1e-6 {
		t.Fatalf("expected support to average top-2 prototypes to 0.7, got %.4f", score.Support)
	}
	if math.Abs(float64(score.Score)-0.925) > 1e-6 {
		t.Fatalf("expected prototype-aware max score 0.925, got %.4f", score.Score)
	}
	if score.PrototypeCount != 2 {
		t.Fatalf("expected 2 prototypes, got %d", score.PrototypeCount)
	}
	if len(detailed.Matches) != 1 || math.Abs(float64(detailed.Matches[0].Score)-0.925) > 1e-6 {
		t.Fatalf("expected matched score to use prototype-aware aggregation, got %+v", detailed.Matches)
	}
}

func softMatchingRules() []config.EmbeddingRule {
	return []config.EmbeddingRule{
		{
			Name:                      "rule_a",
			Candidates:                []string{"candidate_a1", "candidate_a2"},
			SimilarityThreshold:       0.75,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
		{
			Name:                      "rule_b",
			Candidates:                []string{"candidate_b1", "candidate_b2"},
			SimilarityThreshold:       0.75,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
		{
			Name:                      "rule_c",
			Candidates:                []string{"candidate_c1", "candidate_c2"},
			SimilarityThreshold:       0.75,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
	}
}

func topicRules() []config.EmbeddingRule {
	return []config.EmbeddingRule{
		{
			Name:                      "ai",
			Candidates:                []string{"machine learning", "neural network"},
			SimilarityThreshold:       0.70,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
		{
			Name:                      "programming",
			Candidates:                []string{"python code", "software development"},
			SimilarityThreshold:       0.70,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
		{
			Name:                      "cooking",
			Candidates:                []string{"recipe", "ingredients"},
			SimilarityThreshold:       0.70,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
	}
}

func newTestEmbeddingClassifier(
	t *testing.T,
	rules []config.EmbeddingRule,
	hnswConfig config.HNSWConfig,
) *EmbeddingClassifier {
	t.Helper()

	classifier, err := NewEmbeddingClassifier(rules, hnswConfig)
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}
	return classifier
}

func stubEmbeddingLookup(t *testing.T, mockEmbeddings map[string][]float32) {
	t.Helper()

	originalFunc := getEmbeddingWithModelType
	getEmbeddingWithModelType = func(text string, modelType string, targetDim int) (*candle_binding.EmbeddingOutput, error) {
		if emb, ok := mockEmbeddings[text]; ok {
			return &candle_binding.EmbeddingOutput{Embedding: emb}, nil
		}
		return &candle_binding.EmbeddingOutput{Embedding: makeEmbedding(0.0)}, nil
	}
	t.Cleanup(func() {
		getEmbeddingWithModelType = originalFunc
	})
}

func makeEmbedding(values ...float32) []float32 {
	result := make([]float32, 768)
	for i, value := range values {
		if i < len(result) {
			result[i] = value
		}
	}
	return result
}

func intPtr(value int) *int {
	return &value
}

// stubMultiModalImageLookup overrides getMultiModalImageEmbedding for tests.
// keys map raw image payloads (the same string passed to ClassifyDetailedMultimodal)
// to the embedding the multimodal model would have produced. Tests must
// populate every payload they expect the classifier to embed; an unmocked
// payload fails the test loudly so dimension or stub mismatches surface as
// real failures instead of silently classifying against a 1-D zero vector.
func stubMultiModalImageLookup(t *testing.T, mockEmbeddings map[string][]float32) {
	t.Helper()

	originalFunc := getMultiModalImageEmbedding
	getMultiModalImageEmbedding = func(imageRef string, targetDim int) ([]float32, error) {
		if emb, ok := mockEmbeddings[imageRef]; ok {
			return emb, nil
		}
		t.Errorf("stubMultiModalImageLookup called with unmocked payload %q; populate the fixture or stub the embedding explicitly", imageRef)
		t.FailNow()
		return nil, nil
	}
	t.Cleanup(func() {
		getMultiModalImageEmbedding = originalFunc
	})
}

// chipFabImageRules returns a small, fictional anchor pack matching what
// the chip-fab vertical demo will ship as its default. Candidates are
// short text descriptions; the rule's QueryModality is Image so it only
// fires when the request carries an image attachment.
func chipFabImageRules() []config.EmbeddingRule {
	return []config.EmbeddingRule{
		{
			Name:                      "chip_fab_sensitive_imagery",
			Candidates:                []string{"wafer photo", "SEM micrograph"},
			SimilarityThreshold:       0.70,
			AggregationMethodConfiged: config.AggregationMethodMax,
			QueryModality:             config.QueryModalityImage,
		},
		{
			Name:                      "ambient_office_imagery",
			Candidates:                []string{"office whiteboard", "conference room"},
			SimilarityThreshold:       0.70,
			AggregationMethodConfiged: config.AggregationMethodMax,
			QueryModality:             config.QueryModalityImage,
		},
	}
}

// multimodalHNSWConfig returns the HNSWConfig every multimodal test needs.
// Image and audio rules are init-rejected unless ModelType is "multimodal",
// so this helper centralizes that pairing for the test fixtures below.
func multimodalHNSWConfig(preload bool) config.HNSWConfig {
	return config.HNSWConfig{
		ModelType:         "multimodal",
		PreloadEmbeddings: preload,
	}
}

// mixedModalityRules combines text and image rules to verify the classifier
// only scores rules that match the active query modality.
func mixedModalityRules() []config.EmbeddingRule {
	return []config.EmbeddingRule{
		{
			Name:                      "text_topic_ai",
			Candidates:                []string{"machine learning", "neural network"},
			SimilarityThreshold:       0.70,
			AggregationMethodConfiged: config.AggregationMethodMax,
			// QueryModality omitted: defaults to text, exercising the
			// backward-compatible default path.
		},
		{
			Name:                      "chip_fab_sensitive_imagery",
			Candidates:                []string{"wafer photo", "SEM micrograph"},
			SimilarityThreshold:       0.70,
			AggregationMethodConfiged: config.AggregationMethodMax,
			QueryModality:             config.QueryModalityImage,
		},
	}
}

func TestEmbeddingClassifier_ClassifyDetailedMultimodalImageHardMatch(t *testing.T) {
	stubMultiModalImageLookup(t, map[string][]float32{
		"data:image/png;base64,FAKE_WAFER_BYTES": makeEmbedding(0.9, 0.1, 0.0),
	})
	stubEmbeddingLookup(t, map[string][]float32{
		// Text candidates are embedded the same way as for the text-query path
		// because preloading goes through getEmbeddingWithModelType regardless
		// of the rule's query modality. The multimodal model emits text and
		// image embeddings in the same shared space, so this is a valid stub.
		"wafer photo":       makeEmbedding(0.95, 0.0, 0.0),
		"SEM micrograph":    makeEmbedding(0.85, 0.0, 0.0),
		"office whiteboard": makeEmbedding(0.0, 0.95, 0.0),
		"conference room":   makeEmbedding(0.0, 0.85, 0.0),
	})

	classifier := newTestEmbeddingClassifier(t, chipFabImageRules(), multimodalHNSWConfig(true))

	result, err := classifier.ClassifyDetailedMultimodal(
		config.QueryModalityImage,
		"data:image/png;base64,FAKE_WAFER_BYTES",
	)
	if err != nil {
		t.Fatalf("ClassifyDetailedMultimodal failed: %v", err)
	}
	if len(result.Matches) != 1 {
		t.Fatalf("expected exactly one match, got %d: %+v", len(result.Matches), result.Matches)
	}
	if result.Matches[0].RuleName != "chip_fab_sensitive_imagery" {
		t.Errorf("expected chip_fab_sensitive_imagery to match, got %s", result.Matches[0].RuleName)
	}
	if result.Matches[0].Method != "hard" {
		t.Errorf("expected hard match, got %+v", result.Matches[0])
	}
}

func TestEmbeddingClassifier_ClassifyDetailedFiltersOutImageRules(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"machine learning":             makeEmbedding(0.0, 0.0, 0.95),
		"neural network":               makeEmbedding(0.0, 0.0, 0.85),
		"wafer photo":                  makeEmbedding(0.95, 0.0, 0.0),
		"SEM micrograph":               makeEmbedding(0.85, 0.0, 0.0),
		"TensorFlow training pipeline": makeEmbedding(0.0, 0.0, 0.95),
	})

	classifier := newTestEmbeddingClassifier(t, mixedModalityRules(), multimodalHNSWConfig(true))

	result, err := classifier.ClassifyDetailed("TensorFlow training pipeline")
	if err != nil {
		t.Fatalf("ClassifyDetailed failed: %v", err)
	}
	for _, score := range result.Scores {
		if score.Name == "chip_fab_sensitive_imagery" {
			t.Errorf("text classification path should NOT score image-modality rules, got %+v", score)
		}
	}
	if len(result.Matches) != 1 || result.Matches[0].RuleName != "text_topic_ai" {
		t.Fatalf("expected text_topic_ai as sole match, got %+v", result.Matches)
	}
}

func TestEmbeddingClassifier_ClassifyDetailedMultimodalFiltersOutTextRules(t *testing.T) {
	stubMultiModalImageLookup(t, map[string][]float32{
		"data:image/png;base64,FAKE_WAFER_BYTES": makeEmbedding(0.9, 0.0, 0.0),
	})
	stubEmbeddingLookup(t, map[string][]float32{
		"machine learning": makeEmbedding(0.0, 0.0, 0.95),
		"neural network":   makeEmbedding(0.0, 0.0, 0.85),
		"wafer photo":      makeEmbedding(0.95, 0.0, 0.0),
		"SEM micrograph":   makeEmbedding(0.85, 0.0, 0.0),
	})

	classifier := newTestEmbeddingClassifier(t, mixedModalityRules(), multimodalHNSWConfig(true))

	result, err := classifier.ClassifyDetailedMultimodal(
		config.QueryModalityImage,
		"data:image/png;base64,FAKE_WAFER_BYTES",
	)
	if err != nil {
		t.Fatalf("ClassifyDetailedMultimodal failed: %v", err)
	}
	for _, score := range result.Scores {
		if score.Name == "text_topic_ai" {
			t.Errorf("image classification path should NOT score text-modality rules, got %+v", score)
		}
	}
	if len(result.Matches) != 1 || result.Matches[0].RuleName != "chip_fab_sensitive_imagery" {
		t.Fatalf("expected chip_fab_sensitive_imagery as sole match, got %+v", result.Matches)
	}
}

func TestEmbeddingClassifier_ClassifyDetailedMultimodalRejectsTextModality(t *testing.T) {
	classifier := newTestEmbeddingClassifier(t, chipFabImageRules(), multimodalHNSWConfig(false))

	_, err := classifier.ClassifyDetailedMultimodal(config.QueryModalityText, "anything")
	if err == nil {
		t.Fatal("expected error for modality=text passed to ClassifyDetailedMultimodal, got nil")
	}
	_, err = classifier.ClassifyDetailedMultimodal(config.QueryModality(""), "anything")
	if err == nil {
		t.Fatal("expected error for empty modality passed to ClassifyDetailedMultimodal, got nil")
	}
}

func TestEmbeddingClassifier_ClassifyDetailedMultimodalRejectsEmptyPayload(t *testing.T) {
	classifier := newTestEmbeddingClassifier(t, chipFabImageRules(), multimodalHNSWConfig(false))

	_, err := classifier.ClassifyDetailedMultimodal(config.QueryModalityImage, "")
	if err == nil {
		t.Fatal("expected error for empty payload, got nil")
	}
}

func TestEmbeddingClassifier_ClassifyDetailedMultimodalRejectsAudioUntilWired(t *testing.T) {
	classifier := newTestEmbeddingClassifier(t, chipFabImageRules(), multimodalHNSWConfig(false))

	_, err := classifier.ClassifyDetailedMultimodal(config.QueryModalityAudio, "data:audio/wav;base64,XYZ")
	if err == nil {
		t.Fatal("expected error for audio modality (not yet wired), got nil")
	}
}

func TestEmbeddingClassifier_PreloadCoversImageModalityCandidates(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"wafer photo":       makeEmbedding(0.95, 0.0, 0.0),
		"SEM micrograph":    makeEmbedding(0.85, 0.0, 0.0),
		"office whiteboard": makeEmbedding(0.0, 0.95, 0.0),
		"conference room":   makeEmbedding(0.0, 0.85, 0.0),
	})

	classifier, err := NewEmbeddingClassifier(chipFabImageRules(), multimodalHNSWConfig(true))
	if err != nil {
		t.Fatalf("NewEmbeddingClassifier failed: %v", err)
	}

	// All four candidates across both image-modality rules should be preloaded.
	// Preload is what makes the runtime image-query path cheap; without this
	// guarantee, multimodal classification would silently re-embed every anchor
	// per request.
	got := classifier.GetPreloadStats()
	if got != 4 {
		t.Errorf("expected 4 preloaded candidates from image rules, got %d", got)
	}
}

func TestEmbeddingClassifier_ClassifyDetailedMultimodalSurfacesEmbeddingErrors(t *testing.T) {
	wantErr := errors.New("synthetic FFI failure")

	originalFunc := getMultiModalImageEmbedding
	getMultiModalImageEmbedding = func(imageRef string, targetDim int) ([]float32, error) {
		return nil, wantErr
	}
	t.Cleanup(func() {
		getMultiModalImageEmbedding = originalFunc
	})

	classifier, err := NewEmbeddingClassifier(chipFabImageRules(), multimodalHNSWConfig(false))
	if err != nil {
		t.Fatalf("NewEmbeddingClassifier failed: %v", err)
	}

	_, err = classifier.ClassifyDetailedMultimodal(config.QueryModalityImage, "data:image/png;base64,WHATEVER")
	if err == nil {
		t.Fatal("expected error to surface from getMultiModalImageEmbedding, got nil")
	}
	if !errors.Is(err, wantErr) {
		t.Errorf("expected wrapped error to satisfy errors.Is for synthetic failure, got: %v", err)
	}
}

// textOnlyRulesForGracefulDegradation is a fixture for the no-matching-rules
// tests: a classifier configured with text-only rules should return an empty
// result (not an error) when ClassifyDetailedMultimodal is called for image,
// and a classifier configured with image-only rules should return an empty
// result (not an error) when ClassifyDetailed is called for text.
func textOnlyRulesForGracefulDegradation() []config.EmbeddingRule {
	return []config.EmbeddingRule{
		{
			Name:                      "text_only_topic",
			Candidates:                []string{"machine learning"},
			SimilarityThreshold:       0.70,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
	}
}

func TestEmbeddingClassifier_ClassifyDetailedMultimodalNoMatchingRulesReturnsEmpty(t *testing.T) {
	classifier := newTestEmbeddingClassifier(t, textOnlyRulesForGracefulDegradation(), multimodalHNSWConfig(false))

	result, err := classifier.ClassifyDetailedMultimodal(
		config.QueryModalityImage,
		"data:image/png;base64,WHATEVER",
	)
	if err != nil {
		t.Fatalf("expected no error when classifier has no image rules, got: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result on graceful-degradation path, got nil")
	}
	if len(result.Matches) != 0 {
		t.Errorf("expected zero matches, got %+v", result.Matches)
	}
	if len(result.Scores) != 0 {
		t.Errorf("expected zero scores, got %+v", result.Scores)
	}
}

func TestEmbeddingClassifier_ClassifyDetailedNoMatchingTextRulesReturnsEmpty(t *testing.T) {
	classifier := newTestEmbeddingClassifier(t, chipFabImageRules(), multimodalHNSWConfig(false))

	result, err := classifier.ClassifyDetailed("any text query")
	if err != nil {
		t.Fatalf("expected no error when classifier has no text rules, got: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result on graceful-degradation path, got nil")
	}
	if len(result.Matches) != 0 {
		t.Errorf("expected zero matches, got %+v", result.Matches)
	}
	if len(result.Scores) != 0 {
		t.Errorf("expected zero scores, got %+v", result.Scores)
	}
}
