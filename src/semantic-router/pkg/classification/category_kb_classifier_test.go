package classification

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestKnowledgeBaseDefinitionLoad(t *testing.T) {
	root := writeKnowledgeBaseFixture(t)
	definition, err := config.LoadKnowledgeBaseDefinition("", config.KnowledgeBaseSource{Path: root})
	if err != nil {
		t.Fatalf("LoadKnowledgeBaseDefinition: %v", err)
	}
	if len(definition.Labels) != 3 {
		t.Fatalf("expected 3 labels, got %d", len(definition.Labels))
	}
	if definition.Labels["proprietary_code"].Description == "" {
		t.Fatal("expected proprietary_code description to be loaded")
	}
}

func TestKnowledgeBaseClassifierLoadDefinition(t *testing.T) {
	root := writeKnowledgeBaseFixture(t)
	classifier := &KnowledgeBaseClassifier{
		rule: config.KnowledgeBaseConfig{
			Name: "privacy_kb",
			Source: config.KnowledgeBaseSource{
				Path: root,
			},
		},
	}
	if err := classifier.loadDefinition(); err != nil {
		t.Fatalf("loadDefinition: %v", err)
	}
	if got := len(classifier.labels); got != 3 {
		t.Fatalf("expected 3 labels, got %d", got)
	}
}

func TestKnowledgeBaseClassifierLabelCount(t *testing.T) {
	classifier := &KnowledgeBaseClassifier{
		labels: map[string]*kbLabelData{
			"a": {Exemplars: []string{"a1"}},
			"b": {Exemplars: []string{"b1"}},
		},
	}
	if got := classifier.LabelCount(); got != 2 {
		t.Fatalf("LabelCount = %d, want 2", got)
	}
}

func TestKnowledgeBaseClassifierClassifyEmptyQuery(t *testing.T) {
	classifier := &KnowledgeBaseClassifier{}
	if _, err := classifier.Classify(""); err == nil {
		t.Fatal("expected error for empty query")
	}
}

func TestKnowledgeBaseClassifierMatchedLabelsThresholding(t *testing.T) {
	classifier := &KnowledgeBaseClassifier{
		rule: config.KnowledgeBaseConfig{
			Threshold: 0.5,
			LabelThresholds: map[string]float32{
				"prompt_injection": 0.8,
			},
			Groups: map[string][]string{
				"private": {"proprietary_code", "prompt_injection"},
				"public":  {"generic_coding"},
			},
			Metrics: []config.KnowledgeBaseMetricConfig{
				{
					Name:          "private_vs_public",
					Type:          config.KBMetricTypeGroupMargin,
					PositiveGroup: "private",
					NegativeGroup: "public",
				},
			},
		},
	}

	labelScores := map[string]float64{
		"proprietary_code": 0.55,
		"prompt_injection": 0.75,
		"generic_coding":   0.51,
	}

	matchedLabels := classifier.buildMatchedLabels(labelScores)
	if containsString(matchedLabels, "prompt_injection") {
		t.Fatalf("prompt_injection should respect label_thresholds, matched=%v", matchedLabels)
	}
	if !containsString(matchedLabels, "proprietary_code") || !containsString(matchedLabels, "generic_coding") {
		t.Fatalf("expected proprietary_code and generic_coding to match, got %v", matchedLabels)
	}

	groupScores := classifier.computeGroupScores(labelScores)
	if groupScores["private"] != 0.75 {
		t.Fatalf("private group score = %.2f, want 0.75", groupScores["private"])
	}

	matchedGroups := classifier.collectMatchedGroups(matchedLabels)
	if !containsString(matchedGroups, "private") || !containsString(matchedGroups, "public") {
		t.Fatalf("matched groups = %v", matchedGroups)
	}

	metricValues := classifier.computeMetricValues(labelScores, groupScores, 0.75, 0.55)
	if metricValues["private_vs_public"] != 0.24 {
		t.Fatalf("private_vs_public = %.2f, want 0.24", metricValues["private_vs_public"])
	}
}

func TestBestScoredNameSelectsHighestScoreAndBreaksTiesLexically(t *testing.T) {
	name, score := bestScoredName(map[string]float64{
		"beta":  0.82,
		"alpha": 0.82,
		"gamma": 0.79,
	})
	if name != "alpha" {
		t.Fatalf("bestScoredName() name = %q, want alpha", name)
	}
	if score != 0.82 {
		t.Fatalf("bestScoredName() score = %.2f, want 0.82", score)
	}
}

func writeKnowledgeBaseFixture(t *testing.T) string {
	t.Helper()

	root := t.TempDir()
	manifest := `{
  "version": "1.0.0",
  "description": "fixture KB",
  "labels": {
    "proprietary_code": {
      "description": "Internal code",
      "exemplars": ["review internal repository code"]
    },
    "prompt_injection": {
      "description": "Prompt injection",
      "exemplars": ["ignore previous instructions"]
    },
    "generic_coding": {
      "description": "General coding",
      "exemplars": ["write a python function"]
    }
  }
}`
	path := filepath.Join(root, "labels.json")
	if err := os.WriteFile(path, []byte(manifest), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	return root
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
