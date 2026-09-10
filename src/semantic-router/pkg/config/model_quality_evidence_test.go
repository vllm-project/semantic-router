package config

import (
	"testing"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

func TestModelParamsEvidenceScoreAtRequiresExactEffort(t *testing.T) {
	const index = "vllm-sr/coding@1.0.0"
	lowScore := 42.0
	highScore := 91.0
	params := ModelParams{
		QualityIndex: index,
		IndexResults: map[string]modelcatalog.IndexResult{
			index: {Index: index, Status: "available", Score: &highScore},
		},
		IndexResultsByEffort: map[string]map[string]modelcatalog.IndexResult{
			"low": {
				index: {Index: index, Status: "available", Score: &lowScore},
			},
		},
	}

	if score, ok := params.EvidenceScoreAt(index, "low"); !ok || score != lowScore {
		t.Fatalf("low effort score = (%v, %t), want (%v, true)", score, ok, lowScore)
	}
	if _, ok := params.EvidenceScoreAt(index, "high"); ok {
		t.Fatal("missing high-effort evidence borrowed the preferred score")
	}
	if score, ok := params.EvidenceScoreAt(index, ""); !ok || score != highScore {
		t.Fatalf("empty effort score = (%v, %t), want preferred (%v, true)", score, ok, highScore)
	}
}

func TestModelParamsEvidenceScoreAtRejectsIncompleteResult(t *testing.T) {
	const index = "vllm-sr/intelligence@1.0.0"
	score := 75.0
	params := ModelParams{IndexResultsByEffort: map[string]map[string]modelcatalog.IndexResult{
		"default": {
			index: {Index: index, Status: "partial", Score: &score},
		},
	}}

	if _, ok := params.EvidenceScoreAt(index, "default"); ok {
		t.Fatal("partial index result was admitted as routing quality evidence")
	}
}
