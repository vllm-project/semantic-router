package config

import (
	"testing"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

func TestModelParamsEvidenceResultUsesExactEffort(t *testing.T) {
	low, high := 70.0, 80.0
	params := ModelParams{
		QualityIndex: "acme/index@1.0.0",
		IndexResults: map[string]modelcatalog.IndexResult{
			"acme/index@1.0.0": {Status: "available", Score: &low, Coverage: 0.6},
		},
		IndexResultsByEffort: map[string]map[string]modelcatalog.IndexResult{
			"high": {"acme/index@1.0.0": {Status: "available", Score: &high, Coverage: 1}},
		},
	}
	result, ok := params.EvidenceResultAt("", "high")
	if !ok || result.Score == nil || *result.Score != 80 || result.Coverage != 1 {
		t.Fatalf("high evidence = %+v, %v", result, ok)
	}
	if _, found := params.EvidenceResultAt("", "medium"); found {
		t.Fatal("missing exact effort fell back to default")
	}
	result, ok = params.EvidenceResultAt("", "")
	if !ok || result.Score == nil || *result.Score != 70 {
		t.Fatalf("default evidence = %+v, %v", result, ok)
	}
}

func TestCloneCatalogIndexResultsByEffortIsDeep(t *testing.T) {
	score := 80.0
	source := map[string]map[string]modelcatalog.IndexResult{
		"high": {"acme/index@1.0.0": {Score: &score, Domains: map[string]float64{"reasoning": 0.8}}},
	}
	clone := cloneCatalogIndexResultsByEffort(source)
	result := clone["high"]["acme/index@1.0.0"]
	*result.Score = 0
	result.Domains["reasoning"] = 0
	if *source["high"]["acme/index@1.0.0"].Score != 80 || source["high"]["acme/index@1.0.0"].Domains["reasoning"] != 0.8 {
		t.Fatal("effort result clone mutated source")
	}
}

func TestGetModelIndexResultReturnsDeepCopy(t *testing.T) {
	const (
		wantScore      = 84.0
		wantValue      = 0.84
		wantNormalized = 0.84
	)
	score := wantScore
	value := wantValue
	normalized := wantNormalized
	cfg := &RouterConfig{
		BackendModels: BackendModels{
			DefaultQualityIndex: "acme/index@1.0.0",
			ModelConfig: map[string]ModelParams{
				"production": {
					IndexResults: map[string]modelcatalog.IndexResult{
						"acme/index@1.0.0": {
							Score: &score,
							Components: []modelcatalog.IndexComponentResult{{
								Value: &value, Normalized: &normalized, Status: "available",
							}},
							Domains:    map[string]float64{"reasoning": 0.84},
							Provenance: []string{"acme/run-1"},
						},
					},
				},
			},
		},
	}

	first, ok := cfg.GetModelIndexResult("production", "")
	if !ok {
		t.Fatal("index result is missing")
	}
	*first.Score = 0
	*first.Components[0].Value = 0
	*first.Components[0].Normalized = 0
	first.Components[0].Status = "mutated"
	first.Domains["reasoning"] = 0
	first.Provenance[0] = "mutated"

	second, ok := cfg.GetModelIndexResult("production", "")
	if !ok || second.Score == nil || *second.Score != wantScore ||
		second.Components[0].Value == nil || *second.Components[0].Value != wantValue ||
		second.Components[0].Normalized == nil || *second.Components[0].Normalized != wantNormalized ||
		second.Components[0].Status != "available" || second.Domains["reasoning"] != 0.84 ||
		second.Provenance[0] != "acme/run-1" {
		t.Fatalf("stored index result was mutated through lookup: %+v", second)
	}
}
