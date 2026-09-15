package config

import (
	"testing"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

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
