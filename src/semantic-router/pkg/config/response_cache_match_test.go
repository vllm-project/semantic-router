package config

import "testing"

// A decision's semantic lookup runs its written block, or the default; a
// request that matched no decision recalls by the store's threshold.
func TestResponseCacheMatch(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.SemanticCache.SimilarityThreshold = f32(0.75)
	if block := cfg.ResponseCacheMatch(nil); block == nil || block.Recall.TopK != 3 || *block.Recall.MinSimilarity != float64(float32(0.75)) || block.Judge != nil {
		t.Fatalf("decisionless block %+v", block)
	}
	plain := responseCacheDecision(map[string]interface{}{"enabled": true})
	if block := cfg.ResponseCacheMatch(&plain); block != nil {
		t.Fatalf("a decision without semantic.match runs the default, got %+v", block)
	}
	judged := responseCacheDecision(map[string]interface{}{"enabled": true, "semantic": map[string]interface{}{"match": map[string]interface{}{
		"recall": map[string]interface{}{"top_k": 5, "min_similarity": 0.8},
		"judge":  map[string]interface{}{"preset": "same_request", "min_score": 0.9},
	}}})
	if block := cfg.ResponseCacheMatch(&judged); block == nil || block.Recall.TopK != 5 || block.Judge == nil || block.Judge.Preset != "same_request" {
		t.Fatalf("written block %+v", block)
	}
}

// Only a written judge on an enabled semantic tier has a deployment to
// prepare: the default recalls only, and exact reuse never matches.
func TestResponseCacheMatchJudges(t *testing.T) {
	judge := map[string]interface{}{"match": map[string]interface{}{
		"recall": map[string]interface{}{"top_k": 3},
		"judge":  map[string]interface{}{"preset": "same_request"},
	}}
	cases := []struct {
		name          string
		configuration map[string]interface{}
		want          int
	}{
		{name: "default recall", configuration: map[string]interface{}{"enabled": true}},
		{name: "same_request judge", configuration: map[string]interface{}{"enabled": true, "semantic": judge}, want: 1},
		{name: "exact mode", configuration: map[string]interface{}{"enabled": true, "mode": ResponseCacheModeExact, "semantic": judge}},
		{name: "disabled", configuration: map[string]interface{}{"enabled": false, "semantic": judge}},
	}
	for _, tc := range cases {
		decision := responseCacheDecision(tc.configuration)
		if judges := decision.MatchJudges(); len(judges) != tc.want {
			t.Fatalf("%s: judges %+v, want %d", tc.name, judges, tc.want)
		}
	}
}
