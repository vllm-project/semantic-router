package config

import "testing"

// A decision's semantic lookup runs its written block, or the default; a
// request that matched no decision runs the block of global.plugins.
func TestResponseCacheMatch(t *testing.T) {
	global := responseCacheDecision(map[string]interface{}{"enabled": true, "semantic": map[string]interface{}{"match": map[string]interface{}{
		"recall": map[string]interface{}{"top_k": 3, "min_similarity": 0.75},
	}}})
	cfg := layeredRouting(global.Plugins, Decision{Name: "plain"})
	if block := cfg.ResponseCacheMatch(nil); block == nil || block.Recall.TopK != 3 || *block.Recall.MinSimilarity != 0.75 || block.Judge != nil {
		t.Fatalf("decisionless block %+v", block)
	}
	cfg = &RouterConfig{}
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

// An enabled semantic tier has a judge's deployment to prepare: the
// default's same_request, or the one its block writes; a recall-only block
// asks no model, and exact reuse never matches. A request that matches no
// decision prepares the judge of its default plugins.
func TestResponseCacheMatchJudges(t *testing.T) {
	judge := map[string]interface{}{"match": map[string]interface{}{
		"recall": map[string]interface{}{"top_k": 3},
		"judge":  map[string]interface{}{"preset": "same_request"},
	}}
	recallOnly := map[string]interface{}{"match": map[string]interface{}{"recall": map[string]interface{}{"top_k": 3}}}
	cases := []struct {
		name          string
		configuration map[string]interface{}
		want          int
	}{
		{name: "default", configuration: map[string]interface{}{"enabled": true}, want: 1},
		{name: "same_request judge", configuration: map[string]interface{}{"enabled": true, "semantic": judge}, want: 1},
		{name: "recall only", configuration: map[string]interface{}{"enabled": true, "semantic": recallOnly}},
		{name: "exact mode", configuration: map[string]interface{}{"enabled": true, "mode": ResponseCacheModeExact, "semantic": judge}},
		{name: "disabled", configuration: map[string]interface{}{"enabled": false, "semantic": judge}},
	}
	for _, tc := range cases {
		decision := responseCacheDecision(tc.configuration)
		judges := decision.MatchJudges()
		if len(judges) != tc.want {
			t.Fatalf("%s: judges %+v, want %d", tc.name, judges, tc.want)
		}
		if tc.want == 1 && judges[0].Preset != "same_request" {
			t.Fatalf("%s: judge %+v", tc.name, judges[0])
		}
	}
	defaults := responseCacheDecision(map[string]interface{}{"enabled": true})
	if judges := MatchJudgesOf(defaults.Plugins); len(judges) != 1 {
		t.Fatalf("default plugins prepare their judge: %+v", judges)
	}
}
