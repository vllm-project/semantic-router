package config

import (
	"strings"
	"testing"
)

func responseCacheDecision(configuration map[string]interface{}) Decision {
	return Decision{Name: "route", Plugins: []DecisionPlugin{{Type: DecisionPluginResponseCache, Configuration: MustStructuredPayload(configuration)}}}
}

// A decision's semantic.match is checked as the response cache's block, at
// its document path: recall from the cache index, bounded as a cosine.
func TestValidateResponseCacheSemanticMatch(t *testing.T) {
	cases := []struct {
		name    string
		match   map[string]interface{}
		wantErr string
	}{
		{name: "recall", match: map[string]interface{}{"recall": map[string]interface{}{"top_k": 3, "min_similarity": 0.85}}},
		{name: "empty", match: map[string]interface{}{}, wantErr: "recipes[default].decisions[route].plugins[0].configuration.semantic.match: match_empty"},
		{name: "similarity beyond a cosine", match: map[string]interface{}{"recall": map[string]interface{}{"top_k": 3, "min_similarity": 2.0}}, wantErr: "semantic.match.recall.min_similarity must be a cosine similarity"},
		{name: "judge without recall", match: map[string]interface{}{"judge": map[string]interface{}{"preset": "same_request"}}, wantErr: "add recall"},
		{name: "bm25 recall", match: map[string]interface{}{"recall": map[string]interface{}{"method": "bm25", "top_k": 3}}, wantErr: "recalls with embedding only"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{}
			cfg.Decisions = []Decision{responseCacheDecision(map[string]interface{}{"enabled": true, "semantic": map[string]interface{}{"match": tc.match}})}
			err := validateDecisionPluginContracts(cfg)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %v, want containing %q", err, tc.wantErr)
			}
		})
	}
}

func TestValidateSemanticCacheMode(t *testing.T) {
	for _, mode := range []string{"", ResponseCacheModeSemantic, ResponseCacheModeExact, ResponseCacheModeExactThenSemantic} {
		decision := responseCacheDecision(map[string]interface{}{"enabled": true, "mode": mode})
		if err := validateDecisionPluginPayload(decision.Name, 0, decision.Plugins[0]); err != nil {
			t.Fatalf("valid mode %q rejected: %v", mode, err)
		}
	}
	decision := responseCacheDecision(map[string]interface{}{"enabled": true, "mode": "unsafe"})
	if err := validateDecisionPluginPayload(decision.Name, 0, decision.Plugins[0]); err == nil {
		t.Fatal("unknown response_cache mode must be rejected")
	}
}
