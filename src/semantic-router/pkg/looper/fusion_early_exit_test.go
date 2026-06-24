package looper

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func earlyExitCfg(threshold float64) fusionExecutionConfig {
	return fusionExecutionConfig{
		GroundingEarlyExitEnabled:        true,
		GroundingEarlyExitMinConsistency: threshold,
	}
}

func panelScores(vals ...float64) []groundingScore {
	scores := make([]groundingScore, len(vals))
	for i, v := range vals {
		scores[i] = groundingScore{Model: "m", Score: v}
	}
	return scores
}

func TestShouldFusionEarlyExit(t *testing.T) {
	panel := config.FusionGroundingReferencePanel
	ctx := config.FusionGroundingReferenceContext

	tests := []struct {
		name    string
		cfg     fusionExecutionConfig
		mode    string
		scores  []groundingScore
		wantHit bool
	}{
		{"disabled", fusionExecutionConfig{}, panel, panelScores(0.99, 0.99), false},
		{"unanimous above threshold", earlyExitCfg(0.9), panel, panelScores(0.95, 0.92, 0.91), true},
		{"dissenter below threshold keeps full pipeline", earlyExitCfg(0.9), panel, panelScores(0.95, 0.40, 0.92), false},
		{"exactly at threshold counts as agreement", earlyExitCfg(0.9), panel, panelScores(0.90, 0.90), true},
		{"context mode never early-exits", earlyExitCfg(0.0), ctx, panelScores(1.0, 1.0), false},
		{"single response cannot agree", earlyExitCfg(0.0), panel, panelScores(1.0), false},
		{"no scores", earlyExitCfg(0.0), panel, nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := shouldFusionEarlyExit(tt.cfg, tt.mode, tt.scores); got != tt.wantHit {
				t.Fatalf("shouldFusionEarlyExit = %v, want %v", got, tt.wantHit)
			}
		})
	}
}

func TestShouldEscalateToSingleModel(t *testing.T) {
	hard := fusionExecutionConfig{EscalationEnabled: true, EscalationHardRules: []string{"reasoning_complexity:hard"}}

	tests := []struct {
		name string
		cfg  fusionExecutionConfig
		req  *Request
		want bool
	}{
		{"disabled never escalates", fusionExecutionConfig{}, &Request{MatchedComplexity: nil}, false},
		{"easy query (no hard match) -> single model", hard, &Request{MatchedComplexity: []string{"reasoning_complexity:easy"}}, true},
		{"no complexity matched at all -> single model", hard, &Request{MatchedComplexity: nil}, true},
		{"hard rule matched -> full panel", hard, &Request{MatchedComplexity: []string{"reasoning_complexity:hard"}}, false},
		{"cached panel (eval) never escalates", hard, &Request{MatchedComplexity: nil, CachedPanel: []*ModelResponse{{Model: "m"}}}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := shouldEscalateToSingleModel(tt.cfg, tt.req); got != tt.want {
				t.Fatalf("shouldEscalateToSingleModel = %v, want %v", got, tt.want)
			}
		})
	}
}
