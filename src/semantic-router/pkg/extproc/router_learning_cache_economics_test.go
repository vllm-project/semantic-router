package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// cacheEconomicsTestConfig prices premium like a frontier model, cheap at a
// third of it and near at 90% of it, and turns off the stability margins so
// the decision change alone would move the conversation.
func cacheEconomicsTestConfig() *config.RouterConfig {
	cfg := routerLearningProtectionOnlyTestConfig(config.RouterLearningScopeConversation)
	price := func(prompt float64) config.ModelParams {
		write := prompt * 1.25
		return config.ModelParams{Pricing: config.ModelPricing{
			Currency: "USD", PromptPer1M: prompt, CachedInputPer1M: prompt / 10,
			CacheWritePer1M: &write, CompletionPer1M: prompt * 5,
		}}
	}
	cfg.ModelConfig = map[string]config.ModelParams{"premium": price(3), "near": price(2.7), "cheap": price(1)}
	cfg.RouterLearning.Protection.Tuning = config.RouterLearningProtectionTuning{
		MinTurnsBeforeSwitch: extprocIntPtr(0),
		SwitchMargin:         extprocFloat64Ptr(0),
		StabilityWeight:      extprocFloat64Ptr(0),
	}
	return cfg
}

func TestRouterLearningProtectionPricesASwitchFromCacheReceipts(t *testing.T) {
	for _, tc := range []struct {
		proposed, selected, action, reason, decision string
	}{
		{"near", "premium", string(routerLearningActionHoldCurrent), selection.SessionReasonCacheSwitchDoesNotPay, selection.CacheEconomicsHold},
		{"cheap", "cheap", string(routerLearningActionAllowSwitch), selection.SessionReasonCacheSwitchPays, selection.CacheEconomicsSwitch},
	} {
		t.Run(tc.proposed, func(t *testing.T) {
			sessiontelemetry.ResetRouterSessionMemoryForTesting()
			t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
			now := time.Now()
			sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
				SessionID: "session-a/conversation-a", SelectedModel: "premium",
				DecisionName: "seed", TurnIndex: 2, Timestamp: now.Add(-time.Minute),
			})
			sessiontelemetry.RecordSessionUsage(sessiontelemetry.SessionUsageParams{
				SessionID: "session-a/conversation-a", Model: "premium",
				PromptTokens: 20000, CachedPromptTokens: 18000, CacheWriteTokens: 1000, CompletionTokens: 500,
				CacheReported: true, Timestamp: now.Add(-time.Minute),
			})

			router := &OpenAIRouter{Config: cacheEconomicsTestConfig()}
			ctx := routerLearningRequestContext("session-a", "conversation-a")
			ctx.VSRSelectedDecision = &config.Decision{Name: "next"}
			selected, _, err := router.selectModelFromCandidates(&selection.SelectionContext{
				SessionID:       "session-a",
				DecisionName:    "next",
				CandidateModels: []config.ModelRef{{Model: tc.proposed}, {Model: "premium"}},
			}, nil, ctx)
			if err != nil || selected == nil || selected.Model != tc.selected {
				t.Fatalf("selected %+v, err %v; want %s", selected, err, tc.selected)
			}
			policy := ctx.VSRLearningPolicy
			if policy.String("action") != tc.action || policy.String("reason") != tc.reason {
				t.Fatalf("protection said %s/%s, want %s/%s", policy.String("action"), policy.String("reason"), tc.action, tc.reason)
			}
			economics, _ := policy.ToMap()["cache_economics"].(map[string]interface{})
			if economics["decision"] != tc.decision || economics["proposed_model"] != tc.proposed ||
				economics["cached_prefix_tokens"] != 19000 || economics["rewrite_cost"] == 0.0 {
				t.Fatalf("replay economics = %#v", economics)
			}
		})
	}
}
