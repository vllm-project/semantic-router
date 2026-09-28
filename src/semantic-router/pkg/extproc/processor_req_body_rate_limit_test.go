package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestApplyRateLimitEnforcesRequestRulesAndRetainsSettlementContext(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.RateLimit.Providers = []config.RateLimitProviderConfig{{
		Type: "local-limiter",
		Rules: []config.RateLimitRule{{
			Name: "one-per-minute",
			Match: config.RateLimitMatch{
				User:  "user-1",
				Model: "model-1",
			},
			RequestsPerUnit: 1,
			Unit:            "minute",
		}},
	}}
	router := &OpenAIRouter{Config: cfg, RateLimiter: buildRateLimitResolver(cfg)}
	first := &RequestContext{Headers: map[string]string{
		cfg.Authz.Identity.GetUserIDHeader():     "user-1",
		cfg.Authz.Identity.GetUserGroupsHeader(): "team-a, team-b",
	}}
	if response := router.applyRateLimit(first, "model-1"); response != nil {
		t.Fatalf("first request was rejected: %+v", response)
	}
	if first.RateLimitCtx == nil || first.RateLimitCtx.UserID != "user-1" ||
		len(first.RateLimitCtx.Groups) != 2 || first.RateLimitCtx.Model != "model-1" {
		t.Fatalf("settlement context = %+v", first.RateLimitCtx)
	}

	second := &RequestContext{Headers: first.Headers}
	response := router.applyRateLimit(second, "model-1")
	if response == nil || response.GetImmediateResponse().GetStatus().GetCode() != 429 {
		t.Fatalf("second request response = %+v, want 429", response)
	}
	if second.RateLimitCtx != nil {
		t.Fatalf("denied request retained settlement context: %+v", second.RateLimitCtx)
	}
}
