package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
)

// A streamed answer whose backend reported no usage still spent its output
// tokens: the budget is debited with the estimate, not left untouched.
func TestStreamWithoutUsageDebitsItsAnswerFromTheBudget(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.RateLimit.Providers = []config.RateLimitProviderConfig{{
		Type:  "local-limiter",
		Rules: []config.RateLimitRule{{Name: "token-budget", Match: config.RateLimitMatch{User: "user-1"}, TokensPerUnit: 100, Unit: "minute"}},
	}}
	router := &OpenAIRouter{Config: cfg, RateLimiter: buildRateLimitResolver(cfg)}
	ctx := &RequestContext{
		Headers:      map[string]string{cfg.Authz.Identity.GetUserIDHeader(): "user-1"},
		RateLimitCtx: &ratelimit.Context{UserID: "user-1"},
	}
	before, err := router.RateLimiter.Check(*ctx.RateLimitCtx)
	if err != nil || before == nil || !before.Allowed {
		t.Fatalf("initial check = %+v, %v", before, err)
	}

	answer := llmprotocol.Response{Output: []llmprotocol.OutputItem{{
		Role:    llmprotocol.RoleAssistant,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: strings.Repeat("answer ", 500)}},
	}}}
	if tokens := router.reportStreamingAnswerEstimate(ctx, &answer); tokens <= 0 {
		t.Fatal("the answer estimate reported no tokens")
	}

	after, err := router.RateLimiter.Check(*ctx.RateLimitCtx)
	if err != nil || after == nil {
		t.Fatalf("check after the stream = %+v, %v", after, err)
	}
	if after.Remaining >= before.Remaining {
		t.Fatalf("the budget is untouched: remaining %d before, %d after", before.Remaining, after.Remaining)
	}
	if after.Allowed {
		t.Fatalf("a 500-word answer left %d of a 100-token budget", after.Remaining)
	}
}
