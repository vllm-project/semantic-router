package extproc

import (
	"encoding/json"
	"strconv"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
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

// Retry-After is a whole number of seconds, so a fractional wait rounds up: a
// client that waits the advertised time must not be rejected by the same window.
func TestRateLimitResponseRoundsRetryAfterUp(t *testing.T) {
	cases := []struct {
		retryAfter time.Duration
		want       string
	}{
		{500 * time.Millisecond, "1"},
		{time.Second, "1"},
		{1500 * time.Millisecond, "2"},
		{3 * time.Second, "3"},
	}
	for _, tc := range cases {
		t.Run(tc.retryAfter.String(), func(t *testing.T) {
			response := (&OpenAIRouter{}).createRateLimitResponse(&ratelimit.Decision{RetryAfter: tc.retryAfter})
			immediate := response.GetImmediateResponse()
			var header string
			for _, option := range immediate.GetHeaders().GetSetHeaders() {
				if option.GetHeader().GetKey() == "retry-after" {
					header = string(option.GetHeader().GetRawValue())
				}
			}
			if header != tc.want {
				t.Errorf("retry-after = %q, want %q", header, tc.want)
			}
			if want := "Retry after " + tc.want + " seconds."; !strings.Contains(string(immediate.GetBody()), want) {
				t.Errorf("body = %s, want it to contain %q", immediate.GetBody(), want)
			}
		})
	}
}

// The router receives identities after gateway authentication. The gateway's
// separate spoofing test owns stripping untrusted client identity headers.
func TestApplyRateLimitIsolatesPostAuthUsersAndTiers(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.RateLimit.Providers = []config.RateLimitProviderConfig{{
		Type: "local-limiter",
		Rules: []config.RateLimitRule{
			{Name: "free", Match: config.RateLimitMatch{Group: "free"}, RequestsPerUnit: 1, Unit: "minute"},
			{Name: "premium", Match: config.RateLimitMatch{Group: "premium"}, RequestsPerUnit: 3, Unit: "minute"},
		},
	}}
	router := &OpenAIRouter{Config: cfg, RateLimiter: buildRateLimitResolver(cfg)}
	for _, request := range []struct {
		user, group string
		allowed     bool
	}{
		{"alice", "free", true},
		{"alice", "free", false},
		{"bob", "free", true},
		{"bob", "free", false},
		{"carol", "premium", true},
		{"carol", "premium", true},
		{"carol", "premium", true},
		{"carol", "premium", false},
	} {
		ctx := &RequestContext{Headers: map[string]string{
			cfg.Authz.Identity.GetUserIDHeader():     request.user,
			cfg.Authz.Identity.GetUserGroupsHeader(): request.group,
		}}
		response := router.applyRateLimit(ctx, "model")
		if request.allowed {
			if response != nil || ctx.RateLimitCtx == nil || ctx.RateLimitCtx.UserID != request.user {
				t.Fatalf("allowed %s/%s: response=%v, settlement=%v", request.user, request.group, response, ctx.RateLimitCtx)
			}
			continue
		}
		if response == nil || response.GetImmediateResponse().GetStatus().GetCode() != 429 {
			t.Fatalf("exhausted %s/%s did not return 429: %v", request.user, request.group, response)
		}
		immediate := response.GetImmediateResponse()
		headers := map[string]string{}
		for _, option := range immediate.GetHeaders().GetSetHeaders() {
			headers[option.GetHeader().GetKey()] = string(option.GetHeader().GetRawValue())
		}
		limit := "1"
		if request.group == "premium" {
			limit = "3"
		}
		if headers["x-ratelimit-limit"] != limit || headers["x-ratelimit-remaining"] != "0" {
			t.Fatalf("quota headers = %v", headers)
		}
		for _, name := range []string{"retry-after", "x-ratelimit-reset"} {
			value, err := strconv.ParseInt(headers[name], 10, 64)
			if err != nil || value <= 0 {
				t.Fatalf("%s = %q, want positive integer", name, headers[name])
			}
		}
		var payload map[string]map[string]interface{}
		if err := json.Unmarshal(immediate.GetBody(), &payload); err != nil {
			t.Fatalf("rate-limit error is not JSON: %v", err)
		}
		if payload["error"]["code"] != float64(429) || payload["error"]["type"] != "rate_limit_error" || ctx.RateLimitCtx != nil {
			t.Fatalf("invalid denial contract: body=%s, settlement=%v", immediate.GetBody(), ctx.RateLimitCtx)
		}
	}
}

// Header names are case-insensitive: identity.user_id_header may name
// X-User-ID while the gateway hands the Router lowercase keys, as Envoy and
// the standalone listener both do.
func TestApplyRateLimitReadsIdentityHeadersInAnyCase(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Authz.Identity = config.IdentityConfig{UserIDHeader: "X-User-ID", UserGroupsHeader: "X-User-Groups"}
	cfg.RateLimit.Providers = []config.RateLimitProviderConfig{{
		Type: "local-limiter",
		Rules: []config.RateLimitRule{
			{Name: "alice", Match: config.RateLimitMatch{User: "alice"}, RequestsPerUnit: 1, Unit: "minute"},
			{Name: "free", Match: config.RateLimitMatch{Group: "free"}, RequestsPerUnit: 1, Unit: "minute"},
		},
	}}
	router := &OpenAIRouter{Config: cfg, RateLimiter: buildRateLimitResolver(cfg)}
	send := func(user, group string) *ext_proc.ProcessingResponse {
		ctx := &RequestContext{Headers: map[string]string{}}
		captureRequestHeaders(&ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
				{Key: "x-user-id", RawValue: []byte(user)},
				{Key: "x-user-groups", RawValue: []byte(group)},
			}},
		}}, ctx, false)
		return router.applyRateLimit(ctx, "model")
	}
	if response := send("alice", "staff"); response != nil {
		t.Fatalf("alice's first request was rejected: %v", response)
	}
	if response := send("alice", "staff"); response.GetImmediateResponse().GetStatus().GetCode() != 429 {
		t.Fatalf("alice's second request = %v, want 429 from the user rule", response)
	}
	if response := send("bob", "free"); response != nil {
		t.Fatalf("bob's first request was rejected: %v", response)
	}
	if response := send("bob", "free"); response.GetImmediateResponse().GetStatus().GetCode() != 429 {
		t.Fatalf("bob's second request = %v, want 429 from the group rule", response)
	}
}
