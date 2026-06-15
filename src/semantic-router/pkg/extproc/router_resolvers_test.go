package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildCredentialResolverDefaultChain(t *testing.T) {
	resolver := buildCredentialResolver(&config.RouterConfig{})

	if resolver == nil {
		t.Fatal("expected credential resolver")
	}
	if !resolver.FailOpen() {
		t.Fatal("expected default credential resolver to be fail-open")
	}
	names := resolver.ProviderNames()
	if len(names) != 2 || names[0] != "header-injection" || names[1] != "static-config" {
		t.Fatalf("unexpected provider chain: %#v", names)
	}
}

func TestBuildCredentialResolverFallsBackOnInvalidConfig(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Authz.Providers = []config.AuthzProviderConfig{{Type: "unknown-provider"}}

	resolver := buildCredentialResolver(cfg)
	if resolver == nil {
		t.Fatal("expected credential resolver")
	}
	if resolver.FailOpen() {
		t.Fatal("expected invalid config fallback to be fail-closed")
	}
	names := resolver.ProviderNames()
	if len(names) != 2 || names[0] != "header-injection" || names[1] != "static-config" {
		t.Fatalf("unexpected fallback provider chain: %#v", names)
	}
}

func TestBuildRateLimitResolverCreatesLocalLimiter(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.RateLimit.FailOpen = true
	cfg.RateLimit.Providers = []config.RateLimitProviderConfig{
		{
			Type: "local-limiter",
			Rules: []config.RateLimitRule{
				{
					Name: "default",
					Match: config.RateLimitMatch{
						User: "user-1",
					},
					RequestsPerUnit: 5,
					Unit:            "minute",
				},
			},
		},
	}

	resolver := buildRateLimitResolver(cfg)
	if resolver == nil {
		t.Fatal("expected rate limit resolver")
	}
	if !resolver.FailOpen() {
		t.Fatal("expected rate limit resolver to inherit fail_open=true")
	}
	names := resolver.ProviderNames()
	if len(names) != 1 || names[0] != "local-limiter" {
		t.Fatalf("unexpected rate limit providers: %#v", names)
	}
}

func TestBuildRateLimitResolverReturnsNilOnInvalidConfig(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.RateLimit.Providers = []config.RateLimitProviderConfig{{Type: "envoy-ratelimit"}}

	resolver := buildRateLimitResolver(cfg)
	if resolver != nil {
		t.Fatal("expected invalid rate limit config to disable resolver")
	}
}
