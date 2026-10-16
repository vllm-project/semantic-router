package config

import (
	"strings"
	"testing"
)

func identityViolations(cfg *RouterConfig, mode GatewayMode) []CapabilityViolation {
	var found []CapabilityViolation
	for _, violation := range CheckGatewayCapabilities(cfg, mode) {
		if violation.Capability == CapabilityGatewayIdentity {
			found = append(found, violation)
		}
	}
	return found
}

func TestStandaloneRefusesPolicyOnGatewayAssertedIdentity(t *testing.T) {
	authzDecision := Decision{
		Name: "admins",
		Rules: RuleNode{Operator: "OR", Conditions: []RuleNode{
			{Type: SignalTypeKeyword, Name: "urgent"},
			{Operator: "AND", Conditions: []RuleNode{{Type: SignalTypeAuthz, Name: "admin"}}},
		}},
	}
	tests := []struct {
		name string
		cfg  *RouterConfig
		path string
		want string
	}{
		{
			name: "authz signal in a nested rule",
			cfg:  &RouterConfig{IntelligentRouting: IntelligentRouting{Decisions: []Decision{authzDecision}}},
			path: "routing.decisions[admins].rules",
			want: "decision 'admins': the authz signal 'admin'",
		},
		{
			name: "rate limit by group",
			cfg: &RouterConfig{RateLimit: RateLimitConfig{Providers: []RateLimitProviderConfig{{
				Type: "local-limiter", Rules: []RateLimitRule{{Name: "tier", Match: RateLimitMatch{Group: "premium"}}},
			}}}},
			path: "global.services.ratelimit.providers[0].rules[tier].match",
			want: "rate limit rule 'tier'",
		},
		{
			name: "authz providers resolving per-user keys",
			cfg:  &RouterConfig{Authz: AuthzConfig{Providers: []AuthzProviderConfig{{Type: "header"}}}},
			path: "global.services.authz.providers",
			want: "resolve per-user API keys",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			found := identityViolations(tt.cfg, GatewayStandalone)
			if len(found) != 1 || found[0].Path != tt.path || !strings.Contains(found[0].Message, tt.want) ||
				!strings.Contains(found[0].Message, "--gateway extproc") ||
				!strings.Contains(found[0].Message, "listeners[].identity.trust_headers") {
				t.Fatalf("violations = %+v, want one at %s naming %q and both remedies", found, tt.path, tt.want)
			}
			if found := identityViolations(tt.cfg, GatewayExtProc); len(found) != 0 {
				t.Fatalf("ext_proc mode trusts the gateway in front: %+v", found)
			}
			tt.cfg.Listeners = []Listener{
				{Name: "public", Port: 8899},
				{Name: "behind-auth", Port: 8900, Identity: &ListenerIdentity{TrustHeaders: true}},
			}
			if found := identityViolations(tt.cfg, GatewayStandalone); len(found) != 0 {
				t.Fatalf("a listener that trusts identity headers serves the policy: %+v", found)
			}
		})
	}
}

func TestUntrustedIdentityWarningNamesTheFeaturesThatRecordAUser(t *testing.T) {
	cfg := &RouterConfig{
		APIServer:    APIServer{Listeners: []Listener{{Name: "public", Port: 8899}}},
		Memory:       MemoryConfig{Enabled: true},
		RouterReplay: RouterReplayConfig{Enabled: true},
	}
	warning := UntrustedIdentityWarning(cfg)
	if !strings.Contains(warning, "memory and router replay record") ||
		!strings.Contains(warning, "listeners[].identity.trust_headers") {
		t.Fatalf("warning = %q", warning)
	}
	cfg.Decisions = []Decision{{Name: "personal", Algorithm: &AlgorithmConfig{Type: "gmtrouter"}}}
	if warning := UntrustedIdentityWarning(cfg); !strings.Contains(warning,
		"memory, router replay and per-user model selection (gmtrouter) record") {
		t.Fatalf("warning = %q", warning)
	}
	cfg.Listeners[0].Identity = &ListenerIdentity{TrustHeaders: true}
	if warning := UntrustedIdentityWarning(cfg); warning != "" {
		t.Fatalf("a trusting listener needs no warning: %q", warning)
	}
	if warning := UntrustedIdentityWarning(&RouterConfig{}); warning != "" {
		t.Fatalf("nothing records a user: %q", warning)
	}
}

func TestListenerIdentityPeersAreCIDRsThatNeedTrust(t *testing.T) {
	for _, tt := range []struct {
		identity ListenerIdentity
		want     string
	}{
		{ListenerIdentity{TrustHeaders: true, TrustedPeers: []string{"10.0.0.0/8", "fd00::/8", "10.1.2.3/32"}}, ""},
		{ListenerIdentity{TrustHeaders: true, TrustedPeers: []string{"10.1.2.3"}}, "is not a CIDR"},
		{ListenerIdentity{TrustedPeers: []string{"10.0.0.0/8"}}, "applies only with identity.trust_headers"},
	} {
		identity := tt.identity
		cfg := &RouterConfig{APIServer: APIServer{Listeners: []Listener{{Name: "l", Port: 8899, Identity: &identity}}}}
		err := validateListenerContracts(cfg)
		if (tt.want == "") != (err == nil) || (err != nil && !strings.Contains(err.Error(), tt.want)) {
			t.Errorf("%+v: error = %v, want %q", tt.identity, err, tt.want)
		}
	}
}

func TestGatewayIdentityNamesTheConfiguredHeaders(t *testing.T) {
	cfg := &RouterConfig{
		Authz: AuthzConfig{Identity: IdentityConfig{UserIDHeader: "x-user-id", UserGroupsHeader: "x-user-groups"}},
		RateLimit: RateLimitConfig{Providers: []RateLimitProviderConfig{{
			Rules: []RateLimitRule{{Name: "per-user", Match: RateLimitMatch{User: "*"}}},
		}}},
	}
	found := identityViolations(cfg, GatewayStandalone)
	if len(found) != 1 || !strings.Contains(found[0].Message, "x-user-id and x-user-groups") {
		t.Fatalf("violations = %+v", found)
	}
}

func TestStandaloneServesPolicyThatReadsNoIdentity(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{Decisions: []Decision{{
			Name: "code", Rules: RuleNode{Type: SignalTypeKeyword, Name: "python"},
		}}},
		RateLimit: RateLimitConfig{Providers: []RateLimitProviderConfig{{
			Rules: []RateLimitRule{{Name: "per-model", Match: RateLimitMatch{Model: "m"}}},
		}}},
	}
	if found := identityViolations(cfg, GatewayStandalone); len(found) != 0 {
		t.Fatalf("policy that reads no identity works in standalone mode: %+v", found)
	}
}
