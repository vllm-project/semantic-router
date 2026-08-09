package handlers

import (
	"testing"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Edge cases for the canonical deploy merge: empty and global-less bases, repeated apply,
// and the nil-handling of mergeFragmentGlobal. These previously reached mergeDeployPayload
// through the retired security policy fragment generator and now use payloads directly.

const roleBindingAndLimiterPayload = `
routing:
  signals:
    role_bindings:
      - name: g
        subjects:
          - kind: Group
            name: g
        role: r
  decisions:
    - name: g-route
      priority: 5
      rules:
        type: authz
        name: r
      modelRefs:
        - model: gpt-4
global:
  services:
    ratelimit:
      providers:
        - type: local-limiter
          rules:
            - name: rate
              match:
                group: g
              requests_per_unit: 100
              unit: minute
`

func TestMergeDeployPayloadIntoEmptyBase(t *testing.T) {
	t.Parallel()

	merged, err := mergeDeployPayload(nil, DeployRequest{YAML: roleBindingAndLimiterPayload})
	if err != nil {
		t.Fatalf("mergeDeployPayload error: %v", err)
	}

	var doc routingFragmentDocument
	if err := yaml.Unmarshal(merged, &doc); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if len(doc.Routing.Signals.RoleBindings) != 1 {
		t.Fatalf("expected 1 role binding, got %d", len(doc.Routing.Signals.RoleBindings))
	}
	if doc.Global == nil || doc.Global.Services == nil || doc.Global.Services.RateLimit == nil {
		t.Fatal("expected ratelimit in merged output from empty base")
	}
}

func TestMergeDeployPayloadIntoBaseWithNoGlobal(t *testing.T) {
	t.Parallel()

	baseYAML := `
version: "0.3"
routing:
  modelCards:
    - name: gpt-4
  decisions:
    - name: old-decision
      priority: 1
      rules:
        type: keyword
        name: kw
      modelRefs:
        - model: gpt-4
`

	payload := `
global:
  services:
    ratelimit:
      providers:
        - type: local-limiter
          rules:
            - name: new-rate
              match:
                group: new
              requests_per_unit: 50
              unit: minute
`

	merged, err := mergeDeployPayload([]byte(baseYAML), DeployRequest{YAML: payload})
	if err != nil {
		t.Fatalf("mergeDeployPayload error: %v", err)
	}

	var result routerconfig.CanonicalConfig
	if err := yaml.Unmarshal(merged, &result); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if result.Global == nil {
		t.Fatal("expected global to be created")
	}
	if len(result.Global.Services.RateLimit.Providers) != 1 {
		t.Fatalf("expected 1 provider, got %d", len(result.Global.Services.RateLimit.Providers))
	}
	if result.Global.Services.RateLimit.Providers[0].Rules[0].Name != "new-rate" {
		t.Fatalf("expected rule new-rate, got %q", result.Global.Services.RateLimit.Providers[0].Rules[0].Name)
	}
}

func TestMergeDeployPayloadWithoutGlobalLeavesGlobalUnset(t *testing.T) {
	t.Parallel()

	payload := `
routing:
  signals:
    role_bindings:
      - name: g
        subjects:
          - kind: Group
            name: g
        role: r
`

	merged, err := mergeDeployPayload(nil, DeployRequest{YAML: payload})
	if err != nil {
		t.Fatalf("mergeDeployPayload error: %v", err)
	}

	var doc routingFragmentDocument
	if err := yaml.Unmarshal(merged, &doc); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if doc.Global != nil {
		t.Fatalf("expected nil global when the payload carries none, got %+v", doc.Global)
	}
}

func TestMergeDeployPayloadIdempotentDoubleApply(t *testing.T) {
	t.Parallel()

	baseYAML := `
version: "0.3"
routing:
  modelCards:
    - name: gpt-4
`

	first, err := mergeDeployPayload([]byte(baseYAML), DeployRequest{YAML: roleBindingAndLimiterPayload})
	if err != nil {
		t.Fatalf("first merge error: %v", err)
	}
	second, err := mergeDeployPayload(first, DeployRequest{YAML: roleBindingAndLimiterPayload})
	if err != nil {
		t.Fatalf("second merge error: %v", err)
	}

	var result routerconfig.CanonicalConfig
	if err := yaml.Unmarshal(second, &result); err != nil {
		t.Fatalf("unmarshal second: %v", err)
	}

	if len(result.Routing.Signals.RoleBindings) != 1 {
		t.Fatalf("expected 1 role binding after double apply, got %d", len(result.Routing.Signals.RoleBindings))
	}
	if len(result.Routing.Decisions) != 1 {
		t.Fatalf("expected 1 decision after double apply, got %d", len(result.Routing.Decisions))
	}
	if result.Global == nil {
		t.Fatal("expected global after double apply")
	}
	if len(result.Global.Services.RateLimit.Providers) != 1 {
		t.Fatalf("expected 1 provider after double apply, got %d", len(result.Global.Services.RateLimit.Providers))
	}
	if len(result.Global.Services.RateLimit.Providers[0].Rules) != 1 {
		t.Fatalf("expected 1 rule after double apply, got %d", len(result.Global.Services.RateLimit.Providers[0].Rules))
	}
}

func TestMergedDeployPayloadKeepsUserMatchAndRuleOrder(t *testing.T) {
	t.Parallel()

	baseYAML := `
version: "0.3"
routing:
  modelCards:
    - name: gpt-4
`

	payload := `
global:
  services:
    ratelimit:
      providers:
        - type: local-limiter
          rules:
            - name: tier-1
              match:
                group: g1
              requests_per_unit: 10
              unit: minute
            - name: tier-2
              match:
                group: g2
              requests_per_unit: 20
              unit: minute
            - name: tier-3
              match:
                group: g3
              requests_per_unit: 30
              unit: minute
            - name: tier-4
              match:
                user: u4
              requests_per_unit: 40
              unit: minute
            - name: tier-5
              match:
                group: g5
              tokens_per_unit: 50000
              unit: minute
`

	merged, err := mergeDeployPayload([]byte(baseYAML), DeployRequest{YAML: payload})
	if err != nil {
		t.Fatalf("mergeDeployPayload error: %v", err)
	}

	cfg, err := routerconfig.ParseYAMLBytes(merged)
	if err != nil {
		t.Fatalf("ParseYAMLBytes error: %v", err)
	}

	rules := cfg.RateLimit.Providers[0].Rules
	if len(rules) != 5 {
		t.Fatalf("expected 5 rate limit rules, got %d", len(rules))
	}
	for i, name := range []string{"tier-1", "tier-2", "tier-3", "tier-4", "tier-5"} {
		if rules[i].Name != name {
			t.Fatalf("expected rule[%d] name %q, got %q", i, name, rules[i].Name)
		}
	}
	if rules[3].Match.User != "u4" {
		t.Fatalf("expected rule[3] user match u4, got %q", rules[3].Match.User)
	}
	if rules[4].TokensPerUnit != 50000 {
		t.Fatalf("expected rule[4] TPU 50000, got %d", rules[4].TokensPerUnit)
	}
}

func TestMergeFragmentGlobalNilFragmentIsNoOp(t *testing.T) {
	t.Parallel()

	base := routerconfig.CanonicalConfig{
		Version: "0.3",
		Global: &routerconfig.CanonicalGlobal{
			Services: routerconfig.CanonicalServiceGlobal{
				RateLimit: routerconfig.RateLimitConfig{
					Providers: []routerconfig.RateLimitProviderConfig{
						{Type: "local-limiter", Rules: []routerconfig.RateLimitRule{
							{Name: "original", Match: routerconfig.RateLimitMatch{Group: "keep"}, RequestsPerUnit: 1, Unit: "minute"},
						}},
					},
				},
			},
		},
	}

	mergeFragmentGlobal(&base, nil)

	if base.Global.Services.RateLimit.Providers[0].Rules[0].Name != "original" {
		t.Fatal("expected nil fragment to be a no-op")
	}
}

func TestMergeFragmentGlobalNilServicesIsNoOp(t *testing.T) {
	t.Parallel()

	base := routerconfig.CanonicalConfig{
		Global: &routerconfig.CanonicalGlobal{
			Services: routerconfig.CanonicalServiceGlobal{
				RateLimit: routerconfig.RateLimitConfig{
					Providers: []routerconfig.RateLimitProviderConfig{
						{Type: "local-limiter", Rules: []routerconfig.RateLimitRule{
							{Name: "keep", Match: routerconfig.RateLimitMatch{Group: "g"}, RequestsPerUnit: 1, Unit: "minute"},
						}},
					},
				},
			},
		},
	}

	mergeFragmentGlobal(&base, &globalFragment{Services: nil})

	if base.Global.Services.RateLimit.Providers[0].Rules[0].Name != "keep" {
		t.Fatal("expected nil services to be a no-op")
	}
}

func TestMergeFragmentGlobalNilRateLimitIsNoOp(t *testing.T) {
	t.Parallel()

	base := routerconfig.CanonicalConfig{
		Global: &routerconfig.CanonicalGlobal{
			Services: routerconfig.CanonicalServiceGlobal{
				RateLimit: routerconfig.RateLimitConfig{
					Providers: []routerconfig.RateLimitProviderConfig{
						{Type: "local-limiter", Rules: []routerconfig.RateLimitRule{
							{Name: "keep", Match: routerconfig.RateLimitMatch{Group: "g"}, RequestsPerUnit: 1, Unit: "minute"},
						}},
					},
				},
			},
		},
	}

	mergeFragmentGlobal(&base, &globalFragment{Services: &globalServicesFragment{RateLimit: nil}})

	if base.Global.Services.RateLimit.Providers[0].Rules[0].Name != "keep" {
		t.Fatal("expected nil ratelimit to be a no-op")
	}
}

func TestMergeFragmentGlobalCreatesGlobalWhenBaseIsNil(t *testing.T) {
	t.Parallel()

	base := routerconfig.CanonicalConfig{Version: "0.3"}

	rl := &routerconfig.RateLimitConfig{
		Providers: []routerconfig.RateLimitProviderConfig{
			{Type: "local-limiter", Rules: []routerconfig.RateLimitRule{
				{Name: "new", Match: routerconfig.RateLimitMatch{Group: "g"}, RequestsPerUnit: 10, Unit: "minute"},
			}},
		},
	}

	mergeFragmentGlobal(&base, &globalFragment{Services: &globalServicesFragment{RateLimit: rl}})

	if base.Global == nil {
		t.Fatal("expected global to be created")
	}
	if base.Global.Services.RateLimit.Providers[0].Rules[0].Name != "new" {
		t.Fatalf("expected rule name new, got %q", base.Global.Services.RateLimit.Providers[0].Rules[0].Name)
	}
}
