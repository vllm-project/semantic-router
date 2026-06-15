package handlers

import (
	"testing"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestToCanonicalYAMLRateLimitOnlyNoRouting(t *testing.T) {
	t.Parallel()

	fragment := &GeneratedRouterFragment{
		RateLimit: RateLimitFragment{
			Providers: []RateLimitProviderFragment{
				{
					Type: "local-limiter",
					Rules: []RateLimitRuleFragment{
						{Name: "global-rate", Match: RateLimitMatchFragment{Group: "all"}, RPU: 60, Unit: "minute"},
					},
				},
			},
		},
	}

	yamlBytes, err := toCanonicalYAML(fragment)
	if err != nil {
		t.Fatalf("toCanonicalYAML error: %v", err)
	}

	var doc routingFragmentDocument
	if err := yaml.Unmarshal(yamlBytes, &doc); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if len(doc.Routing.Signals.RoleBindings) != 0 {
		t.Fatalf("expected 0 role bindings, got %d", len(doc.Routing.Signals.RoleBindings))
	}
	if len(doc.Routing.Decisions) != 0 {
		t.Fatalf("expected 0 decisions, got %d", len(doc.Routing.Decisions))
	}
	if doc.Global == nil || doc.Global.Services == nil || doc.Global.Services.RateLimit == nil {
		t.Fatal("expected global.services.ratelimit to be present")
	}
	if doc.Global.Services.RateLimit.Providers[0].Rules[0].Name != "global-rate" {
		t.Fatalf("expected rule name global-rate, got %q", doc.Global.Services.RateLimit.Providers[0].Rules[0].Name)
	}
}

func TestToCanonicalYAMLUserBasedRateLimitMatch(t *testing.T) {
	t.Parallel()

	fragment := &GeneratedRouterFragment{
		RateLimit: RateLimitFragment{
			Providers: []RateLimitProviderFragment{
				{
					Type: "local-limiter",
					Rules: []RateLimitRuleFragment{
						{Name: "user-limit", Match: RateLimitMatchFragment{User: "admin@co.com"}, RPU: 999, Unit: "minute"},
					},
				},
			},
		},
	}

	yamlBytes, err := toCanonicalYAML(fragment)
	if err != nil {
		t.Fatalf("toCanonicalYAML error: %v", err)
	}

	var doc routingFragmentDocument
	if err := yaml.Unmarshal(yamlBytes, &doc); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	rule := doc.Global.Services.RateLimit.Providers[0].Rules[0]
	if rule.Match.User != "admin@co.com" {
		t.Fatalf("expected user match admin@co.com, got %q", rule.Match.User)
	}
	if rule.Match.Group != "" {
		t.Fatalf("expected empty group match, got %q", rule.Match.Group)
	}
}

func TestToCanonicalYAMLMultipleSubjectsPerRoleBinding(t *testing.T) {
	t.Parallel()

	fragment := &GeneratedRouterFragment{
		RoleBindings: []RoleBindingFragment{
			{
				Subjects: []Subject{
					{Kind: "User", Name: "alice"},
					{Kind: "Group", Name: "eng"},
					{Kind: "User", Name: "bob"},
				},
				Role: "multi_tier",
			},
		},
		Decisions: []DecisionFragment{
			{Name: "rbac-multi", Priority: 1, Rules: RuleFragment{Type: "authz", Name: "multi_tier"}},
		},
	}

	yamlBytes, err := toCanonicalYAML(fragment)
	if err != nil {
		t.Fatalf("toCanonicalYAML error: %v", err)
	}

	var doc routingFragmentDocument
	if err := yaml.Unmarshal(yamlBytes, &doc); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	subjects := doc.Routing.Signals.RoleBindings[0].Subjects
	if len(subjects) != 3 {
		t.Fatalf("expected 3 subjects, got %d", len(subjects))
	}
	if subjects[0].Kind != "User" || subjects[0].Name != "alice" {
		t.Fatalf("expected first subject User/alice, got %s/%s", subjects[0].Kind, subjects[0].Name)
	}
	if subjects[1].Kind != "Group" || subjects[1].Name != "eng" {
		t.Fatalf("expected second subject Group/eng, got %s/%s", subjects[1].Kind, subjects[1].Name)
	}
}

func TestMergeDeployPayloadIntoEmptyBase(t *testing.T) {
	t.Parallel()

	fragment := &GeneratedRouterFragment{
		RoleBindings: []RoleBindingFragment{
			{Subjects: []Subject{{Kind: "Group", Name: "g"}}, Role: "r"},
		},
		Decisions: []DecisionFragment{
			{Name: "rbac-g", Priority: 1, Rules: RuleFragment{Type: "authz", Name: "r"}},
		},
		RateLimit: RateLimitFragment{
			Providers: []RateLimitProviderFragment{
				{Type: "local-limiter", Rules: []RateLimitRuleFragment{
					{Name: "rate", Match: RateLimitMatchFragment{Group: "g"}, RPU: 10, Unit: "minute"},
				}},
			},
		},
	}

	fragmentYAML, err := toCanonicalYAML(fragment)
	if err != nil {
		t.Fatalf("toCanonicalYAML error: %v", err)
	}

	merged, err := mergeDeployPayload(nil, DeployRequest{YAML: string(fragmentYAML)})
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

	fragment := &GeneratedRouterFragment{
		RateLimit: RateLimitFragment{
			Providers: []RateLimitProviderFragment{
				{Type: "local-limiter", Rules: []RateLimitRuleFragment{
					{Name: "new-rate", Match: RateLimitMatchFragment{Group: "new"}, RPU: 50, Unit: "minute"},
				}},
			},
		},
	}

	fragmentYAML, err := toCanonicalYAML(fragment)
	if err != nil {
		t.Fatalf("toCanonicalYAML error: %v", err)
	}

	merged, err := mergeDeployPayload([]byte(baseYAML), DeployRequest{YAML: string(fragmentYAML)})
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

func TestMergeDeployPayloadEmptyRateLimitProvidersDoesNotCreateGlobal(t *testing.T) {
	t.Parallel()

	fragment := &GeneratedRouterFragment{
		RoleBindings: []RoleBindingFragment{
			{Subjects: []Subject{{Kind: "Group", Name: "g"}}, Role: "r"},
		},
		Decisions: []DecisionFragment{
			{Name: "rbac-g", Priority: 1, Rules: RuleFragment{Type: "authz", Name: "r"}},
		},
		RateLimit: RateLimitFragment{Providers: nil},
	}

	yamlBytes, err := toCanonicalYAML(fragment)
	if err != nil {
		t.Fatalf("toCanonicalYAML error: %v", err)
	}

	var doc routingFragmentDocument
	if err := yaml.Unmarshal(yamlBytes, &doc); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if doc.Global != nil {
		t.Fatal("expected nil global when providers list is empty")
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

	fragment := &GeneratedRouterFragment{
		RoleBindings: []RoleBindingFragment{
			{Subjects: []Subject{{Kind: "Group", Name: "g"}}, Role: "r"},
		},
		Decisions: []DecisionFragment{
			{Name: "rbac-g", Priority: 5, Rules: RuleFragment{Type: "authz", Name: "r"}, ModelRefs: []ModelRefFragment{{Model: "gpt-4"}}},
		},
		RateLimit: RateLimitFragment{
			Providers: []RateLimitProviderFragment{
				{Type: "local-limiter", Rules: []RateLimitRuleFragment{
					{Name: "rate", Match: RateLimitMatchFragment{Group: "g"}, RPU: 100, Unit: "minute"},
				}},
			},
		},
	}

	fragmentYAML, err := toCanonicalYAML(fragment)
	if err != nil {
		t.Fatalf("toCanonicalYAML error: %v", err)
	}

	first, err := mergeDeployPayload([]byte(baseYAML), DeployRequest{YAML: string(fragmentYAML)})
	if err != nil {
		t.Fatalf("first merge error: %v", err)
	}

	second, err := mergeDeployPayload(first, DeployRequest{YAML: string(fragmentYAML)})
	if err != nil {
		t.Fatalf("second merge error: %v", err)
	}

	var result1, result2 routerconfig.CanonicalConfig
	if err := yaml.Unmarshal(first, &result1); err != nil {
		t.Fatalf("unmarshal first: %v", err)
	}
	if err := yaml.Unmarshal(second, &result2); err != nil {
		t.Fatalf("unmarshal second: %v", err)
	}

	if len(result2.Routing.Signals.RoleBindings) != 1 {
		t.Fatalf("expected 1 role binding after double apply, got %d", len(result2.Routing.Signals.RoleBindings))
	}
	if len(result2.Routing.Decisions) != 1 {
		t.Fatalf("expected 1 decision after double apply, got %d", len(result2.Routing.Decisions))
	}
	if result2.Global == nil {
		t.Fatal("expected global after double apply")
	}
	if len(result2.Global.Services.RateLimit.Providers) != 1 {
		t.Fatalf("expected 1 provider after double apply, got %d", len(result2.Global.Services.RateLimit.Providers))
	}
	if len(result2.Global.Services.RateLimit.Providers[0].Rules) != 1 {
		t.Fatalf("expected 1 rule after double apply, got %d", len(result2.Global.Services.RateLimit.Providers[0].Rules))
	}
}

func TestRateLimitRoundTripUserBasedMatch(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{
				Name:      "admin",
				Subjects:  []Subject{{Kind: "User", Name: "admin@co.com"}},
				Role:      "admin_tier",
				ModelRefs: []string{"gpt-4"},
				Priority:  1,
			},
		},
		RateTiers: []RateLimitTier{
			{Name: "admin-rate", User: "admin@co.com", RequestsPerMin: 10000},
		},
	}

	fragment := GenerateRouterFragment(policy)
	fragmentYAML, err := toCanonicalYAML(fragment)
	if err != nil {
		t.Fatalf("toCanonicalYAML error: %v", err)
	}

	baseYAML := `
version: "0.3"
routing:
  modelCards:
    - name: gpt-4
`
	merged, err := mergeDeployPayload([]byte(baseYAML), DeployRequest{YAML: string(fragmentYAML)})
	if err != nil {
		t.Fatalf("mergeDeployPayload error: %v", err)
	}

	cfg, err := routerconfig.ParseYAMLBytes(merged)
	if err != nil {
		t.Fatalf("ParseYAMLBytes error: %v", err)
	}

	if len(cfg.Signals.RoleBindings) != 1 {
		t.Fatalf("expected 1 role binding, got %d", len(cfg.Signals.RoleBindings))
	}
	if cfg.Signals.RoleBindings[0].Subjects[0].Kind != "User" {
		t.Fatalf("expected subject kind User, got %q", cfg.Signals.RoleBindings[0].Subjects[0].Kind)
	}

	rules := cfg.RateLimit.Providers[0].Rules
	if len(rules) != 1 {
		t.Fatalf("expected 1 rate limit rule, got %d", len(rules))
	}
	if rules[0].Match.User != "admin@co.com" {
		t.Fatalf("expected user match admin@co.com, got %q", rules[0].Match.User)
	}
	if rules[0].Match.Group != "" {
		t.Fatalf("expected empty group match, got %q", rules[0].Match.Group)
	}
	if rules[0].RequestsPerUnit != 10000 {
		t.Fatalf("expected RPU 10000, got %d", rules[0].RequestsPerUnit)
	}
}

func TestRateLimitRoundTripManyTiersPreservesOrder(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{Name: "a", Subjects: []Subject{{Kind: "Group", Name: "ga"}}, Role: "a_tier", ModelRefs: []string{"gpt-4"}, Priority: 1},
		},
		RateTiers: []RateLimitTier{
			{Name: "tier-1", Group: "g1", RequestsPerMin: 10},
			{Name: "tier-2", Group: "g2", RequestsPerMin: 20},
			{Name: "tier-3", Group: "g3", RequestsPerMin: 30},
			{Name: "tier-4", User: "u4", RequestsPerMin: 40},
			{Name: "tier-5", Group: "g5", TokensPerMin: 50000},
		},
	}

	fragment := GenerateRouterFragment(policy)
	fragmentYAML, err := toCanonicalYAML(fragment)
	if err != nil {
		t.Fatalf("toCanonicalYAML error: %v", err)
	}

	baseYAML := `
version: "0.3"
routing:
  modelCards:
    - name: gpt-4
`
	merged, err := mergeDeployPayload([]byte(baseYAML), DeployRequest{YAML: string(fragmentYAML)})
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

	expectedNames := []string{"tier-1", "tier-2", "tier-3", "tier-4", "tier-5"}
	for i, name := range expectedNames {
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
