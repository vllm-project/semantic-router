package handlers

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestToCanonicalYAMLRoleBindingsAndDecisions(t *testing.T) {
	t.Parallel()

	fragment := &GeneratedRouterFragment{
		RoleBindings: []RoleBindingFragment{
			{Subjects: []Subject{{Kind: "Group", Name: "eng"}}, Role: "eng_tier"},
		},
		Decisions: []DecisionFragment{
			{
				Name:     "rbac-eng",
				Priority: 10,
				Rules:    RuleFragment{Type: "authz", Name: "eng_tier"},
				ModelRefs: []ModelRefFragment{
					{Model: "gpt-4"},
					{Model: "claude-3"},
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
		t.Fatalf("failed to unmarshal canonical YAML: %v", err)
	}

	if len(doc.Routing.Signals.RoleBindings) != 1 {
		t.Fatalf("expected 1 role_binding, got %d", len(doc.Routing.Signals.RoleBindings))
	}
	if doc.Routing.Signals.RoleBindings[0].Role != "eng_tier" {
		t.Fatalf("expected role eng_tier, got %q", doc.Routing.Signals.RoleBindings[0].Role)
	}
	if len(doc.Routing.Decisions) != 1 {
		t.Fatalf("expected 1 decision, got %d", len(doc.Routing.Decisions))
	}
	if doc.Routing.Decisions[0].Name != "rbac-eng" {
		t.Fatalf("expected decision name rbac-eng, got %q", doc.Routing.Decisions[0].Name)
	}
	if doc.Global != nil {
		t.Fatal("expected nil global when no rate tiers")
	}
}

func TestToCanonicalYAMLWithRateLimit(t *testing.T) {
	t.Parallel()

	fragment := &GeneratedRouterFragment{
		RoleBindings: []RoleBindingFragment{
			{Subjects: []Subject{{Kind: "Group", Name: "premium"}}, Role: "premium_tier"},
		},
		Decisions: []DecisionFragment{
			{
				Name:      "rbac-premium",
				Priority:  10,
				Rules:     RuleFragment{Type: "authz", Name: "premium_tier"},
				ModelRefs: []ModelRefFragment{{Model: "gpt-4"}},
			},
		},
		RateLimit: RateLimitFragment{
			Providers: []RateLimitProviderFragment{
				{
					Type: "local-limiter",
					Rules: []RateLimitRuleFragment{
						{Name: "premium-rate", Match: RateLimitMatchFragment{Group: "premium"}, RPU: 1000, TPU: 100000, Unit: "minute"},
						{Name: "free-rate", Match: RateLimitMatchFragment{Group: "free"}, RPU: 10, TPU: 5000, Unit: "minute"},
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
		t.Fatalf("failed to unmarshal canonical YAML: %v", err)
	}

	if doc.Global == nil || doc.Global.Services == nil || doc.Global.Services.RateLimit == nil {
		t.Fatal("expected global.services.ratelimit to be populated")
	}

	rl := doc.Global.Services.RateLimit
	if len(rl.Providers) != 1 {
		t.Fatalf("expected 1 provider, got %d", len(rl.Providers))
	}
	if rl.Providers[0].Type != "local-limiter" {
		t.Fatalf("expected provider type local-limiter, got %q", rl.Providers[0].Type)
	}
	if len(rl.Providers[0].Rules) != 2 {
		t.Fatalf("expected 2 rules, got %d", len(rl.Providers[0].Rules))
	}
	if rl.Providers[0].Rules[0].Name != "premium-rate" {
		t.Fatalf("expected first rule name premium-rate, got %q", rl.Providers[0].Rules[0].Name)
	}
	if rl.Providers[0].Rules[0].RequestsPerUnit != 1000 {
		t.Fatalf("expected RPU 1000, got %d", rl.Providers[0].Rules[0].RequestsPerUnit)
	}
	if rl.Providers[0].Rules[0].TokensPerUnit != 100000 {
		t.Fatalf("expected TPU 100000, got %d", rl.Providers[0].Rules[0].TokensPerUnit)
	}
	if rl.Providers[0].Rules[1].Match.Group != "free" {
		t.Fatalf("expected second rule match group 'free', got %q", rl.Providers[0].Rules[1].Match.Group)
	}
}

func TestToCanonicalYAMLEmpty(t *testing.T) {
	t.Parallel()

	fragment := &GeneratedRouterFragment{}
	yamlBytes, err := toCanonicalYAML(fragment)
	if err != nil {
		t.Fatalf("toCanonicalYAML error: %v", err)
	}

	var doc routingFragmentDocument
	if err := yaml.Unmarshal(yamlBytes, &doc); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}
	if len(doc.Routing.Signals.RoleBindings) != 0 {
		t.Fatalf("expected 0 role_bindings, got %d", len(doc.Routing.Signals.RoleBindings))
	}
	if len(doc.Routing.Decisions) != 0 {
		t.Fatalf("expected 0 decisions, got %d", len(doc.Routing.Decisions))
	}
	if doc.Global != nil {
		t.Fatal("expected nil global for empty fragment")
	}
}

func TestMergeDeployPayloadWithRateLimit(t *testing.T) {
	t.Parallel()

	baseYAML := `
version: "0.3"
routing:
  modelCards:
    - name: gpt-4
  signals:
    keywords:
      - name: test_kw
        patterns: ["hello"]
        weight: 1.0
  decisions:
    - name: existing-decision
      priority: 1
      rules:
        type: keyword
        name: test_kw
      modelRefs:
        - model: gpt-4
global:
  services:
    observability:
      enabled: true
    ratelimit:
      providers:
        - type: local-limiter
          rules:
            - name: old-rule
              match:
                group: legacy
              requests_per_unit: 5
              unit: minute
`

	fragment := &GeneratedRouterFragment{
		RoleBindings: []RoleBindingFragment{
			{Subjects: []Subject{{Kind: "Group", Name: "vip"}}, Role: "vip_tier"},
		},
		Decisions: []DecisionFragment{
			{
				Name:      "rbac-vip",
				Priority:  10,
				Rules:     RuleFragment{Type: "authz", Name: "vip_tier"},
				ModelRefs: []ModelRefFragment{{Model: "gpt-4"}},
			},
		},
		RateLimit: RateLimitFragment{
			Providers: []RateLimitProviderFragment{
				{
					Type: "local-limiter",
					Rules: []RateLimitRuleFragment{
						{Name: "vip-rate", Match: RateLimitMatchFragment{Group: "vip"}, RPU: 500, Unit: "minute"},
					},
				},
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
		t.Fatalf("failed to unmarshal merged config: %v", err)
	}
	if len(result.Routing.Signals.RoleBindings) != 1 {
		t.Fatalf("expected 1 role binding after merge, got %d", len(result.Routing.Signals.RoleBindings))
	}
	if result.Routing.Signals.RoleBindings[0].Role != "vip_tier" {
		t.Fatalf("expected role vip_tier, got %q", result.Routing.Signals.RoleBindings[0].Role)
	}
	if len(result.Routing.Decisions) != 1 {
		t.Fatalf("expected 1 decision after merge (replaced), got %d", len(result.Routing.Decisions))
	}
	if result.Routing.Decisions[0].Name != "rbac-vip" {
		t.Fatalf("expected decision name rbac-vip, got %q", result.Routing.Decisions[0].Name)
	}
	if result.Global == nil {
		t.Fatal("expected global to be non-nil after merge")
	}
	rl := result.Global.Services.RateLimit
	if len(rl.Providers) != 1 {
		t.Fatalf("expected 1 ratelimit provider, got %d", len(rl.Providers))
	}
	if rl.Providers[0].Rules[0].Name != "vip-rate" {
		t.Fatalf("expected ratelimit rule name vip-rate, got %q", rl.Providers[0].Rules[0].Name)
	}
	if rl.Providers[0].Rules[0].RequestsPerUnit != 500 {
		t.Fatalf("expected RPU 500, got %d", rl.Providers[0].Rules[0].RequestsPerUnit)
	}
}

func TestMergeDeployPayloadPreservesOtherGlobalFields(t *testing.T) {
	t.Parallel()

	baseYAML := `
version: "0.3"
routing:
  modelCards:
    - name: gpt-4
global:
  services:
    observability:
      enabled: true
    ratelimit:
      providers:
        - type: local-limiter
          rules:
            - name: old-rule
              match:
                group: old
              requests_per_unit: 1
              unit: minute
`

	fragment := &GeneratedRouterFragment{
		RateLimit: RateLimitFragment{
			Providers: []RateLimitProviderFragment{
				{
					Type: "local-limiter",
					Rules: []RateLimitRuleFragment{
						{Name: "new-rule", Match: RateLimitMatchFragment{Group: "new"}, RPU: 100, Unit: "minute"},
					},
				},
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

	mergedStr := string(merged)
	if !strings.Contains(mergedStr, "observability") {
		t.Fatal("expected observability to be preserved in merged config")
	}

	var result routerconfig.CanonicalConfig
	if err := yaml.Unmarshal(merged, &result); err != nil {
		t.Fatalf("failed to unmarshal merged config: %v", err)
	}

	if result.Global == nil {
		t.Fatal("expected global to be non-nil")
	}
	rl := result.Global.Services.RateLimit
	if len(rl.Providers) != 1 || rl.Providers[0].Rules[0].Name != "new-rule" {
		t.Fatalf("expected ratelimit to be replaced with new-rule, got %+v", rl)
	}
}

func TestMergeDeployPayloadNoRateLimitPreservesExisting(t *testing.T) {
	t.Parallel()

	baseYAML := `
version: "0.3"
routing:
  modelCards:
    - name: gpt-4
global:
  services:
    ratelimit:
      providers:
        - type: local-limiter
          rules:
            - name: keep-me
              match:
                group: existing
              requests_per_unit: 42
              unit: minute
`

	fragment := &GeneratedRouterFragment{
		RoleBindings: []RoleBindingFragment{
			{Subjects: []Subject{{Kind: "Group", Name: "g"}}, Role: "r"},
		},
		Decisions: []DecisionFragment{
			{
				Name:      "rbac-g",
				Priority:  5,
				Rules:     RuleFragment{Type: "authz", Name: "r"},
				ModelRefs: []ModelRefFragment{{Model: "gpt-4"}},
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
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if result.Global == nil {
		t.Fatal("expected global to be preserved")
	}
	rl := result.Global.Services.RateLimit
	if len(rl.Providers) != 1 || rl.Providers[0].Rules[0].Name != "keep-me" {
		t.Fatalf("expected existing ratelimit to be preserved, got %+v", rl)
	}
}

func TestRateLimitRoundTripFragmentToRouterConfig(t *testing.T) {
	t.Parallel()

	policy := &SecurityPolicyConfig{
		RoleMappings: []RoleMapping{
			{
				Name:      "premium",
				Subjects:  []Subject{{Kind: "Group", Name: "paying"}},
				Role:      "premium_tier",
				ModelRefs: []string{"gpt-4"},
				Priority:  10,
			},
		},
		RateTiers: []RateLimitTier{
			{Name: "premium-rate", Group: "paying", RequestsPerMin: 500, TokensPerMin: 50000},
			{Name: "free-rate", Group: "free", RequestsPerMin: 10, TokensPerMin: 1000},
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
		t.Fatalf("expected 1 role binding in parsed config, got %d", len(cfg.Signals.RoleBindings))
	}
	if cfg.Signals.RoleBindings[0].Role != "premium_tier" {
		t.Fatalf("expected role premium_tier, got %q", cfg.Signals.RoleBindings[0].Role)
	}

	if len(cfg.RateLimit.Providers) != 1 {
		t.Fatalf("expected 1 rate limit provider, got %d", len(cfg.RateLimit.Providers))
	}
	rules := cfg.RateLimit.Providers[0].Rules
	if len(rules) != 2 {
		t.Fatalf("expected 2 rate limit rules, got %d", len(rules))
	}
	if rules[0].Name != "premium-rate" {
		t.Fatalf("expected first rule name premium-rate, got %q", rules[0].Name)
	}
	if rules[0].Match.Group != "paying" {
		t.Fatalf("expected first rule match group 'paying', got %q", rules[0].Match.Group)
	}
	if rules[0].RequestsPerUnit != 500 {
		t.Fatalf("expected RPU 500, got %d", rules[0].RequestsPerUnit)
	}
	if rules[0].TokensPerUnit != 50000 {
		t.Fatalf("expected TPU 50000, got %d", rules[0].TokensPerUnit)
	}
	if rules[1].Name != "free-rate" {
		t.Fatalf("expected second rule name free-rate, got %q", rules[1].Name)
	}
	if rules[1].Match.Group != "free" {
		t.Fatalf("expected second rule match group 'free', got %q", rules[1].Match.Group)
	}
}
