package handlers

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// These cover mergeDeployPayload's canonical merge semantics, which any deploy payload
// relies on. They were previously driven through the retired security policy fragment
// generator; the payloads are now written directly so the coverage does not depend on it.

const mergeBaseWithRateLimit = `
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

func TestMergeDeployPayloadReplacesDecisionsAndRateLimit(t *testing.T) {
	t.Parallel()

	payload := `
routing:
  signals:
    role_bindings:
      - name: vip
        subjects:
          - kind: Group
            name: vip
        role: vip_tier
  decisions:
    - name: vip-route
      priority: 10
      rules:
        type: authz
        name: vip_tier
      modelRefs:
        - model: gpt-4
global:
  services:
    ratelimit:
      providers:
        - type: local-limiter
          rules:
            - name: vip-rate
              match:
                group: vip
              requests_per_unit: 500
              unit: minute
`

	merged, err := mergeDeployPayload([]byte(mergeBaseWithRateLimit), DeployRequest{YAML: payload})
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
	if result.Routing.Decisions[0].Name != "vip-route" {
		t.Fatalf("expected decision name vip-route, got %q", result.Routing.Decisions[0].Name)
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

	payload := `
global:
  services:
    ratelimit:
      providers:
        - type: local-limiter
          rules:
            - name: new-rule
              match:
                group: new
              requests_per_unit: 100
              unit: minute
`

	merged, err := mergeDeployPayload([]byte(mergeBaseWithRateLimit), DeployRequest{YAML: payload})
	if err != nil {
		t.Fatalf("mergeDeployPayload error: %v", err)
	}
	if !strings.Contains(string(merged), "observability") {
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

	payload := `
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
`

	merged, err := mergeDeployPayload([]byte(baseYAML), DeployRequest{YAML: payload})
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

// TestMergedDeployPayloadParsesAsRouterConfig closes the loop: what the dashboard writes
// has to come back out of the router's own parser with the role bindings and limiter rules
// intact, in order.
func TestMergedDeployPayloadParsesAsRouterConfig(t *testing.T) {
	t.Parallel()

	baseYAML := `
version: "0.3"
routing:
  modelCards:
    - name: gpt-4
`

	payload := `
routing:
  signals:
    role_bindings:
      - name: premium
        subjects:
          - kind: Group
            name: paying
        role: premium_tier
global:
  services:
    ratelimit:
      providers:
        - type: local-limiter
          rules:
            - name: premium-rate
              match:
                group: paying
              requests_per_unit: 500
              tokens_per_unit: 50000
              unit: minute
            - name: free-rate
              match:
                group: free
              requests_per_unit: 10
              tokens_per_unit: 1000
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
	if rules[0].Name != "premium-rate" || rules[0].Match.Group != "paying" {
		t.Fatalf("unexpected first rule: %+v", rules[0])
	}
	if rules[0].RequestsPerUnit != 500 || rules[0].TokensPerUnit != 50000 {
		t.Fatalf("expected first rule RPU 500 / TPU 50000, got %d / %d",
			rules[0].RequestsPerUnit, rules[0].TokensPerUnit)
	}
	if rules[1].Name != "free-rate" || rules[1].Match.Group != "free" {
		t.Fatalf("unexpected second rule: %+v", rules[1])
	}
}
