package config

import (
	"strings"
	"testing"
	"time"
)

func TestFallbackConfigDefaultsDisabled(t *testing.T) {
	input := []byte(`version: v0.3
routing:
  decisions:
    - name: route1
      modelRefs: [{model: m1}]
`)
	cfg, err := ParseYAMLBytes(input)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Fallback != nil && cfg.Fallback.Enabled {
		t.Fatal("fallback must be disabled by default when unconfigured")
	}
}

func TestFallbackConfigValidation(t *testing.T) {
	tests := []struct {
		name        string
		yamlSnippet string
		errSubstr   string
	}{
		{
			name: "unsupported version",
			yamlSnippet: `version: v0.3
routing:
  fallback:
    version: 2
    enabled: true
`,
			errSubstr: "unsupported fallback policy version 2",
		},
		{
			name: "max_attempts zero when enabled",
			yamlSnippet: `version: v0.3
routing:
  fallback:
    version: 1
    enabled: true
    max_attempts: 0
`,
			errSubstr: "max_attempts must be >= 1 when enabled",
		},
		{
			name: "per_attempt_timeout exceeds total_timeout",
			yamlSnippet: `version: v0.3
routing:
  fallback:
    version: 1
    enabled: true
    max_attempts: 3
    total_timeout: 5s
    per_attempt_timeout: 10s
`,
			errSubstr: "per_attempt_timeout (10s) cannot exceed total_timeout (5s)",
		},
		{
			name: "invalid retryable status code",
			yamlSnippet: `version: v0.3
routing:
  fallback:
    version: 1
    enabled: true
    max_attempts: 3
    retryable_status_codes: [999]
`,
			errSubstr: "invalid retryable status code 999",
		},
		{
			name: "invalid circuit breaker cooldown",
			yamlSnippet: `version: v0.3
routing:
  fallback:
    version: 1
    enabled: true
    max_attempts: 3
    circuit_breaker:
      cooldown_period: -5s
`,
			errSubstr: "cooldown_period cannot be negative",
		},
		{
			name: "invalid recipe-scoped fallback",
			yamlSnippet: `version: v0.3
routing: {}
recipes:
  - name: broken-recipe
    routing:
      fallback:
        version: 1
        enabled: true
        max_attempts: -1
`,
			errSubstr: "recipes[broken-recipe].routing.fallback: fallback max_attempts must be >= 1 when enabled",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(tt.yamlSnippet))
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tt.errSubstr)
			}
			if !strings.Contains(err.Error(), tt.errSubstr) {
				t.Fatalf("expected error containing %q, got: %v", tt.errSubstr, err)
			}
		})
	}
}

func TestFallbackRecipeInheritanceAndOverrides(t *testing.T) {
	input := []byte(`version: v0.3
routing:
  fallback:
    version: 1
    enabled: false
    max_attempts: 2
    total_timeout: 20s
    per_attempt_timeout: 5s
recipes:
  - name: active-fallback
    routing:
      fallback:
        version: 1
        enabled: true
        max_attempts: 5
        total_timeout: 45s
        per_attempt_timeout: 10s
  - name: inherited-fallback
    routing: {}
`)
	cfg, err := ParseYAMLBytes(input)
	if err != nil {
		t.Fatal(err)
	}

	// Global / top-level fallback
	if cfg.Fallback == nil {
		t.Fatal("expected top-level fallback to be parsed")
	}
	if cfg.Fallback.Enabled {
		t.Fatal("expected top-level fallback to be disabled")
	}
	if cfg.Fallback.MaxAttempts != 2 {
		t.Fatalf("expected max_attempts=2, got %d", cfg.Fallback.MaxAttempts)
	}

	// Active recipe override
	activeRecipe, ok := cfg.RecipeByName("active-fallback")
	if !ok {
		t.Fatal("missing active-fallback recipe")
	}
	activeScoped := cfg.ConfigForRecipe(activeRecipe)
	if activeScoped.Fallback == nil {
		t.Fatal("expected active-fallback recipe to have fallback policy")
	}
	if !activeScoped.Fallback.Enabled {
		t.Fatal("expected active-fallback recipe to have enabled=true")
	}
	if activeScoped.Fallback.MaxAttempts != 5 {
		t.Fatalf("expected max_attempts=5, got %d", activeScoped.Fallback.MaxAttempts)
	}
	if activeScoped.Fallback.TotalTimeout != 45*time.Second {
		t.Fatalf("expected total_timeout=45s, got %v", activeScoped.Fallback.TotalTimeout)
	}

	// Inherited recipe
	inheritedRecipe, ok := cfg.RecipeByName("inherited-fallback")
	if !ok {
		t.Fatal("missing inherited-fallback recipe")
	}
	inheritedScoped := cfg.ConfigForRecipe(inheritedRecipe)
	if inheritedScoped.Fallback == nil {
		t.Fatal("expected inherited recipe to inherit fallback policy")
	}
	if inheritedScoped.Fallback.Enabled {
		t.Fatal("expected inherited recipe fallback to be disabled like parent")
	}
	if inheritedScoped.Fallback.MaxAttempts != 2 {
		t.Fatalf("expected inherited max_attempts=2, got %d", inheritedScoped.Fallback.MaxAttempts)
	}
}

func TestFallbackRoutingFragment(t *testing.T) {
	fragment := []byte(`routing:
  fallback:
    version: 1
    enabled: true
    max_attempts: 4
    total_timeout: 30s
    per_attempt_timeout: 8s
`)
	cfg, err := ParseRoutingYAMLBytes(fragment)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Fallback == nil {
		t.Fatal("expected fallback policy to be parsed in routing fragment")
	}
	if !cfg.Fallback.Enabled {
		t.Fatal("expected fallback enabled=true")
	}
	if cfg.Fallback.MaxAttempts != 4 {
		t.Fatalf("expected max_attempts=4, got %d", cfg.Fallback.MaxAttempts)
	}
	if cfg.Fallback.PerAttemptTimeout != 8*time.Second {
		t.Fatalf("expected per_attempt_timeout=8s, got %v", cfg.Fallback.PerAttemptTimeout)
	}
}

func TestFallbackRecipePartialOverridePreservesGlobalEnabled(t *testing.T) {
	input := []byte(`version: v0.3
routing:
  fallback:
    version: 1
    enabled: true
    max_attempts: 3
    total_timeout: 30s
    per_attempt_timeout: 10s
    circuit_breaker:
      consecutive_failures: 3
      cooldown_period: 30s
      half_open_probes: 1
recipes:
  - name: partial-tuning
    routing:
      fallback:
        max_attempts: 5
        circuit_breaker:
          consecutive_failures: 1
  - name: explicitly-disabled
    routing:
      fallback:
        enabled: false
        max_attempts: 2
  - name: pure-inherit
    routing: {}
`)
	cfg, err := ParseYAMLBytes(input)
	if err != nil {
		t.Fatalf("unexpected error parsing YAML: %v", err)
	}

	// 1. Partial tuning recipe must inherit global enabled: true
	partialRecipe, ok := cfg.RecipeByName("partial-tuning")
	if !ok {
		t.Fatal("missing partial-tuning recipe")
	}
	partialScoped := cfg.ConfigForRecipe(partialRecipe)
	if partialScoped.Fallback == nil {
		t.Fatal("expected partialScoped to have Fallback policy")
	}
	if !partialScoped.Fallback.Enabled {
		t.Errorf("expected partial-tuning recipe to preserve global enabled: true, got false")
	}
	if partialScoped.Fallback.MaxAttempts != 5 {
		t.Errorf("expected partial-tuning max_attempts=5, got %d", partialScoped.Fallback.MaxAttempts)
	}
	if partialScoped.Fallback.CircuitBreaker.ConsecutiveFailures != 1 {
		t.Errorf("expected partial-tuning consecutive_failures=1, got %d", partialScoped.Fallback.CircuitBreaker.ConsecutiveFailures)
	}
	if partialScoped.Fallback.TotalTimeout != 30*time.Second {
		t.Errorf("expected partial-tuning total_timeout=30s, got %v", partialScoped.Fallback.TotalTimeout)
	}
	if partialScoped.Fallback.PerAttemptTimeout != 10*time.Second {
		t.Errorf("expected partial-tuning per_attempt_timeout=10s, got %v", partialScoped.Fallback.PerAttemptTimeout)
	}

	// 2. Explicitly disabled recipe must remain enabled: false despite global enabled: true
	disabledRecipe, ok := cfg.RecipeByName("explicitly-disabled")
	if !ok {
		t.Fatal("missing explicitly-disabled recipe")
	}
	disabledScoped := cfg.ConfigForRecipe(disabledRecipe)
	if disabledScoped.Fallback == nil {
		t.Fatal("expected disabledScoped to have Fallback policy")
	}
	if disabledScoped.Fallback.Enabled {
		t.Errorf("expected explicitly-disabled recipe to remain false, got true")
	}
	if disabledScoped.Fallback.MaxAttempts != 2 {
		t.Errorf("expected explicitly-disabled max_attempts=2, got %d", disabledScoped.Fallback.MaxAttempts)
	}

	// 3. Pure inherit recipe must inherit global enabled: true
	pureRecipe, ok := cfg.RecipeByName("pure-inherit")
	if !ok {
		t.Fatal("missing pure-inherit recipe")
	}
	pureScoped := cfg.ConfigForRecipe(pureRecipe)
	if pureScoped.Fallback == nil {
		t.Fatal("expected pureScoped to have Fallback policy")
	}
	if !pureScoped.Fallback.Enabled {
		t.Errorf("expected pure-inherit recipe to inherit enabled: true, got false")
	}
	if pureScoped.Fallback.MaxAttempts != 3 {
		t.Errorf("expected pure-inherit max_attempts=3, got %d", pureScoped.Fallback.MaxAttempts)
	}
}

func TestFallbackRecipeEnableOnlyOverrideInheritsGlobalLimits(t *testing.T) {
	// 1. Enable-only recipe with global limits defined
	yamlConfigWithGlobal := `
version: v0.3
routing:
  fallback:
    version: 1
    enabled: false
    max_attempts: 3
    total_timeout: 10s
    per_attempt_timeout: 4s
recipes:
  - name: enable-only-with-global
    routing:
      fallback:
        enabled: true
`
	cfg, err := ParseYAMLBytes([]byte(yamlConfigWithGlobal))
	if err != nil {
		t.Fatalf("expected enable-only recipe with global fallback to parse successfully, got: %v", err)
	}

	recipe, ok := cfg.RecipeByName("enable-only-with-global")
	if !ok {
		t.Fatal("missing enable-only-with-global recipe")
	}
	scoped := cfg.ConfigForRecipe(recipe)
	if scoped.Fallback == nil {
		t.Fatal("expected scoped.Fallback to be present")
	}
	if !scoped.Fallback.Enabled {
		t.Errorf("expected recipe fallback to be enabled, got false")
	}
	if scoped.Fallback.MaxAttempts != 3 {
		t.Errorf("expected max_attempts=3 inherited from global, got %d", scoped.Fallback.MaxAttempts)
	}
	if scoped.Fallback.TotalTimeout != 10*time.Second {
		t.Errorf("expected total_timeout=10s inherited from global, got %v", scoped.Fallback.TotalTimeout)
	}
	if scoped.Fallback.PerAttemptTimeout != 4*time.Second {
		t.Errorf("expected per_attempt_timeout=4s inherited from global, got %v", scoped.Fallback.PerAttemptTimeout)
	}

	// 2. Enable-only recipe without any global fallback defined (inherits canonical defaults)
	yamlConfigNoGlobal := `
version: v0.3
recipes:
  - name: enable-only-no-global
    routing:
      fallback:
        enabled: true
`
	cfgNoGlobal, err := ParseYAMLBytes([]byte(yamlConfigNoGlobal))
	if err != nil {
		t.Fatalf("expected enable-only recipe without global fallback to parse successfully, got: %v", err)
	}

	recipeNoGlobal, ok := cfgNoGlobal.RecipeByName("enable-only-no-global")
	if !ok {
		t.Fatal("missing enable-only-no-global recipe")
	}
	scopedNoGlobal := cfgNoGlobal.ConfigForRecipe(recipeNoGlobal)
	if scopedNoGlobal.Fallback == nil {
		t.Fatal("expected scopedNoGlobal.Fallback to be present")
	}
	if !scopedNoGlobal.Fallback.Enabled {
		t.Errorf("expected recipe fallback to be enabled, got false")
	}
	if scopedNoGlobal.Fallback.MaxAttempts != 3 {
		t.Errorf("expected max_attempts=3 from default policy, got %d", scopedNoGlobal.Fallback.MaxAttempts)
	}
	if scopedNoGlobal.Fallback.TotalTimeout != 30*time.Second {
		t.Errorf("expected total_timeout=30s from default policy, got %v", scopedNoGlobal.Fallback.TotalTimeout)
	}

	// 3. Explicitly invalid recipe values must still be rejected during validation
	yamlConfigInvalid := `
version: v0.3
routing:
  fallback:
    version: 1
    enabled: true
    max_attempts: 3
recipes:
  - name: explicitly-invalid-recipe
    routing:
      fallback:
        max_attempts: -1
`
	_, err = ParseYAMLBytes([]byte(yamlConfigInvalid))
	if err == nil {
		t.Fatal("expected ParseYAML to fail for recipe with max_attempts: -1, but got nil error")
	}
}

func TestFallbackRecipeProactiveOverridePermutationsAndValidation(t *testing.T) {
	// 1. Multi-recipe configuration testing independent partial overrides:
	// - recipe-timeouts: overrides timeouts only (inherits max_attempts: 3, breaker settings)
	// - recipe-breaker: overrides circuit breaker only (inherits timeouts and half_open_probes)
	// - recipe-status-codes: overrides retryable status codes only
	// - recipe-unconfigured: omits fallback completely (inherits clean clone of global)
	multiRecipeYAML := `
version: v0.3
routing:
  fallback:
    version: 1
    enabled: true
    max_attempts: 3
    total_timeout: 30s
    per_attempt_timeout: 10s
    retryable_status_codes: [502, 503, 504]
    circuit_breaker:
      consecutive_failures: 3
      cooldown_period: 20s
      half_open_probes: 2
recipes:
  - name: recipe-timeouts
    routing:
      fallback:
        total_timeout: 15s
        per_attempt_timeout: 5s
  - name: recipe-breaker
    routing:
      fallback:
        circuit_breaker:
          consecutive_failures: 5
          cooldown_period: 45s
  - name: recipe-status-codes
    routing:
      fallback:
        retryable_status_codes: [503]
  - name: recipe-unconfigured
    routing: {}
`
	cfg, err := ParseYAMLBytes([]byte(multiRecipeYAML))
	if err != nil {
		t.Fatalf("expected multi-recipe config to parse and validate successfully, got: %v", err)
	}

	// Verify recipe-timeouts
	recTimeouts, ok := cfg.RecipeByName("recipe-timeouts")
	if !ok {
		t.Fatal("missing recipe-timeouts")
	}
	scopedTimeouts := cfg.ConfigForRecipe(recTimeouts)
	if scopedTimeouts.Fallback == nil {
		t.Fatal("expected scopedTimeouts.Fallback to be present")
	}
	if !scopedTimeouts.Fallback.Enabled {
		t.Error("expected recipe-timeouts to inherit enabled: true")
	}
	if scopedTimeouts.Fallback.MaxAttempts != 3 {
		t.Errorf("expected max_attempts=3 inherited from global, got %d", scopedTimeouts.Fallback.MaxAttempts)
	}
	if scopedTimeouts.Fallback.TotalTimeout != 15*time.Second {
		t.Errorf("expected total_timeout=15s, got %v", scopedTimeouts.Fallback.TotalTimeout)
	}
	if scopedTimeouts.Fallback.PerAttemptTimeout != 5*time.Second {
		t.Errorf("expected per_attempt_timeout=5s, got %v", scopedTimeouts.Fallback.PerAttemptTimeout)
	}
	if scopedTimeouts.Fallback.CircuitBreaker.ConsecutiveFailures != 3 {
		t.Errorf("expected consecutive_failures=3 inherited from global, got %d", scopedTimeouts.Fallback.CircuitBreaker.ConsecutiveFailures)
	}

	// Verify recipe-breaker
	recBreaker, ok := cfg.RecipeByName("recipe-breaker")
	if !ok {
		t.Fatal("missing recipe-breaker")
	}
	scopedBreaker := cfg.ConfigForRecipe(recBreaker)
	if scopedBreaker.Fallback == nil {
		t.Fatal("expected scopedBreaker.Fallback to be present")
	}
	if scopedBreaker.Fallback.CircuitBreaker.ConsecutiveFailures != 5 {
		t.Errorf("expected overridden consecutive_failures=5, got %d", scopedBreaker.Fallback.CircuitBreaker.ConsecutiveFailures)
	}
	if scopedBreaker.Fallback.CircuitBreaker.CooldownPeriod != 45*time.Second {
		t.Errorf("expected overridden cooldown_period=45s, got %v", scopedBreaker.Fallback.CircuitBreaker.CooldownPeriod)
	}
	if scopedBreaker.Fallback.CircuitBreaker.HalfOpenProbes != 2 {
		t.Errorf("expected inherited half_open_probes=2, got %d", scopedBreaker.Fallback.CircuitBreaker.HalfOpenProbes)
	}
	if scopedBreaker.Fallback.TotalTimeout != 30*time.Second {
		t.Errorf("expected inherited total_timeout=30s, got %v", scopedBreaker.Fallback.TotalTimeout)
	}

	// Verify recipe-status-codes
	recCodes, ok := cfg.RecipeByName("recipe-status-codes")
	if !ok {
		t.Fatal("missing recipe-status-codes")
	}
	scopedCodes := cfg.ConfigForRecipe(recCodes)
	if scopedCodes.Fallback == nil {
		t.Fatal("expected scopedCodes.Fallback to be present")
	}
	if len(scopedCodes.Fallback.RetryableStatusCodes) != 1 || scopedCodes.Fallback.RetryableStatusCodes[0] != 503 {
		t.Errorf("expected overridden retryable_status_codes=[503], got %v", scopedCodes.Fallback.RetryableStatusCodes)
	}

	// Verify recipe-unconfigured (omitted fallback)
	recUnconf, ok := cfg.RecipeByName("recipe-unconfigured")
	if !ok {
		t.Fatal("missing recipe-unconfigured")
	}
	scopedUnconf := cfg.ConfigForRecipe(recUnconf)
	if scopedUnconf.Fallback == nil {
		t.Fatal("expected scopedUnconf.Fallback to inherit global fallback")
	}
	if scopedUnconf.Fallback.MaxAttempts != 3 || scopedUnconf.Fallback.TotalTimeout != 30*time.Second {
		t.Errorf("expected full global fallback inheritance, got: %#v", scopedUnconf.Fallback)
	}

	// 2. Proactive rejection of invalid combinations in recipe overrides
	invalidSnippets := []struct {
		name      string
		yaml      string
		errSubstr string
	}{
		{
			name: "recipe per_attempt_timeout exceeds inherited total_timeout",
			yaml: `
version: v0.3
routing:
  fallback:
    version: 1
    enabled: true
    max_attempts: 3
    total_timeout: 20s
    per_attempt_timeout: 5s
recipes:
  - name: conflicting-timeouts
    routing:
      fallback:
        per_attempt_timeout: 30s
`,
			errSubstr: "cannot exceed total_timeout",
		},
		{
			name: "recipe total_timeout smaller than inherited per_attempt_timeout",
			yaml: `
version: v0.3
routing:
  fallback:
    version: 1
    enabled: true
    max_attempts: 3
    total_timeout: 30s
    per_attempt_timeout: 10s
recipes:
  - name: clamped-total
    routing:
      fallback:
        total_timeout: 5s
`,
			errSubstr: "cannot exceed total_timeout",
		},
		{
			name: "recipe negative total_timeout",
			yaml: `
version: v0.3
recipes:
  - name: neg-total
    routing:
      fallback:
        total_timeout: -5s
`,
			errSubstr: "total_timeout cannot be negative",
		},
		{
			name: "recipe negative per_attempt_timeout",
			yaml: `
version: v0.3
recipes:
  - name: neg-attempt
    routing:
      fallback:
        per_attempt_timeout: -2s
`,
			errSubstr: "per_attempt_timeout cannot be negative",
		},
		{
			name: "recipe invalid retryable status code",
			yaml: `
version: v0.3
recipes:
  - name: invalid-code
    routing:
      fallback:
        retryable_status_codes: [999]
`,
			errSubstr: "invalid retryable status code 999",
		},
		{
			name: "recipe negative circuit breaker consecutive failures",
			yaml: `
version: v0.3
recipes:
  - name: invalid-breaker-failures
    routing:
      fallback:
        circuit_breaker:
          consecutive_failures: -1
`,
			errSubstr: "consecutive_failures cannot be negative",
		},
		{
			name: "recipe negative circuit breaker cooldown period",
			yaml: `
version: v0.3
recipes:
  - name: invalid-breaker-cooldown
    routing:
      fallback:
        circuit_breaker:
          cooldown_period: -10s
`,
			errSubstr: "cooldown_period cannot be negative",
		},
		{
			name: "recipe negative circuit breaker half open probes",
			yaml: `
version: v0.3
recipes:
  - name: invalid-breaker-probes
    routing:
      fallback:
        circuit_breaker:
          half_open_probes: -1
`,
			errSubstr: "half_open_probes cannot be negative",
		},
		{
			name: "recipe unsupported fallback version",
			yaml: `
version: v0.3
recipes:
  - name: invalid-version
    routing:
      fallback:
        version: 99
`,
			errSubstr: "unsupported fallback policy version 99",
		},
	}

	for _, tc := range invalidSnippets {
		t.Run(tc.name, func(t *testing.T) {
			_, parseErr := ParseYAMLBytes([]byte(tc.yaml))
			if parseErr == nil {
				t.Fatalf("expected error containing %q, got nil", tc.errSubstr)
			}
			if !strings.Contains(parseErr.Error(), tc.errSubstr) {
				t.Fatalf("expected error containing %q, got: %v", tc.errSubstr, parseErr)
			}
		})
	}
}
