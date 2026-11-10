package fallback

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v2"
)

func boolOf(v bool) *bool { return &v }

// The merge goes field by field from the most specific layer down: a step's
// override, the decision's, then the recipe's policy (which carries the
// global one).
func TestResolveMergesFieldByField(t *testing.T) {
	recipe := DefaultEnabledPolicy()
	recipe.MaxAttempts, recipe.TotalTimeout = 4, 20*time.Second
	for name, tc := range map[string]struct {
		decision, step *FallbackOverride
		check          func(FallbackPolicy) bool
	}{
		"no override keeps the recipe's": {
			check: func(p FallbackPolicy) bool { return reflect.DeepEqual(p, recipe) },
		},
		"a decision replaces what it sets": {
			decision: &FallbackOverride{MaxAttempts: 2, RetryableStatusCodes: []int{429}},
			check: func(p FallbackPolicy) bool {
				return p.Enabled && p.MaxAttempts == 2 && p.TotalTimeout == 20*time.Second &&
					reflect.DeepEqual(p.RetryableStatusCodes, []int{429})
			},
		},
		"a step wins over its decision": {
			decision: &FallbackOverride{MaxAttempts: 2, PerAttemptTimeout: 3 * time.Second},
			step:     &FallbackOverride{MaxAttempts: 5},
			check: func(p FallbackPolicy) bool {
				return p.MaxAttempts == 5 && p.PerAttemptTimeout == 3*time.Second
			},
		},
		"enabled is tri-state": {
			decision: Disabled(),
			step:     &FallbackOverride{MaxAttempts: 6},
			check: func(p FallbackPolicy) bool {
				explicit := p.ExplicitEnabled()
				return !p.Enabled && explicit != nil && !*explicit && p.MaxAttempts == 6
			},
		},
		"a step can turn on what its decision turned off": {
			decision: Disabled(),
			step:     &FallbackOverride{Enabled: boolOf(true)},
			check:    func(p FallbackPolicy) bool { return p.Enabled },
		},
	} {
		t.Run(name, func(t *testing.T) {
			got := recipe.Resolve(tc.decision, tc.step)
			if !tc.check(got) {
				t.Fatalf("resolved %+v", got)
			}
		})
	}
	resolved := recipe.Resolve(&FallbackOverride{RetryableStatusCodes: []int{500}})
	resolved.RetryableStatusCodes[0] = 501
	if recipe.RetryableStatusCodes[0] == 501 {
		t.Fatal("the resolved policy shares the recipe's status codes")
	}
}

func TestOverrideDecodesLikeTheRecipePolicy(t *testing.T) {
	var fromYAML FallbackOverride
	if err := yaml.Unmarshal([]byte("enabled: false\nmax_attempts: 2\ntotal_timeout: 30s\nper_attempt_timeout: 5000000000\n"), &fromYAML); err != nil {
		t.Fatal(err)
	}
	want := FallbackOverride{Enabled: boolOf(false), MaxAttempts: 2, TotalTimeout: 30 * time.Second, PerAttemptTimeout: 5 * time.Second}
	if !reflect.DeepEqual(fromYAML, want) {
		t.Fatalf("yaml %+v", fromYAML)
	}
	var fromJSON FallbackOverride
	if err := json.Unmarshal([]byte(`{"enabled":false,"max_attempts":2,"total_timeout":"30s","per_attempt_timeout":5000000000}`), &fromJSON); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(fromJSON, want) {
		t.Fatalf("json %+v", fromJSON)
	}
	if err := yaml.Unmarshal([]byte("total_timeout: soon\n"), &FallbackOverride{}); err == nil || !strings.Contains(err.Error(), "total_timeout") {
		t.Fatalf("a bad duration: %v", err)
	}
}

func TestOverrideValidate(t *testing.T) {
	for _, bad := range []FallbackOverride{
		{MaxAttempts: -1},
		{TotalTimeout: -time.Second},
		{PerAttemptTimeout: -time.Second},
		{RetryableStatusCodes: []int{99}},
	} {
		if err := bad.Validate(); err == nil {
			t.Fatalf("%+v must not validate", bad)
		}
	}
	if err := (*FallbackOverride)(nil).Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestForSharesTheCircuitBreakers(t *testing.T) {
	recipe := NewOrchestrator(DefaultEnabledPolicy(), nil)
	if recipe.For(nil, nil) != recipe {
		t.Fatal("no override keeps the recipe's orchestrator")
	}
	call := recipe.For(Disabled())
	if call.Policy().Enabled || call.CircuitBreaker() != recipe.CircuitBreaker() || !recipe.Policy().Enabled {
		t.Fatalf("call policy %+v", call.Policy())
	}
}
