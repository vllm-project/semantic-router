package config

import (
	"strings"
	"testing"
)

func contextConfig(rules ...ContextRule) *RouterConfig {
	cfg := &RouterConfig{}
	cfg.ContextRules = rules
	return cfg
}

func TestValidateContextContractsAcceptsValidBands(t *testing.T) {
	cases := map[string][]ContextRule{
		"single bounded band": {
			{Name: "long", MinTokens: "32K", MaxTokens: "256K"},
		},
		"equal min and max is an exact match band": {
			{Name: "exact", MinTokens: "1K", MaxTokens: "1K"},
		},
		"omitted max_tokens is open-ended": {
			{Name: "short", MinTokens: "0", MaxTokens: "1K"},
			{Name: "overflow", MinTokens: "1001"},
		},
		"overlapping bands load with a warning": {
			{Name: "wide", MinTokens: "0", MaxTokens: "10K"},
			{Name: "narrow", MinTokens: "2K", MaxTokens: "3K"},
		},
		"gaps between bands load with a warning": {
			{Name: "low", MinTokens: "0", MaxTokens: "1K"},
			{Name: "high", MinTokens: "4K", MaxTokens: "128K"},
		},
		"two open-ended bands load with a warning": {
			{Name: "a", MinTokens: "1K"},
			{Name: "b", MinTokens: "2K"},
		},
		"omitted min_tokens defaults to zero": {
			{Name: "short", MaxTokens: "1K"},
		},
		"fractional suffix values": {
			{Name: "band", MinTokens: "1.5K", MaxTokens: "0.5M"},
		},
	}
	for name, rules := range cases {
		t.Run(name, func(t *testing.T) {
			if err := validateContextContracts(contextConfig(rules...)); err != nil {
				t.Fatalf("expected config to load, got: %v", err)
			}
		})
	}
}

func TestValidateContextContractsRejectsInvalidBands(t *testing.T) {
	cases := []struct {
		name  string
		rules []ContextRule
		want  string
	}{
		{
			name:  "empty name",
			rules: []ContextRule{{Name: " ", MinTokens: "0", MaxTokens: "1K"}},
			want:  "name cannot be empty",
		},
		{
			name: "duplicate name",
			rules: []ContextRule{
				{Name: "band", MinTokens: "0", MaxTokens: "1K"},
				{Name: "band", MinTokens: "2K", MaxTokens: "3K"},
			},
			want: "duplicate rule name",
		},
		{
			name:  "neither limit set",
			rules: []ContextRule{{Name: "band"}},
			want:  "min_tokens or max_tokens must be set",
		},
		{
			name:  "unparsable min_tokens",
			rules: []ContextRule{{Name: "band", MinTokens: "lots", MaxTokens: "1K"}},
			want:  "min_tokens: invalid token count format",
		},
		{
			name:  "unparsable max_tokens",
			rules: []ContextRule{{Name: "band", MinTokens: "0", MaxTokens: "1KB"}},
			want:  "max_tokens: invalid token count format",
		},
		{
			name:  "negative min_tokens",
			rules: []ContextRule{{Name: "band", MinTokens: "-1", MaxTokens: "1K"}},
			want:  "must not be negative",
		},
		{
			name:  "infinite max_tokens",
			rules: []ContextRule{{Name: "band", MinTokens: "0", MaxTokens: "inf"}},
			want:  "invalid token count format",
		},
		{
			name:  "int64 max does not wrap negative",
			rules: []ContextRule{{Name: "band", MinTokens: "9223372036854775807"}},
			want:  "too large",
		},
		{
			name:  "min above max",
			rules: []ContextRule{{Name: "band", MinTokens: "5K", MaxTokens: "1K"}},
			want:  "min_tokens (5K) must not exceed max_tokens (1K)",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateContextContracts(contextConfig(tc.rules...))
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.want)
			}
			if !strings.HasPrefix(err.Error(), "routing.signals.context[") {
				t.Fatalf("error %q is missing the routing.signals.context prefix", err.Error())
			}
		})
	}
}

func TestContextBoundsMatches(t *testing.T) {
	bounded := ContextBounds{Min: 10, Max: 20}
	open := ContextBounds{Min: 10, Unbounded: true}
	exact := ContextBounds{Min: 10, Max: 10}

	checks := []struct {
		bounds ContextBounds
		count  int
		want   bool
	}{
		{bounded, 9, false}, {bounded, 10, true}, {bounded, 20, true}, {bounded, 21, false},
		{open, 9, false}, {open, 10, true}, {open, 1 << 40, true},
		{exact, 9, false}, {exact, 10, true}, {exact, 11, false},
	}
	for _, c := range checks {
		if got := c.bounds.Matches(c.count); got != c.want {
			t.Fatalf("%+v.Matches(%d) = %v, want %v", c.bounds, c.count, got, c.want)
		}
	}
}

// TestParseYAMLLoadsOpenEndedContextBand covers the full YAML path: an
// open-ended final band loads, validates, and keeps its bounds.
func TestParseYAMLLoadsOpenEndedContextBand(t *testing.T) {
	const recipeYAML = `
recipes:
  - name: bands
    routing:
      signals:
        context:
          - name: short_context
            min_tokens: 0
            max_tokens: 8K
          - name: exact_context
            min_tokens: 1K
            max_tokens: 1K
          - name: overflow_context
            min_tokens: 8001
      decisions:
        - name: overflow_route
          rules:
            operator: AND
            conditions:
              - type: context
                name: overflow_context
          modelRefs:
            - model: model-b
              use_reasoning: false
`
	parsed, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + recipeYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	recipe, _ := parsed.RecipeByName("bands")
	if recipe == nil {
		t.Fatal("expected the bands recipe to load")
	}
	rules := recipe.Profile.Signals.ContextRules
	if len(rules) != 3 {
		t.Fatalf("expected 3 context rules, got %d", len(rules))
	}

	overflow := mustContextBounds(t, rules[2])
	if rules[2].MaxTokens.IsSet() || !overflow.Unbounded || overflow.Min != 8001 || !overflow.Matches(1<<40) {
		t.Fatalf("unexpected bounds for overflow band: %+v (max_tokens=%q)", overflow, rules[2].MaxTokens)
	}

	exact := mustContextBounds(t, rules[1])
	if !exact.Matches(1000) || exact.Matches(999) || exact.Matches(1001) {
		t.Fatalf("expected exact band to match only 1000, got %+v", exact)
	}
}

func mustContextBounds(t *testing.T, rule ContextRule) ContextBounds {
	t.Helper()
	bounds, err := rule.Bounds()
	if err != nil {
		t.Fatalf("unexpected bounds error for %q: %v", rule.Name, err)
	}
	return bounds
}

func TestContextBandIssues(t *testing.T) {
	bands := []NamedContextBand{
		{Name: "wide", Bounds: ContextBounds{Min: 16000, Unbounded: true}},
		{Name: "low", Bounds: ContextBounds{Min: 0, Max: 1000}},
		{Name: "mid", Bounds: ContextBounds{Min: 4000, Max: 20000}},
		{Name: "narrow", Bounds: ContextBounds{Min: 120001, Max: 240000}},
	}
	overlaps, gaps := ContextBandIssues(bands)

	if len(gaps) != 1 || gaps[0].From != 1001 || gaps[0].To != 3999 || gaps[0].Before.Name != "mid" {
		t.Fatalf("unexpected gaps: %+v", gaps)
	}

	got := map[string]bool{}
	for _, o := range overlaps {
		got[o.Outer.Name+"/"+o.Inner.Name] = o.Contains
	}
	want := map[string]bool{
		"mid/wide":    false, // partial: mid ends inside wide
		"wide/narrow": true,  // wide fully contains narrow
	}
	if len(got) != len(want) {
		t.Fatalf("unexpected overlaps: %+v", overlaps)
	}
	for key, contains := range want {
		if actual, ok := got[key]; !ok || actual != contains {
			t.Fatalf("overlap %s: got (present=%v, contains=%v), want contains=%v", key, ok, actual, contains)
		}
	}
}
