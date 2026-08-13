package config

import "testing"

func exactPath(v string) *PathMatcher  { return &PathMatcher{Type: PathMatchExact, Value: v} }
func prefixPath(v string) *PathMatcher { return &PathMatcher{Type: PathMatchPrefix, Value: v} }
func header(name, value string) HeaderMatcher {
	return HeaderMatcher{Name: name, Type: HeaderMatchExact, Value: value}
}

func TestPathNarrowerOrEqual(t *testing.T) {
	cases := []struct {
		name         string
		narrow, wide *PathMatcher
		want         bool
	}{
		{"nil is broadest: anything narrower-or-equal to nil", exactPath("/v1/chat/completions"), nil, true},
		{"nil is not narrower than anything", nil, exactPath("/v1/chat/completions"), false},
		{"nil equal to nil", nil, nil, true},
		{"same exact path", exactPath("/v1/responses"), exactPath("/v1/responses"), true},
		{"different exact paths", exactPath("/v1/responses"), exactPath("/v1/messages"), false},
		{"exact under prefix", exactPath("/v1/chat/completions"), prefixPath("/v1"), true},
		// The segment-aware case the issue explicitly calls out: "/v1" must
		// match "/v1/responses" but never "/v10".
		{"exact NOT under a look-alike prefix (/v10 vs /v1)", exactPath("/v10/responses"), prefixPath("/v1"), false},
		{"prefix under wider prefix", prefixPath("/v1/chat"), prefixPath("/v1"), true},
		{"prefix not under a sibling prefix", prefixPath("/v1/chat"), prefixPath("/v2"), false},
		{"prefix look-alike segment rejected", prefixPath("/v10"), prefixPath("/v1"), false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := pathNarrowerOrEqual(c.narrow, c.wide); got != c.want {
				t.Errorf("pathNarrowerOrEqual(%v, %v) = %v, want %v", c.narrow, c.wide, got, c.want)
			}
		})
	}
}

func TestClauseDominates(t *testing.T) {
	tenantA := EntrypointMatch{Headers: []HeaderMatcher{header("x-authz-tenant-id", "A")}}
	tenantAUserB := EntrypointMatch{Headers: []HeaderMatcher{header("x-authz-tenant-id", "A"), header("x-authz-user-id", "B")}}
	catchAll := EntrypointMatch{}
	tenantAWithPath := EntrypointMatch{
		Path:    exactPath("/v1/responses"),
		Headers: []HeaderMatcher{header("x-authz-tenant-id", "A")},
	}
	tenantAWithPrefix := EntrypointMatch{
		Path:    prefixPath("/v1"),
		Headers: []HeaderMatcher{header("x-authz-tenant-id", "A")},
	}

	// The issue's own worked examples: tenant=A,user=B > tenant=A > catch-all;
	// Exact /v1/responses > PathPrefix /v1 > no path constraint.
	if !clauseDominates(tenantAUserB, tenantA) {
		t.Error("tenant=A,user=B should dominate tenant=A")
	}
	if clauseDominates(tenantA, tenantAUserB) {
		t.Error("tenant=A should not dominate tenant=A,user=B")
	}
	if !clauseDominates(tenantA, catchAll) {
		t.Error("tenant=A should dominate the catch-all")
	}
	if clauseDominates(catchAll, tenantA) {
		t.Error("the catch-all should not dominate tenant=A")
	}
	if !clauseDominates(tenantAWithPath, tenantAWithPrefix) {
		t.Error("Exact /v1/responses (same headers) should dominate PathPrefix /v1 (same headers)")
	}
	if clauseDominates(tenantAWithPrefix, tenantAWithPath) {
		t.Error("PathPrefix /v1 should not dominate the narrower Exact /v1/responses")
	}
	// Identical clauses: neither dominates the other.
	if clauseDominates(tenantA, tenantA) {
		t.Error("a clause must not dominate an identical clause")
	}
	// Incomparable: different header values for the same header name never
	// co-match a real request, so this case is tested via headersCompatible
	// instead of dominance (see TestValidateEntrypointRuleAmbiguity).
}

func TestValidateEntrypointRuleAmbiguity(t *testing.T) {
	t.Run("tenant+user dominates tenant-only: not ambiguous", func(t *testing.T) {
		rules := []EntrypointRule{
			{Name: "tenant-a-user-b", Recipe: "recipe-b", Matches: []EntrypointMatch{
				{Headers: []HeaderMatcher{header("x-authz-tenant-id", "A"), header("x-authz-user-id", "B")}},
			}},
			{Name: "tenant-a-default", Recipe: "recipe-a", Matches: []EntrypointMatch{
				{Headers: []HeaderMatcher{header("x-authz-tenant-id", "A")}},
			}},
		}
		if err := validateEntrypointRuleAmbiguity(rules); err != nil {
			t.Errorf("expected no ambiguity error, got: %v", err)
		}
	})

	t.Run("mutually exclusive exact paths: not ambiguous even with identical headers", func(t *testing.T) {
		rules := []EntrypointRule{
			{Name: "chat", Recipe: "recipe-a", Matches: []EntrypointMatch{
				{Path: exactPath("/v1/chat/completions"), Headers: []HeaderMatcher{header("x-authz-tenant-id", "A")}},
			}},
			{Name: "responses", Recipe: "recipe-b", Matches: []EntrypointMatch{
				{Path: exactPath("/v1/responses"), Headers: []HeaderMatcher{header("x-authz-tenant-id", "A")}},
			}},
		}
		if err := validateEntrypointRuleAmbiguity(rules); err != nil {
			t.Errorf("expected no ambiguity error (paths never co-match), got: %v", err)
		}
	})

	t.Run("same header required to different values: not ambiguous", func(t *testing.T) {
		rules := []EntrypointRule{
			{Name: "tenant-a", Recipe: "recipe-a", Matches: []EntrypointMatch{
				{Headers: []HeaderMatcher{header("x-authz-tenant-id", "A")}},
			}},
			{Name: "tenant-b", Recipe: "recipe-b", Matches: []EntrypointMatch{
				{Headers: []HeaderMatcher{header("x-authz-tenant-id", "B")}},
			}},
		}
		if err := validateEntrypointRuleAmbiguity(rules); err != nil {
			t.Errorf("expected no ambiguity error (values disagree, can't co-match), got: %v", err)
		}
	})

	t.Run("incomparable but compatible clauses: rejected as ambiguous", func(t *testing.T) {
		// Two different, non-overlapping header constraints with no path
		// constraint: a request carrying BOTH headers would satisfy both,
		// and neither dominates the other.
		rules := []EntrypointRule{
			{Name: "by-tenant", Recipe: "recipe-a", Matches: []EntrypointMatch{
				{Headers: []HeaderMatcher{header("x-authz-tenant-id", "A")}},
			}},
			{Name: "by-plan", Recipe: "recipe-b", Matches: []EntrypointMatch{
				{Headers: []HeaderMatcher{header("x-plan", "gold")}},
			}},
		}
		err := validateEntrypointRuleAmbiguity(rules)
		if err == nil {
			t.Fatal("expected an ambiguity error, got nil")
		}
	})

	t.Run("clauses within the same rule are never ambiguous with each other", func(t *testing.T) {
		rules := []EntrypointRule{
			{Name: "either-tenant", Recipe: "recipe-a", Matches: []EntrypointMatch{
				{Headers: []HeaderMatcher{header("x-authz-tenant-id", "A")}},
				{Headers: []HeaderMatcher{header("x-authz-tenant-id", "B")}},
			}},
		}
		if err := validateEntrypointRuleAmbiguity(rules); err != nil {
			t.Errorf("OR-composed matches within one rule must not be flagged ambiguous, got: %v", err)
		}
	})

	t.Run("two catch-all rules: rejected as ambiguous", func(t *testing.T) {
		rules := []EntrypointRule{
			{Name: "catch-all-1", Recipe: "recipe-a", Matches: nil},
			{Name: "catch-all-2", Recipe: "recipe-b", Matches: nil},
		}
		if err := validateEntrypointRuleAmbiguity(rules); err == nil {
			t.Error("expected an ambiguity error for two catch-all rules")
		}
	})
}
