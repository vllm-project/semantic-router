package config

import "testing"

func testResolverConfig() *RouterConfig {
	return &RouterConfig{
		Recipes: []RoutingRecipe{
			{Name: DefaultRecipeName},
			{Name: "recipe-a"},
			{Name: "recipe-b"},
		},
		Entrypoints: []EntrypointMapping{
			{ModelNames: []string{"Legacy"}, Recipe: "recipe-a"},
			{
				ModelNames: []string{"Auto"},
				Rules: []EntrypointRule{
					{
						Name:   "tenant-a-user-b",
						Recipe: "recipe-b",
						Matches: []EntrypointMatch{{
							Path:    exactPath("/v1/chat/completions"),
							Headers: []HeaderMatcher{header("x-authz-tenant-id", "A"), header("x-authz-user-id", "B")},
						}},
					},
					{
						Name:   "tenant-a-default",
						Recipe: "recipe-a",
						Matches: []EntrypointMatch{{
							Headers: []HeaderMatcher{header("x-authz-tenant-id", "A")},
						}},
					},
				},
			},
		},
	}
}

func TestResolveEntrypointUnclaimed(t *testing.T) {
	cfg := testResolverConfig()
	res := cfg.ResolveEntrypoint("not-an-entrypoint", MatchContext{})
	if res.Status != EntrypointUnclaimed {
		t.Errorf("status = %v, want EntrypointUnclaimed", res.Status)
	}
}

func TestResolveEntrypointLegacyMatchedRegardlessOfContext(t *testing.T) {
	cfg := testResolverConfig()
	// Legacy entrypoints ignore MatchContext entirely — same behavior for
	// any caller, matching the pre-rules contract exactly.
	res := cfg.ResolveEntrypoint("Legacy", MatchContext{Path: "/v1/responses", Headers: map[string]string{"x-authz-tenant-id": "Z"}})
	if res.Status != EntrypointMatched || res.Recipe == nil || res.Recipe.Name != "recipe-a" {
		t.Errorf("got %+v, want Matched/recipe-a", res)
	}
	if res.Rule != nil {
		t.Errorf("legacy entrypoint resolution should not report a Rule, got %+v", res.Rule)
	}
}

func TestResolveEntrypointConditionalMatched(t *testing.T) {
	cfg := testResolverConfig()
	res := cfg.ResolveEntrypoint("Auto", MatchContext{
		Path:    "/v1/chat/completions",
		Headers: map[string]string{"x-authz-tenant-id": "A", "x-authz-user-id": "B"},
	})
	if res.Status != EntrypointMatched || res.Recipe == nil || res.Recipe.Name != "recipe-b" {
		t.Fatalf("got %+v, want Matched/recipe-b (the more specific rule)", res)
	}
	if res.Rule == nil || res.Rule.Name != "tenant-a-user-b" {
		t.Errorf("got rule %+v, want tenant-a-user-b", res.Rule)
	}
}

func TestResolveEntrypointConditionalFallsBackToLessSpecificRule(t *testing.T) {
	cfg := testResolverConfig()
	// Same tenant, different user: the specific rule's path+user constraint
	// doesn't match, but the broader tenant-only rule does.
	res := cfg.ResolveEntrypoint("Auto", MatchContext{
		Path:    "/v1/chat/completions",
		Headers: map[string]string{"x-authz-tenant-id": "A", "x-authz-user-id": "someone-else"},
	})
	if res.Status != EntrypointMatched || res.Recipe == nil || res.Recipe.Name != "recipe-a" {
		t.Fatalf("got %+v, want Matched/recipe-a", res)
	}
}

func TestResolveEntrypointConditionalClaimedNoMatch(t *testing.T) {
	cfg := testResolverConfig()
	res := cfg.ResolveEntrypoint("Auto", MatchContext{
		Path:    "/v1/chat/completions",
		Headers: map[string]string{"x-authz-tenant-id": "some-other-tenant"},
	})
	if res.Status != EntrypointClaimedNoMatch {
		t.Errorf("status = %v, want EntrypointClaimedNoMatch", res.Status)
	}
	if res.Recipe != nil {
		t.Errorf("a denied resolution must never carry a recipe, got %+v", res.Recipe)
	}
}

func TestResolveEntrypointDefensiveAmbiguous(t *testing.T) {
	// A hand-built config that bypasses normalizeCanonicalEntrypointRules'
	// validation (which would reject this at load time) — proves the
	// resolver fails safe as Ambiguous rather than picking one by
	// declaration order if an invalid rule table is ever reached anyway.
	cfg := &RouterConfig{
		Recipes: []RoutingRecipe{{Name: "recipe-a"}, {Name: "recipe-b"}},
		Entrypoints: []EntrypointMapping{{
			ModelNames: []string{"Auto"},
			Rules: []EntrypointRule{
				{Name: "by-tenant", Recipe: "recipe-a", Matches: []EntrypointMatch{{Headers: []HeaderMatcher{header("x-authz-tenant-id", "A")}}}},
				{Name: "by-plan", Recipe: "recipe-b", Matches: []EntrypointMatch{{Headers: []HeaderMatcher{header("x-plan", "gold")}}}},
			},
		}},
	}
	res := cfg.ResolveEntrypoint("Auto", MatchContext{Headers: map[string]string{"x-authz-tenant-id": "A", "x-plan": "gold"}})
	if res.Status != EntrypointAmbiguous {
		t.Errorf("status = %v, want EntrypointAmbiguous", res.Status)
	}
	if res.Recipe != nil {
		t.Errorf("an ambiguous resolution must never carry a recipe, got %+v", res.Recipe)
	}
}

func TestResolveEntrypointHeaderNameCaseInsensitive(t *testing.T) {
	cfg := testResolverConfig()
	res := cfg.ResolveEntrypoint("Auto", MatchContext{
		Path:    "/v1/chat/completions",
		Headers: map[string]string{"X-Authz-Tenant-Id": "A", "X-AUTHZ-USER-ID": "B"},
	})
	if res.Status != EntrypointMatched {
		t.Errorf("header name matching must be case-insensitive, got status %v", res.Status)
	}
}

func TestResolveEntrypointHeaderValueCaseSensitive(t *testing.T) {
	cfg := testResolverConfig()
	res := cfg.ResolveEntrypoint("Auto", MatchContext{
		Path:    "/v1/chat/completions",
		Headers: map[string]string{"x-authz-tenant-id": "a", "x-authz-user-id": "B"}, // lowercase "a" != "A"
	})
	// Falls through to the tenant-only rule, which also requires "A" exactly
	// — so this should be ClaimedNoMatch, not silently matched.
	if res.Status != EntrypointClaimedNoMatch {
		t.Errorf("header value matching must be case-sensitive, got status %v", res.Status)
	}
}
