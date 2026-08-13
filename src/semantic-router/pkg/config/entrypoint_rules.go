package config

// PathMatchType names a supported path-matching strategy for an entrypoint
// rule. The v1 matcher surface intentionally excludes regex: it makes
// overlap validation and security review substantially harder.
type PathMatchType string

const (
	PathMatchExact  PathMatchType = "exact"
	PathMatchPrefix PathMatchType = "path_prefix"
)

// PathMatcher matches a normalized request path (query string and trailing
// slash removed). Exact matches the full path; PathPrefix is segment-aware,
// so "/v1" matches "/v1/responses" but not "/v10".
type PathMatcher struct {
	Type  PathMatchType
	Value string
}

// HeaderMatchType names a supported header-matching strategy. Exact is the
// only type in v1.
type HeaderMatchType string

const HeaderMatchExact HeaderMatchType = "exact"

// HeaderMatcher matches one trusted request header. Name comparison is
// case-insensitive (Name is stored lowercased by normalization); Value
// comparison is case-sensitive.
type HeaderMatcher struct {
	Name  string
	Type  HeaderMatchType
	Value string
}

// EntrypointMatch is one candidate condition for a rule: an optional path
// constraint AND every header constraint, all of which must hold. A zero
// value (nil Path, no Headers) is unconstrained and matches any request.
type EntrypointMatch struct {
	Path    *PathMatcher
	Headers []HeaderMatcher
}

// IsUnconstrained reports whether this match has no path or header
// conditions, so it matches every request.
func (m EntrypointMatch) IsUnconstrained() bool {
	return m.Path == nil && len(m.Headers) == 0
}

// EntrypointRule is one named candidate action within a conditional
// entrypoint: any of Matches (OR'd) selects Recipe. A rule with no Matches
// is an explicit unconditional catch-all.
type EntrypointRule struct {
	Name    string
	Matches []EntrypointMatch
	Recipe  RecipeName
}

// IsCatchAll reports whether this rule matches every request: either it
// declares no matches at all, or at least one of its declared matches is
// itself unconstrained.
func (r EntrypointRule) IsCatchAll() bool {
	if len(r.Matches) == 0 {
		return true
	}
	for _, m := range r.Matches {
		if m.IsUnconstrained() {
			return true
		}
	}
	return false
}

// entrypointRecipeNames returns every recipe name an entrypoint can route
// to, whether declared as the legacy single Recipe or via Rules. Startup
// resource discovery (ReachableRoutingRecipes) must walk this, not
// entrypoint.Recipe directly, or a recipe reachable only through rules
// would never get its backend models downloaded or classifiers provisioned.
func entrypointRecipeNames(e EntrypointMapping) []RecipeName {
	if len(e.Rules) == 0 {
		if e.Recipe == "" {
			return nil
		}
		return []RecipeName{e.Recipe}
	}
	names := make([]RecipeName, 0, len(e.Rules))
	seen := make(map[RecipeName]struct{}, len(e.Rules))
	for _, rule := range e.Rules {
		if _, ok := seen[rule.Recipe]; ok {
			continue
		}
		seen[rule.Recipe] = struct{}{}
		names = append(names, rule.Recipe)
	}
	return names
}
