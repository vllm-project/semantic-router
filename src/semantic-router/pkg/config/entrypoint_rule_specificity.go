package config

import (
	"fmt"
	"strings"
)

// entrypointRuleClause pairs one rule with one of its matches (or, for a
// rule declared with no matches at all, an implicit unconstrained match) so
// dominance and ambiguity can be computed uniformly per-clause rather than
// per-rule. Multiple clauses from the same rule share one recipe, so they
// are never compared against each other — declaring two overlapping
// `matches` entries under one rule is intentional OR composition, not
// ambiguity.
type entrypointRuleClause struct {
	rule  *EntrypointRule
	match EntrypointMatch
}

func flattenEntrypointRuleClauses(rules []EntrypointRule) []entrypointRuleClause {
	clauses := make([]entrypointRuleClause, 0, len(rules))
	for i := range rules {
		rule := &rules[i]
		if len(rule.Matches) == 0 {
			clauses = append(clauses, entrypointRuleClause{rule: rule, match: EntrypointMatch{}})
			continue
		}
		for _, m := range rule.Matches {
			clauses = append(clauses, entrypointRuleClause{rule: rule, match: m})
		}
	}
	return clauses
}

// pathSegments splits a normalized path ("/v1/chat/completions") into its
// non-empty segments (["v1", "chat", "completions"]).
func pathSegments(path string) []string {
	trimmed := strings.Trim(path, "/")
	if trimmed == "" {
		return nil
	}
	return strings.Split(trimmed, "/")
}

// pathIsSegmentPrefixOf reports whether every segment of prefix appears, in
// order, as a leading run of segments in path. This is what makes
// PathPrefix "/v1" match "/v1/responses" but not "/v10" — string-prefix
// comparison alone would wrongly match the latter.
func pathIsSegmentPrefixOf(prefix, path []string) bool {
	if len(prefix) > len(path) {
		return false
	}
	for i, seg := range prefix {
		if path[i] != seg {
			return false
		}
	}
	return true
}

// pathNarrowerOrEqual reports whether narrow's constraint is at least as
// specific as wide's: every path satisfying narrow also satisfies wide.
// A nil matcher is the broadest possible constraint (no constraint at all).
func pathNarrowerOrEqual(narrow, wide *PathMatcher) bool {
	if wide == nil {
		return true
	}
	if narrow == nil {
		return false
	}
	switch wide.Type {
	case PathMatchExact:
		return narrow.Type == PathMatchExact && narrow.Value == wide.Value
	case PathMatchPrefix:
		wideSegs := pathSegments(wide.Value)
		switch narrow.Type {
		case PathMatchExact:
			return pathIsSegmentPrefixOf(wideSegs, pathSegments(narrow.Value))
		case PathMatchPrefix:
			return pathIsSegmentPrefixOf(wideSegs, pathSegments(narrow.Value))
		}
	}
	return false
}

// pathCompatible reports whether some real request path could satisfy both
// matchers simultaneously.
func pathCompatible(a, b *PathMatcher) bool {
	if a == nil || b == nil {
		return true
	}
	return pathNarrowerOrEqual(a, b) || pathNarrowerOrEqual(b, a)
}

func headerMap(headers []HeaderMatcher) map[string]string {
	m := make(map[string]string, len(headers))
	for _, h := range headers {
		m[strings.ToLower(h.Name)] = h.Value
	}
	return m
}

// headersCompatible reports whether some real request could satisfy both
// header sets simultaneously: every header name required by both sides must
// require the same value.
func headersCompatible(a, b []HeaderMatcher) bool {
	am, bm := headerMap(a), headerMap(b)
	for name, av := range am {
		if bv, ok := bm[name]; ok && bv != av {
			return false
		}
	}
	return true
}

// headersSupersetSameValues reports whether wider's header constraints are a
// superset of narrower's, agreeing on every shared value.
func headersSupersetSameValues(wider, narrower []HeaderMatcher) bool {
	wm := headerMap(wider)
	for name, v := range headerMap(narrower) {
		wv, ok := wm[name]
		if !ok || wv != v {
			return false
		}
	}
	return true
}

// clauseDominates reports whether a is strictly more specific than b: a
// matches a subset of what b matches, and every request satisfying a also
// satisfies b. Per the issue's specificity contract, dominance requires
// a's constraints to be a superset of b's in both dimensions (headers and
// path), with at least one dimension strictly narrower — this is what
// makes "tenant=A,user=B" dominate "tenant=A" dominate a catch-all, and
// "Exact /v1/responses" dominate "PathPrefix /v1" dominate no constraint.
func clauseDominates(a, b EntrypointMatch) bool {
	if !headersSupersetSameValues(a.Headers, b.Headers) {
		return false
	}
	if !pathNarrowerOrEqual(a.Path, b.Path) {
		return false
	}
	strictlyNarrower := len(a.Headers) > len(b.Headers)
	if !strictlyNarrower && a.Path != nil && b.Path == nil {
		strictlyNarrower = true
	}
	if !strictlyNarrower && a.Path != nil && b.Path != nil {
		strictlyNarrower = *a.Path != *b.Path
	}
	return strictlyNarrower
}

// maximalEntrypointClauses reduces a set of matched clauses to the ones no
// other matched clause dominates. Exactly one remaining clause means the
// request has one unambiguous winner; zero or more than one is a config
// invariant violation (rejected at validate time; must fail safe as
// EntrypointAmbiguous, never resolved by declaration order, if it is
// somehow reached at request time despite validation).
func maximalEntrypointClauses(clauses []entrypointRuleClause) []entrypointRuleClause {
	maximal := make([]entrypointRuleClause, 0, len(clauses))
	for i, c := range clauses {
		dominated := false
		for j, other := range clauses {
			if i == j {
				continue
			}
			if clauseDominates(other.match, c.match) {
				dominated = true
				break
			}
		}
		if !dominated {
			maximal = append(maximal, c)
		}
	}
	return maximal
}

// validateEntrypointRuleAmbiguity rejects a rule table containing two
// clauses from different rules that could both match some real request
// without either dominating the other — the config-validation-time half of
// the specificity contract, so ambiguity is rejected at load rather than
// resolved by declaration order at request time.
func validateEntrypointRuleAmbiguity(rules []EntrypointRule) error {
	clauses := flattenEntrypointRuleClauses(rules)
	for i, a := range clauses {
		for _, b := range clauses[i+1:] {
			if a.rule == b.rule {
				continue
			}
			if !pathCompatible(a.match.Path, b.match.Path) {
				continue
			}
			if !headersCompatible(a.match.Headers, b.match.Headers) {
				continue
			}
			if clauseDominates(a.match, b.match) || clauseDominates(b.match, a.match) {
				continue
			}
			return fmt.Errorf("rules %q and %q are ambiguous for overlapping requests", a.rule.Name, b.rule.Name)
		}
	}
	return nil
}
