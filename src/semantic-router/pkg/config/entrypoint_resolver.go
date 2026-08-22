package config

import "strings"

// EntrypointResolutionStatus is the outcome of resolving a request-facing
// model name plus caller context against the entrypoint table.
type EntrypointResolutionStatus uint8

const (
	// EntrypointUnclaimed means modelName is not an entrypoint alias at all.
	EntrypointUnclaimed EntrypointResolutionStatus = iota
	// EntrypointMatched means an entrypoint, rule, and recipe were selected.
	EntrypointMatched
	// EntrypointClaimedNoMatch means modelName is a real entrypoint alias,
	// but no rule permits this caller/path. This must NEVER become
	// passthrough or default-recipe routing — it is a denial, not a miss.
	EntrypointClaimedNoMatch
	// EntrypointAmbiguous means more than one maximally-specific rule
	// matched. Config validation rejects rule tables that can produce this;
	// it exists as a defense-in-depth status that must fail closed if
	// reached anyway, never resolved by declaration order.
	EntrypointAmbiguous
)

// MatchContext is the normalized caller/request context a conditional
// entrypoint's rules are evaluated against. Headers is intentionally
// map[string]string (single value, not multi-value): this mirrors
// RequestContext.Headers' real runtime shape exactly (extproc/request_context.go),
// so this type never has to represent header shapes the router can't
// actually supply.
type MatchContext struct {
	Path    string
	Headers map[string]string
}

// EntrypointResolution is the result of ResolveEntrypoint.
type EntrypointResolution struct {
	Status EntrypointResolutionStatus
	Recipe *RoutingRecipe
	// Rule is the matched rule for a conditional entrypoint, nil for a
	// legacy entrypoint or any non-Matched status. Present for debug
	// logging and future audit/replay use.
	Rule *EntrypointRule
}

// ResolveEntrypoint resolves a request-facing model name against the
// entrypoint table using the full caller/request context. Unlike
// RecipeForRequestModel (which cannot express denial), this distinguishes
// "not an entrypoint" from "a real entrypoint alias this caller may not
// use" — the second case must surface as EntrypointClaimedNoMatch, which
// callers must treat as a hard denial, never a fallback.
func (c *RouterConfig) ResolveEntrypoint(modelName string, ctx MatchContext) EntrypointResolution {
	entrypoint, ok := c.EntrypointByModelName(modelName)
	if !ok {
		return EntrypointResolution{Status: EntrypointUnclaimed}
	}

	if len(entrypoint.Rules) == 0 {
		recipe, found := c.RecipeByName(entrypoint.Recipe)
		if !found {
			// Unreachable once config validation requires every legacy
			// entrypoint to reference a known recipe; fail closed rather
			// than assume, in case this is ever reached anyway.
			return EntrypointResolution{Status: EntrypointClaimedNoMatch}
		}
		return EntrypointResolution{Status: EntrypointMatched, Recipe: recipe}
	}

	var matched []entrypointRuleClause
	for i := range entrypoint.Rules {
		rule := &entrypoint.Rules[i]
		if len(rule.Matches) == 0 {
			matched = append(matched, entrypointRuleClause{rule: rule, match: EntrypointMatch{}})
			continue
		}
		for _, m := range rule.Matches {
			if matchClause(m, ctx) {
				matched = append(matched, entrypointRuleClause{rule: rule, match: m})
				break
			}
		}
	}
	if len(matched) == 0 {
		return EntrypointResolution{Status: EntrypointClaimedNoMatch}
	}

	winners := maximalEntrypointClauses(matched)
	if len(winners) != 1 {
		return EntrypointResolution{Status: EntrypointAmbiguous}
	}
	winner := winners[0]
	recipe, ok := c.RecipeByName(winner.rule.Recipe)
	if !ok {
		return EntrypointResolution{Status: EntrypointClaimedNoMatch}
	}
	return EntrypointResolution{Status: EntrypointMatched, Recipe: recipe, Rule: winner.rule}
}

// matchClause reports whether a match's path (if any) AND every one of its
// headers are satisfied by ctx.
func matchClause(m EntrypointMatch, ctx MatchContext) bool {
	if m.Path != nil && !matchPath(*m.Path, ctx.Path) {
		return false
	}
	for _, h := range m.Headers {
		if !matchHeader(h, ctx.Headers) {
			return false
		}
	}
	return true
}

func matchPath(m PathMatcher, path string) bool {
	switch m.Type {
	case PathMatchExact:
		return path == m.Value
	case PathMatchPrefix:
		return pathIsSegmentPrefixOf(pathSegments(m.Value), pathSegments(path))
	default:
		return false
	}
}

// matchHeader looks up h.Name case-insensitively in headers (single-value
// map, last-write-wins at ingestion — see MatchContext) and compares the
// value case-sensitively.
func matchHeader(h HeaderMatcher, headers map[string]string) bool {
	for name, value := range headers {
		if strings.EqualFold(name, h.Name) {
			return value == h.Value
		}
	}
	return false
}

// NormalizeRequestPath strips the query string and any trailing slash
// (except for the root path itself). pkg/config cannot import pkg/extproc's
// equivalent (processor_req_header_validation.go normalizeRequestPath) since
// the dependency runs the other way; this is a small, deliberate duplication
// so entrypoint rule path matching has a normalization function of its own.
func NormalizeRequestPath(path string) string {
	if idx := strings.IndexByte(path, '?'); idx >= 0 {
		path = path[:idx]
	}
	if len(path) > 1 {
		path = strings.TrimRight(path, "/")
	}
	return path
}
