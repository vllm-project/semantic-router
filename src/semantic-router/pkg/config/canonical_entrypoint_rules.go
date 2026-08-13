package config

import (
	"fmt"
	"strings"
)

// CanonicalEntrypointRule is the wire shape of one conditional entrypoint
// rule: any of Matches (OR'd) selects Recipe.
type CanonicalEntrypointRule struct {
	Name    string                     `yaml:"name"`
	Matches []CanonicalEntrypointMatch `yaml:"matches,omitempty"`
	Recipe  string                     `yaml:"recipe"`
}

// CanonicalEntrypointMatch is the wire shape of one match condition: an
// optional path constraint AND every header constraint.
type CanonicalEntrypointMatch struct {
	Path    *CanonicalPathMatcher    `yaml:"path,omitempty"`
	Headers []CanonicalHeaderMatcher `yaml:"headers,omitempty"`
}

// CanonicalPathMatcher is the wire shape of a path matcher. Type is
// "exact" or "path_prefix".
type CanonicalPathMatcher struct {
	Type  string `yaml:"type"`
	Value string `yaml:"value"`
}

// CanonicalHeaderMatcher is the wire shape of a header matcher. Type
// defaults to "exact", the only supported value in v1.
type CanonicalHeaderMatcher struct {
	Name  string `yaml:"name"`
	Type  string `yaml:"type,omitempty"`
	Value string `yaml:"value"`
}

// normalizeCanonicalEntrypointRules validates and normalizes one
// entrypoint's rules block. entrypointIndex is only used for error context.
func normalizeCanonicalEntrypointRules(rules []CanonicalEntrypointRule, recipes []RoutingRecipe, entrypointIndex int) ([]EntrypointRule, error) {
	if len(rules) == 0 {
		return nil, fmt.Errorf("entrypoints[%d]: rules cannot be empty when recipe is not set", entrypointIndex)
	}

	seenNames := make(map[string]struct{}, len(rules))
	sawCatchAll := false
	normalized := make([]EntrypointRule, 0, len(rules))
	for ruleIndex, rule := range rules {
		name := strings.TrimSpace(rule.Name)
		if name == "" {
			return nil, fmt.Errorf("entrypoints[%d].rules[%d].name cannot be empty", entrypointIndex, ruleIndex)
		}
		if _, exists := seenNames[name]; exists {
			return nil, fmt.Errorf("entrypoints[%d].rules: duplicate rule name %q", entrypointIndex, name)
		}
		seenNames[name] = struct{}{}

		recipeName := RecipeName(strings.TrimSpace(rule.Recipe))
		if recipeName == "" {
			return nil, fmt.Errorf("entrypoints[%d].rules[%q].recipe cannot be empty", entrypointIndex, name)
		}
		if findRecipe(recipes, recipeName) == nil {
			return nil, fmt.Errorf("entrypoints[%d].rules[%q]: unknown recipe %q", entrypointIndex, name, recipeName)
		}

		matches, err := normalizeCanonicalEntrypointMatches(rule.Matches, entrypointIndex, name)
		if err != nil {
			return nil, err
		}

		normalizedRule := EntrypointRule{Name: name, Matches: matches, Recipe: recipeName}
		if normalizedRule.IsCatchAll() {
			if sawCatchAll {
				return nil, fmt.Errorf("entrypoints[%d]: multiple catch-all rules (rule %q matches every request)", entrypointIndex, name)
			}
			sawCatchAll = true
		}
		normalized = append(normalized, normalizedRule)
	}

	if err := validateEntrypointRuleAmbiguity(normalized); err != nil {
		return nil, fmt.Errorf("entrypoints[%d]: %w", entrypointIndex, err)
	}
	return normalized, nil
}

func normalizeCanonicalEntrypointMatches(matches []CanonicalEntrypointMatch, entrypointIndex int, ruleName string) ([]EntrypointMatch, error) {
	normalized := make([]EntrypointMatch, 0, len(matches))
	for matchIndex, m := range matches {
		var path *PathMatcher
		if m.Path != nil {
			p, err := normalizeCanonicalPathMatcher(*m.Path, entrypointIndex, ruleName, matchIndex)
			if err != nil {
				return nil, err
			}
			path = p
		}
		headers, err := normalizeCanonicalHeaderMatchers(m.Headers, entrypointIndex, ruleName, matchIndex)
		if err != nil {
			return nil, err
		}
		normalized = append(normalized, EntrypointMatch{Path: path, Headers: headers})
	}
	return normalized, nil
}

func normalizeCanonicalPathMatcher(p CanonicalPathMatcher, entrypointIndex int, ruleName string, matchIndex int) (*PathMatcher, error) {
	matchType := PathMatchType(strings.TrimSpace(p.Type))
	switch matchType {
	case PathMatchExact, PathMatchPrefix:
	default:
		return nil, fmt.Errorf("entrypoints[%d].rules[%q].matches[%d].path.type must be %q or %q, got %q",
			entrypointIndex, ruleName, matchIndex, PathMatchExact, PathMatchPrefix, p.Type)
	}
	value := strings.TrimSpace(p.Value)
	if !strings.HasPrefix(value, "/") {
		return nil, fmt.Errorf("entrypoints[%d].rules[%q].matches[%d].path.value must start with \"/\", got %q",
			entrypointIndex, ruleName, matchIndex, p.Value)
	}
	if strings.ContainsAny(value, "?#") {
		return nil, fmt.Errorf("entrypoints[%d].rules[%q].matches[%d].path.value must not contain a query string or fragment",
			entrypointIndex, ruleName, matchIndex)
	}
	value = NormalizeRequestPath(value)
	return &PathMatcher{Type: matchType, Value: value}, nil
}

func normalizeCanonicalHeaderMatchers(headers []CanonicalHeaderMatcher, entrypointIndex int, ruleName string, matchIndex int) ([]HeaderMatcher, error) {
	normalized := make([]HeaderMatcher, 0, len(headers))
	seen := make(map[string]struct{}, len(headers))
	for _, h := range headers {
		name := strings.ToLower(strings.TrimSpace(h.Name))
		if name == "" {
			return nil, fmt.Errorf("entrypoints[%d].rules[%q].matches[%d]: header name cannot be empty",
				entrypointIndex, ruleName, matchIndex)
		}
		if _, exists := seen[name]; exists {
			return nil, fmt.Errorf("entrypoints[%d].rules[%q].matches[%d]: duplicate header %q",
				entrypointIndex, ruleName, matchIndex, name)
		}
		seen[name] = struct{}{}

		matchType := HeaderMatchType(strings.TrimSpace(h.Type))
		if matchType == "" {
			matchType = HeaderMatchExact
		}
		if matchType != HeaderMatchExact {
			return nil, fmt.Errorf("entrypoints[%d].rules[%q].matches[%d].headers[%q].type must be %q, got %q",
				entrypointIndex, ruleName, matchIndex, name, HeaderMatchExact, h.Type)
		}
		normalized = append(normalized, HeaderMatcher{Name: name, Type: matchType, Value: h.Value})
	}
	return normalized, nil
}

// canonicalEntrypointRulesFromNormalized exports a normalized rule table
// back to its wire shape, for config export/round-trip.
func canonicalEntrypointRulesFromNormalized(rules []EntrypointRule) []CanonicalEntrypointRule {
	if len(rules) == 0 {
		return nil
	}
	out := make([]CanonicalEntrypointRule, 0, len(rules))
	for _, rule := range rules {
		out = append(out, CanonicalEntrypointRule{
			Name:    rule.Name,
			Matches: canonicalEntrypointMatchesFromNormalized(rule.Matches),
			Recipe:  string(rule.Recipe),
		})
	}
	return out
}

func canonicalEntrypointMatchesFromNormalized(matches []EntrypointMatch) []CanonicalEntrypointMatch {
	if len(matches) == 0 {
		return nil
	}
	out := make([]CanonicalEntrypointMatch, 0, len(matches))
	for _, m := range matches {
		var path *CanonicalPathMatcher
		if m.Path != nil {
			path = &CanonicalPathMatcher{Type: string(m.Path.Type), Value: m.Path.Value}
		}
		var headers []CanonicalHeaderMatcher
		if len(m.Headers) > 0 {
			headers = make([]CanonicalHeaderMatcher, 0, len(m.Headers))
			for _, h := range m.Headers {
				headers = append(headers, CanonicalHeaderMatcher{Name: h.Name, Type: string(h.Type), Value: h.Value})
			}
		}
		out = append(out, CanonicalEntrypointMatch{Path: path, Headers: headers})
	}
	return out
}
