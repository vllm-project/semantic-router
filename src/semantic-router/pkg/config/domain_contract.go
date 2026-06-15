package config

import (
	"slices"
	"strings"
)

var supportedRoutingDomains = []string{
	"biology",
	"business",
	"chemistry",
	"computer science",
	"economics",
	"engineering",
	"health",
	"history",
	"law",
	"math",
	"other",
	"philosophy",
	"physics",
	"psychology",
}

// SupportedRoutingDomainNames returns the repo-supported classifier domain set.
func SupportedRoutingDomainNames() []string {
	return append([]string(nil), supportedRoutingDomains...)
}

// IsSupportedRoutingDomainName reports whether name is one of the supported
// domain labels emitted by the repository's domain classifier.
func IsSupportedRoutingDomainName(name string) bool {
	return slices.Contains(supportedRoutingDomains, strings.TrimSpace(name))
}

// SuggestSupportedRoutingDomainName suggests the canonical supported label when
// the input is close but not exact, for example "computer_science".
func SuggestSupportedRoutingDomainName(name string) string {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return ""
	}

	if strings.Contains(trimmed, "_") {
		candidate := strings.ReplaceAll(trimmed, "_", " ")
		if IsSupportedRoutingDomainName(candidate) {
			return candidate
		}
	}

	best := ""
	bestDistance := len(trimmed) + 1
	for _, candidate := range supportedRoutingDomains {
		distance := levenshtein(strings.ToLower(trimmed), strings.ToLower(candidate))
		if distance < bestDistance && distance <= len(candidate)/2+1 {
			best = candidate
			bestDistance = distance
		}
	}
	return best
}

func levenshtein(a, b string) int {
	la, lb := len(a), len(b)
	if la == 0 {
		return lb
	}
	if lb == 0 {
		return la
	}

	prev := make([]int, lb+1)
	curr := make([]int, lb+1)
	for j := 0; j <= lb; j++ {
		prev[j] = j
	}

	for i := 1; i <= la; i++ {
		curr[0] = i
		for j := 1; j <= lb; j++ {
			cost := 0
			if a[i-1] != b[j-1] {
				cost = 1
			}
			curr[j] = min3(
				prev[j]+1,
				curr[j-1]+1,
				prev[j-1]+cost,
			)
		}
		prev, curr = curr, prev
	}

	return prev[lb]
}

func min3(a, b, c int) int {
	if a < b {
		if a < c {
			return a
		}
		return c
	}
	if b < c {
		return b
	}
	return c
}
