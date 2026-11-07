package config

import (
	"fmt"
	"slices"
	"strings"
)

// KeywordSourceToolResult selects textual tool results as the only content a
// keyword rule matches. It mirrors PIISourceToolResult.
const KeywordSourceToolResult = "tool_result"

var validKeywordSources = []string{"", KeywordSourceToolResult}

// validateKeywordContracts rejects source values the runtime does not
// understand, source/history combinations that contradict each other, and
// scored matching methods on tool results, which only regex rules inspect. An
// omitted source preserves the legacy user-message behavior.
func validateKeywordContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	for _, rule := range cfg.KeywordRules {
		if !slices.Contains(validKeywordSources, rule.Source) {
			return fmt.Errorf(
				"routing.signals.keywords %q: unknown source %q; valid values are %q or omitted",
				rule.Name,
				rule.Source,
				KeywordSourceToolResult,
			)
		}
		if rule.Source == KeywordSourceToolResult && rule.IncludeHistory {
			return fmt.Errorf(
				"routing.signals.keywords %q: source %q never includes the prompt or conversation history; remove include_history",
				rule.Name,
				KeywordSourceToolResult,
			)
		}
		if method := strings.ToLower(rule.Method); rule.Source == KeywordSourceToolResult && method != "" && method != "regex" {
			return fmt.Errorf(
				"routing.signals.keywords %q: source %q supports method regex only, got %q",
				rule.Name,
				KeywordSourceToolResult,
				rule.Method,
			)
		}
	}
	return nil
}
