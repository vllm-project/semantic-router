package nlgen

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// dslTopLevelKeywords are the keywords that can begin a top-level DSL block.
var dslTopLevelKeywords = []string{
	"DECISION_TREE",
	"PROJECTION",
	"SIGNAL",
	"ROUTE",
	"MODEL",
	"PLUGIN",
	"TEST",
}

var (
	routeLanguageConditionPattern = regexp.MustCompile(`(?m)^([ \t]*)(?:CONDITION|WHEN)\s+"language\s*[:=]\s*([A-Za-z0-9_.-]+)"\s*$`)
	conditionKeywordPattern       = regexp.MustCompile(`(?m)^([ \t]*)CONDITION\b`)
	languageSignalDeclPattern     = regexp.MustCompile(`(?m)^\s*SIGNAL\s+language\s+(?:"([^"]+)"|([A-Za-z_][A-Za-z0-9_-]*))\b`)
	languageSignalRefPattern      = regexp.MustCompile(`language\("([^"]+)"\)`)
	firstRoutePattern             = regexp.MustCompile(`(?m)^ROUTE\b`)
	dslIdentifierPattern          = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_-]*$`)
)

// SanitizeLLMOutput extracts valid DSL text from raw LLM output.
// It strips markdown code fences, leading prose, and trailing commentary,
// returning only the DSL program text suitable for parsing.
func SanitizeLLMOutput(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}

	if extracted := extractFromCodeFence(raw); extracted != "" {
		raw = extracted
	}

	if idx := findFirstKeyword(raw); idx > 0 {
		raw = raw[idx:]
	}

	if idx := findLastTopLevelClose(raw); idx >= 0 && idx < len(raw)-1 {
		raw = raw[:idx+1]
	}

	return strings.TrimSpace(raw)
}

// NormalizeGeneratedDSL repairs common LLM mistakes while keeping the shared
// NL-to-DSL pipeline prompt/repair behavior canonical across callers.
func NormalizeGeneratedDSL(source string) (string, []string) {
	normalized := strings.TrimSpace(source)
	if normalized == "" {
		return source, nil
	}

	var notes []string
	if routeLanguageConditionPattern.MatchString(normalized) {
		normalized = routeLanguageConditionPattern.ReplaceAllString(normalized, `${1}WHEN language("${2}")`)
		notes = append(notes, `normalized: rewrote quoted language CONDITION clauses into valid WHEN language("...") guards.`)
	}
	if conditionKeywordPattern.MatchString(normalized) {
		normalized = conditionKeywordPattern.ReplaceAllString(normalized, `${1}WHEN`)
		notes = append(notes, "normalized: rewrote CONDITION into WHEN for route predicates.")
	}

	normalized, missingSignals := ensureLanguageSignals(normalized)
	if len(missingSignals) > 0 {
		notes = append(notes, fmt.Sprintf("normalized: added %d missing SIGNAL language declaration(s) to match generated WHEN clauses.", len(missingSignals)))
	}

	if strings.TrimSpace(normalized) == strings.TrimSpace(source) {
		return source, nil
	}
	return normalized, uniqueNotes(notes)
}

// findTripleBacktick returns the index of the first "```" at or after pos, or -1.
func findTripleBacktick(s string, pos int) int {
	for i := pos; i < len(s)-2; i++ {
		if s[i] == '`' && s[i+1] == '`' && s[i+2] == '`' {
			return i
		}
	}
	return -1
}

func extractFromCodeFence(s string) string {
	fenceStart := findTripleBacktick(s, 0)
	if fenceStart < 0 {
		return ""
	}

	contentStart := fenceStart + 3
	if nl := strings.IndexByte(s[contentStart:], '\n'); nl >= 0 {
		contentStart += nl + 1
	} else {
		contentStart = len(s)
	}

	fenceEnd := findTripleBacktick(s, contentStart)
	if fenceEnd < 0 {
		return strings.TrimSpace(s[contentStart:])
	}
	return strings.TrimSpace(s[contentStart:fenceEnd])
}

// skipQuotedOrComment advances index i past a quoted string or line comment.
// Returns the new index and true if something was skipped.
func skipQuotedOrComment(s string, i int) (int, bool) {
	ch := s[i]
	if ch == '"' {
		i++
		for i < len(s) {
			if s[i] == '\\' {
				i += 2
				continue
			}
			if s[i] == '"' {
				return i + 1, true
			}
			i++
		}
		return i, true
	}
	if ch == '#' {
		for i < len(s) && s[i] != '\n' {
			i++
		}
		return i, true
	}
	return i, false
}

// matchKeywordAt checks whether a top-level DSL keyword starts at position i
// as a whole word. Returns true if matched.
func matchKeywordAt(s string, i int) bool {
	for _, kw := range dslTopLevelKeywords {
		if i+len(kw) > len(s) || s[i:i+len(kw)] != kw {
			continue
		}
		if i > 0 && isIdentPart(rune(s[i-1])) {
			continue
		}
		if end := i + len(kw); end < len(s) && isIdentPart(rune(s[end])) {
			continue
		}
		return true
	}
	return false
}

// findFirstKeyword scans s for the first occurrence of a top-level DSL keyword
// that is NOT inside a quoted string or comment.
func findFirstKeyword(s string) int {
	for i := 0; i < len(s); {
		if next, skipped := skipQuotedOrComment(s, i); skipped {
			i = next
			continue
		}
		if matchKeywordAt(s, i) {
			return i
		}
		i++
	}
	return -1
}

func findLastTopLevelClose(s string) int {
	depth := 0
	lastClose := -1

	for i := 0; i < len(s); {
		if next, skipped := skipQuotedOrComment(s, i); skipped {
			i = next
			continue
		}
		switch s[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				lastClose = i
			}
		}
		i++
	}
	return lastClose
}

func ensureLanguageSignals(source string) (string, []string) {
	existingSignals := make(map[string]struct{})
	for _, match := range languageSignalDeclPattern.FindAllStringSubmatch(source, -1) {
		name := strings.TrimSpace(match[1])
		if name == "" {
			name = strings.TrimSpace(match[2])
		}
		if name == "" {
			continue
		}
		existingSignals[strings.ToLower(name)] = struct{}{}
	}

	var missing []string
	seenMissing := make(map[string]struct{})
	for _, match := range languageSignalRefPattern.FindAllStringSubmatch(source, -1) {
		name := strings.TrimSpace(match[1])
		if name == "" {
			continue
		}
		key := strings.ToLower(name)
		if _, ok := existingSignals[key]; ok {
			continue
		}
		if _, ok := seenMissing[key]; ok {
			continue
		}
		seenMissing[key] = struct{}{}
		missing = append(missing, name)
	}
	if len(missing) == 0 {
		return source, nil
	}

	declarations := make([]string, 0, len(missing))
	for _, name := range missing {
		declarations = append(declarations, fmt.Sprintf(`SIGNAL language %s { description: %q }`, dslName(name), languageDescription(name)))
	}
	block := strings.Join(declarations, "\n\n")

	if routeIndex := firstRoutePattern.FindStringIndex(source); routeIndex != nil {
		prefix := strings.TrimRight(source[:routeIndex[0]], "\n")
		suffix := strings.TrimLeft(source[routeIndex[0]:], "\n")
		if prefix == "" {
			return block + "\n\n" + suffix, missing
		}
		return prefix + "\n\n" + block + "\n\n" + suffix, missing
	}
	return strings.TrimRight(source, "\n") + "\n\n" + block + "\n", missing
}

func dslName(name string) string {
	if dslIdentifierPattern.MatchString(name) {
		return name
	}
	return strconv.Quote(name)
}

func languageDescription(name string) string {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "zh", "zh-cn", "zh-hans", "zh-hant", "chinese":
		return "Chinese prompts"
	case "en", "en-us", "en-gb", "english":
		return "English prompts"
	default:
		return fmt.Sprintf("%s prompts", strings.TrimSpace(name))
	}
}

func uniqueNotes(notes []string) []string {
	if len(notes) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(notes))
	var result []string
	for _, note := range notes {
		note = strings.TrimSpace(note)
		if note == "" {
			continue
		}
		if _, ok := seen[note]; ok {
			continue
		}
		seen[note] = struct{}{}
		result = append(result, note)
	}
	return result
}

// isIdentPart returns true if ch is valid in a DSL identifier.
func isIdentPart(ch rune) bool {
	return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
		(ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.' || ch == '/'
}
