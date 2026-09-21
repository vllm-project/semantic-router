package nlgen

import (
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

// isIdentPart returns true if ch is valid in a DSL identifier.
func isIdentPart(ch rune) bool {
	return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
		(ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.' || ch == '/'
}
