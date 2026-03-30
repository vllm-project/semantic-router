package dsl

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

	// Strip markdown code fences: ```dsl ... ```, ```text ... ```, or bare ``` ... ```
	if extracted := extractFromCodeFence(raw); extracted != "" {
		raw = extracted
	}

	// Find the first top-level DSL keyword and trim leading prose.
	if idx := findFirstKeyword(raw); idx > 0 {
		raw = raw[idx:]
	}

	// Trim trailing prose after the last closing brace at depth 0.
	if idx := findLastTopLevelClose(raw); idx >= 0 && idx < len(raw)-1 {
		raw = raw[:idx+1]
	}

	return strings.TrimSpace(raw)
}

func extractFromCodeFence(s string) string {
	// Find opening fence: ``` optionally followed by a language tag
	fenceStart := -1
	for i := 0; i < len(s)-2; i++ {
		if s[i] == '`' && s[i+1] == '`' && s[i+2] == '`' {
			fenceStart = i
			break
		}
	}
	if fenceStart < 0 {
		return ""
	}

	// Skip past the opening ``` and any language tag on the same line
	contentStart := fenceStart + 3
	for contentStart < len(s) && s[contentStart] != '\n' {
		contentStart++
	}
	if contentStart < len(s) {
		contentStart++ // skip the newline
	}

	// Find closing fence
	fenceEnd := -1
	for i := contentStart; i < len(s)-2; i++ {
		if s[i] == '`' && s[i+1] == '`' && s[i+2] == '`' {
			fenceEnd = i
			break
		}
	}

	if fenceEnd < 0 {
		return strings.TrimSpace(s[contentStart:])
	}
	return strings.TrimSpace(s[contentStart:fenceEnd])
}

func findFirstKeyword(s string) int {
	best := -1
	for _, kw := range dslTopLevelKeywords {
		idx := strings.Index(s, kw)
		if idx < 0 {
			continue
		}
		// Verify word boundary: not preceded by an ident char
		if idx > 0 && isIdentPart(rune(s[idx-1])) {
			continue
		}
		// Not followed by an ident char (except for identifiers that start the next token)
		end := idx + len(kw)
		if end < len(s) && isIdentPart(rune(s[end])) {
			continue
		}
		if best < 0 || idx < best {
			best = idx
		}
	}
	return best
}

func findLastTopLevelClose(s string) int {
	depth := 0
	lastClose := -1
	inString := false
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if ch == '"' && !inString {
			inString = true
			continue
		}
		if inString {
			if ch == '\\' {
				i++
				continue
			}
			if ch == '"' {
				inString = false
			}
			continue
		}
		if ch == '#' {
			for i < len(s) && s[i] != '\n' {
				i++
			}
			continue
		}
		if ch == '{' {
			depth++
		}
		if ch == '}' {
			depth--
			if depth == 0 {
				lastClose = i
			}
		}
	}
	return lastClose
}
