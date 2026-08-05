package looper

import "strings"

func jsonObjectParseCandidates(content string) []string {
	candidate := stripMarkdownJSONFence(strings.TrimSpace(content))
	candidates := appendUniqueNonEmptyString(nil, candidate)

	extracted := extractJSONObject(candidate)
	candidates = appendUniqueNonEmptyString(candidates, extracted)
	candidates = appendUniqueNonEmptyString(candidates, repairLooseJSONObject(candidate))
	candidates = appendUniqueNonEmptyString(candidates, repairLooseJSONObject(extracted))
	return candidates
}

func appendUniqueNonEmptyString(candidates []string, value string) []string {
	value = strings.TrimSpace(value)
	if value == "" {
		return candidates
	}
	for _, existing := range candidates {
		if existing == value {
			return candidates
		}
	}
	return append(candidates, value)
}

func stripMarkdownJSONFence(value string) string {
	value = strings.TrimSpace(value)
	if !strings.HasPrefix(value, "```") {
		return value
	}
	lines := strings.Split(value, "\n")
	if len(lines) > 0 && strings.HasPrefix(strings.TrimSpace(lines[0]), "```") {
		lines = lines[1:]
	}
	if len(lines) > 0 && strings.HasPrefix(strings.TrimSpace(lines[len(lines)-1]), "```") {
		lines = lines[:len(lines)-1]
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func extractJSONObject(value string) string {
	start := strings.Index(value, "{")
	end := strings.LastIndex(value, "}")
	if start >= 0 && end > start {
		return strings.TrimSpace(value[start : end+1])
	}
	return strings.TrimSpace(value)
}

func repairLooseJSONObject(value string) string {
	return removeJSONTrailingCommas(repairInvalidJSONEscapes(repairJSONBackticks(value)))
}

func repairJSONBackticks(value string) string {
	var b strings.Builder
	b.Grow(len(value))
	inString := false
	escaped := false
	for _, r := range value {
		if r == '"' && !escaped {
			inString = !inString
		}
		if r == '`' && !inString {
			b.WriteRune('"')
		} else {
			b.WriteRune(r)
		}
		if r == '\\' && !escaped {
			escaped = true
		} else {
			escaped = false
		}
	}
	return b.String()
}

func removeJSONTrailingCommas(value string) string {
	var b strings.Builder
	b.Grow(len(value))
	inString := false
	escaped := false
	for index, r := range value {
		if r == '"' && !escaped {
			inString = !inString
		}
		if r == ',' && !inString {
			remaining := strings.TrimLeft(value[index+len(string(r)):], " \t\r\n")
			if strings.HasPrefix(remaining, "}") || strings.HasPrefix(remaining, "]") {
				continue
			}
		}
		b.WriteRune(r)
		if r == '\\' && !escaped {
			escaped = true
		} else {
			escaped = false
		}
	}
	return b.String()
}

func repairInvalidJSONEscapes(value string) string {
	var b strings.Builder
	b.Grow(len(value))
	inString := false
	escaped := false
	for i := 0; i < len(value); i++ {
		ch := value[i]
		if !inString {
			if ch == '"' {
				inString = true
			}
			b.WriteByte(ch)
			continue
		}
		if escaped {
			b.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '\\' {
			if i+1 < len(value) && validJSONEscape(value, i+1) {
				b.WriteByte(ch)
				escaped = true
			} else {
				b.WriteString(`\\`)
			}
			continue
		}
		if ch == '"' {
			inString = false
		}
		b.WriteByte(ch)
	}
	return b.String()
}

func validJSONEscape(value string, index int) bool {
	switch value[index] {
	case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
		return true
	case 'u':
		if index+4 >= len(value) {
			return false
		}
		for i := index + 1; i <= index+4; i++ {
			if !isHexByte(value[i]) {
				return false
			}
		}
		return true
	default:
		return false
	}
}

func isHexByte(ch byte) bool {
	return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F')
}
