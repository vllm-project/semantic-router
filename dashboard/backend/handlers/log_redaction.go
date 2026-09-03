package handlers

import (
	"net/url"
	"regexp"
	"strings"
)

var (
	logURLPattern                = regexp.MustCompile(`[A-Za-z][A-Za-z0-9+.-]*://[^\s"'<>]+`)
	runtimeLogCredentialPatterns = []struct {
		pattern     *regexp.Regexp
		replacement string
	}{
		{
			pattern:     regexp.MustCompile(`(?i)(\b(authorization|proxy-authorization|cookie|set-cookie)[ \t]*:[ \t]*)[^\r\n]*`),
			replacement: `$1[REDACTED]`,
		},
		{
			pattern:     regexp.MustCompile(`(?i)\b(bearer|basic)[ \t]+[A-Za-z0-9._~+/=-]{4,}`),
			replacement: `$1 [REDACTED]`,
		},
		{
			pattern: regexp.MustCompile(
				`(?i)(["']?(authorization|proxy-authorization|cookie|set-cookie|([A-Za-z0-9]+[_-])*(api[_-]?key|token|secret|password|passwd|credential))["']?[ \t]*[:=][ \t]*)("[^"\r\n]*"|'[^'\r\n]*'|[^\s,;]+)`,
			),
			replacement: `$1[REDACTED]`,
		},
		{
			pattern:     regexp.MustCompile(`\b(sk|rk|pk)-[A-Za-z0-9_-]{16,}\b`),
			replacement: `[REDACTED API KEY]`,
		},
		{
			pattern:     regexp.MustCompile(`\b(AKIA|ASIA)[A-Z0-9]{16}\b`),
			replacement: `[REDACTED ACCESS KEY]`,
		},
		{
			pattern:     regexp.MustCompile(`\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b`),
			replacement: `[REDACTED JWT]`,
		},
	}
	encodedSecretLinePattern = regexp.MustCompile(`^[A-Za-z0-9+/=]{64,}$`)
)

func redactURLForLog(rawURL string) string {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return "<invalid-url>"
	}

	parsed.User = nil
	if parsed.EscapedPath() != "" && parsed.EscapedPath() != "/" {
		parsed.Path = "/[REDACTED]"
		parsed.RawPath = ""
	}
	if parsed.RawQuery != "" || parsed.ForceQuery {
		parsed.RawQuery = "REDACTED"
		parsed.ForceQuery = false
	}
	if parsed.Fragment != "" || parsed.RawFragment != "" {
		parsed.Fragment = "REDACTED"
		parsed.RawFragment = ""
	}
	return parsed.String()
}

func redactURLsForLog(text string) string {
	return logURLPattern.ReplaceAllStringFunc(text, redactURLForLog)
}

// redactRuntimeLogLines is the final confidentiality boundary before raw
// producer output reaches a logs.read caller. Source loggers must still avoid
// content, but this catches common credential shapes and encoded key blocks.
func redactRuntimeLogLines(lines []string) []string {
	redacted := make([]string, 0, len(lines))
	insidePrivateKey := false
	for _, line := range lines {
		upper := strings.ToUpper(line)
		if strings.Contains(upper, "-----BEGIN") && strings.Contains(upper, "PRIVATE KEY-----") {
			insidePrivateKey = true
			redacted = append(redacted, "[REDACTED PRIVATE KEY]")
			continue
		}
		if insidePrivateKey {
			if strings.Contains(upper, "-----END") && strings.Contains(upper, "PRIVATE KEY-----") {
				insidePrivateKey = false
			}
			continue
		}
		trimmed := strings.TrimSpace(line)
		if encodedSecretLinePattern.MatchString(trimmed) {
			redacted = append(redacted, "[REDACTED ENCODED VALUE]")
			continue
		}
		line = redactURLsForLog(line)
		for _, rule := range runtimeLogCredentialPatterns {
			line = rule.pattern.ReplaceAllString(line, rule.replacement)
		}
		redacted = append(redacted, line)
	}
	return redacted
}
