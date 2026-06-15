package handlers

import (
	"net/url"
	"regexp"
	"strings"
)

var logURLPattern = regexp.MustCompile(`https?://[^\s"'<>]+`)

func redactURLForLog(rawURL string) string {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return "<invalid-url>"
	}

	if parsed.User != nil {
		parsed.User = url.User("redacted")
	}

	query := parsed.Query()
	for key := range query {
		if isSensitiveURLParam(key) {
			query.Set(key, "***")
		}
	}
	parsed.RawQuery = query.Encode()
	return parsed.String()
}

func redactURLsForLog(text string) string {
	return logURLPattern.ReplaceAllStringFunc(text, redactURLForLog)
}

func isSensitiveURLParam(key string) bool {
	normalized := strings.ToLower(strings.TrimSpace(key))
	if normalized == "" {
		return false
	}

	for _, marker := range []string{
		"api_key",
		"apikey",
		"access_token",
		"auth",
		"authorization",
		"bearer",
		"credential",
		"key",
		"password",
		"secret",
		"sig",
		"signature",
		"token",
	} {
		if strings.Contains(normalized, marker) {
			return true
		}
	}
	return false
}
