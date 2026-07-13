package handlers

import (
	"net/url"
	"regexp"
	"strings"
)

var logURLPattern = regexp.MustCompile(`https?://[^\s"'<>]+`)

func redactURLForLog(rawURL string) string {
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil || parsed.Opaque != "" || parsed.Host == "" ||
		(!strings.EqualFold(parsed.Scheme, "http") && !strings.EqualFold(parsed.Scheme, "https")) {
		return "<invalid-url>"
	}

	return (&url.URL{
		Scheme: strings.ToLower(parsed.Scheme),
		Host:   parsed.Host,
	}).String()
}

func redactURLsForLog(text string) string {
	return logURLPattern.ReplaceAllStringFunc(text, redactURLForLog)
}
