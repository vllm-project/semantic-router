package handlers

import (
	"strings"
	"testing"
)

func TestRedactURLForLogMasksSensitiveQueryValues(t *testing.T) {
	t.Parallel()

	got := redactURLForLog("https://user:pass@example.com/path?api_key=abc&token=def&ok=yes")
	if strings.Contains(got, "abc") || strings.Contains(got, "def") || strings.Contains(got, "user:pass") {
		t.Fatalf("redactURLForLog leaked sensitive values: %s", got)
	}
	if !strings.Contains(got, "ok=yes") {
		t.Fatalf("redactURLForLog removed non-sensitive query value: %s", got)
	}
}

func TestRedactURLsForLogMasksURLsInsideErrors(t *testing.T) {
	t.Parallel()

	got := redactURLsForLog(`Get "https://example.com/a?access_token=secret&x=1": timeout`)
	if strings.Contains(got, "secret") {
		t.Fatalf("redactURLsForLog leaked sensitive values: %s", got)
	}
	if !strings.Contains(got, "x=1") {
		t.Fatalf("redactURLsForLog removed non-sensitive query value: %s", got)
	}
}
