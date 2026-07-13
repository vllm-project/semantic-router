package handlers

import (
	"strings"
	"testing"
)

func TestRedactURLForLogKeepsOnlyOrigin(t *testing.T) {
	t.Parallel()

	got := redactURLForLog("https://user:pass@example.com/path?api_key=abc&token=def&ok=yes#fragment")
	if got != "https://example.com" {
		t.Fatalf("redactURLForLog() = %q, want origin only", got)
	}
}

func TestRedactURLsForLogMasksURLsInsideErrors(t *testing.T) {
	t.Parallel()

	got := redactURLsForLog(`Get "https://example.com/a?access_token=secret&x=1": timeout`)
	for _, queryValue := range []string{"secret", "x=1", "access_token"} {
		if strings.Contains(got, queryValue) {
			t.Fatalf("redactURLsForLog leaked query component %q: %s", queryValue, got)
		}
	}
	if !strings.Contains(got, "https://example.com") {
		t.Fatalf("redactURLsForLog removed safe URL components: %s", got)
	}
	if strings.Contains(got, "/a") {
		t.Fatalf("redactURLsForLog leaked a URL path: %s", got)
	}
}
