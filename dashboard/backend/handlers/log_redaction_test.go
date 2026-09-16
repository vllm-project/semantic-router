package handlers

import (
	"strings"
	"testing"
)

func TestRedactURLForLogKeepsOnlyNonSecretOriginDetails(t *testing.T) {
	t.Parallel()

	secret := "capability-canary"
	got := redactURLForLog(
		"https://user:pass@example.com:8443/capability/" + secret +
			"?opaque=" + secret + "#" + secret,
	)
	if strings.Contains(got, secret) || strings.Contains(got, "user:pass") {
		t.Fatalf("redactURLForLog leaked URL capability material: %s", got)
	}
	if got != "https://example.com:8443/%5BREDACTED%5D?REDACTED#REDACTED" {
		t.Fatalf("redactURLForLog returned an unexpected safe view: %s", got)
	}
}

func TestRedactURLsForLogMasksURLsInsideErrors(t *testing.T) {
	t.Parallel()

	secret := "opaque-capability" //nolint:gosec // Canary verifies capability URLs are redacted from logs.
	got := redactURLsForLog(
		`Get "https://example.com/a/` + secret + `?x=` + secret + `#` + secret + `": timeout`,
	)
	if strings.Contains(got, secret) || strings.Contains(got, "x=") {
		t.Fatalf("redactURLsForLog leaked sensitive values: %s", got)
	}
	if !strings.Contains(got, "https://example.com/%5BREDACTED%5D?REDACTED#REDACTED") {
		t.Fatalf("redactURLsForLog omitted its safe URL view: %s", got)
	}
}

func TestRedactURLForLogPreservesRootWithoutCredentials(t *testing.T) {
	t.Parallel()

	if got := redactURLForLog("https://example.com/"); got != "https://example.com/" {
		t.Fatalf("redactURLForLog changed a credential-free origin: %s", got)
	}
}

func TestRedactRuntimeLogLinesMasksCredentialShapes(t *testing.T) {
	t.Parallel()

	secret := "runtime-log-canary"
	lines := redactRuntimeLogLines([]string{
		"Authorization: Bearer " + secret,
		`{"api_key":"` + secret + `"}`,
		"DATABASE_URL=postgres://user:" + secret + "@db.internal/app?token=" + secret,
		"Cookie: session=" + secret + "; harmless=value",
	})
	got := strings.Join(lines, "\n")
	if strings.Contains(got, secret) || strings.Contains(got, "user:") {
		t.Fatalf("runtime log redaction leaked credentials: %s", got)
	}
	if !strings.Contains(got, "[REDACTED]") {
		t.Fatalf("runtime log redaction omitted its marker: %s", got)
	}
}
