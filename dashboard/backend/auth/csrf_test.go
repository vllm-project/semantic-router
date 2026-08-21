package auth

import (
	"crypto/rand"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

var csrfTestSecret = []byte("test-jwt-secret")

func TestCSRFTokenDerivation(t *testing.T) {
	t.Run("deterministic", func(t *testing.T) {
		first := csrfTokenFor(csrfTestSecret, "session-1")
		second := csrfTokenFor(csrfTestSecret, "session-1")
		if first == "" || first != second {
			t.Fatalf("csrfTokenFor not deterministic: %q vs %q", first, second)
		}
	})

	t.Run("session bound", func(t *testing.T) {
		if csrfTokenFor(csrfTestSecret, "session-1") == csrfTokenFor(csrfTestSecret, "session-2") {
			t.Fatal("different sessions produced the same token")
		}
	})

	t.Run("secret bound", func(t *testing.T) {
		if csrfTokenFor(csrfTestSecret, "session-1") == csrfTokenFor([]byte("other-secret"), "session-1") {
			t.Fatal("different secrets produced the same token")
		}
	})

	t.Run("fails closed on empty input", func(t *testing.T) {
		cases := []struct {
			name      string
			secret    []byte
			sessionID string
		}{
			{"empty session id", csrfTestSecret, ""},
			{"whitespace session id", csrfTestSecret, "   "},
			{"empty secret", []byte{}, "session-1"},
			{"nil secret", nil, "session-1"},
		}
		for _, tc := range cases {
			if got := csrfTokenFor(tc.secret, tc.sessionID); got != "" {
				t.Errorf("%s: csrfTokenFor = %q, want empty", tc.name, got)
			}
		}
	})

	t.Run("url safe encoding", func(t *testing.T) {
		for i := 0; i < 200; i++ {
			raw := make([]byte, 16)
			if _, err := rand.Read(raw); err != nil {
				t.Fatal(err)
			}
			token := csrfTokenFor(csrfTestSecret, hex.EncodeToString(raw))
			if strings.ContainsAny(token, "+/=") {
				t.Fatalf("token %q is not URL-safe", token)
			}
		}
	})
}

func TestCSRFTokenValid(t *testing.T) {
	valid := csrfTokenFor(csrfTestSecret, "session-1")

	flipped := []byte(valid)
	flipped[0] ^= 0x01

	cases := []struct {
		name      string
		secret    []byte
		sessionID string
		presented string
		want      bool
	}{
		{"round trip", csrfTestSecret, "session-1", valid, true},
		{"wrong session", csrfTestSecret, "session-2", valid, false},
		{"wrong secret", []byte("other-secret"), "session-1", valid, false},
		{"empty presented", csrfTestSecret, "session-1", "", false},
		{"empty session id", csrfTestSecret, "", valid, false},
		{"empty secret", nil, "session-1", valid, false},
		{"one byte flipped", csrfTestSecret, "session-1", string(flipped), false},
		{"truncated", csrfTestSecret, "session-1", valid[:len(valid)-1], false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := csrfTokenValid(tc.secret, tc.sessionID, tc.presented); got != tc.want {
				t.Fatalf("csrfTokenValid = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestOriginAllowed_NoHeadersIsDenied is the case that proves the check fails closed.
func TestOriginAllowed_NoHeadersIsDenied(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "http://dash.example/api/x", nil)
	if originAllowed(r, nil) {
		t.Fatal("a request with neither Origin nor Referer was allowed")
	}
	if originAllowed(r, []string{"http://localhost:3001"}) {
		t.Fatal("a request with neither Origin nor Referer was allowed with an allowlist")
	}
	if originAllowed(nil, nil) {
		t.Fatal("a nil request was allowed")
	}
}

func TestOriginAllowed(t *testing.T) {
	cases := []struct {
		name    string
		origin  string
		referer string
		headers map[string]string
		host    string
		allowed []string
		want    bool
	}{
		{name: "same origin", origin: "http://dash.example", want: true},
		{name: "cross origin", origin: "https://evil.example"},
		{name: "same host wrong scheme", origin: "https://dash.example"},
		{name: "same host different port", origin: "http://dash.example:9999"},
		{name: "null origin", origin: "null"},
		{name: "null origin even when allowlisted", origin: "null", allowed: []string{"null"}},
		{name: "referer fallback same", referer: "http://dash.example/page?x=1", want: true},
		{name: "referer fallback cross", referer: "https://evil.example/a"},
		{name: "unparseable referer", referer: "::::"},
		{name: "relative referer", referer: "/dashboard"},
		{name: "origin wins over referer", origin: "https://evil.example", referer: "http://dash.example/page"},
		{
			name:    "behind TLS proxy",
			origin:  "https://dash.example",
			headers: map[string]string{"X-Forwarded-Proto": "https"},
			want:    true,
		},
		{
			name:    "behind TLS proxy, http origin",
			origin:  "http://dash.example",
			headers: map[string]string{"X-Forwarded-Proto": "https"},
		},
		{
			name:   "forwarded host",
			origin: "https://public.example",
			headers: map[string]string{
				"X-Forwarded-Proto": "https",
				"X-Forwarded-Host":  "public.example",
			},
			want: true,
		},
		{
			name:   "forwarded host chain",
			origin: "https://public.example",
			headers: map[string]string{
				"X-Forwarded-Proto": "https",
				"X-Forwarded-Host":  "public.example, internal",
			},
			want: true,
		},
		{name: "empty host", origin: "http://dash.example", host: " "},
		{name: "case insensitive", origin: "HTTP://DASH.EXAMPLE", want: true},
		{name: "whitespace padded origin", origin: "  http://dash.example  ", want: true},

		// Allowlist mode (DASHBOARD_ALLOWED_ORIGINS set).
		{
			name:    "allowlisted vite dev origin",
			origin:  "http://localhost:3001",
			allowed: []string{"http://localhost:3001"},
			want:    true,
		},
		{
			name:    "allowlist entry is trimmed and lowercased",
			origin:  "http://localhost:3001",
			allowed: []string{"  HTTP://LOCALHOST:3001 "},
			want:    true,
		},
		{
			name:    "own origin still allowed alongside allowlist",
			origin:  "http://dash.example",
			allowed: []string{"http://localhost:3001"},
			want:    true,
		},
		{
			name:    "cross origin denied with allowlist",
			origin:  "https://evil.example",
			allowed: []string{"http://localhost:3001"},
		},
		{
			// D10: with an allowlist configured, X-Forwarded-Host must not be trusted.
			name:   "forwarded host ignored when allowlist is set",
			origin: "https://public.example",
			headers: map[string]string{
				"X-Forwarded-Proto": "https",
				"X-Forwarded-Host":  "public.example",
			},
			allowed: []string{"http://localhost:3001"},
		},
		{
			name:    "empty allowlist entries are skipped",
			origin:  "https://evil.example",
			allowed: []string{"", "   "},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, "http://dash.example/api/x", nil)
			if tc.host != "" {
				r.Host = strings.TrimSpace(tc.host)
			}
			if tc.origin != "" {
				r.Header.Set("Origin", tc.origin)
			}
			if tc.referer != "" {
				r.Header.Set("Referer", tc.referer)
			}
			for name, value := range tc.headers {
				r.Header.Set(name, value)
			}
			if got := originAllowed(r, tc.allowed); got != tc.want {
				t.Fatalf("originAllowed = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestRequiresCSRFCheck(t *testing.T) {
	cases := map[string]bool{
		http.MethodGet:     false,
		http.MethodHead:    false,
		http.MethodOptions: false,
		"get":              false,
		http.MethodPost:    true,
		http.MethodPut:     true,
		http.MethodPatch:   true,
		http.MethodDelete:  true,
		"post":             true,
		" POST ":           true,
		"":                 true,
	}
	for method, want := range cases {
		if got := requiresCSRFCheck(method); got != want {
			t.Errorf("requiresCSRFCheck(%q) = %v, want %v", method, got, want)
		}
	}
}

func TestCSRFCookieAndHeaderNames(t *testing.T) {
	if csrfCookieName != "vsr_csrf" {
		t.Errorf("csrfCookieName = %q", csrfCookieName)
	}
	if csrfHeaderName != "X-CSRF-Token" {
		t.Errorf("csrfHeaderName = %q", csrfHeaderName)
	}
}
