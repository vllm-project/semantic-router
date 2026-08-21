package auth

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
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
		{name: "unparsable referer", referer: "::::"},
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
	cases := []struct {
		method string
		want   bool
	}{
		{method: http.MethodGet},
		{method: http.MethodHead},
		{method: http.MethodOptions},
		{method: "get"},
		{method: http.MethodPost, want: true},
		{method: http.MethodPut, want: true},
		{method: http.MethodPatch, want: true},
		{method: http.MethodDelete, want: true},
		{method: "post", want: true},
		{method: " POST ", want: true},
		{method: "", want: true},
	}
	for _, tc := range cases {
		method, want := tc.method, tc.want
		if got := requiresCSRFCheck(method); got != want {
			t.Errorf("requiresCSRFCheck(%q) = %v, want %v", method, got, want)
		}
	}
}

// csrfFixture is a real user with a real minted token, plus the CSRF value derived from
// that token's session id.
type csrfFixture struct {
	svc       *Service
	token     string
	sessionID string
	csrf      string
}

func newCSRFFixture(t *testing.T) csrfFixture {
	t.Helper()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "csrf@example.com", RoleAdmin, "active")
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}
	claims, err := svc.ParseToken(token)
	if err != nil {
		t.Fatalf("ParseToken() error = %v", err)
	}
	return csrfFixture{svc: svc, token: token, sessionID: claims.ID, csrf: csrfTokenFor(svc.jwtSecret, claims.ID)}
}

func (f csrfFixture) serve(t *testing.T, r *http.Request) (*httptest.ResponseRecorder, bool) {
	t.Helper()

	handlerRan := false
	handler := AuthenticateRequest(f.svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		handlerRan = true
		w.WriteHeader(http.StatusNoContent)
	}))
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, r)
	return recorder, handlerRan
}

func optionalResponseCookie(recorder *httptest.ResponseRecorder, name string) *http.Cookie {
	for _, cookie := range recorder.Result().Cookies() {
		if cookie.Name == name {
			return cookie
		}
	}
	return nil
}

func TestCSRFEnforcement(t *testing.T) {
	f := newCSRFFixture(t)
	const sameOrigin = "http://example.com"

	cases := []struct {
		name        string
		method      string
		path        string
		authVia     string
		origin      string
		referer     string
		csrfHeader  string
		wantStatus  int
		wantHandler bool
	}{
		{name: "safe read by cookie", method: http.MethodGet, authVia: "cookie", wantStatus: http.StatusNoContent, wantHandler: true},
		{name: "HEAD without origin", method: http.MethodHead, authVia: "cookie", wantStatus: http.StatusNoContent, wantHandler: true},
		{name: "OPTIONS from a hostile origin is a safe method", method: http.MethodOptions, authVia: "cookie", origin: "https://evil.example", wantStatus: http.StatusNoContent, wantHandler: true},

		{name: "the attack", method: http.MethodPost, authVia: "cookie", origin: "https://evil.example", wantStatus: http.StatusForbidden},
		{name: "the attack with a valid token", method: http.MethodPost, authVia: "cookie", origin: "https://evil.example", csrfHeader: f.csrf, wantStatus: http.StatusForbidden},
		{name: "same origin, no token", method: http.MethodPost, authVia: "cookie", origin: sameOrigin, wantStatus: http.StatusForbidden},
		{name: "same origin, garbage token", method: http.MethodPost, authVia: "cookie", origin: sameOrigin, csrfHeader: "garbage", wantStatus: http.StatusForbidden},
		{name: "same origin, token for another session", method: http.MethodPost, authVia: "cookie", origin: sameOrigin, csrfHeader: csrfTokenFor([]byte("test-secret"), "another-session"), wantStatus: http.StatusForbidden},
		{name: "no origin, no referer", method: http.MethodPost, authVia: "cookie", csrfHeader: f.csrf, wantStatus: http.StatusForbidden},

		{name: "the happy path", method: http.MethodPost, authVia: "cookie", origin: sameOrigin, csrfHeader: f.csrf, wantStatus: http.StatusNoContent, wantHandler: true},
		{name: "PUT happy path", method: http.MethodPut, authVia: "cookie", origin: sameOrigin, csrfHeader: f.csrf, wantStatus: http.StatusNoContent, wantHandler: true},
		{name: "DELETE happy path", method: http.MethodDelete, authVia: "cookie", origin: sameOrigin, csrfHeader: f.csrf, wantStatus: http.StatusNoContent, wantHandler: true},
		{name: "referer fallback", method: http.MethodPost, authVia: "cookie", referer: sameOrigin + "/dashboard", csrfHeader: f.csrf, wantStatus: http.StatusNoContent, wantHandler: true},

		{name: "query transport is not exempt", method: http.MethodPost, authVia: "query", origin: "https://evil.example", wantStatus: http.StatusForbidden},
		{name: "query transport needs the token too", method: http.MethodPost, authVia: "query", origin: sameOrigin, csrfHeader: f.csrf, wantStatus: http.StatusNoContent, wantHandler: true},

		{name: "unauthenticated fails before CSRF", method: http.MethodPost, authVia: "none", origin: sameOrigin, csrfHeader: f.csrf, wantStatus: http.StatusUnauthorized},
		{name: "public route is untouched", method: http.MethodPost, path: "/api/auth/login", authVia: "none", origin: "https://evil.example", wantStatus: http.StatusNoContent, wantHandler: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := tc.path
			if path == "" {
				path = "/api/status"
			}
			if tc.authVia == "query" {
				path += "?authToken=" + f.token
			}

			r := httptest.NewRequest(tc.method, path, nil)
			switch tc.authVia {
			case "cookie":
				r.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: f.token})
			case "header":
				r.Header.Set("Authorization", "Bearer "+f.token)
			}
			if tc.origin != "" {
				r.Header.Set("Origin", tc.origin)
			}
			if tc.referer != "" {
				r.Header.Set("Referer", tc.referer)
			}
			if tc.csrfHeader != "" {
				r.Header.Set(csrfHeaderName, tc.csrfHeader)
			}

			recorder, handlerRan := f.serve(t, r)
			if recorder.Code != tc.wantStatus {
				t.Fatalf("status = %d (%s), want %d", recorder.Code, strings.TrimSpace(recorder.Body.String()), tc.wantStatus)
			}
			if handlerRan != tc.wantHandler {
				t.Fatalf("handler ran = %v, want %v", handlerRan, tc.wantHandler)
			}
		})
	}
}

// TestCSRF_BearerAuthenticatedRequestsAreExempt documents why the exemption exists: a
// browser never attaches Authorization on its own, so curl and CI clients need no token.
func TestCSRF_BearerAuthenticatedRequestsAreExempt(t *testing.T) {
	f := newCSRFFixture(t)

	for _, method := range []string{http.MethodPost, http.MethodPut, http.MethodDelete} {
		t.Run(method, func(t *testing.T) {
			r := httptest.NewRequest(method, "/api/status", nil)
			r.Header.Set("Authorization", "Bearer "+f.token)
			r.Header.Set("Origin", "https://evil.example")

			recorder, handlerRan := f.serve(t, r)
			if recorder.Code != http.StatusNoContent || !handlerRan {
				t.Fatalf("status = %d, handler ran = %v; want 204 and true", recorder.Code, handlerRan)
			}
			if cookie := optionalResponseCookie(recorder, csrfCookieName); cookie != nil {
				t.Fatalf("bearer request should not be issued a CSRF cookie, got %q", cookie.Value)
			}
		})
	}
}

func TestCSRFCookieBackfill(t *testing.T) {
	f := newCSRFFixture(t)

	newRequest := func(method string, csrfCookie string) *http.Request {
		r := httptest.NewRequest(method, "/api/status", nil)
		r.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: f.token})
		if csrfCookie != "" {
			r.AddCookie(&http.Cookie{Name: csrfCookieName, Value: csrfCookie})
		}
		return r
	}

	t.Run("a page load repairs a session with no CSRF cookie", func(t *testing.T) {
		recorder, _ := f.serve(t, newRequest(http.MethodGet, ""))
		cookie := optionalResponseCookie(recorder, csrfCookieName)
		if cookie == nil || cookie.Value != f.csrf {
			t.Fatalf("backfilled cookie = %v, want %q", cookie, f.csrf)
		}
		if cookie.HttpOnly {
			t.Fatal("the CSRF cookie must be readable by the frontend")
		}
	})

	t.Run("a stale value is replaced", func(t *testing.T) {
		recorder, _ := f.serve(t, newRequest(http.MethodGet, "stale"))
		if cookie := optionalResponseCookie(recorder, csrfCookieName); cookie == nil || cookie.Value != f.csrf {
			t.Fatalf("stale cookie was not replaced: %v", cookie)
		}
	})

	t.Run("a correct value is not re-issued", func(t *testing.T) {
		recorder, _ := f.serve(t, newRequest(http.MethodGet, f.csrf))
		if cookie := optionalResponseCookie(recorder, csrfCookieName); cookie != nil {
			t.Fatalf("expected no Set-Cookie, got %q", cookie.Value)
		}
	})

	t.Run("the rejected write still repairs the session", func(t *testing.T) {
		r := newRequest(http.MethodPost, "")
		r.Header.Set("Origin", "http://example.com")
		recorder, handlerRan := f.serve(t, r)
		if recorder.Code != http.StatusForbidden || handlerRan {
			t.Fatalf("status = %d, handler ran = %v; want 403 and false", recorder.Code, handlerRan)
		}
		if cookie := optionalResponseCookie(recorder, csrfCookieName); cookie == nil || cookie.Value != f.csrf {
			t.Fatalf("403 response did not backfill the cookie: %v", cookie)
		}
	})
}

func TestCSRFCookieIssuance(t *testing.T) {
	t.Run("login sets both cookies", func(t *testing.T) {
		svc := newTestAuthService(t)
		_ = newTestUser(t, svc, "csrf-login@example.com", RoleWrite, "active")

		recorder := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/api/auth/login",
			strings.NewReader(`{"email":"csrf-login@example.com","password":"secret-password"}`))
		req.Header.Set("Content-Type", "application/json")
		loginHandler(svc).ServeHTTP(recorder, req)

		if recorder.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", recorder.Code)
		}
		session := responseCookie(t, recorder, authSessionCookieName)
		if !session.HttpOnly {
			t.Fatal("session cookie should be HttpOnly")
		}
		csrf := responseCookie(t, recorder, csrfCookieName)
		if csrf.HttpOnly {
			t.Fatal("CSRF cookie must not be HttpOnly")
		}
		if csrf.Secure {
			t.Fatal("CSRF cookie should not be Secure over plain HTTP")
		}
		if csrf.SameSite != http.SameSiteLaxMode || csrf.Path != "/" {
			t.Fatalf("sameSite = %v, path = %q", csrf.SameSite, csrf.Path)
		}

		claims, err := svc.ParseToken(session.Value)
		if err != nil {
			t.Fatalf("ParseToken() error = %v", err)
		}
		if want := csrfTokenFor(svc.jwtSecret, claims.ID); csrf.Value != want {
			t.Fatalf("csrf cookie = %q, want %q", csrf.Value, want)
		}
	})

	t.Run("the CSRF cookie is Secure behind an HTTPS proxy", func(t *testing.T) {
		svc := newTestAuthService(t)
		_ = newTestUser(t, svc, "csrf-secure@example.com", RoleWrite, "active")

		recorder := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/api/auth/login",
			strings.NewReader(`{"email":"csrf-secure@example.com","password":"secret-password"}`))
		req.Header.Set("X-Forwarded-Proto", "https")
		loginHandler(svc).ServeHTTP(recorder, req)

		if csrf := responseCookie(t, recorder, csrfCookieName); !csrf.Secure {
			t.Fatal("CSRF cookie should be Secure behind an HTTPS proxy")
		}
	})

	t.Run("a failed login sets no CSRF cookie", func(t *testing.T) {
		svc := newTestAuthService(t)
		_ = newTestUser(t, svc, "csrf-badpass@example.com", RoleWrite, "active")

		recorder := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/api/auth/login",
			strings.NewReader(`{"email":"csrf-badpass@example.com","password":"wrong-password"}`))
		loginHandler(svc).ServeHTTP(recorder, req)

		if recorder.Code != http.StatusUnauthorized {
			t.Fatalf("status = %d, want 401", recorder.Code)
		}
		if cookie := optionalResponseCookie(recorder, csrfCookieName); cookie != nil {
			t.Fatalf("failed login issued a CSRF cookie: %q", cookie.Value)
		}
	})

	t.Run("bootstrap register sets both cookies", func(t *testing.T) {
		svc := newBootstrapService(t, true)
		recorder := postRegister(svc, "csrf-bootstrap@example.com")
		if recorder.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", recorder.Code)
		}

		session := responseCookie(t, recorder, authSessionCookieName)
		csrf := responseCookie(t, recorder, csrfCookieName)
		claims, err := svc.ParseToken(session.Value)
		if err != nil {
			t.Fatalf("ParseToken() error = %v", err)
		}
		if want := csrfTokenFor(svc.jwtSecret, claims.ID); csrf.Value != want {
			t.Fatalf("csrf cookie = %q, want %q", csrf.Value, want)
		}
	})

	t.Run("logout clears both cookies", func(t *testing.T) {
		svc := newTestAuthService(t)
		recorder := httptest.NewRecorder()
		logoutHandler(svc).ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/api/auth/logout", nil))

		for _, name := range []string{authSessionCookieName, csrfCookieName} {
			cookie := responseCookie(t, recorder, name)
			if cookie.Value != "" || cookie.MaxAge != -1 {
				t.Fatalf("%s: value = %q, maxAge = %d; want empty and -1", name, cookie.Value, cookie.MaxAge)
			}
		}
	})
}

// The deprecation warning fires before the token is verified, so an unauthenticated caller
// controls the path it prints. net/http percent-decodes the path, so a raw %0a would forge
// a second log line if it were printed with %s.
func TestDeprecatedQueryWarningCannotForgeALogLine(t *testing.T) {
	var logged bytes.Buffer
	log.SetOutput(&logged)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })

	r := httptest.NewRequest(http.MethodGet, "/api/x", nil)
	r.URL.Path = "/api/\nWARNING: forged line"
	q := r.URL.Query()
	q.Set("authToken", "not-a-real-token")
	r.URL.RawQuery = q.Encode()

	if _, source := extractAccessTokenWithSource(r); source != tokenSourceQuery {
		t.Fatalf("source = %v, want tokenSourceQuery", source)
	}
	if strings.Contains(logged.String(), "\nWARNING: forged line") {
		t.Fatalf("the path forged a log line: %q", logged.String())
	}
	if !strings.Contains(logged.String(), `\nWARNING: forged line`) {
		t.Fatalf("the path was not logged in escaped form: %q", logged.String())
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
