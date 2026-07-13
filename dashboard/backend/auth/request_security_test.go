package auth

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/jsonunicode"
)

func TestAuthJSONDecoderEnforcesMediaTypeShapeAndBodyLimit(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	tests := []struct {
		name        string
		contentType string
		body        string
		wantStatus  int
	}{
		{
			name:       "missing content type",
			body:       `{"email":"person@example.com","password":"wrong-password-value"}`,
			wantStatus: http.StatusUnsupportedMediaType,
		},
		{
			name:        "unknown field",
			contentType: "application/json",
			body:        `{"email":"person@example.com","password":"wrong-password-value","extra":true}`,
			wantStatus:  http.StatusBadRequest,
		},
		{
			name:        "multiple JSON values",
			contentType: "application/json",
			body:        `{"email":"person@example.com","password":"wrong-password-value"}{}`,
			wantStatus:  http.StatusBadRequest,
		},
		{
			name:        "oversized body",
			contentType: "application/json",
			body:        `{"email":"person@example.com","password":"` + strings.Repeat("x", int(maxAuthJSONBodyBytes)) + `"}`,
			wantStatus:  http.StatusRequestEntityTooLarge,
		},
		{
			name:        "invalid UTF-8 password bytes",
			contentType: "application/json",
			body: `{"email":"person@example.com","password":"` +
				string(bytes.Repeat([]byte{0xff}, minimumPasswordCodePoints)) + `"}`,
			wantStatus: http.StatusBadRequest,
		},
		{
			name:        "unpaired high-surrogate password escapes",
			contentType: "application/json",
			body: `{"email":"person@example.com","password":"` +
				strings.Repeat(`\ud800`, minimumPasswordCodePoints) + `"}`,
			wantStatus: http.StatusBadRequest,
		},
		{
			name:        "unpaired low-surrogate password escapes",
			contentType: "application/json",
			body: `{"email":"person@example.com","password":"` +
				strings.Repeat(`\udc00`, minimumPasswordCodePoints) + `"}`,
			wantStatus: http.StatusBadRequest,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			recorder := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/api/auth/login", strings.NewReader(test.body))
			if test.contentType != "" {
				req.Header.Set("Content-Type", test.contentType)
			}
			loginHandler(svc).ServeHTTP(recorder, req)
			if recorder.Code != test.wantStatus {
				t.Fatalf("status = %d, want %d; body=%s", recorder.Code, test.wantStatus, recorder.Body.String())
			}
			if recorder.Header().Get("Cache-Control") != "no-store" {
				t.Fatalf("Cache-Control = %q, want no-store", recorder.Header().Get("Cache-Control"))
			}
			if recorder.Header().Get("X-Content-Type-Options") != "nosniff" {
				t.Fatalf("X-Content-Type-Options = %q, want nosniff", recorder.Header().Get("X-Content-Type-Options"))
			}
		})
	}
}

func TestValidAuthJSONUnicodeAcceptsSurrogatePairsAndLiteralReplacementCharacter(t *testing.T) {
	t.Parallel()

	for _, body := range [][]byte{
		[]byte(`{"value":"\ud83d\ude00"}`),
		[]byte(`{"value":"�"}`),
		[]byte(`{"value":"\ufffd"}`),
	} {
		if !jsonunicode.Valid(body) {
			t.Fatalf("jsonunicode.Valid(%q) = false, want true", body)
		}
	}
}

func TestLoginRequestSourceUsesDirectPeerNotForwardingHeader(t *testing.T) {
	t.Parallel()

	req := httptest.NewRequest(http.MethodPost, "/api/auth/login", nil)
	req.RemoteAddr = "192.0.2.10:8443"
	req.Header.Set("X-Forwarded-For", "203.0.113.99")
	if got := loginRequestSource(req); got != "192.0.2.10" {
		t.Fatalf("loginRequestSource() = %q, want direct peer", got)
	}
}

func TestLoginRequestSourceDisablesSharedIngressBuckets(t *testing.T) {
	t.Parallel()

	for _, remoteAddr := range []string{
		"127.0.0.1:8443",
		"10.1.2.3:8443",
		"172.16.2.3:8443",
		"192.168.2.3:8443",
		"169.254.2.3:8443",
		"[::1]:8443",
		"[fd00::1]:8443",
	} {
		t.Run(remoteAddr, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/api/auth/login", nil)
			req.RemoteAddr = remoteAddr
			req.Header.Set("X-Forwarded-For", "203.0.113.99")
			if got := loginRequestSource(req); got != "" {
				t.Fatalf("loginRequestSource() = %q, want disabled bucket", got)
			}
		})
	}
}

func TestValidUnsafeRequestOriginProtectsCookieSessions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		source      CredentialSource
		origin      string
		fetchSite   string
		wantAllowed bool
	}{
		{name: "same origin", source: CredentialSourceCookie, origin: "https://dashboard.example.com", wantAllowed: true},
		{name: "same-site sibling", source: CredentialSourceCookie, origin: "https://evil.example.com", wantAllowed: false},
		{name: "fetch metadata fallback", source: CredentialSourceCookie, fetchSite: "same-origin", wantAllowed: true},
		{name: "same-site fetch metadata", source: CredentialSourceCookie, fetchSite: "same-site", wantAllowed: false},
		{name: "cookie without browser evidence", source: CredentialSourceCookie, wantAllowed: false},
		{name: "query without browser evidence", source: CredentialSourceQuery, wantAllowed: false},
		{name: "non-browser bearer", source: CredentialSourceBearer, wantAllowed: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			req := httptest.NewRequest(http.MethodPost, "http://dashboard.example.com/api/settings", nil)
			req.Header.Set("X-Forwarded-Proto", "https")
			if test.origin != "" {
				req.Header.Set("Origin", test.origin)
			}
			if test.fetchSite != "" {
				req.Header.Set("Sec-Fetch-Site", test.fetchSite)
			}
			if got := validUnsafeRequestOrigin(req, test.source); got != test.wantAllowed {
				t.Fatalf("validUnsafeRequestOrigin() = %v, want %v", got, test.wantAllowed)
			}
		})
	}
}
