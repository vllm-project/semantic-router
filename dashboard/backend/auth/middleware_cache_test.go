package auth

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestAuthenticateRequestDisablesCachingForEveryProtectedOutcome(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "protected-cache@example.com", RoleAdmin, "active")
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}
	readUser := newTestUser(t, svc, "protected-cache-read@example.com", RoleRead, "active")
	readToken, err := svc.issueToken(readUser)
	if err != nil {
		t.Fatalf("issue read token: %v", err)
	}

	testCases := []struct {
		name       string
		path       string
		token      string
		nextCode   int
		wantCode   int
		wantCalled bool
		flush      bool
	}{
		{name: "success", path: "/api/status", token: token, nextCode: http.StatusOK, wantCode: http.StatusOK, wantCalled: true},
		{name: "handler error", path: "/api/status", token: token, nextCode: http.StatusInternalServerError, wantCode: http.StatusInternalServerError, wantCalled: true},
		{name: "stream", path: "/api/status", token: token, nextCode: http.StatusOK, wantCode: http.StatusOK, wantCalled: true, flush: true},
		{name: "forbidden", path: "/api/admin/permissions", token: readToken, wantCode: http.StatusForbidden},
		{name: "missing token", path: "/api/status", wantCode: http.StatusUnauthorized},
		{name: "invalid token", path: "/api/status", token: "not-a-token", wantCode: http.StatusUnauthorized},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			nextCalled := false
			handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				nextCalled = true
				w.WriteHeader(tc.nextCode)
				if tc.flush {
					w.(http.Flusher).Flush()
				}
			}))
			req := httptest.NewRequest(http.MethodGet, tc.path, nil)
			if tc.token != "" {
				req.Header.Set("Authorization", "Bearer "+tc.token)
			}
			rec := httptest.NewRecorder()

			handler.ServeHTTP(rec, req)

			if rec.Code != tc.wantCode {
				t.Fatalf("status = %d, want %d", rec.Code, tc.wantCode)
			}
			if nextCalled != tc.wantCalled {
				t.Fatalf("next called = %v, want %v", nextCalled, tc.wantCalled)
			}
			if got := rec.Header().Get("Cache-Control"); got != "no-store" {
				t.Fatalf("Cache-Control = %q, want no-store", got)
			}
			if got := rec.Header().Get("Pragma"); got != "no-cache" {
				t.Fatalf("Pragma = %q, want no-cache", got)
			}
		})
	}
}

func TestAuthenticateRequestLeavesPublicStaticCachePolicyUntouched(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		w.WriteHeader(http.StatusOK)
	}))
	req := httptest.NewRequest(http.MethodGet, "/assets/app.01234567.js", nil)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if got := rec.Header().Get("Cache-Control"); got != "public, max-age=31536000, immutable" {
		t.Fatalf("Cache-Control = %q, want immutable static policy", got)
	}
}

func TestAuthenticateRequestEnforcesNoStoreAfterHandlerOverwritesCachePolicy(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "protected-overwrite@example.com", RoleAdmin, "active")
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}

	testCases := []struct {
		name    string
		handler http.HandlerFunc
	}{
		{
			name: "explicit WriteHeader",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Cache-Control", "public, max-age=3600")
				w.Header().Set("Pragma", "cache")
				w.WriteHeader(http.StatusNoContent)
			},
		},
		{
			name: "implicit WriteHeader through Write",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Cache-Control", "no-cache")
				_, _ = io.WriteString(w, "protected")
			},
		},
		{
			name: "header-only return",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Cache-Control", "public")
				w.Header().Set("Pragma", "cache")
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			handler := AuthenticateRequest(svc)(tc.handler)
			req := httptest.NewRequest(http.MethodGet, "/api/status", nil)
			req.Header.Set("Authorization", "Bearer "+token)
			recorder := httptest.NewRecorder()

			handler.ServeHTTP(recorder, req)

			if got := recorder.Header().Get("Cache-Control"); got != "no-store" {
				t.Fatalf("Cache-Control = %q, want no-store", got)
			}
			if got := recorder.Header().Get("Pragma"); got != "no-cache" {
				t.Fatalf("Pragma = %q, want no-cache", got)
			}
		})
	}
}
