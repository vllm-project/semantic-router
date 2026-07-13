package auth

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func responseCookie(t *testing.T, recorder *httptest.ResponseRecorder, name string) *http.Cookie {
	t.Helper()

	for _, cookie := range recorder.Result().Cookies() {
		if cookie.Name == name {
			return cookie
		}
	}
	t.Fatalf("missing response cookie %q", name)
	return nil
}

func TestAuthenticateRequestAcceptsSessionCookie(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "cookie-session@example.com", RoleRead, "active")
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}

	handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ac, ok := AuthFromContext(r)
		if !ok {
			t.Fatalf("missing auth context")
		}
		if ac.UserID != user.ID {
			t.Fatalf("user id = %q, want %q", ac.UserID, user.ID)
		}
		w.WriteHeader(http.StatusNoContent)
	}))

	req := httptest.NewRequest(http.MethodGet, "/api/status", nil)
	req.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: token})
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNoContent)
	}
}

func TestLoginHandlerSetsHttpOnlySessionCookie(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	_ = newTestUser(t, svc, "cookie-login@example.com", RoleWrite, "active")

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(
		http.MethodPost,
		"/api/auth/login",
		strings.NewReader(`{"email":"cookie-login@example.com","password":"secret-password"}`),
	)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Forwarded-Proto", "https")
	loginHandler(svc).ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}

	var payload LoginResponse
	if err := json.NewDecoder(recorder.Body).Decode(&payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if payload.Token == "" {
		t.Fatalf("expected token in login response")
	}

	cookie := responseCookie(t, recorder, authSessionCookieName)
	if cookie.Value != payload.Token {
		t.Fatalf("cookie token does not match response token")
	}
	if !cookie.HttpOnly {
		t.Fatalf("session cookie should be HttpOnly")
	}
	if !cookie.Secure {
		t.Fatalf("session cookie should be Secure behind HTTPS proxy")
	}
	if cookie.SameSite != http.SameSiteLaxMode {
		t.Fatalf("same site = %v, want %v", cookie.SameSite, http.SameSiteLaxMode)
	}
	if cookie.Path != "/" {
		t.Fatalf("path = %q, want /", cookie.Path)
	}
	if cookie.MaxAge != 3600 {
		t.Fatalf("maxAge = %d, want 3600", cookie.MaxAge)
	}
	if !cookie.Expires.After(time.Now()) {
		t.Fatalf("expires = %v, want a future timestamp", cookie.Expires)
	}
}

func TestLogoutHandlerClearsSessionCookie(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(
		http.MethodPost,
		"https://dashboard.example.com/api/auth/logout",
		nil,
	)
	req.Header.Set("Origin", "https://dashboard.example.com")
	req.Header.Set("X-Forwarded-Proto", "https")
	logoutHandler(svc).ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}

	cookie := responseCookie(t, recorder, authSessionCookieName)
	if cookie.Value != "" {
		t.Fatalf("cookie value = %q, want empty", cookie.Value)
	}
	if cookie.MaxAge != -1 {
		t.Fatalf("maxAge = %d, want -1", cookie.MaxAge)
	}
	if !cookie.Expires.Before(time.Now()) {
		t.Fatalf("expires = %v, want an expired timestamp", cookie.Expires)
	}
	if !cookie.HttpOnly {
		t.Fatalf("session cookie should be HttpOnly")
	}
	if !cookie.Secure {
		t.Fatalf("session cookie should be Secure behind HTTPS proxy")
	}
}

func TestLogoutHandlerRevokesSessionToken(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "revoked-session@example.com", RoleRead, "active")
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}

	logoutRecorder := httptest.NewRecorder()
	logoutReq := httptest.NewRequest(http.MethodPost, "https://dashboard.example.com/api/auth/logout", nil)
	logoutReq.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: token})
	logoutReq.Header.Set("Origin", "https://dashboard.example.com")
	logoutHandler(svc).ServeHTTP(logoutRecorder, logoutReq)
	if logoutRecorder.Code != http.StatusOK {
		t.Fatalf("logout status = %d, want %d", logoutRecorder.Code, http.StatusOK)
	}

	handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	req := httptest.NewRequest(http.MethodGet, "/api/status", nil)
	req.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: token})
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusUnauthorized)
	}
}

func TestLogoutHandlerRequiresSameOriginForCookieSession(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "logout-csrf@example.com", RoleRead, "active")

	for _, test := range []struct {
		name       string
		origin     string
		fetchSite  string
		wantStatus int
	}{
		{name: "same origin", origin: "https://dashboard.example.com", wantStatus: http.StatusOK},
		{name: "same-site sibling", origin: "https://evil.example.com", wantStatus: http.StatusForbidden},
		{name: "same-origin fetch metadata", fetchSite: "same-origin", wantStatus: http.StatusOK},
		{name: "missing browser evidence", wantStatus: http.StatusForbidden},
	} {
		t.Run(test.name, func(t *testing.T) {
			requestToken, issueErr := svc.issueToken(user)
			if issueErr != nil {
				t.Fatalf("issueToken() error = %v", issueErr)
			}
			req := httptest.NewRequest(http.MethodPost, "https://dashboard.example.com/api/auth/logout", nil)
			req.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: requestToken})
			if test.origin != "" {
				req.Header.Set("Origin", test.origin)
			}
			if test.fetchSite != "" {
				req.Header.Set("Sec-Fetch-Site", test.fetchSite)
			}
			recorder := httptest.NewRecorder()
			logoutHandler(svc).ServeHTTP(recorder, req)
			if recorder.Code != test.wantStatus {
				t.Fatalf("status = %d, want %d", recorder.Code, test.wantStatus)
			}
		})
	}
}

func TestLogoutHandlerAllowsMetadataFreeBearerClient(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "logout-api@example.com", RoleRead, "active")
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/auth/logout", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	recorder := httptest.NewRecorder()
	logoutHandler(svc).ServeHTTP(recorder, req)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
}

func TestLogoutHandlerRejectsCredentialFreeCrossSiteRequest(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	req := httptest.NewRequest(
		http.MethodPost,
		"https://dashboard.example.com/api/auth/logout",
		nil,
	)
	req.Header.Set("Origin", "https://evil.example.com")
	recorder := httptest.NewRecorder()

	logoutHandler(svc).ServeHTTP(recorder, req)

	if recorder.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusForbidden)
	}
	if got := recorder.Header().Values("Set-Cookie"); len(got) != 0 {
		t.Fatalf("cross-site logout emitted Set-Cookie: %#v", got)
	}
}

func TestRequestUsesHTTPSUsesOriginalForwardedProtoHop(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name      string
		values    []string
		wantHTTPS bool
	}{
		{name: "direct HTTPS proxy", values: []string{"https"}, wantHTTPS: true},
		{name: "HTTPS client then HTTP internal hop", values: []string{"https, http"}, wantHTTPS: true},
		{name: "HTTP client then HTTPS internal hop", values: []string{"http, https"}, wantHTTPS: false},
		{name: "repeated proxy fields use first field", values: []string{"https", "http"}, wantHTTPS: true},
		{name: "malformed first hop", values: []string{"ftp, https"}, wantHTTPS: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "http://dashboard.example.com/", nil)
			for _, value := range test.values {
				req.Header.Add("X-Forwarded-Proto", value)
			}
			if got := requestUsesHTTPS(req); got != test.wantHTTPS {
				t.Fatalf("requestUsesHTTPS() = %v, want %v", got, test.wantHTTPS)
			}
		})
	}
}
