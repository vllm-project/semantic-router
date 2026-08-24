package auth

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
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
	claims, err := svc.ParseToken(token)
	if err != nil {
		t.Fatalf("ParseToken() error = %v", err)
	}

	handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ac, ok := AuthFromContext(r)
		if !ok {
			t.Fatalf("missing auth context")
		}
		if ac.UserID != user.ID || claims.ExpiresAt == nil || claims.IssuedAt == nil ||
			!ac.ExpiresAt.Equal(claims.ExpiresAt.Time.UTC()) ||
			!ac.AuthenticatedAt.Equal(claims.IssuedAt.Time.UTC()) {
			t.Fatalf("auth context = %#v, claims = %#v", ac, claims)
		}
		w.WriteHeader(http.StatusNoContent)
	}))

	req := httptest.NewRequest(http.MethodGet, "/api/settings", nil)
	req.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: token})
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNoContent)
	}
}

func TestAuthenticateRequestRejectsSessionWithoutExpiry(t *testing.T) {
	t.Parallel()
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "missing-expiry@example.com", RoleRead, "active")
	unsigned := jwt.NewWithClaims(jwt.SigningMethodHS256, TokenClaims{
		UserID: user.ID, Email: user.Email, Role: user.Role,
		RegisteredClaims: jwt.RegisteredClaims{IssuedAt: jwt.NewNumericDate(time.Now().UTC())},
	})
	token, err := unsigned.SignedString(svc.jwtSecret)
	if err != nil {
		t.Fatal(err)
	}
	handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	request := httptest.NewRequest(http.MethodGet, "/api/settings", nil)
	request.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: token})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusUnauthorized)
	}
}

func TestParseTokenRejectsAlternateHMACAlgorithm(t *testing.T) {
	t.Parallel()
	svc := newTestAuthService(t)
	now := svc.currentTime()
	token := jwt.NewWithClaims(jwt.SigningMethodHS384, TokenClaims{
		UserID: "10000000-0000-4000-8000-000000000001",
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(now.Add(time.Hour)),
			IssuedAt:  jwt.NewNumericDate(now),
		},
	})
	signed, err := token.SignedString(svc.jwtSecret)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := svc.ParseToken(signed); err == nil {
		t.Fatal("ParseToken() accepted an alternate HMAC algorithm")
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
	cookie := responseCookie(t, recorder, authSessionCookieName)
	if cookie.Value == "" {
		t.Fatalf("expected opaque HttpOnly session cookie")
	}
	if strings.Contains(recorder.Body.String(), `"token"`) {
		t.Fatalf("login response exposed the browser session credential")
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
	req := httptest.NewRequest(http.MethodPost, "/api/auth/logout", nil)
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
	logoutReq := httptest.NewRequest(http.MethodPost, "/api/auth/logout", nil)
	logoutReq.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: token})
	logoutHandler(svc).ServeHTTP(logoutRecorder, logoutReq)
	if logoutRecorder.Code != http.StatusOK {
		t.Fatalf("logout status = %d, want %d", logoutRecorder.Code, http.StatusOK)
	}

	handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	req := httptest.NewRequest(http.MethodGet, "/api/auth/me", nil)
	req.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: token})
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusUnauthorized)
	}
}
