package auth

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCookieOnlyAuthResponseMode(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name       string
		values     []string
		wantCookie bool
		wantError  bool
	}{
		{name: "API compatibility default"},
		{name: "maintained browser", values: []string{authResponseModeCookie}, wantCookie: true},
		{name: "unknown mode", values: []string{"bearer"}, wantError: true},
		{name: "ambiguous mode", values: []string{authResponseModeCookie, authResponseModeCookie}, wantError: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			req := httptest.NewRequest(http.MethodPost, "/api/auth/login", nil)
			for _, value := range test.values {
				req.Header.Add(authResponseModeHeader, value)
			}
			gotCookie, err := cookieOnlyAuthResponse(req)
			if gotCookie != test.wantCookie || (err != nil) != test.wantError {
				t.Fatalf("cookieOnlyAuthResponse() = (%v, %v), want (%v, error:%v)", gotCookie, err, test.wantCookie, test.wantError)
			}
		})
	}
}

func TestMaintainedBrowserLoginOmitsBearerFromJSON(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "browser-cookie-login@example.com", RoleWrite, defaultUserStatusActive)
	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(
		http.MethodPost,
		"https://dashboard.example.com/api/auth/login",
		strings.NewReader(`{"email":"browser-cookie-login@example.com","password":"secret-password"}`),
	)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(authResponseModeHeader, authResponseModeCookie)
	loginHandler(svc).ServeHTTP(recorder, req)

	assertCookieOnlyLoginResponse(t, svc, recorder, user.ID)
}

func TestMaintainedBrowserBootstrapOmitsBearerFromJSON(t *testing.T) {
	// Bootstrap serialization shares the browser response contract but has its
	// own issuance path, so cover it independently.
	svc := newBootstrapService(t, true)
	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(
		http.MethodPost,
		"https://dashboard.example.com/api/auth/bootstrap/register",
		strings.NewReader(`{"email":"browser-bootstrap@example.com","password":"bootstrap unique password 2026","name":"Admin"}`),
	)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(authResponseModeHeader, authResponseModeCookie)
	bootstrapRegisterHandler(svc).ServeHTTP(recorder, req)

	assertCookieOnlyLoginResponse(t, svc, recorder, "")
}

func TestMaintainedBrowserPasswordChangeOmitsBearerFromJSON(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "browser-password@example.com", RoleRead, defaultUserStatusActive)
	oldToken, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken: %v", err)
	}
	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(
		http.MethodPost,
		"https://dashboard.example.com/api/auth/password",
		strings.NewReader(`{"currentPassword":"secret-password","newPassword":"browser replacement password 2026"}`),
	)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Origin", "https://dashboard.example.com")
	req.Header.Set(authResponseModeHeader, authResponseModeCookie)
	req.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: oldToken})
	AuthenticateRequest(svc)(changePasswordHandler(svc)).ServeHTTP(recorder, req)

	assertCookieOnlyLoginResponse(t, svc, recorder, user.ID)
	assertSessionTokenStatus(t, svc, oldToken, http.StatusUnauthorized)
}

func TestMeReissuesLegacyCookieWithoutMintingSession(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "legacy-cookie@example.com", RoleRead, defaultUserStatusActive)
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken: %v", err)
	}
	countSessions := func() int {
		var count int
		if err := svc.store.db.QueryRowContext(
			context.Background(),
			`SELECT COUNT(*) FROM auth_sessions WHERE user_id = ?`,
			user.ID,
		).Scan(&count); err != nil {
			t.Fatalf("count sessions: %v", err)
		}
		return count
	}
	before := countSessions()

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "https://dashboard.example.com/api/auth/me", nil)
	req.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: token})
	AuthenticateRequest(svc)(meHandler(svc)).ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	cookie := responseCookie(t, recorder, authSessionCookieName)
	if cookie.Value != token || !cookie.HttpOnly || !cookie.Secure ||
		cookie.SameSite != http.SameSiteLaxMode || cookie.Path != "/" || cookie.MaxAge <= 0 {
		t.Fatalf("replacement cookie = %#v, want same JWT with hardened attributes", cookie)
	}
	if after := countSessions(); after != before {
		t.Fatalf("session rows = %d after /me, want unchanged %d", after, before)
	}

	bearerRecorder := httptest.NewRecorder()
	bearerReq := httptest.NewRequest(http.MethodGet, "https://dashboard.example.com/api/auth/me", nil)
	bearerReq.Header.Set("Authorization", "Bearer "+token)
	AuthenticateRequest(svc)(meHandler(svc)).ServeHTTP(bearerRecorder, bearerReq)
	if cookies := bearerRecorder.Result().Cookies(); len(cookies) != 0 {
		t.Fatalf("bearer /me unexpectedly created browser cookies: %#v", cookies)
	}
}

func assertCookieOnlyLoginResponse(
	t *testing.T,
	svc *Service,
	recorder *httptest.ResponseRecorder,
	wantUserID string,
) {
	t.Helper()
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(recorder.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if _, exposed := payload["token"]; exposed {
		t.Fatalf("browser response exposed bearer token: %s", recorder.Body.String())
	}
	if _, ok := payload["user"]; !ok {
		t.Fatalf("browser response omitted user: %s", recorder.Body.String())
	}
	cookie := responseCookie(t, recorder, authSessionCookieName)
	if cookie.Value == "" || !cookie.HttpOnly || !cookie.Secure {
		t.Fatalf("session cookie = %#v, want secure HttpOnly credential", cookie)
	}
	claims, err := svc.ParseToken(cookie.Value)
	if err != nil {
		t.Fatalf("parse cookie token: %v", err)
	}
	if wantUserID != "" && claims.UserID != wantUserID {
		t.Fatalf("cookie user = %q, want %q", claims.UserID, wantUserID)
	}
	if _, _, err := svc.ResolveSessionUser(context.Background(), claims); err != nil {
		t.Fatalf("cookie session is not active: %v", err)
	}
}
