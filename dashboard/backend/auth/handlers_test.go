package auth

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func newTestAuthService(t *testing.T) *Service {
	t.Helper()

	store, err := NewStore(filepath.Join(t.TempDir(), "auth.db"))
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	t.Cleanup(func() {
		_ = store.Close()
	})

	return NewService(store, "test-secret", 1)
}

func newTestUser(t *testing.T, svc *Service, email, role, status string) *User {
	t.Helper()

	hash, err := svc.HashPassword("secret-password")
	if err != nil {
		t.Fatalf("HashPassword() error = %v", err)
	}

	user, err := svc.store.CreateUser(context.Background(), email, "Test User", hash, role, status)
	if err != nil {
		t.Fatalf("CreateUser() error = %v", err)
	}
	return user
}

func newAuthenticatedRequest(t *testing.T, svc *Service, user *User, method, path, body string) *http.Request {
	t.Helper()

	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}

	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+token)
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	return req
}

func TestAuthenticateRequestUsesCurrentDatabaseState(t *testing.T) {
	t.Parallel()

	t.Run("rejects downgraded role even with old admin token", func(t *testing.T) {
		t.Parallel()

		svc := newTestAuthService(t)
		user := newTestUser(t, svc, "admin@example.com", "admin", "active")
		if _, err := svc.store.UpdateUserRoleOrStatus(context.Background(), user.ID, RoleRead, ""); err != nil {
			t.Fatalf("UpdateUserRoleOrStatus() error = %v", err)
		}

		handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusNoContent)
		}))

		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, newAuthenticatedRequest(t, svc, user, http.MethodGet, "/api/admin/users", ""))

		if recorder.Code != http.StatusForbidden {
			t.Fatalf("status = %d, want %d", recorder.Code, http.StatusForbidden)
		}
	})

	t.Run("rejects inactive user even with old active token", func(t *testing.T) {
		t.Parallel()

		svc := newTestAuthService(t)
		user := newTestUser(t, svc, "inactive@example.com", "admin", "active")
		if _, err := svc.store.UpdateUserRoleOrStatus(context.Background(), user.ID, "", "inactive"); err != nil {
			t.Fatalf("UpdateUserRoleOrStatus() error = %v", err)
		}

		handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusNoContent)
		}))

		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, newAuthenticatedRequest(t, svc, user, http.MethodGet, "/api/status", ""))

		if recorder.Code != http.StatusUnauthorized {
			t.Fatalf("status = %d, want %d", recorder.Code, http.StatusUnauthorized)
		}
	})
}

func TestRegisterAdminRoutesHonorsUsersViewAndSelfLockoutGuards(t *testing.T) {
	t.Parallel()

	t.Run("allows users.view on GET /api/admin/users", func(t *testing.T) {
		t.Parallel()

		svc := newTestAuthService(t)
		user := newTestUser(t, svc, "viewer@example.com", RoleRead, "active")
		if _, err := svc.store.db.ExecContext(
			context.Background(),
			`INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES(?,?,1)`,
			user.ID,
			PermUsersView,
		); err != nil {
			t.Fatalf("grant users.view error = %v", err)
		}

		mux := http.NewServeMux()
		RegisterAdminRoutes(mux, svc)
		handler := AuthenticateRequest(svc)(mux)

		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, newAuthenticatedRequest(t, svc, user, http.MethodGet, "/api/admin/users", ""))

		if recorder.Code != http.StatusOK {
			t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
		}
	})

	t.Run("blocks self role and status changes", func(t *testing.T) {
		t.Parallel()

		svc := newTestAuthService(t)
		user := newTestUser(t, svc, "self-update@example.com", "admin", "active")

		mux := http.NewServeMux()
		RegisterAdminRoutes(mux, svc)
		handler := AuthenticateRequest(svc)(mux)

		recorder := httptest.NewRecorder()
		handler.ServeHTTP(
			recorder,
			newAuthenticatedRequest(
				t,
				svc,
				user,
				http.MethodPatch,
				"/api/admin/users/"+user.ID,
				`{"status":"inactive"}`,
			),
		)

		if recorder.Code != http.StatusConflict {
			t.Fatalf("status = %d, want %d", recorder.Code, http.StatusConflict)
		}
	})

	t.Run("blocks self deletion", func(t *testing.T) {
		t.Parallel()

		svc := newTestAuthService(t)
		user := newTestUser(t, svc, "self-delete@example.com", "admin", "active")

		mux := http.NewServeMux()
		RegisterAdminRoutes(mux, svc)
		handler := AuthenticateRequest(svc)(mux)

		recorder := httptest.NewRecorder()
		handler.ServeHTTP(
			recorder,
			newAuthenticatedRequest(t, svc, user, http.MethodDelete, "/api/admin/users/"+user.ID, ""),
		)

		if recorder.Code != http.StatusConflict {
			t.Fatalf("status = %d, want %d", recorder.Code, http.StatusConflict)
		}
	})
}

func TestLoginHandlerReturnsEffectivePermissions(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "login@example.com", RoleWrite, "active")

	mux := http.NewServeMux()
	mux.HandleFunc("/api/auth/login", loginHandler(svc))

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(
		http.MethodPost,
		"/api/auth/login",
		strings.NewReader(`{"email":"login@example.com","password":"secret-password"}`),
	)
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}

	var payload struct {
		Token string `json:"token"`
		User  struct {
			ID          string   `json:"id"`
			Role        string   `json:"role"`
			Permissions []string `json:"permissions"`
		} `json:"user"`
	}
	if err := json.NewDecoder(recorder.Body).Decode(&payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if payload.Token == "" {
		t.Fatalf("expected token in login response")
	}
	if payload.User.ID != user.ID {
		t.Fatalf("user.id = %q, want %q", payload.User.ID, user.ID)
	}
	if payload.User.Role != RoleWrite {
		t.Fatalf("user.role = %q, want %q", payload.User.Role, RoleWrite)
	}
	if !slices.Contains(payload.User.Permissions, PermConfigDeploy) {
		t.Fatalf("login response permissions missing %q: %v", PermConfigDeploy, payload.User.Permissions)
	}
	if !slices.Contains(payload.User.Permissions, PermToolsUse) {
		t.Fatalf("login response permissions missing %q: %v", PermToolsUse, payload.User.Permissions)
	}

	sessionCookie := recorder.Result().Cookies()
	if len(sessionCookie) != 1 {
		t.Fatalf("cookie count = %d, want 1", len(sessionCookie))
	}
	if sessionCookie[0].Name != authSessionCookieName {
		t.Fatalf("cookie name = %q, want %q", sessionCookie[0].Name, authSessionCookieName)
	}
	if !sessionCookie[0].HttpOnly {
		t.Fatalf("expected %q cookie to be HttpOnly", authSessionCookieName)
	}
	if sessionCookie[0].SameSite != http.SameSiteLaxMode {
		t.Fatalf("cookie SameSite = %v, want %v", sessionCookie[0].SameSite, http.SameSiteLaxMode)
	}
	if sessionCookie[0].Path != authSessionCookiePath {
		t.Fatalf("cookie path = %q, want %q", sessionCookie[0].Path, authSessionCookiePath)
	}
	if sessionCookie[0].Value == "" {
		t.Fatalf("expected %q cookie value", authSessionCookieName)
	}
	if time.Until(sessionCookie[0].Expires) <= 0 {
		t.Fatalf("expected %q cookie expiry in the future", authSessionCookieName)
	}
}

func TestMeHandlerReturnsEffectivePermissions(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "viewer@example.com", RoleRead, "active")
	if _, err := svc.store.db.ExecContext(
		context.Background(),
		`INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES(?,?,1)`,
		user.ID,
		PermUsersView,
	); err != nil {
		t.Fatalf("grant users.view error = %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/api/auth/me", meHandler(svc))
	handler := AuthenticateRequest(svc)(mux)

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, newAuthenticatedRequest(t, svc, user, http.MethodGet, "/api/auth/me", ""))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}

	var payload struct {
		User struct {
			ID          string   `json:"id"`
			Permissions []string `json:"permissions"`
		} `json:"user"`
	}
	if err := json.NewDecoder(recorder.Body).Decode(&payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if payload.User.ID != user.ID {
		t.Fatalf("user.id = %q, want %q", payload.User.ID, user.ID)
	}
	if !slices.Contains(payload.User.Permissions, PermConfigRead) {
		t.Fatalf("me response permissions missing %q: %v", PermConfigRead, payload.User.Permissions)
	}
	if !slices.Contains(payload.User.Permissions, PermUsersView) {
		t.Fatalf("me response permissions missing %q: %v", PermUsersView, payload.User.Permissions)
	}
}

func TestBootstrapRegisterHandlerSetsSessionCookie(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)

	mux := http.NewServeMux()
	mux.HandleFunc("/api/auth/bootstrap/register", bootstrapRegisterHandler(svc))

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(
		http.MethodPost,
		"/api/auth/bootstrap/register",
		strings.NewReader(`{"email":"admin@example.com","password":"secret-password","name":"Admin"}`),
	)
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}

	cookies := recorder.Result().Cookies()
	if len(cookies) != 1 {
		t.Fatalf("cookie count = %d, want 1", len(cookies))
	}
	if cookies[0].Name != authSessionCookieName {
		t.Fatalf("cookie name = %q, want %q", cookies[0].Name, authSessionCookieName)
	}
	if !cookies[0].HttpOnly {
		t.Fatalf("expected %q cookie to be HttpOnly", authSessionCookieName)
	}
}

func TestLogoutHandlerClearsSessionCookie(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	mux.HandleFunc("/api/auth/logout", logoutHandler(nil))

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/auth/logout", nil)
	mux.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNoContent)
	}

	cookies := recorder.Result().Cookies()
	if len(cookies) != 1 {
		t.Fatalf("cookie count = %d, want 1", len(cookies))
	}
	if cookies[0].Name != authSessionCookieName {
		t.Fatalf("cookie name = %q, want %q", cookies[0].Name, authSessionCookieName)
	}
	if cookies[0].MaxAge != -1 {
		t.Fatalf("cookie MaxAge = %d, want -1", cookies[0].MaxAge)
	}
	if !cookies[0].Expires.Equal(time.Unix(0, 0)) {
		t.Fatalf("cookie Expires = %v, want Unix(0, 0)", cookies[0].Expires)
	}
}
