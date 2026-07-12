package auth

import (
	"context"
	"errors"
	"net/http"
	"testing"
)

func TestPasswordPolicyAppliesAtEveryEstablishmentPath(t *testing.T) {
	t.Parallel()

	assertPolicyError := func(t *testing.T, err error) {
		t.Helper()
		var policyErr *PasswordPolicyError
		if !errors.As(err, &policyErr) || policyErr.Code != PasswordPolicyTooShort {
			t.Fatalf("error = %#v, want too-short policy error", err)
		}
	}

	t.Run("environment bootstrap", func(t *testing.T) {
		svc := newTestAuthService(t)
		assertPolicyError(t, svc.EnsureBootstrapAdmin(
			context.Background(),
			"bootstrap@example.com",
			"short",
			"Admin",
		))
	})

	t.Run("web bootstrap", func(t *testing.T) {
		svc := newBootstrapService(t, true)
		recorder := authJSONRequest(
			t,
			bootstrapRegisterHandler(svc),
			http.MethodPost,
			"/api/auth/bootstrap/register",
			`{"email":"bootstrap@example.com","password":"short","name":"Admin"}`,
			nil,
		)
		if recorder.Code != http.StatusUnprocessableEntity {
			t.Fatalf("status = %d, want 422; body=%s", recorder.Code, recorder.Body.String())
		}
	})

	t.Run("admin create", func(t *testing.T) {
		svc := newTestAuthService(t)
		recorder := authJSONRequest(
			t,
			adminUsersCollectionHandler(svc),
			http.MethodPost,
			"/api/admin/users",
			`{"email":"created@example.com","name":"Created","password":"short","role":"read"}`,
			&AuthContext{UserID: "admin", Perms: map[string]bool{PermUsersManage: true}},
		)
		if recorder.Code != http.StatusUnprocessableEntity {
			t.Fatalf("status = %d, want 422; body=%s", recorder.Code, recorder.Body.String())
		}
	})

	t.Run("admin reset", func(t *testing.T) {
		svc := newTestAuthService(t)
		target := newTestUser(t, svc, "target@example.com", RoleRead, defaultUserStatusActive)
		recorder := authJSONRequest(
			t,
			adminUserPasswordHandler(svc),
			http.MethodPost,
			"/api/admin/users/password",
			`{"userId":"`+target.ID+`","password":"short"}`,
			&AuthContext{UserID: "admin", Perms: map[string]bool{PermUsersManage: true}},
		)
		if recorder.Code != http.StatusUnprocessableEntity {
			t.Fatalf("status = %d, want 422; body=%s", recorder.Code, recorder.Body.String())
		}
	})

	t.Run("self change", func(t *testing.T) {
		svc := newTestAuthService(t)
		user := newTestUser(t, svc, "self-change@example.com", RoleRead, defaultUserStatusActive)
		_, _, err := svc.ChangePassword(
			context.Background(),
			user.ID,
			"secret-password",
			"short",
		)
		assertPolicyError(t, err)
	})
}

func TestConfiguredBootstrapAdminNormalizesEmailForLogin(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	if err := svc.EnsureBootstrapAdmin(
		context.Background(),
		"  Bootstrap.Admin@Example.COM  ",
		"configured bootstrap password 2026",
		"Bootstrap Admin",
	); err != nil {
		t.Fatalf("EnsureBootstrapAdmin() error = %v", err)
	}
	token, user, err := svc.Login(
		context.Background(),
		"bootstrap.admin@example.com",
		"configured bootstrap password 2026",
	)
	if err != nil {
		t.Fatalf("Login() error = %v", err)
	}
	if token == "" || user == nil || user.Email != "bootstrap.admin@example.com" {
		t.Fatalf("login result = token-present:%v user:%#v", token != "", user)
	}
}
