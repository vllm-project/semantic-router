package auth

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAdminPasswordResetRevokesAllTargetSessions(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "reset-admin@example.com", RoleAdmin, defaultUserStatusActive)
	target := newTestUser(t, svc, "reset-target@example.com", RoleRead, defaultUserStatusActive)
	targetToken, err := svc.issueToken(target)
	if err != nil {
		t.Fatalf("issue target token: %v", err)
	}
	adminToken, err := svc.issueToken(admin)
	if err != nil {
		t.Fatalf("issue admin token: %v", err)
	}
	adminClaims, err := svc.ParseToken(adminToken)
	if err != nil {
		t.Fatalf("parse admin token: %v", err)
	}

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(
		http.MethodPost,
		"/api/admin/users/password",
		strings.NewReader(`{"userId":"`+target.ID+`","password":"admin reset password value"}`),
	)
	req.Header.Set("Content-Type", "application/json")
	req = req.WithContext(WithAuthContext(req.Context(), AuthContext{
		UserID:    admin.ID,
		SessionID: adminClaims.ID,
		Perms:     map[string]bool{PermUsersManage: true},
	}))
	adminUserPasswordHandler(svc).ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	assertSessionTokenStatus(t, svc, targetToken, http.StatusUnauthorized)
	_, hash, err := svc.store.GetUserWithPasswordHash(context.Background(), target.ID)
	if err != nil {
		t.Fatalf("GetUserWithPasswordHash() error = %v", err)
	}
	if !svc.VerifyPassword(hash, "admin reset password value") {
		t.Fatal("reset password does not verify")
	}
}

func TestPasswordChangeCompareAndSwapRejectsStaleHash(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "password-cas@example.com", RoleRead, defaultUserStatusActive)
	passwordState, err := svc.store.getUserPasswordStateByID(context.Background(), user.ID)
	if err != nil {
		t.Fatalf("getUserPasswordStateByID() error = %v", err)
	}
	oldHash := passwordState.hash
	firstHash, err := svc.HashPasswordForUser(user.Email, "first replacement password")
	if err != nil {
		t.Fatalf("first hash: %v", err)
	}
	secondHash, err := svc.HashPasswordForUser(user.Email, "second replacement password")
	if err != nil {
		t.Fatalf("second hash: %v", err)
	}
	firstToken, err := svc.prepareToken(user)
	if err != nil {
		t.Fatalf("prepare first token: %v", err)
	}
	if changeErr := svc.store.ChangePasswordAndReplaceSessions(
		context.Background(),
		user.ID,
		oldHash,
		passwordState.authGeneration,
		firstHash,
		firstToken,
	); changeErr != nil {
		t.Fatalf("first change: %v", changeErr)
	}
	staleToken, err := svc.prepareToken(user)
	if err != nil {
		t.Fatalf("prepare stale token: %v", err)
	}
	if err := svc.store.ChangePasswordAndReplaceSessions(
		context.Background(),
		user.ID,
		oldHash,
		passwordState.authGeneration,
		secondHash,
		staleToken,
	); !errors.Is(err, ErrPasswordChanged) {
		t.Fatalf("stale change error = %v, want ErrPasswordChanged", err)
	}
	assertSessionTokenStatus(t, svc, firstToken.signed, http.StatusNoContent)
	assertSessionTokenStatus(t, svc, staleToken.signed, http.StatusUnauthorized)
}

func TestPasswordChangeCompareAndSwapRejectsInactiveUser(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "inactive-password-cas@example.com", RoleRead, defaultUserStatusActive)
	passwordState, err := svc.store.getUserPasswordStateByID(context.Background(), user.ID)
	if err != nil {
		t.Fatalf("getUserPasswordStateByID() error = %v", err)
	}
	oldHash := passwordState.hash
	newHash, err := svc.HashPasswordForUser(user.Email, "inactive replacement password")
	if err != nil {
		t.Fatalf("HashPasswordForUser() error = %v", err)
	}
	issued, err := svc.prepareToken(user)
	if err != nil {
		t.Fatalf("prepareToken() error = %v", err)
	}
	if _, err := svc.store.UpdateUserRoleOrStatus(
		context.Background(),
		user.ID,
		"",
		"inactive",
	); err != nil {
		t.Fatalf("deactivate user: %v", err)
	}
	if err := svc.store.ChangePasswordAndReplaceSessions(
		context.Background(),
		user.ID,
		oldHash,
		passwordState.authGeneration,
		newHash,
		issued,
	); !errors.Is(err, ErrPasswordChanged) {
		t.Fatalf("inactive change error = %v, want ErrPasswordChanged", err)
	}
	if sessionExists(t, svc.store, issued.sessionID) {
		t.Fatal("inactive password CAS inserted a session")
	}
}
