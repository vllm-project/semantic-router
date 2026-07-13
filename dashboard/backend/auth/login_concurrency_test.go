package auth

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/crypto/bcrypt"
)

type asyncLoginResult struct {
	token string
	user  *User
	err   error
}

type asyncPasswordChangeResult struct {
	token string
	user  *User
	err   error
}

func TestVerifiedLoginCannotCreateSessionAfterSelfPasswordChange(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "self-change-login-race@example.com", RoleRead, defaultUserStatusActive)
	waitForVerification, resumeLogin := pauseFirstSuccessfulVerification(t, svc)
	result := startAsyncLogin(svc, user.Email, "secret-password")
	waitForVerification()

	freshToken, _, err := svc.ChangePassword(
		context.Background(),
		user.ID,
		"secret-password",
		"self change replacement password 2026",
	)
	if err != nil {
		t.Fatalf("ChangePassword() error = %v", err)
	}
	resumeLogin()
	assertRejectedStaleLogin(t, awaitAsyncLogin(t, result))
	assertSessionTokenStatus(t, svc, freshToken, http.StatusNoContent)
	if count := activeSessionCount(t, svc, user.ID); count != 1 {
		t.Fatalf("active session count = %d, want only the password-change session", count)
	}
}

func TestVerifiedLoginCannotCreateSessionAfterAdminPasswordReset(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "admin-reset-login-race@example.com", RoleRead, defaultUserStatusActive)
	waitForVerification, resumeLogin := pauseFirstSuccessfulVerification(t, svc)
	result := startAsyncLogin(svc, user.Email, "secret-password")
	waitForVerification()

	if err := svc.ResetPassword(
		context.Background(),
		user.ID,
		"admin reset replacement password 2026",
	); err != nil {
		t.Fatalf("ResetPassword() error = %v", err)
	}
	resumeLogin()
	assertRejectedStaleLogin(t, awaitAsyncLogin(t, result))
	if count := activeSessionCount(t, svc, user.ID); count != 0 {
		t.Fatalf("active session count = %d, want 0 after reset", count)
	}
}

func TestLegacyLoginUpgradeCannotOverwriteConcurrentPasswordReset(t *testing.T) {
	svc := newTestAuthService(t)
	legacyPassword := "legacy-password-value"
	legacyHash, err := bcrypt.GenerateFromPassword([]byte(legacyPassword), passwordHashCost)
	if err != nil {
		t.Fatalf("GenerateFromPassword() error = %v", err)
	}
	user, err := svc.store.CreateUser(
		context.Background(),
		"legacy-reset-login-race@example.com",
		"Legacy Race",
		string(legacyHash),
		RoleRead,
		defaultUserStatusActive,
	)
	if err != nil {
		t.Fatalf("CreateUser() error = %v", err)
	}
	waitForVerification, resumeLogin := pauseFirstSuccessfulVerification(t, svc)
	result := startAsyncLogin(svc, user.Email, legacyPassword)
	waitForVerification()

	newPassword := "legacy account reset password 2026"
	if resetErr := svc.ResetPassword(context.Background(), user.ID, newPassword); resetErr != nil {
		t.Fatalf("ResetPassword() error = %v", resetErr)
	}
	resumeLogin()
	assertRejectedStaleLogin(t, awaitAsyncLogin(t, result))

	_, storedHash, err := svc.store.GetUserWithPasswordHash(context.Background(), user.ID)
	if err != nil {
		t.Fatalf("GetUserWithPasswordHash() error = %v", err)
	}
	if !strings.HasPrefix(storedHash, passwordHashPrefix) {
		t.Fatalf("stored hash = %q, want reset's versioned hash", storedHash)
	}
	if !svc.VerifyPassword(storedHash, newPassword) {
		t.Fatal("concurrent reset password no longer verifies")
	}
	if svc.VerifyPassword(storedHash, legacyPassword) {
		t.Fatal("legacy login overwrote the concurrent reset hash")
	}
}

func TestDeactivationPermanentlyRevokesExistingSessions(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "deactivation-session@example.com", RoleRead, defaultUserStatusActive)
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}
	assertSessionTokenStatus(t, svc, token, http.StatusNoContent)

	if _, err := svc.store.UpdateUserRoleOrStatus(
		context.Background(),
		user.ID,
		"",
		"inactive",
	); err != nil {
		t.Fatalf("deactivate user: %v", err)
	}
	assertSessionTokenStatus(t, svc, token, http.StatusUnauthorized)

	if _, err := svc.store.UpdateUserRoleOrStatus(
		context.Background(),
		user.ID,
		"",
		defaultUserStatusActive,
	); err != nil {
		t.Fatalf("reactivate user: %v", err)
	}
	assertSessionTokenStatus(t, svc, token, http.StatusUnauthorized)
	if count := activeSessionCount(t, svc, user.ID); count != 0 {
		t.Fatalf("active session count after reactivation = %d, want 0", count)
	}
}

func TestVerifiedLoginCannotCreateSessionAfterDeactivation(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "deactivation-login-race@example.com", RoleRead, defaultUserStatusActive)
	waitForVerification, resumeLogin := pauseFirstSuccessfulVerification(t, svc)
	result := startAsyncLogin(svc, user.Email, "secret-password")
	waitForVerification()

	if _, err := svc.store.UpdateUserRoleOrStatus(
		context.Background(),
		user.ID,
		"",
		"inactive",
	); err != nil {
		t.Fatalf("deactivate user: %v", err)
	}
	resumeLogin()
	assertRejectedStaleLogin(t, awaitAsyncLogin(t, result))
	if count := activeSessionCount(t, svc, user.ID); count != 0 {
		t.Fatalf("active session count = %d, want 0 after deactivation", count)
	}
}

func TestVerifiedLoginCannotCreateSessionAfterDeactivateReactivateABA(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "deactivation-login-aba@example.com", RoleRead, defaultUserStatusActive)
	waitForVerification, resumeLogin := pauseFirstSuccessfulVerification(t, svc)
	result := startAsyncLogin(svc, user.Email, "secret-password")
	waitForVerification()

	deactivateAndReactivateUser(t, svc, user.ID)
	resumeLogin()
	assertRejectedStaleLogin(t, awaitAsyncLogin(t, result))
	if count := activeSessionCount(t, svc, user.ID); count != 0 {
		t.Fatalf("active session count = %d, want 0 after status ABA", count)
	}
}

func TestVerifiedPasswordChangeCannotCreateSessionAfterDeactivateReactivateABA(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "deactivation-password-aba@example.com", RoleRead, defaultUserStatusActive)
	oldToken, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}
	waitForVerification, resumeChange := pauseFirstSuccessfulVerification(t, svc)
	result := startAsyncPasswordChange(
		svc,
		user.ID,
		"secret-password",
		"replacement after status aba 2026",
	)
	waitForVerification()

	deactivateAndReactivateUser(t, svc, user.ID)
	resumeChange()
	change := awaitAsyncPasswordChange(t, result)
	if !errors.Is(change.err, ErrPasswordChanged) {
		t.Fatalf("password change error = %v, want ErrPasswordChanged", change.err)
	}
	if change.token != "" || change.user != nil {
		t.Fatalf("stale password change returned token/user: %#v", change)
	}
	assertSessionTokenStatus(t, svc, oldToken, http.StatusUnauthorized)
	if count := activeSessionCount(t, svc, user.ID); count != 0 {
		t.Fatalf("active session count = %d, want 0 after status ABA", count)
	}
	_, storedHash, err := svc.store.GetUserWithPasswordHash(context.Background(), user.ID)
	if err != nil {
		t.Fatalf("GetUserWithPasswordHash() error = %v", err)
	}
	if !svc.VerifyPassword(storedHash, "secret-password") {
		t.Fatal("failed stale change modified the current password")
	}
}

func pauseFirstSuccessfulVerification(
	t *testing.T,
	svc *Service,
) (wait func(), resume func()) {
	t.Helper()
	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	var successfulCalls atomic.Int32
	originalVerify := svc.verify
	svc.verify = func(hash, password string) bool {
		matched := originalVerify(hash, password)
		if matched && successfulCalls.Add(1) == 1 {
			close(entered)
			<-release
		}
		return matched
	}
	resume = func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(func() {
		resume()
		svc.verify = originalVerify
	})
	wait = func() {
		select {
		case <-entered:
		case <-time.After(20 * time.Second):
			t.Fatal("login did not reach the post-read password verification barrier")
		}
	}
	return wait, resume
}

func startAsyncLogin(svc *Service, email, password string) <-chan asyncLoginResult {
	result := make(chan asyncLoginResult, 1)
	go func() {
		token, user, err := svc.Login(context.Background(), email, password)
		result <- asyncLoginResult{token: token, user: user, err: err}
	}()
	return result
}

func startAsyncPasswordChange(
	svc *Service,
	userID string,
	currentPassword string,
	newPassword string,
) <-chan asyncPasswordChangeResult {
	result := make(chan asyncPasswordChangeResult, 1)
	go func() {
		token, user, err := svc.ChangePassword(
			context.Background(),
			userID,
			currentPassword,
			newPassword,
		)
		result <- asyncPasswordChangeResult{token: token, user: user, err: err}
	}()
	return result
}

func awaitAsyncLogin(t *testing.T, result <-chan asyncLoginResult) asyncLoginResult {
	t.Helper()
	select {
	case value := <-result:
		return value
	case <-time.After(20 * time.Second):
		t.Fatal("login did not complete after the barrier was released")
		return asyncLoginResult{}
	}
}

func awaitAsyncPasswordChange(
	t *testing.T,
	result <-chan asyncPasswordChangeResult,
) asyncPasswordChangeResult {
	t.Helper()
	select {
	case value := <-result:
		return value
	case <-time.After(20 * time.Second):
		t.Fatal("password change did not complete after the barrier was released")
		return asyncPasswordChangeResult{}
	}
}

func deactivateAndReactivateUser(t *testing.T, svc *Service, userID string) {
	t.Helper()
	if _, err := svc.store.UpdateUserRoleOrStatus(
		context.Background(),
		userID,
		"",
		"inactive",
	); err != nil {
		t.Fatalf("deactivate user: %v", err)
	}
	if _, err := svc.store.UpdateUserRoleOrStatus(
		context.Background(),
		userID,
		"",
		defaultUserStatusActive,
	); err != nil {
		t.Fatalf("reactivate user: %v", err)
	}
}

func assertRejectedStaleLogin(t *testing.T, result asyncLoginResult) {
	t.Helper()
	if !errors.Is(result.err, ErrInvalidCredentials) {
		t.Fatalf("Login() error = %v, want generic ErrInvalidCredentials", result.err)
	}
	if result.token != "" || result.user != nil {
		t.Fatalf("rejected login returned token=%q user=%#v", result.token, result.user)
	}
}

func activeSessionCount(t *testing.T, svc *Service, userID string) int {
	t.Helper()
	var count int
	if err := svc.store.db.QueryRowContext(
		context.Background(),
		`SELECT COUNT(*) FROM auth_sessions
		 WHERE user_id = ? AND revoked_at IS NULL AND expires_at > ?`,
		userID,
		time.Now().Unix(),
	).Scan(&count); err != nil {
		t.Fatalf("count active sessions: %v", err)
	}
	return count
}
