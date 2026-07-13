package auth

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestChangePasswordCurrentVerificationIsRateLimited(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	svc.passwordChangeLimiter = NewLoginLimiter(LoginLimiterConfig{
		AccountFailureThreshold: 1,
		SourceFailureThreshold:  100,
		BaseDelay:               3 * time.Second,
		MaxDelay:                3 * time.Second,
	})
	user := newTestUser(t, svc, "change-rate-limit@example.com", RoleRead, defaultUserStatusActive)
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}
	verifyCalls := 0
	originalVerify := svc.verify
	svc.verify = func(hash, password string) bool {
		verifyCalls++
		return originalVerify(hash, password)
	}

	request := func() *httptest.ResponseRecorder {
		recorder := httptest.NewRecorder()
		req := httptest.NewRequest(
			http.MethodPost,
			"/api/auth/password",
			strings.NewReader(`{"currentPassword":"incorrect-password-value","newPassword":"new unique password value 2026"}`),
		)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+token)
		AuthenticateRequest(svc)(changePasswordHandler(svc)).ServeHTTP(recorder, req)
		return recorder
	}

	if first := request(); first.Code != http.StatusForbidden {
		t.Fatalf("first status = %d, want 403; body=%s", first.Code, first.Body.String())
	}
	second := request()
	if second.Code != http.StatusTooManyRequests {
		t.Fatalf("second status = %d, want 429; body=%s", second.Code, second.Body.String())
	}
	if second.Header().Get("Retry-After") != "3" {
		t.Fatalf("Retry-After = %q, want 3", second.Header().Get("Retry-After"))
	}
	if verifyCalls != 1 {
		t.Fatalf("password verifier calls = %d, want 1", verifyCalls)
	}
	assertSessionTokenStatus(t, svc, token, http.StatusNoContent)
}

func TestSuccessfulCurrentPasswordClearsAccountFailuresBeforePolicyValidation(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	svc.passwordChangeLimiter = NewLoginLimiter(LoginLimiterConfig{
		AccountFailureThreshold: 2,
		SourceFailureThreshold:  100,
		BaseDelay:               time.Minute,
		MaxDelay:                time.Minute,
	})
	user := newTestUser(t, svc, "reauth-reset@example.com", RoleRead, defaultUserStatusActive)

	_, _, err := svc.ChangePassword(
		context.Background(),
		user.ID,
		"incorrect-password-value",
		"new unique password value 2026",
	)
	if !errors.Is(err, ErrCurrentPasswordFailed) {
		t.Fatalf("first change error = %v, want ErrCurrentPasswordFailed", err)
	}
	_, _, err = svc.ChangePassword(
		context.Background(),
		user.ID,
		"secret-password",
		"short",
	)
	var policyErr *PasswordPolicyError
	if !errors.As(err, &policyErr) || policyErr.Code != PasswordPolicyTooShort {
		t.Fatalf("reauthenticated change error = %#v, want too-short policy error", err)
	}
	_, _, err = svc.ChangePassword(
		context.Background(),
		user.ID,
		"incorrect-password-value",
		"new unique password value 2026",
	)
	if !errors.Is(err, ErrCurrentPasswordFailed) {
		t.Fatalf("post-reauth change error = %v, want ErrCurrentPasswordFailed", err)
	}
	if retry := svc.passwordChangeLimiter.RetryAfter(user.Email, ""); retry != 0 {
		t.Fatalf("retry after one post-reauth failure = %v, want 0", retry)
	}
}

func TestGlobalLoginBudgetCannotBlockAuthenticatedPasswordChange(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "management-reserve@example.com", RoleRead, defaultUserStatusActive)
	svc.limiter = NewLoginLimiter(LoginLimiterConfig{
		AccountFailureThreshold: 100,
		SourceFailureThreshold:  100,
		GlobalAttemptBurst:      1,
		GlobalRefillInterval:    time.Hour,
	})
	attacker, retry := svc.limiter.Reserve("random-attacker@example.com", "")
	if attacker == nil || retry != 0 {
		t.Fatalf("attacker Reserve() = (%#v, %v), want initial admission", attacker, retry)
	}
	attacker.Fail()
	if attempt, retry := svc.limiter.Reserve("another-random@example.com", ""); attempt != nil || retry <= 0 {
		t.Fatalf("global login budget was not exhausted: (%#v, %v)", attempt, retry)
	}

	token, _, err := svc.ChangePassword(
		context.Background(),
		user.ID,
		"secret-password",
		"management path replacement password 2026",
	)
	if err != nil {
		t.Fatalf("ChangePassword() error under exhausted login budget = %v", err)
	}
	assertSessionTokenStatus(t, svc, token, http.StatusNoContent)
}

func TestAnonymousLoginEvidenceCannotBlockEmergencyPasswordChange(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "emergency-rotation@example.com", RoleRead, defaultUserStatusActive)
	svc.limiter = NewLoginLimiter(LoginLimiterConfig{
		AccountFailureThreshold: 1,
		SourceFailureThreshold:  1,
		MaxAccounts:             1,
		MaxSources:              1,
		GlobalAttemptBurst:      10,
		BaseDelay:               time.Minute,
		MaxDelay:                time.Minute,
	})
	source := "198.51.100.20"
	svc.limiter.RecordFailure(user.Email, source)
	svc.limiter.RecordFailure("overflow-attacker@example.com", "198.51.100.21")
	if retry := svc.limiter.RetryAfter(user.Email, source); retry <= 0 {
		t.Fatal("anonymous login evidence did not block the login path fixture")
	}

	token, _, err := svc.ChangePasswordWithSource(
		context.Background(),
		user.ID,
		"secret-password",
		"emergency rotation replacement 2026",
		source,
	)
	if err != nil {
		t.Fatalf("ChangePasswordWithSource() inherited anonymous limiter state: %v", err)
	}
	assertSessionTokenStatus(t, svc, token, http.StatusNoContent)
}
