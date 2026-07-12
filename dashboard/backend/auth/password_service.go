package auth

import (
	"context"
	"strings"
	"time"
)

const (
	defaultLoginPasswordWorkConcurrency      = 3
	defaultManagementPasswordWorkConcurrency = 1
)

func (s *Service) HashPassword(password string) (string, error) {
	return s.HashPasswordForUser("", password)
}

func (s *Service) HashPasswordForUser(email, password string) (string, error) {
	normalized, err := s.policy.Validate(email, password)
	if err != nil {
		return "", err
	}
	release, ok := acquirePasswordWork(s.managementPasswordWork)
	if !ok {
		return "", ErrPasswordWorkSaturated
	}
	defer release()
	return hashVersionedPassword(normalized)
}

func (s *Service) VerifyPassword(hash, password string) bool {
	return verifyStoredPassword(hash, password)
}

func (s *Service) ChangePassword(
	ctx context.Context,
	userID string,
	currentPassword string,
	newPassword string,
) (string, *User, error) {
	return s.ChangePasswordWithSource(ctx, userID, currentPassword, newPassword, "")
}

func (s *Service) ChangePasswordWithSource(
	ctx context.Context,
	userID string,
	currentPassword string,
	newPassword string,
	source string,
) (string, *User, error) {
	passwordState, err := s.store.getUserPasswordStateByID(ctx, userID)
	if err != nil {
		return "", nil, err
	}
	user := passwordState.user
	currentHash := passwordState.hash
	account := strings.ToLower(strings.TrimSpace(user.Email))
	attempt, retryAfter := s.passwordChangeLimiter.ReservePasswordManagement(account, source)
	if attempt == nil {
		return "", nil, &LoginRateLimitError{RetryAfter: retryAfter}
	}
	defer attempt.Cancel()
	release, ok := acquirePasswordWork(s.managementPasswordWork)
	if !ok {
		return "", nil, &LoginRateLimitError{RetryAfter: time.Second}
	}
	defer release()
	if user.Status != defaultUserStatusActive || !s.verify(currentHash, currentPassword) {
		attempt.Fail()
		return "", nil, ErrCurrentPasswordFailed
	}
	// A successful current-password reauthentication clears prior account
	// failures even if the proposed replacement is later rejected by policy.
	// Source evidence remains independent to preserve password-spray detection.
	attempt.Succeed()
	normalizedNewPassword, err := s.policy.Validate(user.Email, newPassword)
	if err != nil {
		return "", nil, err
	}
	if passwordsEquivalent(currentPassword, normalizedNewPassword) {
		return "", nil, &PasswordPolicyError{
			Code:    PasswordPolicyUnchanged,
			Message: "new password must differ from the current password",
		}
	}
	newHash, err := hashVersionedPassword(normalizedNewPassword)
	if err != nil {
		return "", nil, err
	}
	issued, err := s.prepareToken(user)
	if err != nil {
		return "", nil, err
	}
	if err := s.store.ChangePasswordAndReplaceSessions(
		ctx,
		user.ID,
		currentHash,
		passwordState.authGeneration,
		newHash,
		issued,
	); err != nil {
		return "", nil, err
	}
	return issued.signed, user, nil
}

func (s *Service) ResetPassword(ctx context.Context, userID, newPassword string) error {
	return s.resetPassword(ctx, nil, userID, newPassword)
}

func (s *Service) resetPasswordAuthorized(
	ctx context.Context,
	authorization *mutationAuthorization,
	userID string,
	newPassword string,
) error {
	return s.resetPassword(ctx, authorization, userID, newPassword)
}

func (s *Service) resetPassword(
	ctx context.Context,
	authorization *mutationAuthorization,
	userID string,
	newPassword string,
) error {
	user, err := s.store.GetUserByID(ctx, userID)
	if err != nil {
		return err
	}
	newHash, err := s.HashPasswordForUser(user.Email, newPassword)
	if err != nil {
		return err
	}
	if authorization == nil {
		return s.store.UpdatePasswordAndRevokeSessions(ctx, user.ID, newHash)
	}
	return s.store.updatePasswordAndRevokeSessionsAuthorized(
		ctx,
		authorization,
		user.ID,
		newHash,
	)
}

func acquirePasswordWork(work chan struct{}) (func(), bool) {
	select {
	case work <- struct{}{}:
		return func() { <-work }, true
	default:
		return nil, false
	}
}
