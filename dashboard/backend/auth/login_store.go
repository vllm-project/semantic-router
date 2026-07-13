package auth

import (
	"context"
	"errors"
)

// CompleteLogin commits all state derived from one password verification in a
// single transaction. The expected hash and monotonic auth generation are the
// credential snapshot that was actually verified; if another transaction
// changes the password or crosses an account-status boundary first, no login
// timestamp or session can be written. Legacy bcrypt rows remain byte-stable
// during login so the first dual-read release is safe to roll back.
func (s *Store) CompleteLogin(
	ctx context.Context,
	userID string,
	expectedHash string,
	expectedAuthGeneration int64,
	issued *issuedToken,
) error {
	if issued == nil {
		return errors.New("issued session is required")
	}
	if expectedHash == "" {
		return ErrLoginStateChanged
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()

	now := nowUnix()
	result, err := tx.ExecContext(
		ctx,
		`UPDATE users
		 SET password_hash = ?, last_login_at = ?, updated_at = ?
		 WHERE id = ? AND password_hash = ? AND auth_generation = ? AND status = ?`,
		expectedHash,
		now,
		now,
		userID,
		expectedHash,
		expectedAuthGeneration,
		defaultUserStatusActive,
	)
	if err != nil {
		return err
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if affected != 1 {
		return ErrLoginStateChanged
	}
	if err := insertIssuedSessionTx(ctx, tx, userID, issued); err != nil {
		return err
	}
	return tx.Commit()
}
