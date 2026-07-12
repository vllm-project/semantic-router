package auth

import "context"

func (s *Store) createUserAuthorized(
	ctx context.Context,
	authorization *mutationAuthorization,
	email string,
	name string,
	hash string,
	role string,
	status string,
) (*User, error) {
	user, err := prepareNewUser(email, name, role, status)
	if err != nil {
		return nil, err
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback() }()
	if err := requireMutationAuthorizationTx(ctx, tx, authorization); err != nil {
		return nil, err
	}
	if err := insertPreparedUser(ctx, tx, user, hash); err != nil {
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return user, nil
}
