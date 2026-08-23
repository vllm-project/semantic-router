package auth

import (
	"context"
	"database/sql"
	"strings"
)

const invitationPresentationColumns = `router_invitation_id,router_namespace_id,router_revision,email,name,
token_digest,planned_subject_id,presentation_status,expires_at,accepted_at,revoked_at,created_at,
created_by,updated_at,last_sent_at,delivery_status,delivery_error`

type localSessionDraft struct {
	ID        string
	IssuedAt  int64
	ExpiresAt int64
}

func scanInvitationPresentation(scanner interface{ Scan(...any) error }) (*invitationPresentation, error) {
	item := &invitationPresentation{}
	var acceptedAt, revokedAt, lastSentAt sql.NullInt64
	err := scanner.Scan(
		&item.RouterInvitationID, &item.RouterNamespaceID, &item.RouterRevision,
		&item.Email, &item.Name, &item.TokenDigest, &item.PlannedSubjectID, &item.Status,
		&item.ExpiresAt, &acceptedAt, &revokedAt, &item.CreatedAt, &item.CreatedBy,
		&item.UpdatedAt, &lastSentAt, &item.DeliveryStatus, &item.DeliveryError,
	)
	if err != nil {
		return nil, err
	}
	if acceptedAt.Valid {
		value := acceptedAt.Int64
		item.AcceptedAt = &value
	}
	if revokedAt.Valid {
		value := revokedAt.Int64
		item.RevokedAt = &value
	}
	if lastSentAt.Valid {
		value := lastSentAt.Int64
		item.LastSentAt = &value
	}
	return item, nil
}

func (s *Store) ListInvitationPresentations(ctx context.Context, namespaceID string) (map[string]invitationPresentation, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT `+invitationPresentationColumns+`
FROM dashboard_member_invitations WHERE router_namespace_id=? ORDER BY created_at DESC`, namespaceID)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	items := map[string]invitationPresentation{}
	for rows.Next() {
		item, scanErr := scanInvitationPresentation(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		items[item.RouterInvitationID] = *item
	}
	return items, rows.Err()
}

func (s *Store) CreateInvitationPresentation(ctx context.Context, item invitationPresentation) (*invitationPresentation, error) {
	now := nowUnix()
	_, err := s.db.ExecContext(ctx, `
INSERT INTO dashboard_member_invitations(
 router_invitation_id,router_namespace_id,router_revision,email,name,token_digest,
 planned_subject_id,presentation_status,expires_at,created_at,created_by,updated_at,delivery_status
) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)`, item.RouterInvitationID, item.RouterNamespaceID,
		item.RouterRevision, strings.ToLower(item.Email), item.Name, item.TokenDigest,
		item.PlannedSubjectID, InvitationPending, item.ExpiresAt, now, item.CreatedBy, now,
		item.DeliveryStatus)
	if err != nil {
		return nil, err
	}
	return s.GetInvitationPresentationByID(ctx, item.RouterInvitationID)
}

func (s *Store) GetInvitationPresentationByID(ctx context.Context, id string) (*invitationPresentation, error) {
	return scanInvitationPresentation(s.db.QueryRowContext(ctx, `SELECT `+invitationPresentationColumns+`
FROM dashboard_member_invitations WHERE router_invitation_id=?`, id))
}

func (s *Store) GetInvitationPresentationByDigest(ctx context.Context, digest string) (*invitationPresentation, error) {
	return scanInvitationPresentation(s.db.QueryRowContext(ctx, `SELECT `+invitationPresentationColumns+`
FROM dashboard_member_invitations WHERE token_digest=?`, digest))
}

func (s *Store) RotateInvitationPresentation(ctx context.Context, id string, revision uint64, digest string, expiresAt int64) (*invitationPresentation, error) {
	result, err := s.db.ExecContext(ctx, `
UPDATE dashboard_member_invitations
SET router_revision=?,token_digest=?,expires_at=?,updated_at=?,last_sent_at=NULL,
    delivery_status='not_requested',delivery_error=''
WHERE router_invitation_id=? AND presentation_status='pending'`,
		revision, digest, expiresAt, nowUnix(), id)
	if err != nil {
		return nil, err
	}
	if affected, _ := result.RowsAffected(); affected != 1 {
		return nil, sql.ErrNoRows
	}
	return s.GetInvitationPresentationByID(ctx, id)
}

func (s *Store) UpdateInvitationDelivery(ctx context.Context, id, status, deliveryError string) error {
	_, err := s.db.ExecContext(ctx, `
UPDATE dashboard_member_invitations
SET delivery_status=?,delivery_error=?,last_sent_at=?,updated_at=? WHERE router_invitation_id=?`,
		status, deliveryError, nowUnix(), nowUnix(), id)
	return err
}

func (s *Store) MarkInvitationRevoked(ctx context.Context, id string, revision uint64) (*invitationPresentation, error) {
	now := nowUnix()
	result, err := s.db.ExecContext(ctx, `
UPDATE dashboard_member_invitations
SET router_revision=?,presentation_status='revoked',revoked_at=?,updated_at=?
WHERE router_invitation_id=? AND presentation_status='pending'`, revision, now, now, id)
	if err != nil {
		return nil, err
	}
	if affected, _ := result.RowsAffected(); affected != 1 {
		return nil, sql.ErrNoRows
	}
	return s.GetInvitationPresentationByID(ctx, id)
}

// CompleteRouterInvitation is the only local acceptance mutation. It runs
// strictly after Router onboarding and atomically commits the Dashboard user,
// browser session, and presentation marker. No Router API key is accepted as
// input, so plaintext cannot accidentally cross the SQLite seam.
func (s *Store) CompleteRouterInvitation(
	ctx context.Context,
	digest string,
	user User,
	passwordHash string,
	session localSessionDraft,
) (*User, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback() }()

	var invitationID, plannedSubjectID, status string
	var expiresAt int64
	if queryErr := tx.QueryRowContext(ctx, `SELECT router_invitation_id,planned_subject_id,presentation_status,expires_at
FROM dashboard_member_invitations WHERE token_digest=?`, digest).
		Scan(&invitationID, &plannedSubjectID, &status, &expiresAt); queryErr != nil {
		return nil, queryErr
	}
	if status != InvitationPending || expiresAt <= nowUnix() || plannedSubjectID != user.ID {
		return nil, ErrInvitationUnavailable
	}
	if user.Role != RoleAdmin && user.Role != RoleWrite && user.Role != RoleRead {
		return nil, ErrInvitationAuthorityUnavailable
	}
	_, err = tx.ExecContext(ctx, `
INSERT INTO users(id,email,name,password_hash,role,status,created_at,updated_at)
VALUES(?,?,?,?,?,?,?,?)`, user.ID, strings.ToLower(user.Email), user.Name, passwordHash,
		user.Role, user.Status, user.CreatedAt, user.UpdatedAt)
	if err != nil {
		return nil, err
	}
	if _, err = tx.ExecContext(ctx, `INSERT INTO auth_sessions(id,user_id,issued_at,expires_at)
VALUES(?,?,?,?)`, session.ID, user.ID, session.IssuedAt, session.ExpiresAt); err != nil {
		return nil, err
	}
	now := nowUnix()
	result, err := tx.ExecContext(ctx, `UPDATE dashboard_member_invitations
SET presentation_status='accepted',accepted_at=?,updated_at=?
WHERE router_invitation_id=? AND presentation_status='pending'`, now, now, invitationID)
	if err != nil {
		return nil, err
	}
	if affected, _ := result.RowsAffected(); affected != 1 {
		return nil, ErrInvitationUnavailable
	}
	if s.invitationBeforeCommit != nil {
		if err := s.invitationBeforeCommit(); err != nil {
			return nil, err
		}
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return &user, nil
}

func presentationAvailable(item *invitationPresentation) error {
	if item == nil || item.Status != InvitationPending || item.ExpiresAt <= nowUnix() {
		return ErrInvitationUnavailable
	}
	if item.RouterInvitationID == "" || item.RouterNamespaceID == "" || item.PlannedSubjectID == "" {
		return ErrInvitationAuthorityUnavailable
	}
	return nil
}
