package auth

import (
	"context"
	"database/sql"
	"errors"
	"strings"
	"time"

	"github.com/google/uuid"
)

func scanInvitation(scanner interface{ Scan(...any) error }) (*DashboardMemberInvitation, error) {
	item := &DashboardMemberInvitation{}
	var acceptedAt, revokedAt, lastSentAt sql.NullInt64
	err := scanner.Scan(
		&item.ID, &item.Email, &item.Name, &item.Role, &item.TeamID,
		&item.Status, &item.ExpiresAt, &acceptedAt, &revokedAt, &item.CreatedAt,
		&item.CreatedBy, &item.UpdatedAt, &lastSentAt, &item.DeliveryStatus, &item.DeliveryError,
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
	item.Status = item.effectiveStatus(time.Now())
	return item, nil
}

const invitationSelectColumns = `id,email,name,role,team_id,status,expires_at,accepted_at,revoked_at,created_at,created_by,updated_at,last_sent_at,delivery_status,delivery_error`

func (s *Store) ListInvitations(ctx context.Context) ([]DashboardMemberInvitation, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT `+invitationSelectColumns+` FROM dashboard_member_invitations ORDER BY created_at DESC`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	items := []DashboardMemberInvitation{}
	for rows.Next() {
		item, scanErr := scanInvitation(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		items = append(items, *item)
	}
	return items, rows.Err()
}

func (s *Store) CreateInvitation(ctx context.Context, item DashboardMemberInvitation, digest string) (*DashboardMemberInvitation, error) {
	if _, err := s.db.ExecContext(ctx, `UPDATE dashboard_member_invitations SET status='expired',updated_at=? WHERE status='pending' AND expires_at<=?`, nowUnix(), nowUnix()); err != nil {
		return nil, err
	}
	if item.ID == "" {
		item.ID = uuid.NewString()
	}
	now := nowUnix()
	_, err := s.db.ExecContext(ctx, `
INSERT INTO dashboard_member_invitations(
 id,email,name,role,team_id,token_digest,status,expires_at,created_at,created_by,updated_at,delivery_status
) VALUES(?,?,?,?,?,?,?, ?,?,?,?,?)`, item.ID, strings.ToLower(item.Email), item.Name, item.Role,
		item.TeamID, digest, InvitationPending, item.ExpiresAt, now, item.CreatedBy, now, item.DeliveryStatus)
	if err != nil {
		return nil, err
	}
	return s.GetInvitationByID(ctx, item.ID)
}

func (s *Store) GetInvitationByID(ctx context.Context, id string) (*DashboardMemberInvitation, error) {
	return scanInvitation(s.db.QueryRowContext(ctx, `SELECT `+invitationSelectColumns+` FROM dashboard_member_invitations WHERE id=?`, id))
}

func (s *Store) GetInvitationByDigest(ctx context.Context, digest string) (*DashboardMemberInvitation, error) {
	return scanInvitation(s.db.QueryRowContext(ctx, `SELECT `+invitationSelectColumns+` FROM dashboard_member_invitations WHERE token_digest=?`, digest))
}

func (s *Store) RotateInvitation(ctx context.Context, id, digest string, expiresAt int64) (*DashboardMemberInvitation, error) {
	result, err := s.db.ExecContext(ctx, `
UPDATE dashboard_member_invitations
SET token_digest=?, expires_at=?, updated_at=?, last_sent_at=NULL, delivery_status='not_requested', delivery_error=''
WHERE id=? AND status='pending'`, digest, expiresAt, nowUnix(), id)
	if err != nil {
		return nil, err
	}
	if affected, _ := result.RowsAffected(); affected == 0 {
		return nil, sql.ErrNoRows
	}
	return s.GetInvitationByID(ctx, id)
}

func (s *Store) UpdateInvitationDelivery(ctx context.Context, id, status, deliveryError string) error {
	_, err := s.db.ExecContext(ctx, `
UPDATE dashboard_member_invitations
SET delivery_status=?, delivery_error=?, last_sent_at=?, updated_at=? WHERE id=?`,
		status, deliveryError, nowUnix(), nowUnix(), id)
	return err
}

func (s *Store) RevokeInvitation(ctx context.Context, id string) (*DashboardMemberInvitation, error) {
	now := nowUnix()
	result, err := s.db.ExecContext(ctx, `
UPDATE dashboard_member_invitations SET status='revoked', revoked_at=?, updated_at=?
WHERE id=? AND status='pending'`, now, now, id)
	if err != nil {
		return nil, err
	}
	if affected, _ := result.RowsAffected(); affected == 0 {
		return nil, sql.ErrNoRows
	}
	return s.GetInvitationByID(ctx, id)
}

func (s *Store) AcceptInvitation(ctx context.Context, digest, userID, passwordHash, acceptedName string) (*User, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback() }()

	var invitation DashboardMemberInvitation
	var storedName, storedStatus string
	err = tx.QueryRowContext(ctx, `
SELECT id,email,name,role,team_id,status,expires_at
FROM dashboard_member_invitations WHERE token_digest=?`, digest).
		Scan(&invitation.ID, &invitation.Email, &storedName, &invitation.Role, &invitation.TeamID, &storedStatus, &invitation.ExpiresAt)
	if err != nil {
		return nil, err
	}
	if storedStatus != InvitationPending || invitation.ExpiresAt <= nowUnix() {
		return nil, ErrInvitationUnavailable
	}
	name := strings.TrimSpace(acceptedName)
	if name == "" {
		name = strings.TrimSpace(storedName)
	}
	if name == "" {
		return nil, errors.New("name is required")
	}

	user := &User{
		ID: userID, Email: strings.ToLower(invitation.Email), Name: name,
		Role: invitation.Role, Status: defaultUserStatusActive, CreatedAt: nowUnix(), UpdatedAt: nowUnix(),
		InferenceConsumerID: userID,
	}
	_, err = tx.ExecContext(ctx, `
INSERT INTO users(id,email,name,password_hash,role,status,created_at,updated_at,inference_consumer_id)
VALUES(?,?,?,?,?,?,?,?,?)`, user.ID, user.Email, user.Name, passwordHash, user.Role, user.Status,
		user.CreatedAt, user.UpdatedAt, user.InferenceConsumerID)
	if err != nil {
		return nil, err
	}
	now := nowUnix()
	result, err := tx.ExecContext(ctx, `
UPDATE dashboard_member_invitations SET status='accepted', accepted_at=?, updated_at=?
WHERE id=? AND status='pending'`, now, now, invitation.ID)
	if err != nil {
		return nil, err
	}
	if affected, _ := result.RowsAffected(); affected != 1 {
		return nil, ErrInvitationUnavailable
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return user, nil
}
