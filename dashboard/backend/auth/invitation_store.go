package auth

import (
	"context"
	"database/sql"
	"errors"
	"strings"

	"github.com/google/uuid"
)

const (
	InvitationPending  = "pending"
	InvitationAccepted = "accepted"
	InvitationRevoked  = "revoked"
	InvitationExpired  = "expired"
)

const (
	InvitationPersonal = "personal"
	InvitationShared   = "shared"
)

var ErrInvitationUnavailable = errors.New("invitation is unavailable")

type Invitation struct {
	ID            string `json:"id"`
	Email         string `json:"email,omitempty"`
	Name          string `json:"name,omitempty"`
	Role          string `json:"role"`
	Kind          string `json:"kind"`
	MaxUses       int    `json:"maxUses"`
	UsedCount     int    `json:"usedCount"`
	RemainingUses int    `json:"remainingUses"`
	Status        string `json:"status"`
	ExpiresAt     int64  `json:"expiresAt"`
	AcceptedAt    *int64 `json:"acceptedAt,omitempty"`
	RevokedAt     *int64 `json:"revokedAt,omitempty"`
	CreatedAt     int64  `json:"createdAt"`
	CreatedBy     string `json:"createdBy,omitempty"`
}

func scanInvitation(scanner interface{ Scan(...any) error }) (*Invitation, string, error) {
	item := &Invitation{}
	var acceptedAt, revokedAt sql.NullInt64
	var digest string
	if err := scanner.Scan(
		&item.ID, &item.Email, &item.Name, &item.Role, &item.Kind, &item.MaxUses, &item.UsedCount, &digest, &item.Status,
		&item.ExpiresAt, &acceptedAt, &revokedAt, &item.CreatedAt, &item.CreatedBy,
	); err != nil {
		return nil, "", err
	}
	item.Role = canonicalRole(item.Role)
	if item.MaxUses < 1 {
		item.MaxUses = 1
	}
	item.RemainingUses = item.MaxUses - item.UsedCount
	if item.RemainingUses < 0 {
		item.RemainingUses = 0
	}
	if acceptedAt.Valid {
		value := acceptedAt.Int64
		item.AcceptedAt = &value
	}
	if revokedAt.Valid {
		value := revokedAt.Int64
		item.RevokedAt = &value
	}
	if item.Status == InvitationPending && (item.ExpiresAt <= nowUnix() || item.RemainingUses == 0) {
		if item.RemainingUses == 0 {
			item.Status = InvitationAccepted
		} else {
			item.Status = InvitationExpired
		}
	}
	return item, digest, nil
}

const invitationColumns = `id,email,name,role,kind,max_uses,used_count,token_digest,status,expires_at,accepted_at,revoked_at,created_at,created_by`

func (s *Store) HasUserEmail(ctx context.Context, email string) (bool, error) {
	var exists bool
	err := s.db.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM users WHERE email=?)`, strings.ToLower(strings.TrimSpace(email))).Scan(&exists)
	return exists, err
}

func (s *Store) CreateInvitation(
	ctx context.Context,
	kind, email, name, role, digest, createdBy string,
	maxUses int,
	expiresAt int64,
) (*Invitation, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback() }()
	now := nowUnix()
	if kind == InvitationPersonal {
		if _, updateErr := tx.ExecContext(ctx, `UPDATE dashboard_invitations SET status=?,revoked_at=? WHERE email=? AND kind=? AND status=?`, InvitationRevoked, now, email, InvitationPersonal, InvitationPending); updateErr != nil {
			return nil, updateErr
		}
	}
	id := uuid.NewString()
	if _, insertErr := tx.ExecContext(ctx, `INSERT INTO dashboard_invitations(id,email,name,role,kind,max_uses,used_count,token_digest,status,expires_at,created_at,created_by) VALUES(?,?,?,?,?,?,0,?,?,?,?,?)`, id, email, name, role, kind, maxUses, digest, InvitationPending, expiresAt, now, createdBy); insertErr != nil {
		return nil, insertErr
	}
	if commitErr := tx.Commit(); commitErr != nil {
		return nil, commitErr
	}
	item, _, lookupErr := s.GetInvitationByID(ctx, id)
	return item, lookupErr
}

func (s *Store) GetInvitationByID(ctx context.Context, id string) (*Invitation, string, error) {
	return scanInvitation(s.db.QueryRowContext(ctx, `SELECT `+invitationColumns+` FROM dashboard_invitations WHERE id=?`, id))
}

func (s *Store) GetInvitationByDigest(ctx context.Context, digest string) (*Invitation, string, error) {
	return scanInvitation(s.db.QueryRowContext(ctx, `SELECT `+invitationColumns+` FROM dashboard_invitations WHERE token_digest=?`, digest))
}

func (s *Store) ListInvitations(ctx context.Context) ([]*Invitation, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT `+invitationColumns+` FROM dashboard_invitations ORDER BY created_at DESC`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	items := make([]*Invitation, 0)
	for rows.Next() {
		item, _, scanErr := scanInvitation(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (s *Store) RotateInvitation(ctx context.Context, id, digest string, expiresAt int64) (*Invitation, error) {
	result, err := s.db.ExecContext(ctx, `UPDATE dashboard_invitations SET token_digest=?,expires_at=? WHERE id=? AND status=?`, digest, expiresAt, id, InvitationPending)
	if err != nil {
		return nil, err
	}
	if affected, _ := result.RowsAffected(); affected != 1 {
		return nil, ErrInvitationUnavailable
	}
	item, _, err := s.GetInvitationByID(ctx, id)
	return item, err
}

func (s *Store) RevokeInvitation(ctx context.Context, id string) error {
	now := nowUnix()
	result, err := s.db.ExecContext(ctx, `UPDATE dashboard_invitations SET status=?,revoked_at=? WHERE id=? AND status=?`, InvitationRevoked, now, id, InvitationPending)
	if err != nil {
		return err
	}
	if affected, _ := result.RowsAffected(); affected != 1 {
		return ErrInvitationUnavailable
	}
	return nil
}

func (s *Store) AcceptInvitation(ctx context.Context, digest, email, name, passwordHash string) (*User, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback() }()
	item, _, invitationErr := scanInvitation(tx.QueryRowContext(ctx, `SELECT `+invitationColumns+` FROM dashboard_invitations WHERE token_digest=?`, digest))
	if invitationErr != nil {
		return nil, invitationErr
	}
	if item.Status != InvitationPending || item.ExpiresAt <= nowUnix() || item.UsedCount >= item.MaxUses {
		return nil, ErrInvitationUnavailable
	}
	now := nowUnix()
	user := &User{ID: uuid.NewString(), Email: strings.ToLower(email), Name: name, Role: item.Role, Status: defaultUserStatusActive, CreatedAt: now, UpdatedAt: now}
	if _, insertErr := tx.ExecContext(ctx, `INSERT INTO users(id,email,name,password_hash,role,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)`, user.ID, user.Email, user.Name, passwordHash, user.Role, user.Status, now, now); insertErr != nil {
		return nil, insertErr
	}
	result, err := tx.ExecContext(ctx, `UPDATE dashboard_invitations SET used_count=used_count+1,status=CASE WHEN used_count+1>=max_uses THEN ? ELSE status END,accepted_at=CASE WHEN used_count+1>=max_uses THEN ? ELSE accepted_at END WHERE id=? AND status=? AND used_count<max_uses`, InvitationAccepted, now, item.ID, InvitationPending)
	if err != nil {
		return nil, err
	}
	if affected, _ := result.RowsAffected(); affected != 1 {
		return nil, ErrInvitationUnavailable
	}
	if commitErr := tx.Commit(); commitErr != nil {
		return nil, commitErr
	}
	return user, nil
}
