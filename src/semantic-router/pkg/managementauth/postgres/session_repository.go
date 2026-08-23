package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

func (s *Store) Create(ctx context.Context, draft managementauth.SessionDraft) (managementauth.LiveSession, error) {
	return inTransaction(ctx, s, sql.LevelSerializable, func(tx *sql.Tx) (managementauth.LiveSession, error) {
		return s.CreateInTransaction(ctx, tx, draft)
	})
}

// CreateInTransaction is the narrow shared-transaction seam used by the
// atomic identity-exchange coordinator. The caller owns commit and rollback.
func (s *Store) CreateInTransaction(ctx context.Context, tx *sql.Tx,
	draft managementauth.SessionDraft,
) (managementauth.LiveSession, error) {
	if s == nil || s.db == nil || tx == nil {
		return managementauth.LiveSession{}, managementauth.ErrAuthenticationUnavailable
	}
	var principalStatus string
	if err := tx.QueryRowContext(ctx, lockPrincipalQuery, draft.PrincipalID).Scan(&principalStatus); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return managementauth.LiveSession{}, managementauth.ErrSessionNotFound
		}
		return managementauth.LiveSession{}, fmt.Errorf("lock management principal: %w", err)
	}
	if principalStatus != string(managementauth.ResourceActive) {
		return managementauth.LiveSession{}, managementauth.ErrSessionInactive
	}

	var (
		serverNow  time.Time
		sessionTTL int64
		maxActive  int
	)
	if err := tx.QueryRowContext(ctx, loadSessionPolicyQuery).Scan(&serverNow, &sessionTTL, &maxActive); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return managementauth.LiveSession{}, fmt.Errorf("management session policy is unavailable")
		}
		return managementauth.LiveSession{}, fmt.Errorf("load management session policy: %w", err)
	}
	const maximumDurationSeconds = int64((time.Duration(1<<63 - 1)) / time.Second)
	if sessionTTL <= 0 || sessionTTL > maximumDurationSeconds || maxActive <= 0 {
		return managementauth.LiveSession{}, fmt.Errorf("management session policy is invalid")
	}
	serverNow = serverNow.UTC()
	if draft.EvidenceExpiresAt.IsZero() || !draft.EvidenceExpiresAt.After(serverNow) {
		return managementauth.LiveSession{}, managementauth.ErrSessionInactive
	}
	if _, err := tx.ExecContext(ctx, expireSessionsQuery, draft.PrincipalID, serverNow); err != nil {
		return managementauth.LiveSession{}, fmt.Errorf("expire stale management sessions: %w", err)
	}
	var activeCount int
	if err := tx.QueryRowContext(ctx, countActiveSessionsQuery, draft.PrincipalID, serverNow).Scan(&activeCount); err != nil {
		return managementauth.LiveSession{}, fmt.Errorf("count active management sessions: %w", err)
	}
	if activeCount >= maxActive {
		return managementauth.LiveSession{}, managementauth.ErrSessionLimitExceeded
	}

	expiresAt := serverNow.Add(time.Duration(sessionTTL) * time.Second)
	if draft.EvidenceExpiresAt.Before(expiresAt) {
		expiresAt = draft.EvidenceExpiresAt.UTC()
	}
	issuerSessionID := cloneStringPointer(draft.IssuerSessionID)
	human, workload := cloneEvidence(draft.Human, draft.Workload)
	session := managementauth.Session{
		ID: draft.ID, PrincipalID: draft.PrincipalID, IssuerSessionID: issuerSessionID,
		TokenID: draft.TokenID, Audience: draft.Audience, AuthSourceKind: draft.AuthSourceKind,
		AuthSourceID: draft.AuthSourceID, EvidenceKind: draft.EvidenceKind,
		Human: human, Workload: workload, AuthenticatedAt: draft.AuthenticatedAt.UTC(),
		ExpiresAt: expiresAt, Status: managementauth.SessionActive, CreatedAt: serverNow,
	}
	if err := session.Validate(); err != nil {
		return managementauth.LiveSession{}, fmt.Errorf("validate management session draft: %w", err)
	}
	assurance, sourceAssuredAt, createInTransactionErr := encodeAssurance(session)
	if createInTransactionErr != nil {
		return managementauth.LiveSession{}, createInTransactionErr
	}
	result, createInTransactionErr := tx.ExecContext(
		ctx, insertSessionQuery, session.ID, session.PrincipalID, session.IssuerSessionID,
		session.TokenID, session.Audience, session.AuthSourceKind, session.AuthSourceID,
		session.EvidenceKind, assurance, session.AuthenticatedAt, sourceAssuredAt,
		session.ExpiresAt, session.CreatedAt,
	)
	if createInTransactionErr != nil {
		if isUniqueViolation(createInTransactionErr) {
			return managementauth.LiveSession{}, managementauth.ErrSessionConflict
		}
		return managementauth.LiveSession{}, fmt.Errorf("insert management session: %w", createInTransactionErr)
	}
	if err := requireOneRow(result); err != nil {
		return managementauth.LiveSession{}, fmt.Errorf("insert management session: %w", err)
	}
	live, createInTransactionErr := getWith(ctx, tx, session.ID)
	if createInTransactionErr != nil {
		return managementauth.LiveSession{}, createInTransactionErr
	}
	if err := live.ValidateAt(serverNow); err != nil {
		return managementauth.LiveSession{}, fmt.Errorf("validate committed management session source: %w", err)
	}
	return live, nil
}

func (s *Store) Get(ctx context.Context, sessionID string) (managementauth.LiveSession, error) {
	if _, err := uuid.Parse(sessionID); err != nil {
		return managementauth.LiveSession{}, managementauth.ErrSessionNotFound
	}
	return getWith(ctx, s.db, sessionID)
}

// GetInTransaction loads a committed or transaction-local session for atomic
// exchange replay. The caller owns the transaction lifecycle.
func (s *Store) GetInTransaction(ctx context.Context, tx *sql.Tx, sessionID string) (managementauth.LiveSession, error) {
	if s == nil || s.db == nil || tx == nil {
		return managementauth.LiveSession{}, managementauth.ErrAuthenticationUnavailable
	}
	if _, err := uuid.Parse(sessionID); err != nil {
		return managementauth.LiveSession{}, managementauth.ErrSessionNotFound
	}
	return getWith(ctx, tx, sessionID)
}
