package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

const managementSessionListColumns = `id::text,principal_id::text,auth_source_kind,evidence_kind,
       authenticated_at,expires_at,
       CASE WHEN status='active' AND expires_at<=clock_timestamp() THEN 'expired' ELSE status END,
       revoked_at,created_at`

func (store *Store) ListManagementSessions(
	ctx context.Context,
	principalID string,
	request managementidentity.ListRequest,
) (managementidentity.ManagementSessionPage, error) {
	if !canonicalUUID(principalID) {
		return managementidentity.ManagementSessionPage{}, managementidentity.ErrNotFound
	}
	if err := validateList(request); err != nil {
		return managementidentity.ManagementSessionPage{}, err
	}
	var exists bool
	if err := store.database.QueryRowContext(ctx,
		`SELECT EXISTS(SELECT 1 FROM management_principals WHERE id=$1)`, principalID,
	).Scan(&exists); err != nil {
		return managementidentity.ManagementSessionPage{}, fmt.Errorf("check Management principal sessions: %w", err)
	}
	if !exists {
		return managementidentity.ManagementSessionPage{}, managementidentity.ErrNotFound
	}
	rows, err := store.database.QueryContext(ctx, `SELECT `+managementSessionListColumns+`
FROM management_sessions
WHERE principal_id=$1 AND ($2='' OR id>NULLIF($2,'')::uuid)
ORDER BY id LIMIT $3`, principalID, request.AfterID, request.Limit+1)
	if err != nil {
		return managementidentity.ManagementSessionPage{}, fmt.Errorf("list Management sessions: %w", err)
	}
	defer rows.Close()
	items := make([]managementidentity.ManagementSession, 0, request.Limit+1)
	for rows.Next() {
		item, err := scanManagementSession(rows)
		if err != nil {
			return managementidentity.ManagementSessionPage{}, err
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return managementidentity.ManagementSessionPage{}, fmt.Errorf("iterate Management sessions: %w", err)
	}
	page := managementidentity.ManagementSessionPage{Items: items}
	if len(items) > request.Limit {
		page.Items = items[:request.Limit]
		page.NextCursor = page.Items[len(page.Items)-1].ID
	}
	return page, nil
}

func (store *Store) RevokeSelfManagementSession(
	ctx context.Context,
	sessionID string,
	principalID string,
	actor managementidentity.MutationActor,
) (managementauth.SessionMutation, error) {
	if !canonicalUUID(sessionID) || !canonicalUUID(principalID) {
		return managementauth.SessionMutation{}, managementidentity.ErrInvalidLifecycleRequest
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementauth.SessionMutation, error) {
		var storedPrincipal, status string
		var tokenID string
		var expiresAt, databaseNow time.Time
		var revokedAt sql.NullTime
		if err := tx.QueryRowContext(ctx, `SELECT principal_id::text,token_id,status,expires_at,revoked_at,clock_timestamp()
FROM management_sessions WHERE id=$1 FOR UPDATE`, sessionID).Scan(
			&storedPrincipal, &tokenID, &status, &expiresAt, &revokedAt, &databaseNow,
		); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return managementauth.SessionMutation{}, managementidentity.ErrNotFound
			}
			return managementauth.SessionMutation{}, err
		}
		if storedPrincipal != principalID {
			return managementauth.SessionMutation{}, managementidentity.ErrNotFound
		}
		mutation := managementauth.SessionMutation{SessionID: sessionID, TokenID: tokenID}
		switch status {
		case string(managementauth.SessionActive):
			if !databaseNow.Before(expiresAt) {
				return managementauth.SessionMutation{}, managementauth.ErrSessionInactive
			}
			if err := tx.QueryRowContext(ctx, `UPDATE management_sessions
SET status='revoked',revoked_at=clock_timestamp()
WHERE id=$1 AND principal_id=$2 AND status='active' RETURNING revoked_at`, sessionID, principalID).Scan(&mutation.ChangedAt); err != nil {
				return managementauth.SessionMutation{}, managementauth.ErrSessionConflict
			}
			mutation.Changed = true
			mutation.ChangedAt = mutation.ChangedAt.UTC()
		case string(managementauth.SessionRevoked):
			if !revokedAt.Valid {
				return managementauth.SessionMutation{}, managementauth.ErrSessionConflict
			}
			mutation.ChangedAt = revokedAt.Time.UTC()
			return mutation, nil
		default:
			return managementauth.SessionMutation{}, managementauth.ErrSessionInactive
		}
		if err := appendAudit(ctx, tx, auditMutation{
			Action: "management_session.self_revoked", ResourceType: "management_session",
			ResourceID: sessionID, AfterRevision: 1, Actor: actor,
		}); err != nil {
			return managementauth.SessionMutation{}, err
		}
		return mutation, nil
	})
}

func (store *Store) RevokeManagementSession(
	ctx context.Context,
	request managementidentity.SessionRevocationCommand,
) (managementauth.SessionMutation, managementidentity.MutationResult, error) {
	if !canonicalUUID(request.SessionID) ||
		request.Command.Scope.Kind != managementcommand.ScopeCluster ||
		request.Command.PrincipalID != request.Actor.PrincipalID {
		return managementauth.SessionMutation{}, managementidentity.MutationResult{}, managementidentity.ErrInvalidLifecycleRequest
	}
	type revocation struct {
		mutation managementauth.SessionMutation
		command  managementidentity.MutationResult
	}
	value, err := inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (revocation, error) {
		if replay, found, err := commandpostgres.Lock(ctx, tx, request.Command); err != nil {
			return revocation{}, err
		} else if found {
			commandResult, err := replayMutation(replay, "management_session")
			if err != nil || commandResult.ID != request.SessionID {
				return revocation{}, managementcommand.ErrConflict
			}
			var tokenID, status string
			var revokedAt sql.NullTime
			if err := tx.QueryRowContext(ctx, `SELECT token_id,status,revoked_at
FROM management_sessions WHERE id=$1 FOR SHARE`, request.SessionID).Scan(&tokenID, &status, &revokedAt); err != nil {
				if errors.Is(err, sql.ErrNoRows) {
					return revocation{}, managementidentity.ErrNotFound
				}
				return revocation{}, err
			}
			if status != string(managementauth.SessionRevoked) || !revokedAt.Valid {
				return revocation{}, managementauth.ErrSessionConflict
			}
			return revocation{
				mutation: managementauth.SessionMutation{
					SessionID: request.SessionID, TokenID: tokenID,
					ChangedAt: revokedAt.Time.UTC(),
				},
				command: commandResult,
			}, nil
		}
		var tokenID, status string
		var expiresAt, databaseNow time.Time
		var revokedAt sql.NullTime
		if err := tx.QueryRowContext(ctx, `SELECT token_id,status,expires_at,revoked_at,clock_timestamp()
FROM management_sessions WHERE id=$1 FOR UPDATE`, request.SessionID).Scan(
			&tokenID, &status, &expiresAt, &revokedAt, &databaseNow,
		); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return revocation{}, managementidentity.ErrNotFound
			}
			return revocation{}, err
		}
		mutation := managementauth.SessionMutation{SessionID: request.SessionID, TokenID: tokenID}
		switch status {
		case string(managementauth.SessionActive):
			if !databaseNow.Before(expiresAt) {
				return revocation{}, managementauth.ErrSessionInactive
			}
			if err := tx.QueryRowContext(ctx, `UPDATE management_sessions
SET status='revoked',revoked_at=clock_timestamp()
WHERE id=$1 AND status='active' RETURNING revoked_at`, request.SessionID).Scan(&mutation.ChangedAt); err != nil {
				return revocation{}, managementauth.ErrSessionConflict
			}
			mutation.Changed = true
			mutation.ChangedAt = mutation.ChangedAt.UTC()
		case string(managementauth.SessionRevoked):
			if !revokedAt.Valid {
				return revocation{}, managementauth.ErrSessionConflict
			}
			mutation.ChangedAt = revokedAt.Time.UTC()
		default:
			return revocation{}, managementauth.ErrSessionInactive
		}
		if err := appendAudit(ctx, tx, auditMutation{
			Action: "management_session.revoked", ResourceType: "management_session",
			ResourceID: request.SessionID, AfterRevision: 1, Actor: request.Actor,
		}); err != nil {
			return revocation{}, err
		}
		stored := managementcommand.ResourceResult{
			ResourceType: "management_session", ResourceID: request.SessionID,
			ResourceRevision: 1, ResponseStatus: 200,
		}
		if err := commandpostgres.CompleteResource(ctx, tx, request.Command, stored); err != nil {
			return revocation{}, err
		}
		return revocation{
			mutation: mutation,
			command: managementidentity.MutationResult{
				Kind: stored.ResourceType, ID: stored.ResourceID, Revision: 1,
				ResponseStatus: 200,
			},
		}, nil
	})
	return value.mutation, value.command, err
}

func (store *Store) RevokePrincipalManagementSessions(
	ctx context.Context,
	request managementidentity.PrincipalSessionRevocationCommand,
) (managementidentity.PrincipalSessionRevocation, error) {
	if !canonicalUUID(request.PrincipalID) ||
		request.Command.Scope.Kind != managementcommand.ScopeCluster ||
		request.Command.PrincipalID != request.Actor.PrincipalID {
		return managementidentity.PrincipalSessionRevocation{}, managementidentity.ErrInvalidLifecycleRequest
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.PrincipalSessionRevocation, error) {
		replayed := false
		if stored, found, err := commandpostgres.Lock(ctx, tx, request.Command); err != nil {
			return managementidentity.PrincipalSessionRevocation{}, err
		} else if found {
			result, err := replayMutation(stored, "management_principal_sessions")
			if err != nil || result.ID != request.PrincipalID {
				return managementidentity.PrincipalSessionRevocation{}, managementcommand.ErrConflict
			}
			replayed = true
		}
		var principalRevision uint64
		if err := tx.QueryRowContext(ctx, `SELECT revision FROM management_principals
WHERE id=$1 FOR UPDATE`, request.PrincipalID).Scan(&principalRevision); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return managementidentity.PrincipalSessionRevocation{}, managementidentity.ErrNotFound
			}
			return managementidentity.PrincipalSessionRevocation{}, err
		}
		changed := 0
		if !replayed {
			if _, err := tx.ExecContext(ctx, `UPDATE management_sessions SET status='expired'
WHERE principal_id=$1 AND status='active' AND expires_at<=clock_timestamp()`, request.PrincipalID); err != nil {
				return managementidentity.PrincipalSessionRevocation{}, fmt.Errorf("expire principal Management sessions: %w", err)
			}
			result, err := tx.ExecContext(ctx, `UPDATE management_sessions
SET status='revoked',revoked_at=COALESCE(revoked_at,clock_timestamp())
WHERE principal_id=$1 AND status='active'`, request.PrincipalID)
			if err != nil {
				return managementidentity.PrincipalSessionRevocation{}, fmt.Errorf("revoke principal Management sessions: %w", err)
			}
			count, err := result.RowsAffected()
			if err != nil {
				return managementidentity.PrincipalSessionRevocation{}, err
			}
			changed = int(count)
			if err := appendAudit(ctx, tx, auditMutation{
				Action: "management_principal.sessions_revoked", ResourceType: "management_principal",
				ResourceID: request.PrincipalID, AfterRevision: principalRevision, Actor: request.Actor,
			}); err != nil {
				return managementidentity.PrincipalSessionRevocation{}, err
			}
			stored := managementcommand.ResourceResult{
				ResourceType: "management_principal_sessions", ResourceID: request.PrincipalID,
				ResourceRevision: principalRevision, ResponseStatus: 200,
			}
			if err := commandpostgres.CompleteResource(ctx, tx, request.Command, stored); err != nil {
				return managementidentity.PrincipalSessionRevocation{}, err
			}
		}
		sessionIDs, err := revokedLiveSessionIDs(ctx, tx, request.PrincipalID)
		if err != nil {
			return managementidentity.PrincipalSessionRevocation{}, err
		}
		already := len(sessionIDs) - changed
		if already < 0 {
			already = 0
		}
		return managementidentity.PrincipalSessionRevocation{
			Result: managementidentity.MutationResult{
				Kind: "management_principal_sessions", ID: request.PrincipalID,
				Revision: principalRevision, ResponseStatus: 200, Replayed: replayed,
			},
			SessionIDs: sessionIDs, RevokedCount: changed, AlreadyRevoked: already,
		}, nil
	})
}

func revokedLiveSessionIDs(ctx context.Context, tx *sql.Tx, principalID string) ([]string, error) {
	rows, err := tx.QueryContext(ctx, `SELECT id::text FROM management_sessions
WHERE principal_id=$1 AND status='revoked' AND expires_at>clock_timestamp()
ORDER BY id`, principalID)
	if err != nil {
		return nil, fmt.Errorf("list revoked Management sessions: %w", err)
	}
	defer rows.Close()
	ids := make([]string, 0)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		if !canonicalUUID(id) {
			return nil, errors.New("stored Management session identifier is invalid")
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

func scanManagementSession(row scanner) (managementidentity.ManagementSession, error) {
	var session managementidentity.ManagementSession
	var revokedAt sql.NullTime
	if err := row.Scan(
		&session.ID, &session.PrincipalID, &session.AuthSourceKind, &session.EvidenceKind,
		&session.AuthenticatedAt, &session.ExpiresAt, &session.Status, &revokedAt,
		&session.CreatedAt,
	); err != nil {
		return managementidentity.ManagementSession{}, fmt.Errorf("scan Management session: %w", err)
	}
	if !canonicalUUID(session.ID) || !canonicalUUID(session.PrincipalID) {
		return managementidentity.ManagementSession{}, errors.New("stored Management session is invalid")
	}
	session.AuthenticatedAt = session.AuthenticatedAt.UTC()
	session.ExpiresAt = session.ExpiresAt.UTC()
	session.CreatedAt = session.CreatedAt.UTC()
	if revokedAt.Valid {
		value := revokedAt.Time.UTC()
		session.RevokedAt = &value
	}
	validSource := session.AuthSourceKind == managementauth.AuthSourceIssuer ||
		session.AuthSourceKind == managementauth.AuthSourceServiceCredential ||
		session.AuthSourceKind == managementauth.AuthSourceMTLS
	validEvidence := session.EvidenceKind == managementauth.EvidenceHuman ||
		session.EvidenceKind == managementauth.EvidenceWorkload
	validStatus := session.Status == managementauth.SessionActive ||
		session.Status == managementauth.SessionRevoked || session.Status == managementauth.SessionExpired
	validPair := (session.AuthSourceKind == managementauth.AuthSourceIssuer && session.EvidenceKind == managementauth.EvidenceHuman) ||
		(session.AuthSourceKind != managementauth.AuthSourceIssuer && session.EvidenceKind == managementauth.EvidenceWorkload)
	validRevocation := (session.Status == managementauth.SessionRevoked) == (session.RevokedAt != nil)
	if !validSource || !validEvidence || !validStatus || !validPair || !validRevocation ||
		session.AuthenticatedAt.IsZero() || session.ExpiresAt.IsZero() || session.CreatedAt.IsZero() ||
		!session.ExpiresAt.After(session.CreatedAt) {
		return managementidentity.ManagementSession{}, errors.New("stored Management session is invalid")
	}
	return session, nil
}
