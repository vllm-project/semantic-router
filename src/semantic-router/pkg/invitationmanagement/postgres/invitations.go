package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
)

const invitationColumns = `id::text, namespace_id::text, created_by_principal_id::text,
       expected_issuer, COALESCE(expected_subject,''), COALESCE(expected_email,''), display_name,
       grants, expires_at, status, accepted_principal_id::text, accepted_user_id::text,
       accepted_management_session_id::text,
       accepted_at, revision, created_at, updated_at`

func (store *Store) Get(ctx context.Context, namespaceID, invitationID string) (invitationmanagement.Invitation, error) {
	value, err := scanInvitation(store.database.QueryRowContext(ctx,
		`SELECT `+invitationColumns+` FROM management_invitations WHERE namespace_id=$1 AND id=$2`, namespaceID, invitationID))
	return invitationResult(value, err)
}

func (store *Store) GetByID(ctx context.Context, invitationID string) (invitationmanagement.Invitation, []byte, string, error) {
	var digest []byte
	var pepper string
	value, err := scanInvitationWithToken(store.database.QueryRowContext(ctx,
		`SELECT `+invitationColumns+`, token_hmac, pepper_version FROM management_invitations WHERE id=$1`, invitationID), &digest, &pepper)
	if errors.Is(err, sql.ErrNoRows) {
		return invitationmanagement.Invitation{}, nil, "", invitationmanagement.ErrNotFound
	}
	if err != nil {
		return invitationmanagement.Invitation{}, nil, "", fmt.Errorf("get invitation by token identifier: %w", err)
	}
	return value, digest, pepper, nil
}

func (store *Store) List(
	ctx context.Context,
	query invitationmanagement.InvitationQuery,
) (_ invitationmanagement.RepositoryPage, returnErr error) {
	var afterTime any
	afterID := ""
	if query.After != nil {
		afterTime, afterID = query.After.ExpiresAt, query.After.ID
	}
	rows, err := store.database.QueryContext(ctx, `SELECT `+invitationColumns+`
FROM management_invitations
WHERE namespace_id=$1
  AND ($2='' OR CASE WHEN status='pending' AND expires_at<=$3 THEN 'expired' ELSE status END=$2)
  AND ($4::timestamptz IS NULL OR expires_at>$4 OR (expires_at=$4 AND id>NULLIF($5,'')::uuid))
ORDER BY expires_at,id LIMIT $6`, query.NamespaceID, query.Status, query.Now, afterTime, afterID, query.Limit+1)
	if err != nil {
		return invitationmanagement.RepositoryPage{}, fmt.Errorf("list invitations: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	items := make([]invitationmanagement.Invitation, 0, query.Limit+1)
	for rows.Next() {
		value, err := scanInvitation(rows)
		if err != nil {
			return invitationmanagement.RepositoryPage{}, fmt.Errorf("scan invitation page: %w", err)
		}
		items = append(items, value)
	}
	if err := rows.Err(); err != nil {
		return invitationmanagement.RepositoryPage{}, err
	}
	page := invitationmanagement.RepositoryPage{Items: items}
	if len(items) > query.Limit {
		page.Items, page.HasMore = items[:query.Limit], true
	}
	return page, nil
}

func (store *Store) Create(ctx context.Context, mutation invitationmanagement.CreateMutation) (invitationmanagement.MutationResult, error) {
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (invitationmanagement.MutationResult, error) {
		stored, replayed, createErr := commandpostgres.Lock(ctx, tx, mutation.Command)
		if createErr != nil {
			return invitationmanagement.MutationResult{}, mapCommandError(createErr)
		}
		if replayed {
			secret, err := storedSecret(stored, "invitation")
			if err != nil {
				return invitationmanagement.MutationResult{}, err
			}
			return invitationmanagement.MutationResult{
				HTTPStatus: secret.Result.ResponseStatus,
				Replayed:   true, Stored: &secret,
			}, err
		}
		now, createErr := databaseNow(ctx, tx)
		if createErr != nil {
			return invitationmanagement.MutationResult{}, createErr
		}
		if !mutation.Invitation.ExpiresAt.After(now) {
			return invitationmanagement.MutationResult{}, invitationmanagement.ErrExpired
		}
		if err := verifySnapshot(ctx, tx, mutation.Invitation.NamespaceID,
			mutation.Actor.PrincipalID, "future-user", mutation.Invitation.Snapshot); err != nil {
			return invitationmanagement.MutationResult{}, err
		}
		snapshot, createErr := json.Marshal(mutation.Invitation.Snapshot)
		if createErr != nil {
			return invitationmanagement.MutationResult{}, invitationmanagement.ErrUnavailable
		}
		var teamID, teamRole any
		if mutation.Invitation.Snapshot.Team != nil {
			teamID, teamRole = mutation.Invitation.Snapshot.Team.TeamID, mutation.Invitation.Snapshot.Team.Role
		}
		created, createErr := scanInvitation(tx.QueryRowContext(ctx, `INSERT INTO management_invitations
  (id,namespace_id,created_by_principal_id,expected_issuer,expected_subject,expected_email,
   display_name,token_hmac,pepper_version,grants,team_id,team_role,
	   pinned_access_policy_id,pinned_access_policy_revision,
	   pinned_rate_limit_policy_id,pinned_rate_limit_policy_revision,
	   expires_at,status,revision,created_at,updated_at)
	VALUES ($1,$2,$3,$4,NULLIF($5,''),NULLIF($6,''),$7,$8,$9,$10,$11,$12,
	        NULLIF($13,'')::uuid,NULLIF($14,0)::bigint,NULLIF($15,'')::uuid,NULLIF($16,0)::bigint,
	        $17,'pending',1,$18,$18)
RETURNING `+invitationColumns,
			mutation.Invitation.ID, mutation.Invitation.NamespaceID, mutation.Invitation.CreatedByPrincipalID,
			mutation.Invitation.Expected.Issuer, mutation.Invitation.Expected.Subject, mutation.Invitation.Expected.Email,
			mutation.Invitation.DisplayName, mutation.TokenHMAC, mutation.PepperVersion, snapshot, teamID, teamRole,
			mutation.Invitation.Snapshot.AccessPolicyID, mutation.Invitation.Snapshot.AccessPolicyRevision,
			mutation.Invitation.Snapshot.RateLimitPolicyID, mutation.Invitation.Snapshot.RateLimitPolicyRevision,
			mutation.Invitation.ExpiresAt, now))
		if createErr != nil {
			return invitationmanagement.MutationResult{}, mapWriteError(createErr, "create invitation")
		}
		if err := appendAudit(ctx, tx, created.NamespaceID, nil, "invitation.created", "invitation",
			created.ID, nil, created.Revision, mutation.Actor); err != nil {
			return invitationmanagement.MutationResult{}, err
		}
		resource := managementcommand.ResourceResult{
			ResourceType: "invitation", ResourceID: created.ID,
			ResourceRevision: created.Revision, ResponseStatus: 201,
		}
		if err := commandpostgres.CompleteSecretResource(ctx, tx, mutation.Command, resource,
			managementcommand.SecretResponse{
				Ciphertext: mutation.Response.Ciphertext, Nonce: mutation.Response.Nonce,
				KEKVersion: mutation.Response.KeyVersion, ExpiresAt: mutation.ResponseExpiresAt,
			}); err != nil {
			return invitationmanagement.MutationResult{}, err
		}
		return invitationmanagement.MutationResult{Invitation: created, HTTPStatus: 201}, nil
	})
}

func (store *Store) Rotate(ctx context.Context, mutation invitationmanagement.RotateMutation) (invitationmanagement.MutationResult, error) {
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (invitationmanagement.MutationResult, error) {
		stored, replayed, rotateErr := commandpostgres.Lock(ctx, tx, mutation.Command)
		if rotateErr != nil {
			return invitationmanagement.MutationResult{}, mapCommandError(rotateErr)
		}
		if replayed {
			secret, err := storedSecret(stored, "invitation")
			if err != nil {
				return invitationmanagement.MutationResult{}, err
			}
			return invitationmanagement.MutationResult{
				HTTPStatus: secret.Result.ResponseStatus,
				Replayed:   true, Stored: &secret,
			}, err
		}
		now, rotateErr := databaseNow(ctx, tx)
		if rotateErr != nil {
			return invitationmanagement.MutationResult{}, rotateErr
		}
		var expiry any
		if mutation.ExpiresAt != nil {
			expiry = mutation.ExpiresAt.UTC()
		}
		updated, rotateErr := scanInvitation(tx.QueryRowContext(ctx, `UPDATE management_invitations
SET token_hmac=$4,pepper_version=$5,expires_at=COALESCE($6,expires_at),
    revision=revision+1,updated_at=$7
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND status='pending' AND expires_at>$7
RETURNING `+invitationColumns, mutation.NamespaceID, mutation.InvitationID, mutation.ExpectedRevision,
			mutation.TokenHMAC, mutation.PepperVersion, expiry, now))
		if errors.Is(rotateErr, sql.ErrNoRows) {
			return invitationmanagement.MutationResult{}, classifyInvitationMutation(ctx, tx,
				mutation.NamespaceID, mutation.InvitationID, mutation.ExpectedRevision, now)
		}
		if rotateErr != nil {
			return invitationmanagement.MutationResult{}, mapWriteError(rotateErr, "rotate invitation token")
		}
		before := mutation.ExpectedRevision
		if err := appendAudit(ctx, tx, updated.NamespaceID, nil, "invitation.token_rotated", "invitation",
			updated.ID, &before, updated.Revision, mutation.Actor); err != nil {
			return invitationmanagement.MutationResult{}, err
		}
		resource := managementcommand.ResourceResult{
			ResourceType: "invitation", ResourceID: updated.ID,
			ResourceRevision: updated.Revision, ResponseStatus: 200,
		}
		if err := commandpostgres.CompleteSecretResource(ctx, tx, mutation.Command, resource,
			managementcommand.SecretResponse{
				Ciphertext: mutation.Response.Ciphertext, Nonce: mutation.Response.Nonce,
				KEKVersion: mutation.Response.KeyVersion, ExpiresAt: mutation.ResponseExpiresAt,
			}); err != nil {
			return invitationmanagement.MutationResult{}, err
		}
		return invitationmanagement.MutationResult{Invitation: updated, HTTPStatus: 200}, nil
	})
}

func (store *Store) Revoke(ctx context.Context, request invitationmanagement.RevokeRequest) (invitationmanagement.MutationResult, error) {
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (invitationmanagement.MutationResult, error) {
		now, err := databaseNow(ctx, tx)
		if err != nil {
			return invitationmanagement.MutationResult{}, err
		}
		updated, err := scanInvitation(tx.QueryRowContext(ctx, `UPDATE management_invitations
SET status='revoked',revision=revision+1,updated_at=$4
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND status='pending' AND expires_at>$4
RETURNING `+invitationColumns, request.NamespaceID, request.InvitationID, request.ExpectedRevision, now))
		if errors.Is(err, sql.ErrNoRows) {
			return invitationmanagement.MutationResult{}, classifyInvitationMutation(ctx, tx,
				request.NamespaceID, request.InvitationID, request.ExpectedRevision, now)
		}
		if err != nil {
			return invitationmanagement.MutationResult{}, mapWriteError(err, "revoke invitation")
		}
		before := request.ExpectedRevision
		if err := appendAudit(ctx, tx, updated.NamespaceID, nil, "invitation.revoked", "invitation",
			updated.ID, &before, updated.Revision, request.Actor); err != nil {
			return invitationmanagement.MutationResult{}, err
		}
		return invitationmanagement.MutationResult{Invitation: updated, HTTPStatus: 204}, nil
	})
}

type rowScanner interface{ Scan(...any) error }

func scanInvitation(scanner rowScanner) (invitationmanagement.Invitation, error) {
	return scanInvitationWithToken(scanner, nil, nil)
}

func scanInvitationWithToken(scanner rowScanner, tokenHMAC *[]byte, pepperVersion *string) (invitationmanagement.Invitation, error) {
	var (
		value                                            invitationmanagement.Invitation
		snapshotJSON                                     []byte
		acceptedPrincipal, acceptedUser, acceptedSession sql.NullString
		acceptedAt                                       sql.NullTime
	)
	destinations := []any{
		&value.ID, &value.NamespaceID, &value.CreatedByPrincipalID,
		&value.Expected.Issuer, &value.Expected.Subject, &value.Expected.Email, &value.DisplayName,
		&snapshotJSON, &value.ExpiresAt, &value.Status, &acceptedPrincipal, &acceptedUser, &acceptedSession,
		&acceptedAt, &value.Revision, &value.CreatedAt, &value.UpdatedAt,
	}
	if tokenHMAC != nil && pepperVersion != nil {
		destinations = append(destinations, tokenHMAC, pepperVersion)
	}
	if err := scanner.Scan(destinations...); err != nil {
		return invitationmanagement.Invitation{}, err
	}
	if err := json.Unmarshal(snapshotJSON, &value.Snapshot); err != nil {
		return invitationmanagement.Invitation{}, invitationmanagement.ErrUnavailable
	}
	value.AcceptedPrincipalID, value.AcceptedUserID = acceptedPrincipal.String, acceptedUser.String
	value.AcceptedManagementSessionID = acceptedSession.String
	if acceptedAt.Valid {
		accepted := acceptedAt.Time.UTC()
		value.AcceptedAt = &accepted
	}
	value.ExpiresAt, value.CreatedAt, value.UpdatedAt = value.ExpiresAt.UTC(), value.CreatedAt.UTC(), value.UpdatedAt.UTC()
	return value, nil
}

func invitationResult(value invitationmanagement.Invitation, err error) (invitationmanagement.Invitation, error) {
	if errors.Is(err, sql.ErrNoRows) {
		return invitationmanagement.Invitation{}, invitationmanagement.ErrNotFound
	}
	if err != nil {
		return invitationmanagement.Invitation{}, fmt.Errorf("read invitation: %w", err)
	}
	return value, nil
}

func classifyInvitationMutation(ctx context.Context, tx *sql.Tx, namespaceID, invitationID string, expected uint64, now time.Time) error {
	var revision uint64
	var status invitationmanagement.Status
	var expiresAt time.Time
	err := tx.QueryRowContext(ctx, `SELECT revision,status,expires_at FROM management_invitations
WHERE namespace_id=$1 AND id=$2`, namespaceID, invitationID).Scan(&revision, &status, &expiresAt)
	if errors.Is(err, sql.ErrNoRows) {
		return invitationmanagement.ErrNotFound
	}
	if err != nil {
		return fmt.Errorf("classify invitation mutation: %w", err)
	}
	if revision != expected {
		return invitationmanagement.ErrRevisionConflict
	}
	if !now.Before(expiresAt) {
		return invitationmanagement.ErrExpired
	}
	return invitationmanagement.ErrConflict
}

func mapWriteError(err error, action string) error {
	var pqError *pq.Error
	if errors.As(err, &pqError) {
		switch pqError.Code {
		case "23505":
			return invitationmanagement.ErrConflict
		case "23503":
			return invitationmanagement.ErrNotFound
		case "23514":
			return invitationmanagement.ErrInvalidRequest
		}
	}
	return fmt.Errorf("%s: %w", action, err)
}
