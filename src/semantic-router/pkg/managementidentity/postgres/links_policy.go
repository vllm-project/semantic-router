package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

const linkColumns = `principal_id::text, namespace_id::text, user_id::text,
       revision, created_at, updated_at`

func (store *Store) GetPrincipalUserLink(ctx context.Context, principalID, namespaceID string) (managementidentity.PrincipalUserLink, error) {
	if !canonicalUUID(principalID) || !canonicalUUID(namespaceID) {
		return managementidentity.PrincipalUserLink{}, managementidentity.ErrNotFound
	}
	link, err := scanLink(store.database.QueryRowContext(ctx, `SELECT `+linkColumns+`
FROM management_principal_user_links WHERE principal_id=$1 AND namespace_id=$2`, principalID, namespaceID))
	return linkResult(link, err)
}

func (store *Store) PutPrincipalUserLink(ctx context.Context, request managementidentity.LinkMutation) (managementidentity.MutationResult, error) {
	if err := validateCreateCommand(request.Command, request.Actor, managementcommand.ScopeNamespace); err != nil ||
		request.Command.Scope.NamespaceID != request.NamespaceID {
		return managementidentity.MutationResult{}, errors.New("principal User-link command scope does not match the namespace")
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.MutationResult, error) {
		if replay, found, err := commandpostgres.Lock(ctx, tx, request.Command); err != nil {
			return managementidentity.MutationResult{}, mapCommandError(err)
		} else if found {
			return replayMutation(replay, "management_principal_user_link")
		}
		var currentUser string
		var currentRevision uint64
		putPrincipalUserLinkErr := tx.QueryRowContext(ctx, `SELECT user_id::text,revision
FROM management_principal_user_links WHERE principal_id=$1 AND namespace_id=$2 FOR UPDATE`,
			request.PrincipalID, request.NamespaceID).Scan(&currentUser, &currentRevision)
		if putPrincipalUserLinkErr != nil && !errors.Is(putPrincipalUserLinkErr, sql.ErrNoRows) {
			return managementidentity.MutationResult{}, putPrincipalUserLinkErr
		}
		if errors.Is(putPrincipalUserLinkErr, sql.ErrNoRows) {
			if request.ExpectedRevision != nil {
				return managementidentity.MutationResult{}, managementidentity.ErrNotFound
			}
		} else {
			if request.ExpectedRevision == nil || *request.ExpectedRevision != currentRevision {
				return managementidentity.MutationResult{}, managementidentity.ErrRevisionConflict
			}
			if currentUser != request.UserID {
				blocked, err := hasUserScopedBinding(ctx, tx, request.PrincipalID, request.NamespaceID)
				if err != nil {
					return managementidentity.MutationResult{}, err
				}
				if blocked {
					return managementidentity.MutationResult{}, managementidentity.ErrPrincipalLinkInUse
				}
			}
		}
		var link managementidentity.PrincipalUserLink
		if request.ExpectedRevision == nil {
			link, putPrincipalUserLinkErr = scanLink(tx.QueryRowContext(ctx, `INSERT INTO management_principal_user_links
  (principal_id,namespace_id,user_id,revision) VALUES ($1,$2,$3,1)
RETURNING `+linkColumns, request.PrincipalID, request.NamespaceID, request.UserID))
		} else {
			link, putPrincipalUserLinkErr = scanLink(tx.QueryRowContext(ctx, `UPDATE management_principal_user_links
SET user_id=$4,revision=revision+1,updated_at=clock_timestamp()
WHERE principal_id=$1 AND namespace_id=$2 AND revision=$3 RETURNING `+linkColumns,
				request.PrincipalID, request.NamespaceID, *request.ExpectedRevision, request.UserID))
		}
		if putPrincipalUserLinkErr != nil {
			return managementidentity.MutationResult{}, mapWriteError("put principal User link", putPrincipalUserLinkErr)
		}
		resourceID := linkResourceID(request.PrincipalID, request.NamespaceID)
		var before *uint64
		if request.ExpectedRevision != nil {
			value := *request.ExpectedRevision
			before = &value
		}
		if err := appendAudit(ctx, tx, auditMutation{
			NamespaceID: request.NamespaceID, Action: "management_principal_user_link.put",
			ResourceType: "management_principal_user_link", ResourceID: resourceID,
			BeforeRevision: before, AfterRevision: uint64(link.Revision), Actor: request.Actor,
		}); err != nil {
			return managementidentity.MutationResult{}, err
		}
		status := 201
		if before != nil {
			status = 200
		}
		result := managementcommand.ResourceResult{ResourceType: "management_principal_user_link", ResourceID: resourceID, ResourceRevision: uint64(link.Revision), ResponseStatus: status}
		if err := commandpostgres.CompleteResource(ctx, tx, request.Command, result); err != nil {
			return managementidentity.MutationResult{}, err
		}
		return managementidentity.MutationResult{Kind: result.ResourceType, ID: result.ResourceID, Revision: result.ResourceRevision, ResponseStatus: status}, nil
	})
}

func (store *Store) DeletePrincipalUserLink(ctx context.Context, request managementidentity.LinkMutation) (managementidentity.MutationResult, error) {
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.MutationResult, error) {
		blocked, err := hasUserScopedBinding(ctx, tx, request.PrincipalID, request.NamespaceID)
		if err != nil {
			return managementidentity.MutationResult{}, err
		}
		if blocked {
			return managementidentity.MutationResult{}, managementidentity.ErrPrincipalLinkInUse
		}
		result, err := tx.ExecContext(ctx, `DELETE FROM management_principal_user_links
WHERE principal_id=$1 AND namespace_id=$2 AND revision=$3`, request.PrincipalID, request.NamespaceID, *request.ExpectedRevision)
		if err != nil {
			return managementidentity.MutationResult{}, mapWriteError("delete principal User link", err)
		}
		count, _ := result.RowsAffected()
		if count != 1 {
			var exists bool
			if err := tx.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM management_principal_user_links
WHERE principal_id=$1 AND namespace_id=$2)`, request.PrincipalID, request.NamespaceID).Scan(&exists); err != nil {
				return managementidentity.MutationResult{}, err
			}
			if !exists {
				return managementidentity.MutationResult{}, managementidentity.ErrNotFound
			}
			return managementidentity.MutationResult{}, managementidentity.ErrRevisionConflict
		}
		after := *request.ExpectedRevision + 1
		resourceID := linkResourceID(request.PrincipalID, request.NamespaceID)
		if err := appendAudit(ctx, tx, auditMutation{
			NamespaceID: request.NamespaceID, Action: "management_principal_user_link.deleted",
			ResourceType: "management_principal_user_link", ResourceID: resourceID,
			BeforeRevision: request.ExpectedRevision, AfterRevision: after, Actor: request.Actor,
		}); err != nil {
			return managementidentity.MutationResult{}, err
		}
		return managementidentity.MutationResult{Kind: "management_principal_user_link", ID: resourceID, Revision: after, ResponseStatus: 204}, nil
	})
}

func (store *Store) LoadSessionPolicy(ctx context.Context) (managementauth.SessionPolicy, error) {
	return store.sessionPolicy.LoadSessionPolicy(ctx)
}

func (store *Store) UpdateSessionPolicy(ctx context.Context, policy managementauth.SessionPolicy, expected uint64, actor managementidentity.MutationActor) (managementidentity.MutationResult, error) {
	actions, err := json.Marshal(policy.ActionRequirements)
	if err != nil {
		return managementidentity.MutationResult{}, errors.New("management session policy requirements are invalid")
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.MutationResult, error) {
		var revision uint64
		err := tx.QueryRowContext(ctx, `UPDATE management_session_policy SET
  access_token_ttl_seconds=$2,session_ttl_seconds=$3,max_active_sessions=$4,
  action_requirements=$5,revision=revision+1,updated_at=clock_timestamp()
WHERE singleton=TRUE AND revision=$1 RETURNING revision`, expected,
			int64(policy.AccessTokenTTL.Seconds()), int64(policy.SessionTTL.Seconds()),
			policy.MaxActiveSessions, actions).Scan(&revision)
		if errors.Is(err, sql.ErrNoRows) {
			return managementidentity.MutationResult{}, managementidentity.ErrRevisionConflict
		}
		if err != nil {
			return managementidentity.MutationResult{}, mapWriteError("update Management session policy", err)
		}
		if _, err := tx.ExecContext(ctx, `UPDATE management_sessions SET
  status='revoked',revoked_at=COALESCE(revoked_at,clock_timestamp())
WHERE status='active' AND expires_at>clock_timestamp()+($1 * interval '1 second')`,
			int64(policy.SessionTTL.Seconds())); err != nil {
			return managementidentity.MutationResult{}, fmt.Errorf("enforce Management session lifetime: %w", err)
		}
		if _, err := tx.ExecContext(ctx, `WITH ranked AS (
  SELECT id,row_number() OVER (PARTITION BY principal_id ORDER BY created_at DESC,id DESC) AS ordinal
  FROM management_sessions WHERE status='active'
)
UPDATE management_sessions session SET status='revoked',revoked_at=COALESCE(session.revoked_at,clock_timestamp())
FROM ranked WHERE session.id=ranked.id AND ranked.ordinal>$1`, policy.MaxActiveSessions); err != nil {
			return managementidentity.MutationResult{}, fmt.Errorf("enforce Management active-session limit: %w", err)
		}
		if err := appendAudit(ctx, tx, auditMutation{
			Action: "management_session_policy.updated", ResourceType: "management_session_policy",
			ResourceID: "singleton", BeforeRevision: &expected, AfterRevision: revision, Actor: actor,
		}); err != nil {
			return managementidentity.MutationResult{}, err
		}
		return managementidentity.MutationResult{Kind: "management_session_policy", ID: "singleton", Revision: revision, ResponseStatus: 200}, nil
	})
}

func scanLink(row scanner) (managementidentity.PrincipalUserLink, error) {
	var link managementidentity.PrincipalUserLink
	if err := row.Scan(&link.PrincipalID, &link.NamespaceID, &link.UserID, &link.Revision, &link.CreatedAt, &link.UpdatedAt); err != nil {
		return managementidentity.PrincipalUserLink{}, err
	}
	if !canonicalUUID(string(link.PrincipalID)) || !canonicalUUID(string(link.NamespaceID)) || !canonicalUUID(string(link.UserID)) || link.Revision == 0 {
		return managementidentity.PrincipalUserLink{}, errors.New("stored principal User link is invalid")
	}
	link.CreatedAt, link.UpdatedAt = link.CreatedAt.UTC(), link.UpdatedAt.UTC()
	return link, nil
}

func linkResult(value managementidentity.PrincipalUserLink, err error) (managementidentity.PrincipalUserLink, error) {
	if errors.Is(err, sql.ErrNoRows) {
		return managementidentity.PrincipalUserLink{}, managementidentity.ErrNotFound
	}
	if err != nil {
		return managementidentity.PrincipalUserLink{}, fmt.Errorf("load principal User link: %w", err)
	}
	return value, nil
}

func hasUserScopedBinding(ctx context.Context, tx *sql.Tx, principalID, namespaceID string) (bool, error) {
	var exists bool
	err := tx.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM management_role_bindings
WHERE principal_id=$1 AND namespace_id=$2 AND scope_kind='user' AND status='active')`, principalID, namespaceID).Scan(&exists)
	return exists, err
}

func linkResourceID(principalID, namespaceID string) string {
	// The link row is composite; a deterministic v5 UUID gives command receipts
	// and audit rows one stable non-secret resource identity.
	return uuid.NewSHA1(uuid.NameSpaceOID, []byte("management_principal_user_link:"+principalID+":"+namespaceID)).String()
}
