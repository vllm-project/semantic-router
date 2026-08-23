package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

const principalColumns = `id::text, issuer, subject, display_name,
       COALESCE(verified_email, ''), attributes, status, revision,
       created_at, updated_at`

func (store *Store) GetPrincipal(ctx context.Context, id string) (managementidentity.Principal, error) {
	if !canonicalUUID(id) {
		return managementidentity.Principal{}, managementidentity.ErrNotFound
	}
	principal, err := scanPrincipal(store.database.QueryRowContext(ctx,
		`SELECT `+principalColumns+` FROM management_principals WHERE id=$1`, id))
	return principalResult(principal, err)
}

func (store *Store) ListPrincipals(ctx context.Context, request managementidentity.ListRequest) (managementidentity.PrincipalPage, error) {
	if err := validateList(request); err != nil {
		return managementidentity.PrincipalPage{}, err
	}
	rows, err := store.database.QueryContext(ctx, `SELECT `+principalColumns+`
FROM management_principals WHERE ($1='' OR id > NULLIF($1,'')::uuid) ORDER BY id LIMIT $2`, request.AfterID, request.Limit+1)
	if err != nil {
		return managementidentity.PrincipalPage{}, fmt.Errorf("list Management principals: %w", err)
	}
	defer rows.Close()
	items := make([]managementidentity.Principal, 0, request.Limit+1)
	for rows.Next() {
		item, err := scanPrincipal(rows)
		if err != nil {
			return managementidentity.PrincipalPage{}, fmt.Errorf("scan Management principal page: %w", err)
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return managementidentity.PrincipalPage{}, fmt.Errorf("iterate Management principal page: %w", err)
	}
	page := managementidentity.PrincipalPage{Items: items}
	if len(items) > request.Limit {
		page.Items = items[:request.Limit]
		page.NextCursor = string(page.Items[len(page.Items)-1].Identity.ID)
	}
	return page, nil
}

func (store *Store) CreatePrincipal(ctx context.Context, request managementidentity.CreatePrincipal) (managementidentity.MutationResult, error) {
	if err := validateCreateCommand(request.Command, request.Actor, managementcommand.ScopeCluster); err != nil {
		return managementidentity.MutationResult{}, err
	}
	attributes, err := json.Marshal(request.Attributes)
	if err != nil {
		return managementidentity.MutationResult{}, errors.New("management principal attributes are invalid")
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.MutationResult, error) {
		if replay, found, err := commandpostgres.Lock(ctx, tx, request.Command); err != nil {
			return managementidentity.MutationResult{}, mapCommandError(err)
		} else if found {
			return replayMutation(replay, "management_principal")
		}
		principal, err := scanPrincipal(tx.QueryRowContext(ctx, `INSERT INTO management_principals
  (id,issuer,subject,display_name,verified_email,attributes,status,revision)
VALUES ($1,$2,$3,$4,NULLIF($5,''),$6,'active',1)
RETURNING `+principalColumns, request.ID, request.Issuer, request.Subject,
			request.DisplayName, request.VerifiedEmail, attributes))
		if err != nil {
			return managementidentity.MutationResult{}, mapWriteError("create Management principal", err)
		}
		if err := appendAudit(ctx, tx, auditMutation{
			Action: "management_principal.created", ResourceType: "management_principal",
			ResourceID: request.ID, AfterRevision: uint64(principal.Revision), Actor: request.Actor,
		}); err != nil {
			return managementidentity.MutationResult{}, err
		}
		result := managementcommand.ResourceResult{
			ResourceType: "management_principal", ResourceID: request.ID,
			ResourceRevision: uint64(principal.Revision), ResponseStatus: 201,
		}
		if err := commandpostgres.CompleteResource(ctx, tx, request.Command, result); err != nil {
			return managementidentity.MutationResult{}, err
		}
		return managementidentity.MutationResult{Kind: result.ResourceType, ID: result.ResourceID, Revision: result.ResourceRevision, ResponseStatus: 201}, nil
	})
}

func (store *Store) UpdatePrincipal(ctx context.Context, request managementidentity.UpdatePrincipal) (managementidentity.MutationResult, error) {
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.MutationResult, error) {
		if err := rejectReservedServiceAccountPrincipal(ctx, tx, request.ID); err != nil {
			return managementidentity.MutationResult{}, err
		}
		principal, err := scanPrincipal(tx.QueryRowContext(ctx, `UPDATE management_principals SET
  display_name=COALESCE($3,display_name),
  verified_email=CASE WHEN $4::boolean THEN NULLIF($5,'') ELSE verified_email END,
  status=COALESCE($6,status), revision=revision+1, updated_at=clock_timestamp()
WHERE id=$1 AND revision=$2 RETURNING `+principalColumns,
			request.ID, request.ExpectedRevision, nullableString(request.DisplayName), request.VerifiedEmail != nil,
			stringValue(request.VerifiedEmail), nullablePrincipalStatus(request.Status)))
		if errors.Is(err, sql.ErrNoRows) {
			return managementidentity.MutationResult{}, classifyRevision(ctx, tx, "management_principals", request.ID)
		}
		if err != nil {
			return managementidentity.MutationResult{}, mapWriteError("update Management principal", err)
		}
		before := request.ExpectedRevision
		if err := appendAudit(ctx, tx, auditMutation{
			Action: "management_principal.updated", ResourceType: "management_principal", ResourceID: request.ID,
			BeforeRevision: &before, AfterRevision: uint64(principal.Revision), Actor: request.Actor,
		}); err != nil {
			return managementidentity.MutationResult{}, err
		}
		return managementidentity.MutationResult{Kind: "management_principal", ID: request.ID, Revision: uint64(principal.Revision), ResponseStatus: 200}, nil
	})
}

func (store *Store) DeletePrincipal(ctx context.Context, id string, expected uint64, actor managementidentity.MutationActor) (managementidentity.MutationResult, error) {
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.MutationResult, error) {
		if err := rejectReservedServiceAccountPrincipal(ctx, tx, id); err != nil {
			return managementidentity.MutationResult{}, err
		}
		result, err := tx.ExecContext(ctx, `DELETE FROM management_principals
WHERE id=$1 AND revision=$2 AND status='disabled'`, id, expected)
		if err != nil {
			return managementidentity.MutationResult{}, mapWriteError("delete Management principal", err)
		}
		count, err := result.RowsAffected()
		if err != nil || count != 1 {
			return managementidentity.MutationResult{}, classifyRevision(ctx, tx, "management_principals", id)
		}
		after := expected + 1
		if err := appendAudit(ctx, tx, auditMutation{
			Action: "management_principal.deleted", ResourceType: "management_principal", ResourceID: id,
			BeforeRevision: &expected, AfterRevision: after, Actor: actor,
		}); err != nil {
			return managementidentity.MutationResult{}, err
		}
		return managementidentity.MutationResult{Kind: "management_principal", ID: id, Revision: after, ResponseStatus: 204}, nil
	})
}

func rejectReservedServiceAccountPrincipal(ctx context.Context, tx *sql.Tx, id string) error {
	var issuer string
	if err := tx.QueryRowContext(ctx,
		`SELECT issuer FROM management_principals WHERE id=$1 FOR UPDATE`, id,
	).Scan(&issuer); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return managementidentity.ErrNotFound
		}
		return fmt.Errorf("load Management principal subtype: %w", err)
	}
	if issuer == managementidentity.ServiceAccountIssuer {
		return managementidentity.ErrWorkloadDependency
	}
	return nil
}

func (store *Store) RevokePrincipalSessions(ctx context.Context, principalID string) ([]managementauth.SessionMutation, error) {
	rows, err := store.database.QueryContext(ctx, `UPDATE management_sessions
SET status='revoked', revoked_at=COALESCE(revoked_at,clock_timestamp())
WHERE principal_id=$1 AND status='active'
RETURNING id::text, token_id, revoked_at`, principalID)
	if err != nil {
		return nil, fmt.Errorf("revoke principal Management sessions: %w", err)
	}
	defer rows.Close()
	mutations := make([]managementauth.SessionMutation, 0)
	for rows.Next() {
		var mutation managementauth.SessionMutation
		if err := rows.Scan(&mutation.SessionID, &mutation.TokenID, &mutation.ChangedAt); err != nil {
			return nil, fmt.Errorf("scan revoked principal Management session: %w", err)
		}
		mutation.Changed = true
		mutation.ChangedAt = mutation.ChangedAt.UTC()
		mutations = append(mutations, mutation)
	}
	return mutations, rows.Err()
}

func principalResult(value managementidentity.Principal, err error) (managementidentity.Principal, error) {
	if errors.Is(err, sql.ErrNoRows) {
		return managementidentity.Principal{}, managementidentity.ErrNotFound
	}
	if err != nil {
		return managementidentity.Principal{}, fmt.Errorf("load Management principal: %w", err)
	}
	return value, nil
}

func validateCreateCommand(command managementcommand.Command, actor managementidentity.MutationActor, expected managementcommand.ScopeKind) error {
	if command.Scope.Kind != expected || command.PrincipalID != actor.PrincipalID {
		return errors.New("management command scope or actor does not match the mutation")
	}
	return nil
}

func replayMutation(stored managementcommand.StoredResult, kind string) (managementidentity.MutationResult, error) {
	if stored.Resource == nil || stored.Operation != nil || stored.Resource.ResourceType != kind {
		return managementidentity.MutationResult{}, errors.New("stored Management command result does not match the resource")
	}
	return managementidentity.MutationResult{
		Kind: stored.Resource.ResourceType, ID: stored.Resource.ResourceID,
		Revision: stored.Resource.ResourceRevision, ResponseStatus: stored.Resource.ResponseStatus, Replayed: true,
	}, nil
}

func mapCommandError(err error) error {
	if errors.Is(err, managementcommand.ErrConflict) {
		return managementidentity.ErrAlreadyExists
	}
	return err
}

func mapWriteError(action string, err error) error {
	var pqError *pq.Error
	if errors.As(err, &pqError) {
		switch pqError.Code {
		case "23505":
			return managementidentity.ErrAlreadyExists
		case "23503":
			return managementidentity.ErrNotFound
		case "23514":
			return errors.New("management identity relationship is invalid")
		}
	}
	if strings.Contains(err.Error(), "violates foreign key") {
		return managementidentity.ErrNotFound
	}
	return fmt.Errorf("%s: %w", action, err)
}

func classifyRevision(ctx context.Context, tx *sql.Tx, table, id string) error {
	var exists bool
	query := `SELECT EXISTS(SELECT 1 FROM ` + table + ` WHERE id=$1)` // table is a package constant selected by callers.
	if err := tx.QueryRowContext(ctx, query, id).Scan(&exists); err != nil {
		return fmt.Errorf("classify Management identity mutation: %w", err)
	}
	if !exists {
		return managementidentity.ErrNotFound
	}
	return managementidentity.ErrRevisionConflict
}

func nullableString(value *string) any {
	if value == nil {
		return nil
	}
	return *value
}

func stringValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func nullablePrincipalStatus(value *accesscontrol.PrincipalStatus) any {
	if value == nil {
		return nil
	}
	return string(*value)
}
