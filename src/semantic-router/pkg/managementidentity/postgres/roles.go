package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

const roleColumns = `id::text, namespace_id::text, name, display_name,
       description, permissions, permissions_digest, builtin, status,
       revision, created_at, updated_at`

func (store *Store) GetRole(ctx context.Context, id string) (managementidentity.Role, error) {
	if !canonicalUUID(id) {
		return managementidentity.Role{}, managementidentity.ErrNotFound
	}
	role, err := scanRole(store.database.QueryRowContext(ctx,
		`SELECT `+roleColumns+` FROM management_roles WHERE id=$1`, id))
	return roleResult(role, err)
}

func (store *Store) ListRoles(ctx context.Context, namespaceID string, request managementidentity.ListRequest) (managementidentity.RolePage, error) {
	if err := validateList(request); err != nil {
		return managementidentity.RolePage{}, err
	}
	rows, err := store.database.QueryContext(ctx, `SELECT `+roleColumns+`
FROM management_roles
WHERE ($1='' OR namespace_id IS NULL OR namespace_id=NULLIF($1,'')::uuid)
  AND ($2='' OR id>NULLIF($2,'')::uuid)
ORDER BY id LIMIT $3`, namespaceID, request.AfterID, request.Limit+1)
	if err != nil {
		return managementidentity.RolePage{}, fmt.Errorf("list Management roles: %w", err)
	}
	defer rows.Close()
	items := make([]managementidentity.Role, 0, request.Limit+1)
	for rows.Next() {
		item, err := scanRole(rows)
		if err != nil {
			return managementidentity.RolePage{}, fmt.Errorf("scan Management role page: %w", err)
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return managementidentity.RolePage{}, fmt.Errorf("iterate Management role page: %w", err)
	}
	page := managementidentity.RolePage{Items: items}
	if len(items) > request.Limit {
		page.Items = items[:request.Limit]
		page.NextCursor = string(page.Items[len(page.Items)-1].Role.ID)
	}
	return page, nil
}

func (store *Store) CreateRole(ctx context.Context, request managementidentity.CreateRole) (managementidentity.MutationResult, error) {
	if err := validateCreateCommand(request.Command, request.Actor, managementcommand.ScopeNamespace); err != nil ||
		request.Command.Scope.NamespaceID != request.NamespaceID {
		return managementidentity.MutationResult{}, errors.New("management role command scope does not match the owner namespace")
	}
	permissions, digest, err := encodePermissionSet(request.Permissions)
	if err != nil {
		return managementidentity.MutationResult{}, err
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.MutationResult, error) {
		if replay, found, err := commandpostgres.Lock(ctx, tx, request.Command); err != nil {
			return managementidentity.MutationResult{}, mapCommandError(err)
		} else if found {
			return replayMutation(replay, "management_role")
		}
		candidate := accesscontrol.ManagementRole{
			ID: accesscontrol.ManagementRoleID(request.ID), NamespaceID: accesscontrol.NamespaceID(request.NamespaceID),
			Name: request.Name, DisplayName: request.DisplayName, Permissions: request.Permissions,
			BuiltIn: false, Status: accesscontrol.RoleStatusActive, Revision: 1,
		}
		if err := store.canCreateRole(ctx, tx, request.Actor.PrincipalID, candidate); err != nil {
			return managementidentity.MutationResult{}, err
		}
		role, err := scanRole(tx.QueryRowContext(ctx, `INSERT INTO management_roles
  (id,namespace_id,name,display_name,description,permissions,permissions_digest,builtin,status,revision)
VALUES ($1,$2,$3,$4,$5,$6,$7,FALSE,'active',1)
RETURNING `+roleColumns, request.ID, request.NamespaceID, request.Name, request.DisplayName,
			request.Description, permissions, digest[:]))
		if err != nil {
			return managementidentity.MutationResult{}, mapWriteError("create Management role", err)
		}
		if err := appendAudit(ctx, tx, auditMutation{
			NamespaceID: request.NamespaceID, Action: "management_role.created", ResourceType: "management_role",
			ResourceID: request.ID, AfterRevision: uint64(role.Role.Revision), Actor: request.Actor,
		}); err != nil {
			return managementidentity.MutationResult{}, err
		}
		result := managementcommand.ResourceResult{ResourceType: "management_role", ResourceID: request.ID, ResourceRevision: uint64(role.Role.Revision), ResponseStatus: 201}
		if err := commandpostgres.CompleteResource(ctx, tx, request.Command, result); err != nil {
			return managementidentity.MutationResult{}, err
		}
		return managementidentity.MutationResult{Kind: result.ResourceType, ID: result.ResourceID, Revision: result.ResourceRevision, ResponseStatus: 201}, nil
	})
}

func (store *Store) UpdateRole(ctx context.Context, request managementidentity.UpdateRole) (managementidentity.MutationResult, error) {
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.MutationResult, error) {
		role, err := scanRole(tx.QueryRowContext(ctx, `UPDATE management_roles SET
  display_name=COALESCE($3,display_name), description=COALESCE($4,description),
  revision=revision+1,updated_at=clock_timestamp()
WHERE id=$1 AND revision=$2 AND builtin=FALSE RETURNING `+roleColumns,
			request.ID, request.ExpectedRevision, nullableString(request.DisplayName), nullableString(request.Description)))
		if errors.Is(err, sql.ErrNoRows) {
			var builtIn bool
			lookupErr := tx.QueryRowContext(ctx, `SELECT builtin FROM management_roles WHERE id=$1`, request.ID).Scan(&builtIn)
			if lookupErr == nil && builtIn {
				return managementidentity.MutationResult{}, managementidentity.ErrBuiltInImmutable
			}
			return managementidentity.MutationResult{}, classifyRevision(ctx, tx, "management_roles", request.ID)
		}
		if err != nil {
			return managementidentity.MutationResult{}, mapWriteError("update Management role", err)
		}
		before := request.ExpectedRevision
		if err := appendAudit(ctx, tx, auditMutation{
			NamespaceID: string(role.Role.NamespaceID), Action: "management_role.updated", ResourceType: "management_role",
			ResourceID: request.ID, BeforeRevision: &before, AfterRevision: uint64(role.Role.Revision), Actor: request.Actor,
		}); err != nil {
			return managementidentity.MutationResult{}, err
		}
		return managementidentity.MutationResult{Kind: "management_role", ID: request.ID, Revision: uint64(role.Role.Revision), ResponseStatus: 200}, nil
	})
}

func (store *Store) DeleteRole(ctx context.Context, id string, expected uint64, actor managementidentity.MutationActor) (managementidentity.MutationResult, error) {
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.MutationResult, error) {
		role, err := scanRole(tx.QueryRowContext(ctx, `SELECT `+roleColumns+` FROM management_roles WHERE id=$1 FOR UPDATE`, id))
		if errors.Is(err, sql.ErrNoRows) {
			return managementidentity.MutationResult{}, managementidentity.ErrNotFound
		}
		if err != nil {
			return managementidentity.MutationResult{}, err
		}
		if role.Role.BuiltIn {
			return managementidentity.MutationResult{}, managementidentity.ErrBuiltInImmutable
		}
		if uint64(role.Role.Revision) != expected {
			return managementidentity.MutationResult{}, managementidentity.ErrRevisionConflict
		}
		var activeBindings int
		if err := tx.QueryRowContext(ctx, `SELECT count(*) FROM management_role_bindings
WHERE role_id=$1 AND status='active'`, id).Scan(&activeBindings); err != nil {
			return managementidentity.MutationResult{}, err
		}
		if activeBindings != 0 {
			return managementidentity.MutationResult{}, managementidentity.ErrRoleInUse
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM management_roles WHERE id=$1`, id); err != nil {
			return managementidentity.MutationResult{}, mapWriteError("delete Management role", err)
		}
		after := expected + 1
		if err := appendAudit(ctx, tx, auditMutation{
			NamespaceID: string(role.Role.NamespaceID), Action: "management_role.deleted", ResourceType: "management_role",
			ResourceID: id, BeforeRevision: &expected, AfterRevision: after, Actor: actor,
		}); err != nil {
			return managementidentity.MutationResult{}, err
		}
		return managementidentity.MutationResult{Kind: "management_role", ID: id, Revision: after, ResponseStatus: 204}, nil
	})
}

func (store *Store) canCreateRole(ctx context.Context, tx *sql.Tx, actorID string, target accesscontrol.ManagementRole) error {
	sources, err := loadDelegationSources(ctx, tx, actorID)
	if err != nil {
		return err
	}
	for _, source := range sources {
		if accesscontrol.CanCreateCustomRole(source.binding.Binding, source.role.Role, target) == nil {
			return nil
		}
	}
	return managementidentity.ErrDelegationDenied
}

func roleResult(value managementidentity.Role, err error) (managementidentity.Role, error) {
	if errors.Is(err, sql.ErrNoRows) {
		return managementidentity.Role{}, managementidentity.ErrNotFound
	}
	if err != nil {
		return managementidentity.Role{}, fmt.Errorf("load Management role: %w", err)
	}
	return value, nil
}
