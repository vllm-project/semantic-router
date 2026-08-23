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

const bindingColumns = `id::text, principal_id::text, role_id::text,
       scope_kind, namespace_id::text, resource_type, resource_id,
       delegation_ceiling, status, revision, created_at, updated_at`

type delegationSource struct {
	binding managementidentity.RoleBinding
	role    managementidentity.Role
}

func (store *Store) GetRoleBinding(ctx context.Context, id string) (managementidentity.RoleBinding, error) {
	if !canonicalUUID(id) {
		return managementidentity.RoleBinding{}, managementidentity.ErrNotFound
	}
	binding, err := scanRoleBinding(store.database.QueryRowContext(ctx,
		`SELECT `+bindingColumns+` FROM management_role_bindings WHERE id=$1`, id))
	return bindingResult(binding, err)
}

func (store *Store) ListRoleBindings(ctx context.Context, principalID string, request managementidentity.ListRequest) (managementidentity.RoleBindingPage, error) {
	if err := validateList(request); err != nil {
		return managementidentity.RoleBindingPage{}, err
	}
	rows, err := store.database.QueryContext(ctx, `SELECT `+bindingColumns+`
FROM management_role_bindings
WHERE ($1='' OR principal_id=NULLIF($1,'')::uuid) AND ($2='' OR id>NULLIF($2,'')::uuid)
ORDER BY id LIMIT $3`, principalID, request.AfterID, request.Limit+1)
	if err != nil {
		return managementidentity.RoleBindingPage{}, fmt.Errorf("list Management role bindings: %w", err)
	}
	defer rows.Close()
	items := make([]managementidentity.RoleBinding, 0, request.Limit+1)
	for rows.Next() {
		item, err := scanRoleBinding(rows)
		if err != nil {
			return managementidentity.RoleBindingPage{}, fmt.Errorf("scan Management role-binding page: %w", err)
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return managementidentity.RoleBindingPage{}, fmt.Errorf("iterate Management role-binding page: %w", err)
	}
	page := managementidentity.RoleBindingPage{Items: items}
	if len(items) > request.Limit {
		page.Items = items[:request.Limit]
		page.NextCursor = string(page.Items[len(page.Items)-1].Binding.ID)
	}
	return page, nil
}

func (store *Store) CreateRoleBinding(ctx context.Context, request managementidentity.CreateRoleBinding) (managementidentity.MutationResult, error) {
	expectedScope := managementcommand.ScopeNamespace
	if request.Scope.Kind == accesscontrol.ScopeKindCluster {
		expectedScope = managementcommand.ScopeCluster
	}
	if err := validateCreateCommand(request.Command, request.Actor, expectedScope); err != nil ||
		(expectedScope == managementcommand.ScopeNamespace && request.Command.Scope.NamespaceID != string(request.Scope.NamespaceID)) {
		return managementidentity.MutationResult{}, errors.New("management role-binding command scope does not match the binding")
	}
	ceiling, _, err := encodePermissionSet(request.DelegationCeiling)
	if request.DelegationCeiling.Empty() {
		ceiling = []byte("[]")
		err = nil
	}
	if err != nil {
		return managementidentity.MutationResult{}, err
	}
	namespace, resourceType, resourceID, err := scopeColumns(request.Scope)
	if err != nil {
		return managementidentity.MutationResult{}, err
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.MutationResult, error) {
		if replay, found, err := commandpostgres.Lock(ctx, tx, request.Command); err != nil {
			return managementidentity.MutationResult{}, mapCommandError(err)
		} else if found {
			return replayMutation(replay, "management_role_binding")
		}
		principal, createRoleBindingErr := scanPrincipal(tx.QueryRowContext(ctx, `SELECT `+principalColumns+` FROM management_principals WHERE id=$1`, request.PrincipalID))
		if createRoleBindingErr != nil {
			return managementidentity.MutationResult{}, mapReferenceError(createRoleBindingErr)
		}
		if err := validateServiceAccountBindingScope(ctx, tx, request.PrincipalID, request.Scope); err != nil {
			return managementidentity.MutationResult{}, err
		}
		role, createRoleBindingErr := scanRole(tx.QueryRowContext(ctx, `SELECT `+roleColumns+` FROM management_roles WHERE id=$1`, request.RoleID))
		if createRoleBindingErr != nil {
			return managementidentity.MutationResult{}, mapReferenceError(createRoleBindingErr)
		}
		target := accesscontrol.ManagementRoleBinding{
			ID: accesscontrol.ManagementRoleBindingID(request.ID), PrincipalID: principal.Identity.ID,
			RoleID: role.Role.ID, Scope: request.Scope, DelegationCeiling: request.DelegationCeiling,
			Status: accesscontrol.BindingStatusActive, Revision: 1,
		}
		if err := accesscontrol.ValidateManagementRoleBindingReferences(target, principal.Identity, role.Role); err != nil {
			return managementidentity.MutationResult{}, errors.New("management role-binding references are invalid")
		}
		if request.Scope.Kind == accesscontrol.ScopeKindUser {
			var linked bool
			if err := tx.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM management_principal_user_links
WHERE principal_id=$1 AND namespace_id=$2 AND user_id=$3)`, request.PrincipalID,
				request.Scope.NamespaceID, request.Scope.UserID).Scan(&linked); err != nil || !linked {
				return managementidentity.MutationResult{}, errors.New("user-scoped role binding requires the principal's exact User link")
			}
		}
		if err := store.canDelegateBinding(ctx, tx, request.Actor.PrincipalID, role.Role,
			accesscontrol.ScopedTarget{Scope: request.Scope}, request.DelegationCeiling); err != nil {
			return managementidentity.MutationResult{}, err
		}
		binding, createRoleBindingErr := scanRoleBinding(tx.QueryRowContext(ctx, `INSERT INTO management_role_bindings
  (id,principal_id,role_id,scope_kind,namespace_id,resource_type,resource_id,
   delegation_ceiling,status,revision)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'active',1)
RETURNING `+bindingColumns, request.ID, request.PrincipalID, request.RoleID,
			request.Scope.Kind, namespace, resourceType, resourceID, ceiling))
		if createRoleBindingErr != nil {
			return managementidentity.MutationResult{}, mapWriteError("create Management role binding", createRoleBindingErr)
		}
		if err := appendAudit(ctx, tx, auditMutation{
			NamespaceID: string(binding.Binding.Scope.NamespaceID), Action: "management_role_binding.created",
			ResourceType: "management_role_binding", ResourceID: request.ID,
			AfterRevision: uint64(binding.Binding.Revision), Actor: request.Actor,
		}); err != nil {
			return managementidentity.MutationResult{}, err
		}
		result := managementcommand.ResourceResult{ResourceType: "management_role_binding", ResourceID: request.ID, ResourceRevision: uint64(binding.Binding.Revision), ResponseStatus: 201}
		if err := commandpostgres.CompleteResource(ctx, tx, request.Command, result); err != nil {
			return managementidentity.MutationResult{}, err
		}
		return managementidentity.MutationResult{Kind: result.ResourceType, ID: result.ResourceID, Revision: result.ResourceRevision, ResponseStatus: 201}, nil
	})
}

func validateServiceAccountBindingScope(
	ctx context.Context,
	tx *sql.Tx,
	principalID string,
	scope accesscontrol.Scope,
) error {
	var ownerScope string
	var namespaceID sql.NullString
	err := tx.QueryRowContext(ctx, `SELECT owner_scope,namespace_id::text
FROM management_service_accounts WHERE principal_id=$1`, principalID).Scan(&ownerScope, &namespaceID)
	if errors.Is(err, sql.ErrNoRows) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("load Management service-account owner scope: %w", err)
	}
	if ownerScope == string(managementidentity.ServiceAccountOwnerNamespace) &&
		(scope.Kind == accesscontrol.ScopeKindCluster || !namespaceID.Valid ||
			string(scope.NamespaceID) != namespaceID.String) {
		return managementidentity.ErrInvalidLifecycleRequest
	}
	return nil
}

func (store *Store) UpdateRoleBinding(ctx context.Context, request managementidentity.UpdateRoleBinding) (managementidentity.MutationResult, error) {
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.MutationResult, error) {
		binding, err := scanRoleBinding(tx.QueryRowContext(ctx, `UPDATE management_role_bindings SET
  status=$3,revision=revision+1,updated_at=clock_timestamp()
WHERE id=$1 AND revision=$2 RETURNING `+bindingColumns, request.ID, request.ExpectedRevision, request.Status))
		if errors.Is(err, sql.ErrNoRows) {
			return managementidentity.MutationResult{}, classifyRevision(ctx, tx, "management_role_bindings", request.ID)
		}
		if err != nil {
			return managementidentity.MutationResult{}, mapWriteError("update Management role binding", err)
		}
		before := request.ExpectedRevision
		if err := appendAudit(ctx, tx, auditMutation{
			NamespaceID: string(binding.Binding.Scope.NamespaceID), Action: "management_role_binding.updated",
			ResourceType: "management_role_binding", ResourceID: request.ID,
			BeforeRevision: &before, AfterRevision: uint64(binding.Binding.Revision), Actor: request.Actor,
		}); err != nil {
			return managementidentity.MutationResult{}, err
		}
		return managementidentity.MutationResult{Kind: "management_role_binding", ID: request.ID, Revision: uint64(binding.Binding.Revision), ResponseStatus: 200}, nil
	})
}

func (store *Store) DeleteRoleBinding(ctx context.Context, id string, expected uint64, actor managementidentity.MutationActor) (managementidentity.MutationResult, error) {
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.MutationResult, error) {
		binding, err := scanRoleBinding(tx.QueryRowContext(ctx, `SELECT `+bindingColumns+`
FROM management_role_bindings WHERE id=$1 FOR UPDATE`, id))
		if errors.Is(err, sql.ErrNoRows) {
			return managementidentity.MutationResult{}, managementidentity.ErrNotFound
		}
		if err != nil {
			return managementidentity.MutationResult{}, err
		}
		if uint64(binding.Binding.Revision) != expected {
			return managementidentity.MutationResult{}, managementidentity.ErrRevisionConflict
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM management_role_bindings WHERE id=$1`, id); err != nil {
			return managementidentity.MutationResult{}, mapWriteError("delete Management role binding", err)
		}
		after := expected + 1
		if err := appendAudit(ctx, tx, auditMutation{
			NamespaceID: string(binding.Binding.Scope.NamespaceID), Action: "management_role_binding.deleted",
			ResourceType: "management_role_binding", ResourceID: id,
			BeforeRevision: &expected, AfterRevision: after, Actor: actor,
		}); err != nil {
			return managementidentity.MutationResult{}, err
		}
		return managementidentity.MutationResult{Kind: "management_role_binding", ID: id, Revision: after, ResponseStatus: 204}, nil
	})
}

func (store *Store) canDelegateBinding(ctx context.Context, tx *sql.Tx, actorID string, targetRole accesscontrol.ManagementRole, target accesscontrol.ScopedTarget, ceiling accesscontrol.PermissionSet) error {
	sources, err := loadDelegationSources(ctx, tx, actorID)
	if err != nil {
		return err
	}
	for _, source := range sources {
		if accesscontrol.CanDelegateRoleBinding(source.binding.Binding, source.role.Role, targetRole, target, ceiling) == nil {
			return nil
		}
	}
	return managementidentity.ErrDelegationDenied
}

func loadDelegationSources(ctx context.Context, tx *sql.Tx, actorID string) ([]delegationSource, error) {
	rows, err := tx.QueryContext(ctx, `SELECT binding.id::text,binding.role_id::text
FROM management_role_bindings binding
JOIN management_roles role ON role.id=binding.role_id
WHERE binding.principal_id=$1 AND binding.status='active' AND role.status='active'
ORDER BY binding.id LIMIT 201`, actorID)
	if err != nil {
		return nil, fmt.Errorf("load Management delegation sources: %w", err)
	}
	type pair struct{ bindingID, roleID string }
	ids := make([]pair, 0)
	for rows.Next() {
		var value pair
		if err := rows.Scan(&value.bindingID, &value.roleID); err != nil {
			rows.Close()
			return nil, err
		}
		ids = append(ids, value)
	}
	if err := rows.Close(); err != nil {
		return nil, err
	}
	if len(ids) > 200 {
		return nil, errors.New("management principal has too many delegation sources")
	}
	sources := make([]delegationSource, 0, len(ids))
	for _, id := range ids {
		binding, err := scanRoleBinding(tx.QueryRowContext(ctx, `SELECT `+bindingColumns+` FROM management_role_bindings WHERE id=$1`, id.bindingID))
		if err != nil {
			return nil, err
		}
		role, err := scanRole(tx.QueryRowContext(ctx, `SELECT `+roleColumns+` FROM management_roles WHERE id=$1`, id.roleID))
		if err != nil {
			return nil, err
		}
		sources = append(sources, delegationSource{binding: binding, role: role})
	}
	return sources, nil
}

func bindingResult(value managementidentity.RoleBinding, err error) (managementidentity.RoleBinding, error) {
	if errors.Is(err, sql.ErrNoRows) {
		return managementidentity.RoleBinding{}, managementidentity.ErrNotFound
	}
	if err != nil {
		return managementidentity.RoleBinding{}, fmt.Errorf("load Management role binding: %w", err)
	}
	return value, nil
}

func mapReferenceError(err error) error {
	if errors.Is(err, sql.ErrNoRows) {
		return managementidentity.ErrNotFound
	}
	return err
}
