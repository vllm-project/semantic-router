package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/namespacemanagement"
)

const namespaceResultScopeQuery = `SELECT principal.status,binding.scope_kind,binding.namespace_id,
       binding.status,role.namespace_id,role.permissions,role.status
FROM management_principals AS principal
LEFT JOIN management_role_bindings AS binding ON binding.principal_id=principal.id
LEFT JOIN management_roles AS role ON role.id=binding.role_id
WHERE principal.id=$1
ORDER BY binding.id`

// ResolveNamespaceResultScope compiles the Namespace collection predicate from
// current durable role bindings. The returned scope is pushed into SQL; callers
// must never enumerate all Namespaces and filter rows in memory.
func (store *Store) ResolveNamespaceResultScope(
	ctx context.Context,
	principalID string,
) (_ namespacemanagement.ResultScope, returnErr error) {
	if store == nil || store.database == nil {
		return namespacemanagement.ResultScope{}, ErrStateInvalid
	}
	parsed, resolveNamespaceResultScopeErr := uuid.Parse(principalID)
	if resolveNamespaceResultScopeErr != nil || parsed.String() != principalID {
		return namespacemanagement.ResultScope{}, managementauthorization.ErrInvalidContext
	}
	rows, resolveNamespaceResultScopeErr := store.database.QueryContext(ctx, namespaceResultScopeQuery, principalID)
	if resolveNamespaceResultScopeErr != nil {
		return namespacemanagement.ResultScope{}, fmt.Errorf("load Namespace result scope: %w", resolveNamespaceResultScopeErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := namespacemanagement.ResultScope{}
	foundPrincipal := false
	for rows.Next() {
		var principalStatus string
		var scopeKind, bindingNamespace, bindingStatus, roleNamespace, roleStatus sql.NullString
		var permissions []byte
		if err := rows.Scan(&principalStatus, &scopeKind, &bindingNamespace, &bindingStatus, &roleNamespace, &permissions, &roleStatus); err != nil {
			return namespacemanagement.ResultScope{}, err
		}
		foundPrincipal = true
		if principalStatus != "active" {
			return namespacemanagement.ResultScope{}, ErrPrincipalInactive
		}
		if !scopeKind.Valid {
			continue
		}
		if !bindingStatus.Valid || bindingStatus.String != "active" || !roleStatus.Valid || roleStatus.String != "active" {
			continue
		}
		var granted []string
		if err := json.Unmarshal(permissions, &granted); err != nil {
			return namespacemanagement.ResultScope{}, ErrStateInvalid
		}
		slices.Sort(granted)
		if slices.Contains(granted, "cluster.read") || slices.Contains(granted, "namespace.read") {
			switch scopeKind.String {
			case "cluster":
				if bindingNamespace.Valid || roleNamespace.Valid {
					return namespacemanagement.ResultScope{}, ErrStateInvalid
				}
				result.All = true
				result.NamespaceIDs = nil
			case "namespace":
				if !bindingNamespace.Valid {
					return namespacemanagement.ResultScope{}, ErrStateInvalid
				}
				if roleNamespace.Valid && roleNamespace.String != bindingNamespace.String {
					return namespacemanagement.ResultScope{}, ErrStateInvalid
				}
				if !result.All && slices.Contains(granted, "namespace.read") {
					result.NamespaceIDs = append(result.NamespaceIDs, bindingNamespace.String)
				}
			}
		}
	}
	if err := rows.Err(); err != nil {
		return namespacemanagement.ResultScope{}, err
	}
	if !foundPrincipal {
		return namespacemanagement.ResultScope{}, ErrPrincipalNotFound
	}
	canonical, resolveNamespaceResultScopeErr := result.Canonical()
	if resolveNamespaceResultScopeErr != nil {
		return namespacemanagement.ResultScope{}, ErrStateInvalid
	}
	if !canonical.All && len(canonical.NamespaceIDs) == 0 {
		return namespacemanagement.ResultScope{}, managementauthorization.ErrDenied
	}
	return canonical, nil
}

var _ interface {
	ResolveNamespaceResultScope(context.Context, string) (namespacemanagement.ResultScope, error)
} = (*Store)(nil)
