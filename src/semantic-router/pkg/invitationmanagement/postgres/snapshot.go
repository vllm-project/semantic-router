package postgres

import (
	"context"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement"
)

type roleRecord struct {
	role   accesscontrol.ManagementRole
	digest string
}

type delegationRecord struct {
	binding accesscontrol.ManagementRoleBinding
	role    accesscontrol.ManagementRole
	digest  string
}

func (store *Store) ResolveSnapshot(ctx context.Context, namespaceID, actorID string, requested []invitationmanagement.RequestedRoleGrant, team *invitationmanagement.TeamAssignment) (invitationmanagement.OnboardingSnapshot, error) {
	return inTransaction(ctx, store, sql.LevelRepeatableRead, func(tx *sql.Tx) (invitationmanagement.OnboardingSnapshot, error) {
		return resolveSnapshot(ctx, tx, namespaceID, actorID, requested, team, "future-user")
	})
}

func resolveSnapshot(ctx context.Context, tx *sql.Tx, namespaceID, actorID string, requested []invitationmanagement.RequestedRoleGrant, team *invitationmanagement.TeamAssignment, userID string) (invitationmanagement.OnboardingSnapshot, error) {
	var snapshot invitationmanagement.OnboardingSnapshot
	resolveSnapshotErr := tx.QueryRowContext(ctx, `SELECT sp.revision, sp.automatic_first_key,
       ap.id::text, ap.revision, rp.id::text, rp.revision
FROM self_service_policies sp
JOIN access_policies ap ON ap.id=sp.default_access_policy_id
  AND ap.namespace_id=sp.namespace_id AND ap.status='active'
JOIN rate_limit_policies rp ON rp.id=sp.default_rate_limit_policy_id
  AND rp.namespace_id=sp.namespace_id AND rp.status='active'
WHERE sp.namespace_id=$1
FOR SHARE OF sp,ap,rp`, namespaceID).Scan(
		&snapshot.SelfServicePolicyRevision, &snapshot.AutomaticFirstKey,
		&snapshot.AccessPolicyID, &snapshot.AccessPolicyRevision,
		&snapshot.RateLimitPolicyID, &snapshot.RateLimitPolicyRevision,
	)
	if errors.Is(resolveSnapshotErr, sql.ErrNoRows) {
		return invitationmanagement.OnboardingSnapshot{}, invitationmanagement.ErrDefaultsChanged
	}
	if resolveSnapshotErr != nil {
		return invitationmanagement.OnboardingSnapshot{}, fmt.Errorf("resolve invitation defaults: %w", resolveSnapshotErr)
	}
	if team != nil {
		var active bool
		if err := tx.QueryRowContext(ctx, `SELECT status='active' AND deleted_at IS NULL
FROM access_teams WHERE namespace_id=$1 AND id=$2 FOR SHARE`, namespaceID, team.TeamID).Scan(&active); err != nil || !active {
			if errors.Is(err, sql.ErrNoRows) || err == nil {
				return invitationmanagement.OnboardingSnapshot{}, invitationmanagement.ErrConflict
			}
			return invitationmanagement.OnboardingSnapshot{}, fmt.Errorf("resolve invitation Team: %w", err)
		}
		value := *team
		snapshot.Team = &value
	}
	sources, resolveSnapshotErr := loadDelegationSources(ctx, tx, actorID)
	if resolveSnapshotErr != nil {
		return invitationmanagement.OnboardingSnapshot{}, resolveSnapshotErr
	}
	snapshot.RoleGrants = make([]invitationmanagement.RoleGrant, len(requested))
	for index, candidate := range requested {
		target, err := loadRole(ctx, tx, namespaceID, candidate.RoleID)
		if err != nil {
			return invitationmanagement.OnboardingSnapshot{}, err
		}
		ceiling, err := permissionSet(candidate.DelegationCeiling)
		if err != nil {
			return invitationmanagement.OnboardingSnapshot{}, invitationmanagement.ErrInvalidRequest
		}
		targetScope := accesscontrol.NamespaceScope(accesscontrol.NamespaceID(namespaceID))
		if candidate.ScopeKind == "user" {
			targetScope = accesscontrol.UserScope(accesscontrol.NamespaceID(namespaceID), accesscontrol.UserID(userID))
		}
		var selected *delegationRecord
		for sourceIndex := range sources {
			if accesscontrol.CanDelegateRoleBinding(sources[sourceIndex].binding, sources[sourceIndex].role,
				target.role, accesscontrol.ScopedTarget{Scope: targetScope}, ceiling) == nil {
				selected = &sources[sourceIndex]
				break
			}
		}
		if selected == nil {
			return invitationmanagement.OnboardingSnapshot{}, invitationmanagement.ErrDelegationDenied
		}
		snapshot.RoleGrants[index] = invitationmanagement.RoleGrant{
			RoleID: candidate.RoleID, RoleRevision: uint64(target.role.Revision),
			RolePermissionsDigest: target.digest, ScopeKind: candidate.ScopeKind,
			DelegationCeiling: append([]string(nil), candidate.DelegationCeiling...),
			SourceBindingID:   string(selected.binding.ID), SourceBindingRevision: uint64(selected.binding.Revision),
			SourceRoleID: string(selected.role.ID), SourcePermissionsDigest: selected.digest,
		}
	}
	return snapshot, nil
}

func verifySnapshot(ctx context.Context, tx *sql.Tx, namespaceID, actorID, userID string, expected invitationmanagement.OnboardingSnapshot) error {
	actual, err := resolveSnapshot(ctx, tx, namespaceID, actorID, nil, expected.Team, userID)
	if err != nil {
		return err
	}
	// Privileged onboarding may deliberately override automatic-first-key.
	actual.AutomaticFirstKey = expected.AutomaticFirstKey
	actual.RoleGrants = expected.RoleGrants
	expectedJSON, _ := json.Marshal(expected)
	actualJSON, _ := json.Marshal(actual)
	if string(expectedJSON) != string(actualJSON) {
		return invitationmanagement.ErrDefaultsChanged
	}
	for _, grant := range expected.RoleGrants {
		target, err := loadRole(ctx, tx, namespaceID, grant.RoleID)
		if err != nil || uint64(target.role.Revision) != grant.RoleRevision || target.digest != grant.RolePermissionsDigest {
			return invitationmanagement.ErrDefaultsChanged
		}
		source, err := loadDelegationSource(ctx, tx, actorID, grant.SourceBindingID)
		if err != nil || uint64(source.binding.Revision) != grant.SourceBindingRevision ||
			string(source.role.ID) != grant.SourceRoleID || source.digest != grant.SourcePermissionsDigest {
			return invitationmanagement.ErrDelegationDenied
		}
		ceiling, err := permissionSet(grant.DelegationCeiling)
		if err != nil {
			return invitationmanagement.ErrDefaultsChanged
		}
		targetScope := accesscontrol.NamespaceScope(accesscontrol.NamespaceID(namespaceID))
		if grant.ScopeKind == "user" {
			targetScope = accesscontrol.UserScope(accesscontrol.NamespaceID(namespaceID), accesscontrol.UserID(userID))
		}
		if accesscontrol.CanDelegateRoleBinding(source.binding, source.role, target.role,
			accesscontrol.ScopedTarget{Scope: targetScope}, ceiling) != nil {
			return invitationmanagement.ErrDelegationDenied
		}
	}
	return nil
}

func loadRole(ctx context.Context, tx *sql.Tx, namespaceID, roleID string) (roleRecord, error) {
	var (
		record          roleRecord
		namespace       sql.NullString
		permissionsJSON []byte
		digest          []byte
		status          string
	)
	err := tx.QueryRowContext(ctx, `SELECT id::text, namespace_id::text, name, display_name,
       permissions, permissions_digest, builtin, status, revision
FROM management_roles
WHERE id=$1 AND status='active' AND (namespace_id IS NULL OR namespace_id=$2)`, roleID, namespaceID).Scan(
		&record.role.ID, &namespace, &record.role.Name, &record.role.DisplayName,
		&permissionsJSON, &digest, &record.role.BuiltIn, &status, &record.role.Revision,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return roleRecord{}, invitationmanagement.ErrConflict
	}
	if err != nil {
		return roleRecord{}, fmt.Errorf("load invitation role: %w", err)
	}
	record.role.NamespaceID = accesscontrol.NamespaceID(namespace.String)
	record.role.Status = accesscontrol.RoleStatus(status)
	permissions, err := decodePermissions(permissionsJSON)
	if err != nil || len(digest) != 32 {
		return roleRecord{}, invitationmanagement.ErrUnavailable
	}
	record.role.Permissions, record.digest = permissions, hex.EncodeToString(digest)
	if err := record.role.Validate(); err != nil {
		return roleRecord{}, invitationmanagement.ErrUnavailable
	}
	return record, nil
}

func loadDelegationSources(
	ctx context.Context,
	tx *sql.Tx,
	actorID string,
) (_ []delegationRecord, returnErr error) {
	rows, loadDelegationSourcesErr := tx.QueryContext(ctx, `SELECT b.id::text,b.role_id::text,b.scope_kind,b.namespace_id::text,
       b.resource_type,b.resource_id,b.delegation_ceiling,b.status,b.revision,
       r.namespace_id::text,r.name,r.display_name,r.permissions,r.permissions_digest,
       r.builtin,r.status,r.revision
FROM management_role_bindings b
JOIN management_roles r ON r.id=b.role_id
WHERE b.principal_id=$1 AND b.status='active' AND r.status='active'
ORDER BY b.id LIMIT 201`, actorID)
	if loadDelegationSourcesErr != nil {
		return nil, fmt.Errorf("load invitation delegation sources: %w", loadDelegationSourcesErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make([]delegationRecord, 0)
	for rows.Next() {
		var (
			value                                                     delegationRecord
			scopeKind, bindingStatus, roleStatus                      string
			bindingNamespace, resourceType, resourceID, roleNamespace sql.NullString
			ceilingJSON, permissionsJSON, digest                      []byte
		)
		if err := rows.Scan(&value.binding.ID, &value.binding.RoleID, &scopeKind,
			&bindingNamespace, &resourceType, &resourceID, &ceilingJSON, &bindingStatus, &value.binding.Revision,
			&roleNamespace, &value.role.Name, &value.role.DisplayName, &permissionsJSON, &digest,
			&value.role.BuiltIn, &roleStatus, &value.role.Revision); err != nil {
			return nil, fmt.Errorf("scan invitation delegation source: %w", err)
		}
		value.binding.PrincipalID = accesscontrol.ManagementPrincipalID(actorID)
		value.binding.Status = accesscontrol.BindingStatus(bindingStatus)
		value.binding.Scope = decodeScope(scopeKind, bindingNamespace.String, resourceType.String, resourceID.String)
		value.binding.DelegationCeiling, loadDelegationSourcesErr = decodePermissions(ceilingJSON)
		if loadDelegationSourcesErr != nil {
			return nil, invitationmanagement.ErrUnavailable
		}
		value.role.ID = value.binding.RoleID
		value.role.NamespaceID = accesscontrol.NamespaceID(roleNamespace.String)
		value.role.Status = accesscontrol.RoleStatus(roleStatus)
		value.role.Permissions, loadDelegationSourcesErr = decodePermissions(permissionsJSON)
		if loadDelegationSourcesErr != nil || len(digest) != 32 {
			return nil, invitationmanagement.ErrUnavailable
		}
		value.digest = hex.EncodeToString(digest)
		result = append(result, value)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(result) > 200 {
		return nil, invitationmanagement.ErrUnavailable
	}
	return result, nil
}

func loadDelegationSource(
	ctx context.Context,
	tx *sql.Tx,
	actorID string,
	bindingID string,
) (_ delegationRecord, returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT b.id::text,b.role_id::text,b.scope_kind,b.namespace_id::text,
       b.resource_type,b.resource_id,b.delegation_ceiling,b.status,b.revision,
       r.namespace_id::text,r.name,r.display_name,r.permissions,r.permissions_digest,
       r.builtin,r.status,r.revision
FROM management_role_bindings b
JOIN management_roles r ON r.id=b.role_id
WHERE b.id=$1 AND b.principal_id=$2 AND b.status='active' AND r.status='active'`, bindingID, actorID)
	if err != nil {
		return delegationRecord{}, err
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	values, err := scanDelegationRows(rows, actorID, 1)
	if err != nil {
		return delegationRecord{}, err
	}
	if len(values) != 1 {
		return delegationRecord{}, invitationmanagement.ErrDelegationDenied
	}
	return values[0], nil
}

func scanDelegationRows(rows *sql.Rows, actorID string, maximum int) ([]delegationRecord, error) {
	result := make([]delegationRecord, 0)
	for rows.Next() {
		var (
			value                                                     delegationRecord
			scopeKind, bindingStatus, roleStatus                      string
			bindingNamespace, resourceType, resourceID, roleNamespace sql.NullString
			ceilingJSON, permissionsJSON, digest                      []byte
		)
		if err := rows.Scan(&value.binding.ID, &value.binding.RoleID, &scopeKind,
			&bindingNamespace, &resourceType, &resourceID, &ceilingJSON, &bindingStatus, &value.binding.Revision,
			&roleNamespace, &value.role.Name, &value.role.DisplayName, &permissionsJSON, &digest,
			&value.role.BuiltIn, &roleStatus, &value.role.Revision); err != nil {
			return nil, fmt.Errorf("scan invitation delegation source: %w", err)
		}
		value.binding.PrincipalID = accesscontrol.ManagementPrincipalID(actorID)
		value.binding.Status = accesscontrol.BindingStatus(bindingStatus)
		value.binding.Scope = decodeScope(scopeKind, bindingNamespace.String, resourceType.String, resourceID.String)
		var err error
		value.binding.DelegationCeiling, err = decodePermissions(ceilingJSON)
		if err != nil {
			return nil, invitationmanagement.ErrUnavailable
		}
		value.role.ID = value.binding.RoleID
		value.role.NamespaceID = accesscontrol.NamespaceID(roleNamespace.String)
		value.role.Status = accesscontrol.RoleStatus(roleStatus)
		value.role.Permissions, err = decodePermissions(permissionsJSON)
		if err != nil || len(digest) != 32 {
			return nil, invitationmanagement.ErrUnavailable
		}
		value.digest = hex.EncodeToString(digest)
		result = append(result, value)
		if len(result) > maximum {
			return nil, invitationmanagement.ErrUnavailable
		}
	}
	return result, rows.Err()
}

func decodeScope(kind, namespaceID, resourceType, resourceID string) accesscontrol.Scope {
	switch accesscontrol.ScopeKind(kind) {
	case accesscontrol.ScopeKindCluster:
		return accesscontrol.ClusterScope()
	case accesscontrol.ScopeKindNamespace:
		return accesscontrol.NamespaceScope(accesscontrol.NamespaceID(namespaceID))
	case accesscontrol.ScopeKindTeam:
		return accesscontrol.TeamScope(accesscontrol.NamespaceID(namespaceID), accesscontrol.TeamID(resourceID))
	case accesscontrol.ScopeKindUser:
		return accesscontrol.UserScope(accesscontrol.NamespaceID(namespaceID), accesscontrol.UserID(resourceID))
	case accesscontrol.ScopeKindResource:
		return accesscontrol.ResourceScope(accesscontrol.NamespaceID(namespaceID), accesscontrol.ScopeResourceType(resourceType), accesscontrol.ResourceID(resourceID))
	default:
		return accesscontrol.Scope{}
	}
}

func decodePermissions(payload []byte) (accesscontrol.PermissionSet, error) {
	var values []string
	if err := json.Unmarshal(payload, &values); err != nil {
		return accesscontrol.PermissionSet{}, err
	}
	return permissionSet(values)
}

func permissionSet(values []string) (accesscontrol.PermissionSet, error) {
	permissions := make([]accesscontrol.Permission, len(values))
	for index, value := range values {
		permissions[index] = accesscontrol.Permission(value)
	}
	return accesscontrol.NewPermissionSet(permissions...)
}
