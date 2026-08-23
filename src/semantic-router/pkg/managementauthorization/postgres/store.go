// Package postgres loads authoritative Management authorization facts from a
// single repeatable-read PostgreSQL snapshot. It does not cache grants: every
// Management request observes current principal, role, binding, User-link,
// Team-membership, and namespace self-service state.
package postgres

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

var (
	ErrPrincipalNotFound = errors.New("management principal was not found")
	ErrPrincipalInactive = errors.New("management principal is inactive")
	ErrStateInvalid      = errors.New("management authorization state is invalid")
)

const principalQuery = `SELECT issuer, subject, status, attributes, revision, created_at, updated_at
FROM management_principals
WHERE id = $1`

const roleGrantsQuery = `SELECT
  binding.id, binding.role_id, binding.scope_kind, binding.namespace_id,
  binding.resource_type, binding.resource_id, binding.delegation_ceiling,
  binding.status, binding.revision,
  role.namespace_id, role.name, role.display_name, role.permissions,
  role.builtin, role.status, role.revision
FROM management_role_bindings AS binding
JOIN management_roles AS role ON role.id = binding.role_id
WHERE binding.principal_id = $1
  AND (
    binding.scope_kind = 'cluster'
    OR ($2::uuid IS NOT NULL AND binding.namespace_id = $2::uuid)
  )
ORDER BY binding.id`

const principalUserLinkQuery = `SELECT link.user_id, link.revision, linked_user.status
FROM management_principal_user_links AS link
JOIN access_users AS linked_user
  ON linked_user.namespace_id = link.namespace_id AND linked_user.id = link.user_id
WHERE link.principal_id = $1 AND link.namespace_id = $2`

const selfServicePolicyQuery = `SELECT allow_team_key_delegation, team_admin_capabilities, revision
FROM self_service_policies
WHERE namespace_id = $1`

const teamMembershipsQuery = `SELECT
  membership.team_id, membership.role, membership.status,
  membership.revision, membership.created_at, membership.updated_at,
  team.status
FROM access_team_memberships AS membership
JOIN access_teams AS team
  ON team.namespace_id = membership.namespace_id AND team.id = membership.team_id
WHERE membership.namespace_id = $1 AND membership.user_id = $2
ORDER BY membership.team_id`

type Store struct {
	database *sql.DB
}

func New(database *sql.DB) (*Store, error) {
	if database == nil {
		return nil, errors.New("management authorization PostgreSQL database is required")
	}
	return &Store{database: database}, nil
}

// Load returns all authority that can cover the requested Namespace plus
// cluster-scoped bindings. An empty namespace intentionally loads only cluster
// bindings. The resulting digest covers every fact used to authorize a request
// and is suitable for binding short-lived discovery claims.
func (store *Store) Load(
	ctx context.Context,
	principalID accesscontrol.ManagementPrincipalID,
	namespaceID accesscontrol.NamespaceID,
) (_ managementauthorization.Snapshot, resultErr error) {
	if store == nil || store.database == nil {
		return managementauthorization.Snapshot{}, errors.New("management authorization PostgreSQL store is unavailable")
	}
	if _, err := uuid.Parse(string(principalID)); err != nil {
		return managementauthorization.Snapshot{}, ErrPrincipalNotFound
	}
	var namespaceArgument any
	if namespaceID != "" {
		if _, err := uuid.Parse(string(namespaceID)); err != nil {
			return managementauthorization.Snapshot{}, fmt.Errorf("%w: namespace identifier", ErrStateInvalid)
		}
		namespaceArgument = string(namespaceID)
	}

	transaction, err := store.database.BeginTx(ctx, &sql.TxOptions{
		Isolation: sql.LevelRepeatableRead,
		ReadOnly:  true,
	})
	if err != nil {
		return managementauthorization.Snapshot{}, fmt.Errorf("begin Management authorization snapshot: %w", err)
	}
	defer func() {
		if resultErr != nil {
			_ = transaction.Rollback()
		}
	}()

	snapshot, err := loadSnapshot(ctx, transaction, principalID, namespaceID, namespaceArgument)
	if err != nil {
		return managementauthorization.Snapshot{}, err
	}
	if err := transaction.Commit(); err != nil {
		return managementauthorization.Snapshot{}, fmt.Errorf("commit Management authorization snapshot: %w", err)
	}

	return snapshot, nil
}

// LoadInTransaction evaluates Management authority from the caller's
// transaction. Domains that atomically create security-sensitive resources
// use this seam so authorization facts and the mutation share one PostgreSQL
// snapshot; it must never commit or roll back the borrowed transaction.
func (store *Store) LoadInTransaction(
	ctx context.Context,
	transaction *sql.Tx,
	principalID accesscontrol.ManagementPrincipalID,
	namespaceID accesscontrol.NamespaceID,
) (managementauthorization.Snapshot, error) {
	if store == nil || store.database == nil || transaction == nil {
		return managementauthorization.Snapshot{}, errors.New("management authorization PostgreSQL store is unavailable")
	}
	if _, err := uuid.Parse(string(principalID)); err != nil {
		return managementauthorization.Snapshot{}, ErrPrincipalNotFound
	}
	var namespaceArgument any
	if namespaceID != "" {
		if _, err := uuid.Parse(string(namespaceID)); err != nil {
			return managementauthorization.Snapshot{}, fmt.Errorf("%w: namespace identifier", ErrStateInvalid)
		}
		namespaceArgument = string(namespaceID)
	}
	return loadSnapshot(ctx, transaction, principalID, namespaceID, namespaceArgument)
}

func loadSnapshot(
	ctx context.Context,
	transaction *sql.Tx,
	principalID accesscontrol.ManagementPrincipalID,
	namespaceID accesscontrol.NamespaceID,
	namespaceArgument any,
) (managementauthorization.Snapshot, error) {
	principal, principalRevision, err := loadPrincipal(ctx, transaction, principalID)
	if err != nil {
		return managementauthorization.Snapshot{}, err
	}
	if principal.Status != accesscontrol.PrincipalStatusActive {
		return managementauthorization.Snapshot{}, ErrPrincipalInactive
	}
	roleGrants, roleDigest, err := loadRoleGrants(ctx, transaction, principal, namespaceArgument)
	if err != nil {
		return managementauthorization.Snapshot{}, err
	}
	teamGrants, teamDigest, err := loadTeamGrants(ctx, transaction, principal.ID, namespaceID)
	if err != nil {
		return managementauthorization.Snapshot{}, err
	}
	digest, err := authorizationDigest(authorizationDigestInput{
		PrincipalID:       string(principal.ID),
		PrincipalStatus:   string(principal.Status),
		PrincipalRevision: uint64(principalRevision),
		NamespaceID:       string(namespaceID),
		Roles:             roleDigest,
		Teams:             teamDigest,
	})
	if err != nil {
		return managementauthorization.Snapshot{}, err
	}
	return managementauthorization.Snapshot{
		Principal: principal, RoleGrants: roleGrants, TeamGrants: teamGrants,
		AuthorityDigest: digest,
	}, nil
}

func loadPrincipal(
	ctx context.Context,
	transaction *sql.Tx,
	principalID accesscontrol.ManagementPrincipalID,
) (accesscontrol.ManagementPrincipal, accesscontrol.Revision, error) {
	var principal accesscontrol.ManagementPrincipal
	var attributes []byte
	var revision int64
	principal.ID = principalID
	loadPrincipalErr := transaction.QueryRowContext(ctx, principalQuery, principalID).Scan(
		&principal.Issuer,
		&principal.Subject,
		&principal.Status,
		&attributes,
		&revision,
		&principal.CreatedAt,
		&principal.UpdatedAt,
	)
	if errors.Is(loadPrincipalErr, sql.ErrNoRows) {
		return accesscontrol.ManagementPrincipal{}, 0, ErrPrincipalNotFound
	}
	if loadPrincipalErr != nil {
		return accesscontrol.ManagementPrincipal{}, 0, fmt.Errorf("load Management principal: %w", loadPrincipalErr)
	}
	if err := json.Unmarshal(attributes, &principal.Attributes); err != nil {
		return accesscontrol.ManagementPrincipal{}, 0, fmt.Errorf("%w: principal attributes", ErrStateInvalid)
	}
	parsedRevision, loadPrincipalErr := revisionValue(revision)
	if loadPrincipalErr != nil {
		return accesscontrol.ManagementPrincipal{}, 0, loadPrincipalErr
	}
	if err := principal.Validate(); err != nil {
		return accesscontrol.ManagementPrincipal{}, 0, fmt.Errorf("%w: principal: %w", ErrStateInvalid, err)
	}
	return principal, parsedRevision, nil
}

type roleDigestEntry struct {
	BindingID       string   `json:"bindingId"`
	BindingRevision uint64   `json:"bindingRevision"`
	BindingStatus   string   `json:"bindingStatus"`
	ScopeKind       string   `json:"scopeKind"`
	NamespaceID     string   `json:"namespaceId,omitempty"`
	ResourceType    string   `json:"resourceType,omitempty"`
	ResourceID      string   `json:"resourceId,omitempty"`
	Ceiling         []string `json:"ceiling"`
	RoleID          string   `json:"roleId"`
	RoleRevision    uint64   `json:"roleRevision"`
	RoleStatus      string   `json:"roleStatus"`
	RolePermissions []string `json:"rolePermissions"`
}

func loadRoleGrants(
	ctx context.Context,
	transaction *sql.Tx,
	principal accesscontrol.ManagementPrincipal,
	namespaceArgument any,
) (_ []managementauthorization.RoleGrant, _ []roleDigestEntry, returnErr error) {
	rows, loadRoleGrantsErr := transaction.QueryContext(ctx, roleGrantsQuery, principal.ID, namespaceArgument)
	if loadRoleGrantsErr != nil {
		return nil, nil, fmt.Errorf("load Management role grants: %w", loadRoleGrantsErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()

	grants := make([]managementauthorization.RoleGrant, 0)
	digestEntries := make([]roleDigestEntry, 0)
	for rows.Next() {
		var (
			binding                  accesscontrol.ManagementRoleBinding
			role                     accesscontrol.ManagementRole
			scopeKind                string
			bindingNamespace         sql.NullString
			resourceType, resourceID sql.NullString
			ceilingJSON              []byte
			bindingRevision          int64
			roleNamespace            sql.NullString
			permissionsJSON          []byte
			roleRevision             int64
		)
		binding.PrincipalID = principal.ID
		if err := rows.Scan(
			&binding.ID,
			&binding.RoleID,
			&scopeKind,
			&bindingNamespace,
			&resourceType,
			&resourceID,
			&ceilingJSON,
			&binding.Status,
			&bindingRevision,
			&roleNamespace,
			&role.Name,
			&role.DisplayName,
			&permissionsJSON,
			&role.BuiltIn,
			&role.Status,
			&roleRevision,
		); err != nil {
			return nil, nil, fmt.Errorf("scan Management role grant: %w", err)
		}
		role.ID = binding.RoleID
		if roleNamespace.Valid {
			role.NamespaceID = accesscontrol.NamespaceID(roleNamespace.String)
		}
		binding.Revision, loadRoleGrantsErr = revisionValue(bindingRevision)
		if loadRoleGrantsErr != nil {
			return nil, nil, loadRoleGrantsErr
		}
		role.Revision, loadRoleGrantsErr = revisionValue(roleRevision)
		if loadRoleGrantsErr != nil {
			return nil, nil, loadRoleGrantsErr
		}
		binding.DelegationCeiling, loadRoleGrantsErr = permissionSet(ceilingJSON)
		if loadRoleGrantsErr != nil {
			return nil, nil, fmt.Errorf("%w: role-binding delegation ceiling: %w", ErrStateInvalid, loadRoleGrantsErr)
		}
		role.Permissions, loadRoleGrantsErr = permissionSet(permissionsJSON)
		if loadRoleGrantsErr != nil {
			return nil, nil, fmt.Errorf("%w: role permissions: %w", ErrStateInvalid, loadRoleGrantsErr)
		}
		binding.Scope, loadRoleGrantsErr = databaseScope(scopeKind, bindingNamespace, resourceType, resourceID)
		if loadRoleGrantsErr != nil {
			return nil, nil, loadRoleGrantsErr
		}
		if err := accesscontrol.ValidateManagementRoleBindingReferences(binding, principal, role); err != nil {
			return nil, nil, fmt.Errorf("%w: role grant: %w", ErrStateInvalid, err)
		}

		grants = append(grants, managementauthorization.RoleGrant{Binding: binding, Role: role})
		digestEntries = append(digestEntries, roleDigestEntry{
			BindingID: string(binding.ID), BindingRevision: uint64(binding.Revision),
			BindingStatus: string(binding.Status), ScopeKind: string(binding.Scope.Kind),
			NamespaceID:  string(binding.Scope.NamespaceID),
			ResourceType: string(binding.Scope.ResourceType), ResourceID: string(binding.Scope.ResourceID),
			Ceiling: permissionsAsStrings(binding.DelegationCeiling), RoleID: string(role.ID),
			RoleRevision: uint64(role.Revision), RoleStatus: string(role.Status),
			RolePermissions: permissionsAsStrings(role.Permissions),
		})
	}
	if err := rows.Err(); err != nil {
		return nil, nil, fmt.Errorf("iterate Management role grants: %w", err)
	}
	return grants, digestEntries, nil
}

type teamDigestEntry struct {
	UserID                 string `json:"userId"`
	UserLinkRevision       uint64 `json:"userLinkRevision"`
	UserStatus             string `json:"userStatus"`
	SelfServiceRevision    uint64 `json:"selfServiceRevision"`
	AllowTeamKeyDelegation bool   `json:"allowTeamKeyDelegation"`
	AllowMembershipManage  bool   `json:"allowMembershipManage"`
	AllowKeyManage         bool   `json:"allowKeyManage"`
	TeamID                 string `json:"teamId,omitempty"`
	TeamStatus             string `json:"teamStatus,omitempty"`
	MembershipRole         string `json:"membershipRole,omitempty"`
	MembershipStatus       string `json:"membershipStatus,omitempty"`
	MembershipRevision     uint64 `json:"membershipRevision,omitempty"`
}

func loadTeamGrants(
	ctx context.Context,
	transaction *sql.Tx,
	principalID accesscontrol.ManagementPrincipalID,
	namespaceID accesscontrol.NamespaceID,
) (_ []managementauthorization.TeamGrant, _ []teamDigestEntry, returnErr error) {
	if namespaceID == "" {
		return []managementauthorization.TeamGrant{}, []teamDigestEntry{}, nil
	}
	var userID accesscontrol.UserID
	var linkRevision int64
	var userStatus accesscontrol.UserStatus
	loadTeamGrantsErr := transaction.QueryRowContext(
		ctx, principalUserLinkQuery, principalID, namespaceID,
	).Scan(&userID, &linkRevision, &userStatus)
	if errors.Is(loadTeamGrantsErr, sql.ErrNoRows) {
		return []managementauthorization.TeamGrant{}, []teamDigestEntry{}, nil
	}
	if loadTeamGrantsErr != nil {
		return nil, nil, fmt.Errorf("load principal User link: %w", loadTeamGrantsErr)
	}
	parsedLinkRevision, loadTeamGrantsErr := revisionValue(linkRevision)
	if loadTeamGrantsErr != nil {
		return nil, nil, loadTeamGrantsErr
	}
	if !userStatus.Valid() {
		return nil, nil, fmt.Errorf("%w: linked User status", ErrStateInvalid)
	}

	var allowTeamKeyDelegation bool
	var capabilityJSON []byte
	var selfServiceRevision int64
	loadTeamGrantsErr = transaction.QueryRowContext(ctx, selfServicePolicyQuery, namespaceID).Scan(
		&allowTeamKeyDelegation, &capabilityJSON, &selfServiceRevision,
	)
	if errors.Is(loadTeamGrantsErr, sql.ErrNoRows) {
		return nil, nil, fmt.Errorf("%w: namespace SelfServicePolicy is missing", ErrStateInvalid)
	}
	if loadTeamGrantsErr != nil {
		return nil, nil, fmt.Errorf("load namespace SelfServicePolicy: %w", loadTeamGrantsErr)
	}
	parsedSelfServiceRevision, loadTeamGrantsErr := revisionValue(selfServiceRevision)
	if loadTeamGrantsErr != nil {
		return nil, nil, loadTeamGrantsErr
	}
	var capabilities []accesscontrol.TeamAdminCapability
	if err := json.Unmarshal(capabilityJSON, &capabilities); err != nil {
		return nil, nil, fmt.Errorf("%w: Team admin capabilities", ErrStateInvalid)
	}
	options, loadTeamGrantsErr := accesscontrol.TeamEntitlementOptionsFromPolicy(
		allowTeamKeyDelegation, capabilities,
	)
	if loadTeamGrantsErr != nil {
		return nil, nil, fmt.Errorf("%w: %w", ErrStateInvalid, loadTeamGrantsErr)
	}

	baseDigest := teamDigestEntry{
		UserID: string(userID), UserLinkRevision: uint64(parsedLinkRevision),
		UserStatus: string(userStatus), SelfServiceRevision: uint64(parsedSelfServiceRevision),
		AllowTeamKeyDelegation: options.AllowTeamKeyDelegation,
		AllowMembershipManage:  options.AllowAdminMembershipManage,
		AllowKeyManage:         options.AllowAdminKeyManage,
	}
	if userStatus != accesscontrol.UserStatusActive {
		return []managementauthorization.TeamGrant{}, []teamDigestEntry{baseDigest}, nil
	}

	rows, loadTeamGrantsErr := transaction.QueryContext(ctx, teamMembershipsQuery, namespaceID, userID)
	if loadTeamGrantsErr != nil {
		return nil, nil, fmt.Errorf("load Team memberships: %w", loadTeamGrantsErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	grants := make([]managementauthorization.TeamGrant, 0)
	digestEntries := []teamDigestEntry{baseDigest}
	for rows.Next() {
		membership := accesscontrol.TeamMembership{NamespaceID: namespaceID, UserID: userID}
		var membershipRevision int64
		var teamStatus accesscontrol.TeamStatus
		if err := rows.Scan(
			&membership.TeamID, &membership.Role, &membership.Status,
			&membershipRevision, &membership.CreatedAt, &membership.UpdatedAt,
			&teamStatus,
		); err != nil {
			return nil, nil, fmt.Errorf("scan Team membership: %w", err)
		}
		parsedMembershipRevision, err := revisionValue(membershipRevision)
		if err != nil {
			return nil, nil, err
		}
		if err := membership.Validate(); err != nil || !teamStatus.Valid() {
			return nil, nil, fmt.Errorf("%w: Team membership", ErrStateInvalid)
		}
		digestEntry := baseDigest
		digestEntry.TeamID = string(membership.TeamID)
		digestEntry.TeamStatus = string(teamStatus)
		digestEntry.MembershipRole = string(membership.Role)
		digestEntry.MembershipStatus = string(membership.Status)
		digestEntry.MembershipRevision = uint64(parsedMembershipRevision)
		digestEntries = append(digestEntries, digestEntry)
		if teamStatus == accesscontrol.TeamStatusActive {
			grants = append(grants, managementauthorization.TeamGrant{
				Membership: membership,
				Options:    options,
			})
		}
	}
	if err := rows.Err(); err != nil {
		return nil, nil, fmt.Errorf("iterate Team memberships: %w", err)
	}
	return grants, digestEntries, nil
}

func databaseScope(
	kind string,
	namespace, resourceType, resourceID sql.NullString,
) (accesscontrol.Scope, error) {
	switch accesscontrol.ScopeKind(kind) {
	case accesscontrol.ScopeKindCluster:
		return accesscontrol.ClusterScope(), nil
	case accesscontrol.ScopeKindNamespace:
		return accesscontrol.NamespaceScope(accesscontrol.NamespaceID(namespace.String)), nil
	case accesscontrol.ScopeKindTeam:
		return accesscontrol.TeamScope(
			accesscontrol.NamespaceID(namespace.String), accesscontrol.TeamID(resourceID.String),
		), nil
	case accesscontrol.ScopeKindUser:
		return accesscontrol.UserScope(
			accesscontrol.NamespaceID(namespace.String), accesscontrol.UserID(resourceID.String),
		), nil
	case accesscontrol.ScopeKindResource:
		return accesscontrol.ResourceScope(
			accesscontrol.NamespaceID(namespace.String),
			accesscontrol.ScopeResourceType(resourceType.String),
			accesscontrol.ResourceID(resourceID.String),
		), nil
	default:
		return accesscontrol.Scope{}, fmt.Errorf("%w: role-binding scope kind", ErrStateInvalid)
	}
}

func permissionSet(document []byte) (accesscontrol.PermissionSet, error) {
	var values []string
	if err := json.Unmarshal(document, &values); err != nil {
		return accesscontrol.PermissionSet{}, err
	}
	permissions := make([]accesscontrol.Permission, len(values))
	for index, value := range values {
		permissions[index] = accesscontrol.Permission(value)
	}
	return accesscontrol.NewPermissionSet(permissions...)
}

func permissionsAsStrings(set accesscontrol.PermissionSet) []string {
	permissions := set.Permissions()
	values := make([]string, len(permissions))
	for index, permission := range permissions {
		values[index] = string(permission)
	}
	return values
}

func revisionValue(value int64) (accesscontrol.Revision, error) {
	if value <= 0 {
		return 0, fmt.Errorf("%w: non-positive revision", ErrStateInvalid)
	}
	return accesscontrol.Revision(value), nil
}

type authorizationDigestInput struct {
	PrincipalID       string            `json:"principalId"`
	PrincipalStatus   string            `json:"principalStatus"`
	PrincipalRevision uint64            `json:"principalRevision"`
	NamespaceID       string            `json:"namespaceId,omitempty"`
	Roles             []roleDigestEntry `json:"roles"`
	Teams             []teamDigestEntry `json:"teams"`
}

func authorizationDigest(input authorizationDigestInput) (string, error) {
	document, err := json.Marshal(input)
	if err != nil {
		return "", fmt.Errorf("encode Management authority digest: %w", err)
	}
	digest := sha256.Sum256(document)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}
