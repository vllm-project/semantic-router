package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

type selfRoleGrant struct {
	binding managementidentity.RoleBinding
	role    managementidentity.Role
}

func (store *Store) LoadSelf(
	ctx context.Context,
	principalID string,
	sessionID string,
) (_ managementidentity.SelfView, resultErr error) {
	if !canonicalUUID(principalID) || !canonicalUUID(sessionID) {
		return managementidentity.SelfView{}, managementidentity.ErrNotFound
	}
	tx, err := store.database.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead, ReadOnly: true})
	if err != nil {
		return managementidentity.SelfView{}, fmt.Errorf("begin Management self view: %w", err)
	}
	defer func() {
		if resultErr != nil {
			_ = tx.Rollback()
		}
	}()
	principal, err := scanPrincipal(tx.QueryRowContext(ctx,
		`SELECT `+principalColumns+` FROM management_principals WHERE id=$1`, principalID))
	if errors.Is(err, sql.ErrNoRows) {
		return managementidentity.SelfView{}, managementidentity.ErrNotFound
	}
	if err != nil {
		return managementidentity.SelfView{}, fmt.Errorf("load Management self principal: %w", err)
	}
	session, err := scanManagementSession(tx.QueryRowContext(ctx, `SELECT `+managementSessionListColumns+`
FROM management_sessions WHERE id=$1 AND principal_id=$2`, sessionID, principalID))
	if errors.Is(err, sql.ErrNoRows) {
		return managementidentity.SelfView{}, managementidentity.ErrNotFound
	}
	if err != nil {
		return managementidentity.SelfView{}, err
	}
	if session.Status != managementauth.SessionActive {
		return managementidentity.SelfView{}, managementauth.ErrSessionInactive
	}
	grants, err := loadSelfRoleGrants(ctx, tx, principalID)
	if err != nil {
		return managementidentity.SelfView{}, err
	}
	clusterPermissions := permissionUnion(grants, "")
	bindingsByNamespace := make(map[string][]selfRoleGrant)
	linkedNamespaces, err := loadLinkedNamespaceIDs(ctx, tx, principalID)
	if err != nil {
		return managementidentity.SelfView{}, err
	}
	for _, grant := range grants {
		if namespaceID := string(grant.binding.Binding.Scope.NamespaceID); namespaceID != "" {
			bindingsByNamespace[namespaceID] = append(bindingsByNamespace[namespaceID], grant)
		}
	}
	includeAllNamespaces := clusterGrantReachesNamespaces(grants)
	namespaces, err := loadSelfNamespaces(ctx, tx, principalID, grants, bindingsByNamespace, linkedNamespaces, includeAllNamespaces)
	if err != nil {
		return managementidentity.SelfView{}, err
	}
	if err := tx.Commit(); err != nil {
		return managementidentity.SelfView{}, fmt.Errorf("commit Management self view: %w", err)
	}
	return managementidentity.SelfView{
		Principal: principal, Session: session,
		ClusterPermissions: clusterPermissions, Namespaces: namespaces,
	}, nil
}

func loadSelfRoleGrants(ctx context.Context, tx *sql.Tx, principalID string) ([]selfRoleGrant, error) {
	rows, err := tx.QueryContext(ctx, `SELECT binding.id::text,binding.role_id::text
FROM management_role_bindings binding
JOIN management_roles role ON role.id=binding.role_id
WHERE binding.principal_id=$1
ORDER BY binding.id LIMIT 201`, principalID)
	if err != nil {
		return nil, fmt.Errorf("list Management self role bindings: %w", err)
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
		return nil, errors.New("management principal has too many role bindings for /me")
	}
	grants := make([]selfRoleGrant, 0, len(ids))
	for _, id := range ids {
		binding, err := scanRoleBinding(tx.QueryRowContext(ctx,
			`SELECT `+bindingColumns+` FROM management_role_bindings WHERE id=$1`, id.bindingID))
		if err != nil {
			return nil, err
		}
		role, err := scanRole(tx.QueryRowContext(ctx,
			`SELECT `+roleColumns+` FROM management_roles WHERE id=$1`, id.roleID))
		if err != nil {
			return nil, err
		}
		grants = append(grants, selfRoleGrant{binding: binding, role: role})
	}
	return grants, nil
}

func loadLinkedNamespaceIDs(ctx context.Context, tx *sql.Tx, principalID string) (map[string]bool, error) {
	rows, err := tx.QueryContext(ctx, `SELECT namespace_id::text
FROM management_principal_user_links WHERE principal_id=$1 ORDER BY namespace_id`, principalID)
	if err != nil {
		return nil, fmt.Errorf("list Management self User links: %w", err)
	}
	defer rows.Close()
	result := make(map[string]bool)
	for rows.Next() {
		var namespaceID string
		if err := rows.Scan(&namespaceID); err != nil {
			return nil, err
		}
		if !canonicalUUID(namespaceID) {
			return nil, errors.New("stored Management User link namespace is invalid")
		}
		result[namespaceID] = true
	}
	return result, rows.Err()
}

func loadSelfNamespaces(
	ctx context.Context,
	tx *sql.Tx,
	principalID string,
	grants []selfRoleGrant,
	bindingsByNamespace map[string][]selfRoleGrant,
	linkedNamespaces map[string]bool,
	includeAll bool,
) ([]managementidentity.SelfNamespace, error) {
	rows, err := tx.QueryContext(ctx, `SELECT id::text,name,status
FROM access_namespaces ORDER BY id`)
	if err != nil {
		return nil, fmt.Errorf("list Management self namespaces: %w", err)
	}
	type namespaceRow struct{ id, name, status string }
	values := make([]namespaceRow, 0)
	for rows.Next() {
		var value namespaceRow
		if err := rows.Scan(&value.id, &value.name, &value.status); err != nil {
			rows.Close()
			return nil, err
		}
		if includeAll || linkedNamespaces[value.id] || len(bindingsByNamespace[value.id]) != 0 {
			values = append(values, value)
		}
	}
	if err := rows.Close(); err != nil {
		return nil, err
	}
	result := make([]managementidentity.SelfNamespace, 0, len(values))
	for _, value := range values {
		namespace, options, err := loadSelfNamespace(ctx, tx, principalID, value.id, value.name, value.status, grants)
		if err != nil {
			return nil, err
		}
		if namespace.User != nil {
			teamPermissions, err := loadSelfTeams(ctx, tx, namespace.User.ID, &namespace, options)
			if err != nil {
				return nil, err
			}
			namespace.Permissions = mergePermissions(namespace.Permissions, teamPermissions)
		}
		result = append(result, namespace)
	}
	return result, nil
}

func loadSelfNamespace(
	ctx context.Context,
	tx *sql.Tx,
	principalID string,
	namespaceID string,
	name string,
	status string,
	grants []selfRoleGrant,
) (managementidentity.SelfNamespace, accesscontrol.TeamEntitlementOptions, error) {
	var desired, applied int64
	if err := tx.QueryRowContext(ctx, `SELECT
  COALESCE((SELECT max(revision) FROM policy_revisions WHERE namespace_id=$1),0),
  COALESCE((SELECT min(applied_revision) FROM projector_watermarks WHERE namespace_id=$1),0)`, namespaceID).Scan(&desired, &applied); err != nil {
		return managementidentity.SelfNamespace{}, accesscontrol.TeamEntitlementOptions{}, err
	}
	if desired < 0 || applied < 0 || applied > desired {
		return managementidentity.SelfNamespace{}, accesscontrol.TeamEntitlementOptions{}, errors.New("stored Management namespace revisions are invalid")
	}
	var policy managementidentity.SelfServicePolicy
	var capabilitiesJSON []byte
	if err := tx.QueryRowContext(ctx, `SELECT max_keys_per_user,max_delegated_sessions,
delegated_session_ttl_seconds,allow_team_key_delegation,automatic_first_key,
team_admin_capabilities,revision FROM self_service_policies WHERE namespace_id=$1`, namespaceID).Scan(
		&policy.MaxKeysPerUser, &policy.MaxDelegatedSessions,
		&policy.DelegatedSessionTTLSeconds, &policy.AllowTeamKeyDelegation,
		&policy.AutomaticFirstKey, &capabilitiesJSON, &policy.Revision,
	); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return managementidentity.SelfNamespace{}, accesscontrol.TeamEntitlementOptions{}, errors.New("management self-service policy is missing")
		}
		return managementidentity.SelfNamespace{}, accesscontrol.TeamEntitlementOptions{}, err
	}
	var rawCapabilities []string
	if err := json.Unmarshal(capabilitiesJSON, &rawCapabilities); err != nil {
		return managementidentity.SelfNamespace{}, accesscontrol.TeamEntitlementOptions{}, errors.New("stored Team admin capabilities are invalid")
	}
	capabilities := make([]accesscontrol.TeamAdminCapability, len(rawCapabilities))
	for index := range rawCapabilities {
		capabilities[index] = accesscontrol.TeamAdminCapability(rawCapabilities[index])
	}
	options, err := accesscontrol.TeamEntitlementOptionsFromPolicy(policy.AllowTeamKeyDelegation, capabilities)
	if err != nil {
		return managementidentity.SelfNamespace{}, accesscontrol.TeamEntitlementOptions{}, err
	}
	relevant := relevantRoleGrants(grants, namespaceID)
	bindings := make([]managementidentity.RoleBinding, len(relevant))
	for index := range relevant {
		bindings[index] = relevant[index].binding
	}
	namespace := managementidentity.SelfNamespace{
		ID: namespaceID, Name: name, Status: status,
		DesiredRevision: uint64(desired), AppliedRevision: uint64(applied),
		Permissions: permissionUnion(grants, namespaceID), RoleBindings: bindings,
		SelfServicePolicy: policy,
	}
	var user managementidentity.SelfUser
	err = tx.QueryRowContext(ctx, `SELECT linked_user.id::text,linked_user.email,linked_user.display_name,linked_user.status
FROM management_principal_user_links link
JOIN access_users linked_user ON linked_user.namespace_id=link.namespace_id AND linked_user.id=link.user_id
WHERE link.principal_id=$1 AND link.namespace_id=$2`, principalID, namespaceID).Scan(
		&user.ID, &user.Email, &user.DisplayName, &user.Status,
	)
	if err == nil {
		namespace.User = &user
	} else if !errors.Is(err, sql.ErrNoRows) {
		return managementidentity.SelfNamespace{}, accesscontrol.TeamEntitlementOptions{}, err
	}
	return namespace, options, nil
}

func loadSelfTeams(
	ctx context.Context,
	tx *sql.Tx,
	userID string,
	namespace *managementidentity.SelfNamespace,
	options accesscontrol.TeamEntitlementOptions,
) ([]string, error) {
	rows, err := tx.QueryContext(ctx, `SELECT membership.team_id::text,team.name,
membership.role,membership.status
FROM access_team_memberships membership
JOIN access_teams team ON team.namespace_id=membership.namespace_id AND team.id=membership.team_id
WHERE membership.namespace_id=$1 AND membership.user_id=$2
ORDER BY membership.team_id`, namespace.ID, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	permissions := make([]string, 0)
	for rows.Next() {
		var membership managementidentity.SelfTeamMembership
		if err := rows.Scan(&membership.TeamID, &membership.Name, &membership.Role, &membership.Status); err != nil {
			return nil, err
		}
		namespace.Teams = append(namespace.Teams, membership)
		if membership.Status != string(accesscontrol.MembershipStatusActive) {
			continue
		}
		set, err := accesscontrol.TeamRolePermissions(accesscontrol.TeamRole(membership.Role), options)
		if err != nil {
			return nil, err
		}
		for _, permission := range set.Permissions() {
			permissions = append(permissions, string(permission))
		}
	}
	return permissions, rows.Err()
}

func relevantRoleGrants(grants []selfRoleGrant, namespaceID string) []selfRoleGrant {
	result := make([]selfRoleGrant, 0)
	for _, grant := range grants {
		scope := grant.binding.Binding.Scope
		if scope.Kind == accesscontrol.ScopeKindCluster || string(scope.NamespaceID) == namespaceID {
			result = append(result, grant)
		}
	}
	return result
}

func permissionUnion(grants []selfRoleGrant, namespaceID string) []string {
	values := make([]string, 0)
	for _, grant := range relevantRoleGrants(grants, namespaceID) {
		if grant.binding.Binding.Status != accesscontrol.BindingStatusActive ||
			grant.role.Role.Status != accesscontrol.RoleStatusActive {
			continue
		}
		for _, permission := range grant.role.Role.Permissions.Permissions() {
			values = append(values, string(permission))
		}
	}
	return mergePermissions(values, nil)
}

func mergePermissions(left, right []string) []string {
	set := make(map[string]struct{}, len(left)+len(right))
	for _, values := range [][]string{left, right} {
		for _, value := range values {
			set[value] = struct{}{}
		}
	}
	result := make([]string, 0, len(set))
	for value := range set {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

func clusterGrantReachesNamespaces(grants []selfRoleGrant) bool {
	for _, grant := range grants {
		if grant.binding.Binding.Scope.Kind != accesscontrol.ScopeKindCluster ||
			grant.binding.Binding.Status != accesscontrol.BindingStatusActive ||
			grant.role.Role.Status != accesscontrol.RoleStatusActive {
			continue
		}
		for _, permission := range grant.role.Role.Permissions.Permissions() {
			value := string(permission)
			if !strings.HasPrefix(value, "cluster.") && !strings.HasPrefix(value, "identity_issuer.") &&
				!strings.HasPrefix(value, "principal.") {
				return true
			}
		}
	}
	return false
}
