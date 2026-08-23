package managementauthorization

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

// ResultScope remains the authorization package's public result-set contract.
// The value lives in accesscontrol so application and repository packages can
// consume an authorized scope without importing the evaluator layer.
type ResultScope = accesscontrol.ResultScope

// ResolveResultScope compiles the exact set of result rows for which the
// principal currently holds permission. Namespace/cluster grants collapse to
// All; narrower grants remain a union of typed subject and resource dimensions.
func (runtime Runtime) ResolveResultScope(
	ctx context.Context,
	principalID accesscontrol.ManagementPrincipalID,
	namespaceID accesscontrol.NamespaceID,
	permission accesscontrol.Permission,
) (ResultScope, error) {
	if runtime.Loader == nil {
		return ResultScope{}, fmt.Errorf("management authorization loader is unavailable")
	}
	if !permission.Valid() || permission.Intrinsic() || namespaceID == "" || principalID == "" {
		return ResultScope{}, ErrInvalidContext
	}
	snapshot, resolveResultScopeErr := runtime.Loader.Load(ctx, principalID, namespaceID)
	if resolveResultScopeErr != nil {
		return ResultScope{}, resolveResultScopeErr
	}
	if snapshot.Principal.ID != principalID || snapshot.AuthorityDigest == "" {
		return ResultScope{}, ErrInvalidContext
	}

	scope := ResultScope{NamespaceID: namespaceID}
	for _, grant := range snapshot.RoleGrants {
		if err := collectRoleResultScope(&scope, snapshot.Principal, grant, namespaceID, permission); err != nil {
			return ResultScope{}, fmt.Errorf("%w: role grant: %w", ErrInvalidContext, err)
		}
		if scope.All {
			canonical, err := scope.Canonical()
			if err != nil {
				return ResultScope{}, fmt.Errorf("%w: normalize result scope: %w", ErrInvalidContext, err)
			}
			return canonical, nil
		}
	}
	for _, grant := range snapshot.TeamGrants {
		if err := grant.Membership.Validate(); err != nil {
			return ResultScope{}, fmt.Errorf("%w: Team grant: %w", ErrInvalidContext, err)
		}
		permissions, err := accesscontrol.TeamRolePermissions(grant.Membership.Role, grant.Options)
		if err != nil {
			return ResultScope{}, fmt.Errorf("%w: Team grant: %w", ErrInvalidContext, err)
		}
		if grant.Membership.NamespaceID == namespaceID &&
			grant.Membership.Status == accesscontrol.MembershipStatusActive &&
			permissions.Contains(permission) {
			scope.TeamIDs = append(scope.TeamIDs, grant.Membership.TeamID)
		}
	}
	canonical, resolveResultScopeErr := scope.Canonical()
	if resolveResultScopeErr != nil {
		return ResultScope{}, fmt.Errorf("%w: normalize result scope: %w", ErrInvalidContext, resolveResultScopeErr)
	}
	if canonical.Empty() {
		return ResultScope{}, ErrDenied
	}
	return canonical, nil
}

func collectRoleResultScope(
	result *ResultScope,
	principal accesscontrol.ManagementPrincipal,
	grant RoleGrant,
	namespaceID accesscontrol.NamespaceID,
	permission accesscontrol.Permission,
) error {
	if err := accesscontrol.ValidateManagementRoleBindingReferences(grant.Binding, principal, grant.Role); err != nil {
		return err
	}
	if grant.Binding.RoleID != grant.Role.ID ||
		grant.Binding.Status != accesscontrol.BindingStatusActive ||
		grant.Role.Status != accesscontrol.RoleStatusActive ||
		!grant.Role.Permissions.Contains(permission) {
		return nil
	}
	scope := grant.Binding.Scope
	switch scope.Kind {
	case accesscontrol.ScopeKindCluster:
		result.All = true
	case accesscontrol.ScopeKindNamespace:
		if scope.NamespaceID == namespaceID {
			result.All = true
		}
	case accesscontrol.ScopeKindTeam:
		if scope.NamespaceID == namespaceID {
			result.TeamIDs = append(result.TeamIDs, scope.TeamID)
		}
	case accesscontrol.ScopeKindUser:
		if scope.NamespaceID == namespaceID {
			result.UserIDs = append(result.UserIDs, scope.UserID)
		}
	case accesscontrol.ScopeKindResource:
		if scope.NamespaceID == namespaceID {
			if scope.ResourceType == accesscontrol.ScopeResourceAPIKey {
				result.APIKeyIDs = append(result.APIKeyIDs, accesscontrol.APIKeyID(scope.ResourceID))
			} else {
				if result.ResourceIDs == nil {
					result.ResourceIDs = make(map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID)
				}
				result.ResourceIDs[scope.ResourceType] = append(result.ResourceIDs[scope.ResourceType], scope.ResourceID)
			}
		}
	}
	return nil
}
