package accesscontrol

import "sort"

type Permission string

const (
	PermissionSelfRead                    Permission = "self.read"
	PermissionSelfManage                  Permission = "self.manage"
	PermissionClusterRead                 Permission = "cluster.read"
	PermissionClusterManage               Permission = "cluster.manage"
	PermissionNamespaceRead               Permission = "namespace.read"
	PermissionNamespaceManage             Permission = "namespace.manage"
	PermissionIdentityIssuerRead          Permission = "identity_issuer.read"
	PermissionIdentityIssuerManage        Permission = "identity_issuer.manage"
	PermissionPrincipalRead               Permission = "principal.read"
	PermissionPrincipalManage             Permission = "principal.manage"
	PermissionPrincipalDirectoryRead      Permission = "principal_directory.read"
	PermissionPrincipalLinkRead           Permission = "principal_link.read"
	PermissionPrincipalLinkManage         Permission = "principal_link.manage"
	PermissionManagementRoleRead          Permission = "management_role.read"
	PermissionManagementRoleManage        Permission = "management_role.manage"
	PermissionRoleBindingRead             Permission = "role_binding.read"
	PermissionRoleBindingManage           Permission = "role_binding.manage"
	PermissionServiceAccountRead          Permission = "service_account.read"
	PermissionServiceAccountManage        Permission = "service_account.manage"
	PermissionInvitationRead              Permission = "invitation.read"
	PermissionInvitationManage            Permission = "invitation.manage"
	PermissionOnboardingManage            Permission = "onboarding.manage"
	PermissionUserRead                    Permission = "user.read"
	PermissionUserManage                  Permission = "user.manage"
	PermissionTeamRead                    Permission = "team.read"
	PermissionTeamManage                  Permission = "team.manage"
	PermissionMembershipManage            Permission = "membership.manage"
	PermissionKeyRead                     Permission = "key.read"
	PermissionKeyManage                   Permission = "key.manage"
	PermissionKeyReveal                   Permission = "key.reveal"
	PermissionDelegationUse               Permission = "delegation.use"
	PermissionDelegationManage            Permission = "delegation.manage"
	PermissionAccessPolicyRead            Permission = "access_policy.read"
	PermissionAccessPolicyManage          Permission = "access_policy.manage"
	PermissionRatePolicyRead              Permission = "rate_policy.read"
	PermissionRatePolicyManage            Permission = "rate_policy.manage"
	PermissionRoutingContextRead          Permission = "routing_context.read"
	PermissionRoutingContextManage        Permission = "routing_context.manage"
	PermissionRoutingRead                 Permission = "routing.read"
	PermissionRoutingManage               Permission = "routing.manage"
	PermissionRoutingPublish              Permission = "routing.publish"
	PermissionEvaluationRun               Permission = "evaluation.run"
	PermissionAgentRead                   Permission = "agent.read"
	PermissionAgentUse                    Permission = "agent.use"
	PermissionAgentManage                 Permission = "agent.manage"
	PermissionToolRead                    Permission = "tool.read"
	PermissionToolInvoke                  Permission = "tool.invoke"
	PermissionToolManage                  Permission = "tool.manage"
	PermissionProviderCatalogRead         Permission = "provider_catalog.read"
	PermissionProviderCredentialRead      Permission = "provider_credential.read"
	PermissionProviderCredentialManage    Permission = "provider_credential.manage"
	PermissionProviderCredentialUse       Permission = "provider_credential.use"
	PermissionQuotaRead                   Permission = "quota.read"
	PermissionQuotaReconcile              Permission = "quota.reconcile"
	PermissionUsageRead                   Permission = "usage.read"
	PermissionUsageInternalDimensionsRead Permission = "usage.internal_dimensions.read"
	PermissionLogRead                     Permission = "log.read"
	PermissionLogPayloadRead              Permission = "log_payload.read"
	PermissionAuditRead                   Permission = "audit.read"
	PermissionOperationRead               Permission = "operation.read"
	PermissionOperationManage             Permission = "operation.manage"
	PermissionHealthRead                  Permission = "health.read"
)

var registeredPermissions = []Permission{
	PermissionSelfRead,
	PermissionSelfManage,
	PermissionClusterRead,
	PermissionClusterManage,
	PermissionNamespaceRead,
	PermissionNamespaceManage,
	PermissionIdentityIssuerRead,
	PermissionIdentityIssuerManage,
	PermissionPrincipalRead,
	PermissionPrincipalManage,
	PermissionPrincipalDirectoryRead,
	PermissionPrincipalLinkRead,
	PermissionPrincipalLinkManage,
	PermissionManagementRoleRead,
	PermissionManagementRoleManage,
	PermissionRoleBindingRead,
	PermissionRoleBindingManage,
	PermissionServiceAccountRead,
	PermissionServiceAccountManage,
	PermissionInvitationRead,
	PermissionInvitationManage,
	PermissionOnboardingManage,
	PermissionUserRead,
	PermissionUserManage,
	PermissionTeamRead,
	PermissionTeamManage,
	PermissionMembershipManage,
	PermissionKeyRead,
	PermissionKeyManage,
	PermissionKeyReveal,
	PermissionDelegationUse,
	PermissionDelegationManage,
	PermissionAccessPolicyRead,
	PermissionAccessPolicyManage,
	PermissionRatePolicyRead,
	PermissionRatePolicyManage,
	PermissionRoutingContextRead,
	PermissionRoutingContextManage,
	PermissionRoutingRead,
	PermissionRoutingManage,
	PermissionRoutingPublish,
	PermissionEvaluationRun,
	PermissionAgentRead,
	PermissionAgentUse,
	PermissionAgentManage,
	PermissionToolRead,
	PermissionToolInvoke,
	PermissionToolManage,
	PermissionProviderCatalogRead,
	PermissionProviderCredentialRead,
	PermissionProviderCredentialManage,
	PermissionProviderCredentialUse,
	PermissionQuotaRead,
	PermissionQuotaReconcile,
	PermissionUsageRead,
	PermissionUsageInternalDimensionsRead,
	PermissionLogRead,
	PermissionLogPayloadRead,
	PermissionAuditRead,
	PermissionOperationRead,
	PermissionOperationManage,
	PermissionHealthRead,
}

var permissionRegistry = func() map[Permission]struct{} {
	registry := make(map[Permission]struct{}, len(registeredPermissions))
	for _, permission := range registeredPermissions {
		registry[permission] = struct{}{}
	}
	return registry
}()

func (p Permission) Valid() bool {
	_, exists := permissionRegistry[p]
	return exists
}

func (p Permission) Intrinsic() bool {
	return p == PermissionSelfRead || p == PermissionSelfManage
}

// RegisteredPermissions returns a deterministic copy of the authoritative
// permission registry.
func RegisteredPermissions() []Permission {
	permissions := append([]Permission(nil), registeredPermissions...)
	sort.Slice(permissions, func(i, j int) bool { return permissions[i] < permissions[j] })
	return permissions
}

// PermissionSet keeps the map private so callers cannot mutate built-in role
// definitions returned by this package.
type PermissionSet struct {
	values map[Permission]struct{}
}

func NewPermissionSet(permissions ...Permission) (PermissionSet, error) {
	set := PermissionSet{values: make(map[Permission]struct{}, len(permissions))}
	for _, permission := range permissions {
		if !permission.Valid() {
			return PermissionSet{}, invalid("permission", "is not registered: "+string(permission))
		}
		set.values[permission] = struct{}{}
	}
	return set, nil
}

func mustPermissionSet(permissions ...Permission) PermissionSet {
	set, err := NewPermissionSet(permissions...)
	if err != nil {
		panic(err)
	}
	return set
}

func (s PermissionSet) Validate() error {
	for permission := range s.values {
		if !permission.Valid() {
			return invalid("permission", "is not registered: "+string(permission))
		}
	}
	return nil
}

func (s PermissionSet) ValidateDelegable() error {
	if err := s.Validate(); err != nil {
		return err
	}
	for permission := range s.values {
		if permission.Intrinsic() {
			return invalid("delegation_ceiling", "intrinsic self permissions are not delegable")
		}
	}
	return nil
}

func (s PermissionSet) Contains(permission Permission) bool {
	_, exists := s.values[permission]
	return exists
}

func (s PermissionSet) ContainsAll(other PermissionSet) bool {
	for permission := range other.values {
		if !s.Contains(permission) {
			return false
		}
	}
	return true
}

func (s PermissionSet) Equal(other PermissionSet) bool {
	return len(s.values) == len(other.values) && s.ContainsAll(other)
}

func (s PermissionSet) Empty() bool { return len(s.values) == 0 }

func (s PermissionSet) Permissions() []Permission {
	permissions := make([]Permission, 0, len(s.values))
	for permission := range s.values {
		permissions = append(permissions, permission)
	}
	sort.Slice(permissions, func(i, j int) bool { return permissions[i] < permissions[j] })
	return permissions
}

func (s PermissionSet) Clone() PermissionSet {
	clone := PermissionSet{values: make(map[Permission]struct{}, len(s.values))}
	for permission := range s.values {
		clone.values[permission] = struct{}{}
	}
	return clone
}

// DelegablePermissions is the complete registered permission set excluding
// intrinsic self entitlements, which can never be put in a role or ceiling.
func DelegablePermissions() PermissionSet {
	permissions := make([]Permission, 0, len(registeredPermissions))
	for _, permission := range registeredPermissions {
		if !permission.Intrinsic() {
			permissions = append(permissions, permission)
		}
	}
	return mustPermissionSet(permissions...)
}
