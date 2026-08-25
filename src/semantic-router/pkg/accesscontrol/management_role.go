package accesscontrol

type BuiltInRoleName string

const (
	BuiltInRoleClusterAdmin  BuiltInRoleName = "cluster_admin"
	BuiltInRolePlatformAdmin BuiltInRoleName = "platform_admin"
	BuiltInRoleOperator      BuiltInRoleName = "operator"
	BuiltInRoleAccessAdmin   BuiltInRoleName = "access_admin"
	// #nosec G101 -- this is an authorization role identifier, not a credential value.
	BuiltInRoleCredentialRevealer BuiltInRoleName = "credential_revealer"
	BuiltInRoleAnalyst            BuiltInRoleName = "analyst"
	BuiltInRoleViewer             BuiltInRoleName = "viewer"
	BuiltInRoleConsumer           BuiltInRoleName = "consumer"
)

var builtInRolePermissions = map[BuiltInRoleName]PermissionSet{
	BuiltInRoleClusterAdmin: mustPermissionSet(
		PermissionClusterRead, PermissionClusterManage,
		PermissionNamespaceRead, PermissionNamespaceManage,
		PermissionIdentityIssuerRead, PermissionIdentityIssuerManage,
		PermissionPrincipalRead, PermissionPrincipalManage,
		PermissionManagementRoleRead, PermissionManagementRoleManage,
		PermissionRoleBindingRead, PermissionRoleBindingManage,
		PermissionServiceAccountRead, PermissionServiceAccountManage,
		PermissionAuditRead, PermissionHealthRead,
	),
	BuiltInRolePlatformAdmin: mustPermissionSet(
		PermissionNamespaceRead, PermissionNamespaceManage,
		PermissionPrincipalDirectoryRead, PermissionPrincipalLinkRead, PermissionPrincipalLinkManage,
		PermissionManagementRoleRead, PermissionManagementRoleManage,
		PermissionRoleBindingRead, PermissionRoleBindingManage,
		PermissionServiceAccountRead, PermissionServiceAccountManage,
		PermissionInvitationRead, PermissionInvitationManage, PermissionOnboardingManage,
		PermissionUserRead, PermissionUserManage,
		PermissionTeamRead, PermissionTeamManage, PermissionMembershipManage,
		PermissionKeyRead, PermissionKeyManage,
		PermissionDelegationUse, PermissionDelegationManage,
		PermissionAccessPolicyRead, PermissionAccessPolicyManage,
		PermissionRatePolicyRead, PermissionRatePolicyManage,
		PermissionRoutingContextRead, PermissionRoutingContextManage,
		PermissionRoutingRead, PermissionRoutingManage, PermissionRoutingPublish, PermissionEvaluationRun,
		PermissionAgentRead, PermissionAgentUse, PermissionAgentManage,
		PermissionToolRead, PermissionToolInvoke, PermissionToolManage,
		PermissionProviderCatalogRead,
		PermissionProviderCredentialRead, PermissionProviderCredentialManage, PermissionProviderCredentialUse,
		PermissionQuotaRead, PermissionQuotaReconcile,
		PermissionUsageRead, PermissionUsageInternalDimensionsRead,
		PermissionLogRead, PermissionLogPayloadRead, PermissionAuditRead,
		PermissionOperationRead, PermissionOperationManage, PermissionHealthRead,
	),
	BuiltInRoleOperator: mustPermissionSet(
		PermissionNamespaceRead,
		PermissionRoutingRead, PermissionRoutingManage, PermissionRoutingPublish, PermissionEvaluationRun,
		PermissionAgentRead, PermissionAgentUse, PermissionAgentManage,
		PermissionToolRead, PermissionToolInvoke, PermissionToolManage,
		PermissionProviderCatalogRead,
		PermissionProviderCredentialRead, PermissionProviderCredentialManage, PermissionProviderCredentialUse,
		PermissionAccessPolicyRead, PermissionRatePolicyRead,
		PermissionRoutingContextRead, PermissionRoutingContextManage,
		PermissionQuotaRead, PermissionUsageRead, PermissionUsageInternalDimensionsRead,
		PermissionLogRead, PermissionLogPayloadRead,
		PermissionOperationRead, PermissionOperationManage, PermissionHealthRead,
	),
	BuiltInRoleAccessAdmin: mustPermissionSet(
		PermissionNamespaceRead,
		PermissionPrincipalDirectoryRead, PermissionPrincipalLinkRead, PermissionPrincipalLinkManage,
		PermissionInvitationRead, PermissionInvitationManage, PermissionOnboardingManage,
		PermissionUserRead, PermissionUserManage,
		PermissionTeamRead, PermissionTeamManage, PermissionMembershipManage,
		PermissionKeyRead, PermissionKeyManage,
		PermissionDelegationUse, PermissionDelegationManage,
		PermissionAccessPolicyRead, PermissionAccessPolicyManage,
		PermissionRatePolicyRead, PermissionRatePolicyManage,
		PermissionRoutingContextRead, PermissionRoutingContextManage,
		PermissionRoutingRead,
		PermissionQuotaRead, PermissionQuotaReconcile,
		PermissionUsageRead, PermissionAuditRead,
		PermissionOperationRead, PermissionOperationManage,
	),
	BuiltInRoleCredentialRevealer: mustPermissionSet(PermissionKeyRead, PermissionKeyReveal),
	BuiltInRoleAnalyst: mustPermissionSet(
		PermissionNamespaceRead, PermissionUserRead, PermissionTeamRead, PermissionKeyRead,
		PermissionQuotaRead, PermissionUsageRead, PermissionLogRead, PermissionAuditRead,
	),
	BuiltInRoleViewer: mustPermissionSet(
		PermissionRoutingRead, PermissionProviderCatalogRead,
		PermissionAgentRead, PermissionToolRead,
	),
	BuiltInRoleConsumer: mustPermissionSet(
		PermissionUserRead, PermissionTeamRead, PermissionKeyRead,
		PermissionDelegationUse, PermissionAccessPolicyRead, PermissionRatePolicyRead,
		PermissionRoutingContextRead, PermissionQuotaRead, PermissionUsageRead, PermissionOperationRead,
		PermissionAgentRead, PermissionAgentUse, PermissionToolRead, PermissionToolInvoke,
	),
}

func (n BuiltInRoleName) Valid() bool {
	_, exists := builtInRolePermissions[n]
	return exists
}

type ManagementRole struct {
	ID          ManagementRoleID
	NamespaceID NamespaceID
	Name        string
	DisplayName string
	BuiltIn     bool
	Permissions PermissionSet
	Status      RoleStatus
	Revision    Revision
}

func (r ManagementRole) Validate() error {
	var statusErr, permissionErr, ownershipErr error
	if !r.Status.Valid() {
		statusErr = invalid("status", "is not a valid role status")
	}
	if err := r.Permissions.ValidateDelegable(); err != nil {
		permissionErr = err
	} else if r.Permissions.Empty() {
		permissionErr = invalid("permissions", "must not be empty")
	}
	if r.BuiltIn {
		roleName := BuiltInRoleName(r.Name)
		expected, exists := builtInRolePermissions[roleName]
		switch {
		case !exists:
			ownershipErr = invalid("name", "is not a registered built-in role")
		case r.NamespaceID != "":
			ownershipErr = invalid("namespace_id", "must be empty for a built-in role")
		case !r.Permissions.Equal(expected):
			ownershipErr = invalid("permissions", "must exactly match the immutable built-in permission set")
		case r.Status != RoleStatusActive:
			ownershipErr = invalid("status", "built-in roles must remain active")
		}
	} else {
		if r.NamespaceID == "" {
			ownershipErr = invalid("namespace_id", "is required for a custom role")
		} else if BuiltInRoleName(r.Name).Valid() {
			ownershipErr = invalid("name", "is reserved for a built-in role")
		}
	}
	return joinValidation(
		validateRequired("id", string(r.ID)),
		validateRequired("name", r.Name),
		validateRequired("display_name", r.DisplayName),
		permissionErr,
		statusErr,
		ownershipErr,
		validateRevision(r.Revision),
	)
}

// BuiltInRole returns an immutable preset by value. Its PermissionSet is cloned
// so callers cannot mutate the registry.
func BuiltInRole(name BuiltInRoleName) (ManagementRole, bool) {
	permissions, exists := builtInRolePermissions[name]
	if !exists {
		return ManagementRole{}, false
	}
	return ManagementRole{
		ID:          ManagementRoleID("builtin:" + string(name)),
		Name:        string(name),
		DisplayName: string(name),
		BuiltIn:     true,
		Permissions: permissions.Clone(),
		Status:      RoleStatusActive,
		Revision:    1,
	}, true
}

type TeamEntitlementOptions struct {
	AllowTeamKeyDelegation     bool
	AllowAdminMembershipManage bool
	AllowAdminKeyManage        bool
}

// TeamAdminCapability is the durable, closed vocabulary stored by a
// Namespace SelfServicePolicy. Keeping this conversion in the domain avoids
// authorization loaders interpreting free-form JSON differently.
type TeamAdminCapability string

const (
	TeamAdminCapabilityMembershipManage TeamAdminCapability = "membership.manage"
	TeamAdminCapabilityKeyManage        TeamAdminCapability = "key.manage"
)

func TeamEntitlementOptionsFromPolicy(
	allowTeamKeyDelegation bool,
	capabilities []TeamAdminCapability,
) (TeamEntitlementOptions, error) {
	options := TeamEntitlementOptions{AllowTeamKeyDelegation: allowTeamKeyDelegation}
	seen := make(map[TeamAdminCapability]struct{}, len(capabilities))
	for _, capability := range capabilities {
		if _, duplicate := seen[capability]; duplicate {
			return TeamEntitlementOptions{}, invalid("team_admin_capabilities", "must not contain duplicates")
		}
		seen[capability] = struct{}{}
		switch capability {
		case TeamAdminCapabilityMembershipManage:
			options.AllowAdminMembershipManage = true
		case TeamAdminCapabilityKeyManage:
			options.AllowAdminKeyManage = true
		default:
			return TeamEntitlementOptions{}, invalid("team_admin_capabilities", "contains an unknown capability")
		}
	}
	return options, nil
}

// TeamRolePermissions synthesizes fixed, non-delegable Team entitlements. The
// caller must still evaluate them only against the membership's Team scope.
func TeamRolePermissions(role TeamRole, options TeamEntitlementOptions) (PermissionSet, error) {
	if !role.Valid() {
		return PermissionSet{}, invalid("team_role", "is not member or admin")
	}
	permissions := []Permission{
		PermissionTeamRead,
		PermissionAccessPolicyRead,
		PermissionRatePolicyRead,
		PermissionQuotaRead,
		PermissionUsageRead,
	}
	if options.AllowTeamKeyDelegation {
		permissions = append(permissions, PermissionDelegationUse)
	}
	if role == TeamRoleAdmin {
		permissions = append(permissions, PermissionKeyRead)
		if options.AllowAdminMembershipManage {
			permissions = append(permissions, PermissionMembershipManage)
		}
		if options.AllowAdminKeyManage {
			permissions = append(permissions, PermissionKeyManage)
		}
	}
	return NewPermissionSet(permissions...)
}
