package accesscontrol

import (
	"errors"
	"testing"
)

func TestBuiltInRolesHaveExactPermissionSets(t *testing.T) {
	tests := []struct {
		name        BuiltInRoleName
		permissions []Permission
	}{
		{BuiltInRoleClusterAdmin, []Permission{
			PermissionClusterRead, PermissionClusterManage, PermissionNamespaceRead, PermissionNamespaceManage,
			PermissionIdentityIssuerRead, PermissionIdentityIssuerManage, PermissionPrincipalRead, PermissionPrincipalManage,
			PermissionManagementRoleRead, PermissionManagementRoleManage, PermissionRoleBindingRead, PermissionRoleBindingManage,
			PermissionServiceAccountRead, PermissionServiceAccountManage, PermissionAuditRead, PermissionHealthRead,
		}},
		{BuiltInRolePlatformAdmin, []Permission{
			PermissionNamespaceRead, PermissionNamespaceManage, PermissionPrincipalDirectoryRead, PermissionPrincipalLinkRead,
			PermissionPrincipalLinkManage, PermissionManagementRoleRead, PermissionManagementRoleManage,
			PermissionRoleBindingRead, PermissionRoleBindingManage, PermissionServiceAccountRead, PermissionServiceAccountManage,
			PermissionInvitationRead, PermissionInvitationManage, PermissionOnboardingManage, PermissionUserRead, PermissionUserManage,
			PermissionTeamRead, PermissionTeamManage, PermissionMembershipManage, PermissionKeyRead, PermissionKeyManage,
			PermissionDelegationUse, PermissionDelegationManage, PermissionAccessPolicyRead, PermissionAccessPolicyManage,
			PermissionRatePolicyRead, PermissionRatePolicyManage, PermissionRoutingContextRead, PermissionRoutingContextManage,
			PermissionRoutingRead, PermissionRoutingManage, PermissionRoutingPublish, PermissionEvaluationRun,
			PermissionAgentRead, PermissionAgentUse, PermissionAgentManage,
			PermissionToolRead, PermissionToolInvoke, PermissionToolManage,
			PermissionProviderCatalogRead, PermissionProviderCredentialRead,
			PermissionProviderCredentialManage, PermissionProviderCredentialUse, PermissionQuotaRead, PermissionQuotaReconcile,
			PermissionUsageRead, PermissionUsageInternalDimensionsRead, PermissionLogRead, PermissionLogPayloadRead,
			PermissionAuditRead, PermissionOperationRead, PermissionOperationManage, PermissionHealthRead,
		}},
		{BuiltInRoleOperator, []Permission{
			PermissionNamespaceRead, PermissionRoutingRead, PermissionRoutingManage, PermissionRoutingPublish,
			PermissionEvaluationRun, PermissionAgentRead, PermissionAgentUse, PermissionAgentManage,
			PermissionToolRead, PermissionToolInvoke, PermissionToolManage, PermissionProviderCatalogRead,
			PermissionProviderCredentialRead, PermissionProviderCredentialManage, PermissionProviderCredentialUse,
			PermissionAccessPolicyRead, PermissionRatePolicyRead, PermissionRoutingContextRead, PermissionRoutingContextManage,
			PermissionQuotaRead, PermissionUsageRead, PermissionUsageInternalDimensionsRead, PermissionLogRead,
			PermissionLogPayloadRead, PermissionOperationRead, PermissionOperationManage, PermissionHealthRead,
		}},
		{BuiltInRoleAccessAdmin, []Permission{
			PermissionNamespaceRead, PermissionPrincipalDirectoryRead, PermissionPrincipalLinkRead, PermissionPrincipalLinkManage,
			PermissionInvitationRead, PermissionInvitationManage, PermissionOnboardingManage, PermissionUserRead,
			PermissionUserManage, PermissionTeamRead, PermissionTeamManage, PermissionMembershipManage, PermissionKeyRead,
			PermissionKeyManage, PermissionDelegationUse, PermissionDelegationManage, PermissionAccessPolicyRead,
			PermissionAccessPolicyManage, PermissionRatePolicyRead, PermissionRatePolicyManage, PermissionRoutingContextRead,
			PermissionRoutingContextManage, PermissionRoutingRead, PermissionQuotaRead, PermissionQuotaReconcile,
			PermissionUsageRead, PermissionAuditRead, PermissionOperationRead, PermissionOperationManage,
		}},
		{BuiltInRoleCredentialRevealer, []Permission{PermissionKeyRead, PermissionKeyReveal}},
		{BuiltInRoleAnalyst, []Permission{
			PermissionNamespaceRead, PermissionUserRead, PermissionTeamRead, PermissionKeyRead,
			PermissionQuotaRead, PermissionUsageRead, PermissionLogRead, PermissionAuditRead,
		}},
		{BuiltInRoleViewer, []Permission{
			PermissionRoutingRead, PermissionProviderCatalogRead, PermissionAgentRead, PermissionToolRead,
		}},
		{BuiltInRoleConsumer, []Permission{
			PermissionUserRead, PermissionTeamRead, PermissionKeyRead, PermissionKeyReveal,
			PermissionDelegationUse, PermissionAccessPolicyRead, PermissionRatePolicyRead,
			PermissionRoutingContextRead, PermissionQuotaRead, PermissionUsageRead, PermissionOperationRead,
			PermissionAgentRead, PermissionAgentUse, PermissionToolRead, PermissionToolInvoke,
		}},
	}

	for _, test := range tests {
		t.Run(string(test.name), func(t *testing.T) {
			role, ok := BuiltInRole(test.name)
			if !ok {
				t.Fatal("built-in role not found")
			}
			want, err := NewPermissionSet(test.permissions...)
			if err != nil {
				t.Fatalf("invalid test permission: %v", err)
			}
			if !role.Permissions.Equal(want) {
				t.Fatalf("permissions = %v, want %v", role.Permissions.Permissions(), want.Permissions())
			}
			if err := role.Validate(); err != nil {
				t.Fatalf("role validation failed: %v", err)
			}
		})
	}
}

func TestViewerCannotReadNamespaceIdentityOrAccessState(t *testing.T) {
	viewer, _ := BuiltInRole(BuiltInRoleViewer)
	for _, permission := range []Permission{
		PermissionNamespaceRead,
		PermissionHealthRead,
		PermissionUserRead,
		PermissionTeamRead,
		PermissionKeyRead,
		PermissionAccessPolicyRead,
		PermissionRatePolicyRead,
		PermissionQuotaRead,
		PermissionUsageRead,
		PermissionLogRead,
	} {
		if viewer.Permissions.Contains(permission) {
			t.Fatalf("viewer unexpectedly includes %s", permission)
		}
	}
	for _, permission := range []Permission{
		PermissionRoutingRead,
		PermissionProviderCatalogRead,
		PermissionAgentRead,
		PermissionToolRead,
	} {
		if !viewer.Permissions.Contains(permission) {
			t.Fatalf("viewer omits %s", permission)
		}
	}
}

func TestConsumerIsReadOnlyOutsideDelegatedInference(t *testing.T) {
	consumer, _ := BuiltInRole(BuiltInRoleConsumer)
	for _, permission := range []Permission{
		PermissionKeyManage,
		PermissionUserManage,
		PermissionTeamManage,
		PermissionMembershipManage,
		PermissionAccessPolicyManage,
		PermissionRatePolicyManage,
		PermissionRoutingManage,
		PermissionRoutingPublish,
		PermissionEvaluationRun,
		PermissionAgentManage,
		PermissionToolManage,
		PermissionOperationManage,
	} {
		if consumer.Permissions.Contains(permission) {
			t.Fatalf("consumer unexpectedly includes %s", permission)
		}
	}
	for _, permission := range []Permission{
		PermissionDelegationUse,
		PermissionAgentRead,
		PermissionAgentUse,
		PermissionToolRead,
		PermissionToolInvoke,
	} {
		if !consumer.Permissions.Contains(permission) {
			t.Fatalf("consumer omits %s", permission)
		}
	}
}

func TestAccessAdminDoesNotInheritAgentOrToolAuthority(t *testing.T) {
	role, _ := BuiltInRole(BuiltInRoleAccessAdmin)
	for _, permission := range []Permission{
		PermissionAgentRead, PermissionAgentUse, PermissionAgentManage,
		PermissionToolRead, PermissionToolInvoke, PermissionToolManage,
		PermissionRoutingPublish, PermissionEvaluationRun,
	} {
		if role.Permissions.Contains(permission) {
			t.Fatalf("access_admin unexpectedly includes %s", permission)
		}
	}
}

func TestBuiltInPermissionsAreDefensiveCopies(t *testing.T) {
	first, _ := BuiltInRole(BuiltInRoleViewer)
	first.Permissions.values[PermissionKeyReveal] = struct{}{}
	second, _ := BuiltInRole(BuiltInRoleViewer)
	if second.Permissions.Contains(PermissionKeyReveal) {
		t.Fatal("mutating a returned role changed the built-in registry")
	}
}

func TestKeyRevealIsSeparatelyAssigned(t *testing.T) {
	for _, roleName := range []BuiltInRoleName{
		BuiltInRoleClusterAdmin, BuiltInRolePlatformAdmin, BuiltInRoleOperator,
		BuiltInRoleAccessAdmin, BuiltInRoleAnalyst, BuiltInRoleViewer,
	} {
		role, _ := BuiltInRole(roleName)
		if role.Permissions.Contains(PermissionKeyReveal) {
			t.Fatalf("%s unexpectedly includes key.reveal", roleName)
		}
	}
	if !DelegablePermissions().Contains(PermissionKeyReveal) {
		t.Fatal("bootstrap delegation ceiling must be able to delegate key.reveal")
	}
	consumer, _ := BuiltInRole(BuiltInRoleConsumer)
	if !consumer.Permissions.Contains(PermissionKeyReveal) {
		t.Fatal("consumer must be able to reveal credentials within its user-scoped binding")
	}
}

func TestTeamRolePermissions(t *testing.T) {
	member, err := TeamRolePermissions(TeamRoleMember, TeamEntitlementOptions{AllowTeamKeyDelegation: true})
	if err != nil {
		t.Fatal(err)
	}
	if !member.Contains(PermissionDelegationUse) || member.Contains(PermissionKeyRead) {
		t.Fatalf("unexpected member entitlements: %v", member.Permissions())
	}
	admin, err := TeamRolePermissions(TeamRoleAdmin, TeamEntitlementOptions{
		AllowAdminMembershipManage: true,
		AllowAdminKeyManage:        true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !admin.Contains(PermissionKeyRead) || !admin.Contains(PermissionKeyManage) || !admin.Contains(PermissionMembershipManage) {
		t.Fatalf("unexpected admin entitlements: %v", admin.Permissions())
	}
	if admin.Contains(PermissionRoleBindingManage) || admin.Contains(PermissionKeyReveal) {
		t.Fatal("team entitlements must not grant role administration or key reveal")
	}
}

func TestTeamEntitlementOptionsFromPolicyUsesClosedCapabilities(t *testing.T) {
	options, err := TeamEntitlementOptionsFromPolicy(true, []TeamAdminCapability{
		TeamAdminCapabilityMembershipManage,
		TeamAdminCapabilityKeyManage,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !options.AllowTeamKeyDelegation || !options.AllowAdminMembershipManage || !options.AllowAdminKeyManage {
		t.Fatalf("unexpected policy options: %+v", options)
	}

	for name, capabilities := range map[string][]TeamAdminCapability{
		"unknown":   {"role_binding.manage"},
		"duplicate": {TeamAdminCapabilityKeyManage, TeamAdminCapabilityKeyManage},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := TeamEntitlementOptionsFromPolicy(false, capabilities); !errors.Is(err, ErrInvalid) {
				t.Fatalf("expected closed-vocabulary validation error, got %v", err)
			}
		})
	}
}

func TestIntrinsicPermissionsCannotBeDelegated(t *testing.T) {
	set, err := NewPermissionSet(PermissionSelfRead)
	if err != nil {
		t.Fatal(err)
	}
	if err := set.ValidateDelegable(); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected validation error, got %v", err)
	}
}
