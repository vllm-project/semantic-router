package managementapi_test

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

func TestPolicyBindingRequiresManagePermissionForItsSubjectKind(t *testing.T) {
	namespaceID := accesscontrol.NamespaceID("ns-1")
	subjects := []struct {
		name       string
		condition  string
		permission accesscontrol.Permission
		target     accesscontrol.ScopedTarget
	}{
		{
			name: "user", condition: "user_owner", permission: accesscontrol.PermissionUserManage,
			target: accesscontrol.ScopedTarget{Scope: accesscontrol.UserScope(namespaceID, "user-1")},
		},
		{
			name: "team", condition: "team_owner", permission: accesscontrol.PermissionTeamManage,
			target: accesscontrol.ScopedTarget{Scope: accesscontrol.TeamScope(namespaceID, "team-1")},
		},
		{
			name: "api_key", condition: "key_owner", permission: accesscontrol.PermissionKeyManage,
			target: accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(namespaceID, accesscontrol.ScopeResourceAPIKey, "key-1")},
		},
	}
	operations := []struct {
		name             string
		path             string
		policyPermission accesscontrol.Permission
		policyTarget     accesscontrol.ScopedTarget
	}{
		{
			name: "access", path: managementapi.BasePath + "/access-policy-bindings",
			policyPermission: accesscontrol.PermissionAccessPolicyManage,
			policyTarget:     accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(namespaceID, accesscontrol.ScopeResourceAccessPolicy, "policy-1")},
		},
		{
			name: "rate", path: managementapi.BasePath + "/rate-limit-bindings",
			policyPermission: accesscontrol.PermissionRatePolicyManage,
			policyTarget:     accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(namespaceID, accesscontrol.ScopeResourceRateLimitPolicy, "policy-1")},
		},
	}

	for _, operation := range operations {
		contract, found := managementapi.LookupOperation(managementapi.MethodPOST, operation.path)
		if !found {
			t.Fatalf("operation %s is not registered", operation.path)
		}
		for _, subject := range subjects {
			t.Run(operation.name+"_"+subject.name, func(t *testing.T) {
				context := managementauthorization.EvaluationContext{
					Authenticated: true,
					Targets: map[string][]accesscontrol.ScopedTarget{
						"policy":  {operation.policyTarget},
						"subject": {subject.target},
					},
					Conditions: map[string]bool{subject.condition: true},
				}
				context.RoleGrants = []managementauthorization.RoleGrant{
					managementRoleGrant(t, namespaceID, operation.policyPermission),
				}
				if err := managementauthorization.Evaluate(contract.Permission, context); !errors.Is(err, managementauthorization.ErrDenied) {
					t.Fatalf("policy manage without %s error = %v, want ErrDenied", subject.permission, err)
				}

				context.RoleGrants = []managementauthorization.RoleGrant{
					managementRoleGrant(t, namespaceID, operation.policyPermission, subject.permission),
				}
				if err := managementauthorization.Evaluate(contract.Permission, context); err != nil {
					t.Fatalf("matching subject manage error = %v", err)
				}
			})
		}
	}
}

func managementRoleGrant(
	t *testing.T,
	namespaceID accesscontrol.NamespaceID,
	permissions ...accesscontrol.Permission,
) managementauthorization.RoleGrant {
	t.Helper()
	permissionSet, err := accesscontrol.NewPermissionSet(permissions...)
	if err != nil {
		t.Fatal(err)
	}
	role := accesscontrol.ManagementRole{
		ID: "role-1", NamespaceID: namespaceID, Name: "policy-binding-test",
		DisplayName: "Policy binding test", Permissions: permissionSet,
		Status: accesscontrol.RoleStatusActive, Revision: 1,
	}
	binding := accesscontrol.ManagementRoleBinding{
		ID: "role-binding-1", PrincipalID: "principal-1", RoleID: role.ID,
		Scope: accesscontrol.NamespaceScope(namespaceID), Status: accesscontrol.BindingStatusActive, Revision: 1,
	}
	return managementauthorization.RoleGrant{Binding: binding, Role: role}
}
