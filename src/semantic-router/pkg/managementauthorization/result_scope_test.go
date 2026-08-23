package managementauthorization

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

func TestResolveResultScopeCompilesNarrowAuthority(t *testing.T) {
	principal := resultScopePrincipal()
	namespaceID := accesscontrol.NamespaceID("11111111-1111-4111-8111-111111111111")
	userID := accesscontrol.UserID("22222222-2222-4222-8222-222222222222")
	teamID := accesscontrol.TeamID("33333333-3333-4333-8333-333333333333")
	keyID := accesscontrol.APIKeyID("44444444-4444-4444-8444-444444444444")
	credentialID := accesscontrol.ResourceID("55555555-5555-4555-8555-555555555555")
	policyID := accesscontrol.ResourceID("66666666-6666-4666-8666-666666666666")
	operationID := accesscontrol.ResourceID("77777777-7777-4777-8777-777777777777")
	runtime := Runtime{Loader: staticSnapshotLoader{snapshot: Snapshot{
		Principal: principal, AuthorityDigest: "authority",
		RoleGrants: []RoleGrant{
			resultScopeRoleGrant(t, principal.ID, accesscontrol.UserScope(namespaceID, userID)),
			resultScopeRoleGrant(t, principal.ID, accesscontrol.ResourceScope(namespaceID, accesscontrol.ScopeResourceAPIKey, accesscontrol.ResourceID(keyID))),
			resultScopeRoleGrant(t, principal.ID, accesscontrol.ResourceScope(namespaceID,
				accesscontrol.ScopeResourceProviderCredential, credentialID)),
			resultScopeRoleGrant(t, principal.ID, accesscontrol.ResourceScope(namespaceID,
				accesscontrol.ScopeResourceAccessPolicy, policyID)),
			resultScopeRoleGrant(t, principal.ID, accesscontrol.ResourceScope(namespaceID,
				accesscontrol.ScopeResourceOperation, operationID)),
		},
		TeamGrants: []TeamGrant{{
			Membership: accesscontrol.TeamMembership{
				NamespaceID: namespaceID, TeamID: teamID,
				UserID: userID, Role: accesscontrol.TeamRoleMember, Status: accesscontrol.MembershipStatusActive,
				CreatedAt: time.Unix(1, 0).UTC(), UpdatedAt: time.Unix(1, 0).UTC(),
			},
		}},
	}}}

	scope, err := runtime.ResolveResultScope(context.Background(), principal.ID, namespaceID, accesscontrol.PermissionUsageRead)
	if err != nil {
		t.Fatalf("ResolveResultScope() error = %v", err)
	}
	if scope.All || len(scope.TeamIDs) != 1 || scope.TeamIDs[0] != teamID ||
		len(scope.UserIDs) != 1 || scope.UserIDs[0] != userID ||
		len(scope.APIKeyIDs) != 1 || scope.APIKeyIDs[0] != keyID ||
		len(scope.IDs(accesscontrol.ScopeResourceProviderCredential)) != 1 ||
		scope.IDs(accesscontrol.ScopeResourceProviderCredential)[0] != credentialID ||
		len(scope.IDs(accesscontrol.ScopeResourceAccessPolicy)) != 1 ||
		scope.IDs(accesscontrol.ScopeResourceAccessPolicy)[0] != policyID ||
		len(scope.IDs(accesscontrol.ScopeResourceOperation)) != 1 ||
		scope.IDs(accesscontrol.ScopeResourceOperation)[0] != operationID {
		t.Fatalf("ResolveResultScope() = %#v", scope)
	}
}

func TestResultScopeDigestCanonicalizesAndBindsTypedResources(t *testing.T) {
	namespaceID := accesscontrol.NamespaceID("11111111-1111-4111-8111-111111111111")
	first := ResultScope{
		NamespaceID: namespaceID,
		TeamIDs:     []accesscontrol.TeamID{"bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"},
		ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
			accesscontrol.ScopeResourceProviderCredential: {
				"dddddddd-dddd-4ddd-8ddd-dddddddddddd",
				"cccccccc-cccc-4ccc-8ccc-cccccccccccc",
			},
		},
	}
	second := ResultScope{
		NamespaceID: namespaceID,
		TeamIDs: []accesscontrol.TeamID{
			"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
			"bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
		},
		ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
			accesscontrol.ScopeResourceProviderCredential: {
				"cccccccc-cccc-4ccc-8ccc-cccccccccccc",
				"dddddddd-dddd-4ddd-8ddd-dddddddddddd",
			},
		},
	}
	firstDigest, firstErr := first.Digest()
	secondDigest, secondErr := second.Digest()
	if firstErr != nil || secondErr != nil || firstDigest == "" || firstDigest != secondDigest {
		t.Fatalf("canonical digests = %q/%q, errors = %v/%v", firstDigest, secondDigest, firstErr, secondErr)
	}
	second.ResourceIDs[accesscontrol.ScopeResourceProviderCredential][0] = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
	changed, err := second.Digest()
	if err != nil || changed == firstDigest {
		t.Fatalf("changed digest = %q, error = %v", changed, err)
	}
	if !first.Covers(ResultScope{
		NamespaceID: namespaceID,
		ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
			accesscontrol.ScopeResourceProviderCredential: {"cccccccc-cccc-4ccc-8ccc-cccccccccccc"},
		},
	}) {
		t.Fatal("typed resource subset was not covered")
	}
}

func TestResultScopeDigestRejectsAmbiguousResourceDimensions(t *testing.T) {
	namespaceID := accesscontrol.NamespaceID("11111111-1111-4111-8111-111111111111")
	for name, scope := range map[string]ResultScope{
		"api key in generic map": {
			NamespaceID: namespaceID,
			ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
				accesscontrol.ScopeResourceAPIKey: {"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"},
			},
		},
		"empty typed id": {
			NamespaceID: namespaceID,
			ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
				accesscontrol.ScopeResourceOperation: {""},
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := scope.Digest(); !errors.Is(err, accesscontrol.ErrInvalidResultScope) {
				t.Fatalf("Digest() error = %v", err)
			}
		})
	}
}

func TestResolveResultScopeNamespaceGrantCollapsesToAll(t *testing.T) {
	principal := resultScopePrincipal()
	namespaceID := accesscontrol.NamespaceID("11111111-1111-4111-8111-111111111111")
	runtime := Runtime{Loader: staticSnapshotLoader{snapshot: Snapshot{
		Principal: principal, AuthorityDigest: "authority",
		RoleGrants: []RoleGrant{resultScopeRoleGrant(t, principal.ID, accesscontrol.NamespaceScope(namespaceID))},
	}}}

	scope, err := runtime.ResolveResultScope(context.Background(), principal.ID, namespaceID, accesscontrol.PermissionUsageRead)
	if err != nil || !scope.All || !scope.Covers(ResultScope{NamespaceID: namespaceID, TeamIDs: []accesscontrol.TeamID{"team"}}) {
		t.Fatalf("ResolveResultScope() = %#v, %v", scope, err)
	}
}

func TestResolveResultScopeDeniesUncoveredPermission(t *testing.T) {
	principal := resultScopePrincipal()
	namespaceID := accesscontrol.NamespaceID("11111111-1111-4111-8111-111111111111")
	runtime := Runtime{Loader: staticSnapshotLoader{snapshot: Snapshot{Principal: principal, AuthorityDigest: "authority"}}}
	_, err := runtime.ResolveResultScope(context.Background(), principal.ID, namespaceID, accesscontrol.PermissionLogRead)
	if !errors.Is(err, ErrDenied) {
		t.Fatalf("ResolveResultScope() error = %v, want ErrDenied", err)
	}
}

func TestInvitedReadIdentityKeepsDirectoryAndUsageNarrow(t *testing.T) {
	principal := resultScopePrincipal()
	namespaceID := accesscontrol.NamespaceID("11111111-1111-4111-8111-111111111111")
	userID := accesscontrol.UserID("22222222-2222-4222-8222-222222222222")
	teamID := accesscontrol.TeamID("33333333-3333-4333-8333-333333333333")
	viewer, _ := accesscontrol.BuiltInRole(accesscontrol.BuiltInRoleViewer)
	consumer, _ := accesscontrol.BuiltInRole(accesscontrol.BuiltInRoleConsumer)
	runtime := Runtime{Loader: staticSnapshotLoader{snapshot: Snapshot{
		Principal: principal, AuthorityDigest: "invited-read-authority",
		RoleGrants: []RoleGrant{
			builtInResultScopeGrant(principal.ID, viewer, accesscontrol.NamespaceScope(namespaceID), "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"),
			builtInResultScopeGrant(principal.ID, consumer, accesscontrol.UserScope(namespaceID, userID), "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2"),
		},
		TeamGrants: []TeamGrant{{
			Membership: accesscontrol.TeamMembership{
				NamespaceID: namespaceID, TeamID: teamID, UserID: userID,
				Role: accesscontrol.TeamRoleMember, Status: accesscontrol.MembershipStatusActive,
				CreatedAt: time.Unix(1, 0).UTC(), UpdatedAt: time.Unix(1, 0).UTC(),
			},
		}},
	}}}

	userScope := resolveResultScopeForTest(t, runtime, principal.ID, namespaceID, accesscontrol.PermissionUserRead)
	if userScope.All || len(userScope.UserIDs) != 1 || userScope.UserIDs[0] != userID || len(userScope.TeamIDs) != 0 {
		t.Fatalf("user.read scope = %#v", userScope)
	}
	keyScope := resolveResultScopeForTest(t, runtime, principal.ID, namespaceID, accesscontrol.PermissionKeyRead)
	if keyScope.All || len(keyScope.UserIDs) != 1 || keyScope.UserIDs[0] != userID || len(keyScope.TeamIDs) != 0 {
		t.Fatalf("key.read scope = %#v", keyScope)
	}
	for _, permission := range []accesscontrol.Permission{
		accesscontrol.PermissionTeamRead,
		accesscontrol.PermissionAccessPolicyRead,
		accesscontrol.PermissionRatePolicyRead,
		accesscontrol.PermissionQuotaRead,
		accesscontrol.PermissionUsageRead,
	} {
		scope := resolveResultScopeForTest(t, runtime, principal.ID, namespaceID, permission)
		if scope.All || len(scope.UserIDs) != 1 || scope.UserIDs[0] != userID ||
			len(scope.TeamIDs) != 1 || scope.TeamIDs[0] != teamID {
			t.Fatalf("%s scope = %#v", permission, scope)
		}
	}
	routingScope := resolveResultScopeForTest(t, runtime, principal.ID, namespaceID, accesscontrol.PermissionRoutingRead)
	if !routingScope.All {
		t.Fatalf("routing.read scope = %#v, want namespace read-only visibility", routingScope)
	}
	for _, permission := range []accesscontrol.Permission{
		accesscontrol.PermissionKeyManage,
		accesscontrol.PermissionRoutingManage,
		accesscontrol.PermissionLogRead,
		accesscontrol.PermissionAuditRead,
	} {
		if _, err := runtime.ResolveResultScope(context.Background(), principal.ID, namespaceID, permission); !errors.Is(err, ErrDenied) {
			t.Fatalf("%s error = %v, want ErrDenied", permission, err)
		}
	}
}

func resolveResultScopeForTest(
	t *testing.T,
	runtime Runtime,
	principalID accesscontrol.ManagementPrincipalID,
	namespaceID accesscontrol.NamespaceID,
	permission accesscontrol.Permission,
) ResultScope {
	t.Helper()
	scope, err := runtime.ResolveResultScope(context.Background(), principalID, namespaceID, permission)
	if err != nil {
		t.Fatalf("ResolveResultScope(%s) error = %v", permission, err)
	}
	return scope
}

func builtInResultScopeGrant(
	principalID accesscontrol.ManagementPrincipalID,
	role accesscontrol.ManagementRole,
	scope accesscontrol.Scope,
	bindingID accesscontrol.ManagementRoleBindingID,
) RoleGrant {
	return RoleGrant{Binding: accesscontrol.ManagementRoleBinding{
		ID: bindingID, PrincipalID: principalID, RoleID: role.ID, Scope: scope,
		Status: accesscontrol.BindingStatusActive, Revision: 1,
	}, Role: role}
}

type staticSnapshotLoader struct{ snapshot Snapshot }

func (loader staticSnapshotLoader) Load(
	context.Context,
	accesscontrol.ManagementPrincipalID,
	accesscontrol.NamespaceID,
) (Snapshot, error) {
	return loader.snapshot, nil
}

func resultScopePrincipal() accesscontrol.ManagementPrincipal {
	now := time.Unix(1, 0).UTC()
	return accesscontrol.ManagementPrincipal{
		ID: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", Issuer: "issuer", Subject: "subject",
		Status: accesscontrol.PrincipalStatusActive, CreatedAt: now, UpdatedAt: now,
	}
}

func resultScopeRoleGrant(
	t *testing.T,
	principalID accesscontrol.ManagementPrincipalID,
	scope accesscontrol.Scope,
) RoleGrant {
	t.Helper()
	permissions, err := accesscontrol.NewPermissionSet(accesscontrol.PermissionUsageRead)
	if err != nil {
		t.Fatal(err)
	}
	ceiling, err := accesscontrol.NewPermissionSet()
	if err != nil {
		t.Fatal(err)
	}
	role := accesscontrol.ManagementRole{
		ID: "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", NamespaceID: scope.NamespaceID,
		Name: "usage_reader", DisplayName: "Usage reader", Permissions: permissions,
		Status: accesscontrol.RoleStatusActive, Revision: 1,
	}
	if scope.Kind == accesscontrol.ScopeKindCluster {
		role.NamespaceID = ""
	}
	return RoleGrant{
		Binding: accesscontrol.ManagementRoleBinding{
			ID: "cccccccc-cccc-4ccc-8ccc-cccccccccccc", PrincipalID: principalID,
			RoleID: role.ID, Scope: scope, DelegationCeiling: ceiling,
			Status: accesscontrol.BindingStatusActive, Revision: 1,
		},
		Role: role,
	}
}
