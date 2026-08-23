package accesscontrol

import (
	"errors"
	"testing"
)

func TestDelegationCeilingDoesNotGrantRuntimeAuthority(t *testing.T) {
	role, _ := BuiltInRole(BuiltInRoleClusterAdmin)
	binding := validRoleBinding(role, ClusterScope())
	target := ScopedTarget{Scope: ResourceScope("ns-1", ScopeResourceAPIKey, "key-1")}
	allowed, err := Authorizes(binding, role, PermissionKeyReveal, target)
	if err != nil {
		t.Fatal(err)
	}
	if allowed {
		t.Fatal("delegation ceiling must not grant key.reveal to cluster_admin")
	}
}

func TestAuthorizesPermissionAndScope(t *testing.T) {
	role, _ := BuiltInRole(BuiltInRoleViewer)
	binding := validRoleBinding(role, NamespaceScope("ns-1"))

	tests := []struct {
		name       string
		permission Permission
		target     ScopedTarget
		want       bool
	}{
		{name: "allowed in namespace", permission: PermissionRoutingRead, target: ScopedTarget{Scope: ResourceScope("ns-1", ScopeResourceModel, "model-1")}, want: true},
		{name: "permission absent", permission: PermissionRoutingManage, target: ScopedTarget{Scope: ResourceScope("ns-1", ScopeResourceModel, "model-1")}},
		{name: "outside namespace", permission: PermissionRoutingRead, target: ScopedTarget{Scope: ResourceScope("ns-2", ScopeResourceModel, "model-1")}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			allowed, err := Authorizes(binding, role, test.permission, test.target)
			if err != nil {
				t.Fatal(err)
			}
			if allowed != test.want {
				t.Fatalf("allowed = %v, want %v", allowed, test.want)
			}
		})
	}
}

func TestConsumerRoleRequiresUserScope(t *testing.T) {
	role, _ := BuiltInRole(BuiltInRoleConsumer)
	binding := validRoleBinding(role, TeamScope("ns-1", "team-1"))
	target := ScopedTarget{Scope: ResourceScope("ns-1", ScopeResourceAPIKey, "key-1"), Ancestors: []Scope{binding.Scope}}
	if _, err := Authorizes(binding, role, PermissionKeyRead, target); !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected user-scope validation error, got %v", err)
	}
}

func TestCanDelegateRoleBinding(t *testing.T) {
	sourceRole, _ := BuiltInRole(BuiltInRolePlatformAdmin)
	source := validRoleBinding(sourceRole, NamespaceScope("ns-1"))
	targetRole, _ := BuiltInRole(BuiltInRoleViewer)

	tests := []struct {
		name          string
		source        ManagementRoleBinding
		target        ScopedTarget
		targetRole    ManagementRole
		targetCeiling PermissionSet
		wantErr       bool
	}{
		{
			name:          "contained delegation",
			source:        source,
			target:        ScopedTarget{Scope: TeamScope("ns-1", "team-1")},
			targetRole:    targetRole,
			targetCeiling: mustPermissionSet(PermissionRoutingRead),
		},
		{
			name:          "scope expansion",
			source:        source,
			target:        ScopedTarget{Scope: NamespaceScope("ns-2")},
			targetRole:    targetRole,
			targetCeiling: PermissionSet{},
			wantErr:       true,
		},
		{
			name: "permission exceeds ceiling",
			source: func() ManagementRoleBinding {
				restricted := source
				restricted.DelegationCeiling = mustPermissionSet(PermissionRoutingRead)
				return restricted
			}(),
			target:        ScopedTarget{Scope: TeamScope("ns-1", "team-1")},
			targetRole:    targetRole,
			targetCeiling: PermissionSet{},
			wantErr:       true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := CanDelegateRoleBinding(test.source, sourceRole, test.targetRole, test.target, test.targetCeiling)
			if test.wantErr && !errors.Is(err, ErrInvalid) {
				t.Fatalf("expected validation error, got %v", err)
			}
			if !test.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
