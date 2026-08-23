package managementauthorization

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementpermission"
)

type snapshotLoaderStub struct {
	snapshot Snapshot
	err      error
	loads    int
}

func (loader *snapshotLoaderStub) Load(
	_ context.Context,
	_ accesscontrol.ManagementPrincipalID,
	_ accesscontrol.NamespaceID,
) (Snapshot, error) {
	loader.loads++
	return loader.snapshot, loader.err
}

func TestRuntimeLoadsFreshAuthorityForEveryDecision(t *testing.T) {
	role, _ := accesscontrol.BuiltInRole(accesscontrol.BuiltInRoleViewer)
	principalID := accesscontrol.ManagementPrincipalID("principal-1")
	namespaceID := accesscontrol.NamespaceID("namespace-1")
	loader := &snapshotLoaderStub{snapshot: Snapshot{
		Principal: accesscontrol.ManagementPrincipal{ID: principalID},
		RoleGrants: []RoleGrant{{
			Binding: accesscontrol.ManagementRoleBinding{
				ID: "binding-1", PrincipalID: principalID, RoleID: role.ID,
				Scope:  accesscontrol.NamespaceScope(namespaceID),
				Status: accesscontrol.BindingStatusActive, Revision: 1,
			},
			Role: role,
		}},
		AuthorityDigest: "sha256:current",
	}}
	runtime := Runtime{Loader: loader}
	request := Request{
		PrincipalID: principalID, NamespaceID: namespaceID, Authenticated: true,
		Permission: managementpermission.Require("provider_catalog.read", "request_namespace"),
		Targets: map[string][]accesscontrol.ScopedTarget{
			"request_namespace": {{Scope: accesscontrol.NamespaceScope(namespaceID)}},
		},
	}
	for range 2 {
		decision, err := runtime.Authorize(context.Background(), request)
		if err != nil {
			t.Fatal(err)
		}
		if decision.AuthorityDigest != "sha256:current" {
			t.Fatalf("unexpected authority digest %q", decision.AuthorityDigest)
		}
	}
	if loader.loads != 2 {
		t.Fatalf("authorization snapshot loads = %d, want 2", loader.loads)
	}
}

func TestRuntimeFailsClosedOnLoaderAndSnapshotErrors(t *testing.T) {
	principalID := accesscontrol.ManagementPrincipalID("principal-1")
	request := Request{
		PrincipalID: principalID, Authenticated: true,
		Permission: managementpermission.Require("self.read", "intrinsic_self"),
	}
	for name, runtime := range map[string]Runtime{
		"missing loader": {},
		"load failure":   {Loader: &snapshotLoaderStub{err: errors.New("storage unavailable")}},
		"wrong principal": {Loader: &snapshotLoaderStub{snapshot: Snapshot{
			Principal: accesscontrol.ManagementPrincipal{ID: "other"}, AuthorityDigest: "sha256:current",
		}}},
		"missing digest": {Loader: &snapshotLoaderStub{snapshot: Snapshot{
			Principal: accesscontrol.ManagementPrincipal{ID: principalID},
		}}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := runtime.Authorize(context.Background(), request); err == nil {
				t.Fatal("expected authorization failure")
			}
		})
	}
}
