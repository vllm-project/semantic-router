package managementserver

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

type authorizationRuntimeStub struct {
	request managementauthorization.Request
}

func (stub *authorizationRuntimeStub) Authorize(
	_ context.Context,
	request managementauthorization.Request,
) (managementauthorization.Decision, error) {
	stub.request = request
	return managementauthorization.Decision{AuthorityDigest: "sha256:authority"}, nil
}

func TestRuntimeAuthorizerBuildsProviderCredentialTargets(t *testing.T) {
	stub := &authorizationRuntimeStub{}
	authorizer, err := NewRuntimeAuthorizer(stub)
	if err != nil {
		t.Fatal(err)
	}
	operation, found := managementapi.LookupOperation(
		managementapi.MethodPOST,
		managementapi.BasePath+"/providers/{providerId}:discover-models",
	)
	if !found {
		t.Fatal("Provider discovery operation is missing")
	}
	request := AuthorizationRequest{
		Operation: operation,
		Session: managementauth.AuthenticatedSession{Session: managementauth.LiveSession{
			Session: managementauth.Session{PrincipalID: "11111111-1111-4111-8111-111111111111"},
		}},
		NamespaceID: "22222222-2222-4222-8222-222222222222",
		Targets: map[string][]accesscontrol.ScopedTarget{
			"credential": {{Scope: accesscontrol.ResourceScope(
				"22222222-2222-4222-8222-222222222222",
				accesscontrol.ScopeResourceProviderCredential,
				"33333333-3333-4333-8333-333333333333",
			)}},
		},
		Conditions: map[string]bool{"provider_credential_supplied": true},
	}
	decision, err := authorizer.Authorize(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if decision.AuthorityDigest != "sha256:authority" {
		t.Fatalf("authority digest = %q", decision.AuthorityDigest)
	}
	credential := stub.request.Targets["credential"]
	if len(credential) != 1 || credential[0].Scope != accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(request.NamespaceID),
		accesscontrol.ScopeResourceProviderCredential,
		"33333333-3333-4333-8333-333333333333",
	) {
		t.Fatalf("credential targets = %#v", credential)
	}
	if !stub.request.Authenticated || !stub.request.Conditions["provider_credential_supplied"] {
		t.Fatalf("authorization request lost trusted facts: %#v", stub.request)
	}
}

func TestRuntimeAuthorizerUsesNamespaceForCollection(t *testing.T) {
	stub := &authorizationRuntimeStub{}
	authorizer, err := NewRuntimeAuthorizer(stub)
	if err != nil {
		t.Fatal(err)
	}
	operation, found := managementapi.LookupOperation(
		managementapi.MethodGET,
		managementapi.BasePath+"/provider-credentials",
	)
	if !found {
		t.Fatal("ProviderCredential list operation is missing")
	}
	request := AuthorizationRequest{
		Operation: operation,
		Session: managementauth.AuthenticatedSession{Session: managementauth.LiveSession{
			Session: managementauth.Session{PrincipalID: "11111111-1111-4111-8111-111111111111"},
		}},
		NamespaceID: "22222222-2222-4222-8222-222222222222",
		Targets: map[string][]accesscontrol.ScopedTarget{
			"credential": {{Scope: accesscontrol.NamespaceScope("22222222-2222-4222-8222-222222222222")}},
		},
	}
	if _, err := authorizer.Authorize(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	target := stub.request.Targets["credential"]
	if len(target) != 1 || target[0].Scope != accesscontrol.NamespaceScope(accesscontrol.NamespaceID(request.NamespaceID)) {
		t.Fatalf("collection target = %#v", target)
	}
	cluster := stub.request.Targets["cluster"]
	if len(cluster) != 1 || cluster[0].Scope != accesscontrol.ClusterScope() {
		t.Fatalf("cluster-or-Namespace operand = %#v", cluster)
	}
}

func TestRuntimeAuthorizerRejectsCallerOwnedNamespaceScopes(t *testing.T) {
	authorizer, err := NewRuntimeAuthorizer(&authorizationRuntimeStub{})
	if err != nil {
		t.Fatal(err)
	}
	operation, found := managementapi.LookupOperation(
		managementapi.MethodGET,
		managementapi.BasePath+"/routing/exports/current",
	)
	if !found {
		t.Fatal("Routing manifest export operation is missing")
	}
	request := AuthorizationRequest{
		Operation: operation,
		Session: managementauth.AuthenticatedSession{Session: managementauth.LiveSession{
			Session: managementauth.Session{PrincipalID: "11111111-1111-4111-8111-111111111111"},
		}},
		NamespaceID: "22222222-2222-4222-8222-222222222222",
		Targets: map[string][]accesscontrol.ScopedTarget{
			"request_namespace": {{Scope: accesscontrol.NamespaceScope("22222222-2222-4222-8222-222222222222")}},
		},
	}
	if _, err := authorizer.Authorize(context.Background(), request); !errors.Is(err, managementauthorization.ErrInvalidContext) {
		t.Fatalf("caller-owned request_namespace error = %v", err)
	}
}
