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

type capturingAuthorizationRuntime struct {
	request managementauthorization.Request
	err     error
}

func (runtime *capturingAuthorizationRuntime) Authorize(_ context.Context, request managementauthorization.Request) (managementauthorization.Decision, error) {
	runtime.request = request
	if runtime.err != nil {
		return managementauthorization.Decision{}, runtime.err
	}
	return managementauthorization.Decision{AuthorityDigest: "sha256:test"}, nil
}

func TestIdentityClusterAuthorizationUsesNoNamespace(t *testing.T) {
	runtime := &capturingAuthorizationRuntime{}
	authorizer, err := NewIdentityRuntimeAuthorizer(runtime)
	if err != nil {
		t.Fatal(err)
	}
	principal := "10000000-0000-4000-8000-000000000001"
	_, err = authorizer.Authorize(context.Background(), AuthorizationRequest{
		Operation: managementapi.OperationContract{Permission: managementapi.Require("principal.read", "cluster")},
		Session:   managementauth.AuthenticatedSession{Session: managementauth.LiveSession{Session: managementauth.Session{PrincipalID: principal}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if runtime.request.NamespaceID != "" || runtime.request.PrincipalID != accesscontrol.ManagementPrincipalID(principal) {
		t.Fatalf("cluster authorization was rebound to a namespace: %#v", runtime.request)
	}
	if target := runtime.request.Targets["cluster"]; len(target) != 1 || target[0].Scope.Kind != accesscontrol.ScopeKindCluster {
		t.Fatalf("cluster target missing: %#v", runtime.request.Targets)
	}
}

func TestIdentityClusterAuthorizationRejectsNamespaceTarget(t *testing.T) {
	authorizer, _ := NewIdentityRuntimeAuthorizer(&capturingAuthorizationRuntime{})
	_, err := authorizer.Authorize(context.Background(), AuthorizationRequest{
		Operation: managementapi.OperationContract{Permission: managementapi.Require("principal.read", "cluster")},
		Session:   managementauth.AuthenticatedSession{Session: managementauth.LiveSession{Session: managementauth.Session{PrincipalID: "10000000-0000-4000-8000-000000000001"}}},
		Targets:   map[string][]accesscontrol.ScopedTarget{"target": {{Scope: accesscontrol.NamespaceScope("20000000-0000-4000-8000-000000000001")}}},
	})
	if !errors.Is(err, managementauthorization.ErrInvalidContext) {
		t.Fatalf("expected fail-closed cluster scope rejection, got %v", err)
	}
}

func TestIdentityClusterAuthorizationRejectsNamespaceSession(t *testing.T) {
	authorizer, _ := NewIdentityRuntimeAuthorizer(&capturingAuthorizationRuntime{})
	_, err := authorizer.Authorize(context.Background(), AuthorizationRequest{
		Operation: managementapi.OperationContract{Permission: managementapi.Require("principal.read", "cluster")},
		Session: managementauth.AuthenticatedSession{
			Session:     managementauth.LiveSession{Session: managementauth.Session{PrincipalID: "10000000-0000-4000-8000-000000000001"}},
			NamespaceID: "20000000-0000-4000-8000-000000000001",
		},
	})
	if !errors.Is(err, managementauthorization.ErrInvalidContext) {
		t.Fatalf("expected namespace-bound session rejection, got %v", err)
	}
}
