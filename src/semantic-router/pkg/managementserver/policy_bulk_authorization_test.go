package managementserver

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

type policyBulkAuthorizationRuntimeStub struct {
	request managementauthorization.Request
	err     error
}

func (runtime *policyBulkAuthorizationRuntimeStub) Authorize(
	_ context.Context,
	request managementauthorization.Request,
) (managementauthorization.Decision, error) {
	runtime.request = request
	return managementauthorization.Decision{}, runtime.err
}

func TestPolicyBulkExecutionAuthorizerRebuildsCurrentItemScope(t *testing.T) {
	runtime := &policyBulkAuthorizationRuntimeStub{}
	authorizer, err := NewPolicyBulkExecutionAuthorizer(runtime)
	if err != nil {
		t.Fatal(err)
	}
	request := policybulk.AuthorizationRequest{
		NamespaceID: testNamespaceID, PrincipalID: testPrincipalID,
		Kind: policybulk.ItemKindAccessBinding, ItemID: bindingOneID,
		PolicyID: policyOneID,
		Subject:  policymanagement.Subject{Type: accesscontrol.SubjectKindUser, ID: policyUserOne},
	}
	if err := authorizer.AuthorizePolicyBulkItem(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if runtime.request.PrincipalID != accesscontrol.ManagementPrincipalID(testPrincipalID) ||
		runtime.request.NamespaceID != accesscontrol.NamespaceID(testNamespaceID) ||
		!runtime.request.Authenticated || !runtime.request.Conditions["user_owner"] ||
		runtime.request.Conditions["team_owner"] || runtime.request.Conditions["key_owner"] {
		t.Fatalf("execution authorization request = %#v", runtime.request)
	}
	policyTargets := runtime.request.Targets["policy"]
	if len(policyTargets) != 1 || policyTargets[0].Scope != accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(testNamespaceID), accesscontrol.ScopeResourceAccessPolicy, policyOneID,
	) {
		t.Fatalf("policy targets = %#v", policyTargets)
	}
	subjectTargets := runtime.request.Targets["subject"]
	if len(subjectTargets) != 1 || subjectTargets[0].Scope != accesscontrol.UserScope(
		accesscontrol.NamespaceID(testNamespaceID), accesscontrol.UserID(policyUserOne),
	) {
		t.Fatalf("subject targets = %#v", subjectTargets)
	}
	for _, operand := range []string{"request_namespace", "path_namespace"} {
		targets := runtime.request.Targets[operand]
		if len(targets) != 1 || targets[0].Scope != accesscontrol.NamespaceScope(testNamespaceID) {
			t.Fatalf("%s targets = %#v", operand, targets)
		}
	}
}

func TestPolicyBulkExecutionAuthorizerDistinguishesRevocationFromInfrastructureFailure(t *testing.T) {
	runtime := &policyBulkAuthorizationRuntimeStub{err: managementauthorization.ErrDenied}
	authorizer, err := NewPolicyBulkExecutionAuthorizer(runtime)
	if err != nil {
		t.Fatal(err)
	}
	request := policybulk.AuthorizationRequest{
		NamespaceID: testNamespaceID, PrincipalID: testPrincipalID,
		Kind: policybulk.ItemKindRateBinding, ItemID: bindingOneID,
		InlinePolicy: true,
		Subject:      policymanagement.Subject{Type: accesscontrol.SubjectKindTeam, ID: policyUserOne},
	}
	if err := authorizer.AuthorizePolicyBulkItem(context.Background(), request); !errors.Is(err, policybulk.ErrExecutionDenied) {
		t.Fatalf("revoked authority error = %v", err)
	}
	infrastructure := errors.New("authorization backend unavailable")
	runtime.err = infrastructure
	if err := authorizer.AuthorizePolicyBulkItem(context.Background(), request); !errors.Is(err, infrastructure) {
		t.Fatalf("infrastructure error = %v", err)
	}
}
