package managementserver

import (
	"context"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
)

// PolicyBulkExecutionAuthorizer re-evaluates current authority immediately
// before a worker applies an item. It does not reconstruct or trust an expired
// HTTP session: the shared runtime reloads the principal's current grants.
type PolicyBulkExecutionAuthorizer struct {
	runtime AuthorizationRuntime
	access  managementapi.OperationContract
	rate    managementapi.OperationContract
}

func NewPolicyBulkExecutionAuthorizer(runtime AuthorizationRuntime) (*PolicyBulkExecutionAuthorizer, error) {
	if runtime == nil {
		return nil, errors.New("policy bulk execution authorization runtime is required")
	}
	accessOperation, accessFound := managementapi.LookupOperation(managementapi.MethodPOST, accessBindingBulkPath)
	rateOperation, rateFound := managementapi.LookupOperation(managementapi.MethodPOST, rateBindingBulkPath)
	if !accessFound || !rateFound {
		return nil, errors.New("policy bulk operation contracts are unavailable")
	}
	return &PolicyBulkExecutionAuthorizer{runtime: runtime, access: accessOperation, rate: rateOperation}, nil
}

func (authorizer *PolicyBulkExecutionAuthorizer) AuthorizePolicyBulkItem(
	ctx context.Context,
	request policybulk.AuthorizationRequest,
) error {
	if authorizer == nil || authorizer.runtime == nil || !canonicalUUID(request.NamespaceID) ||
		!canonicalUUID(request.PrincipalID) || !canonicalUUID(request.ItemID) {
		return managementauthorization.ErrInvalidContext
	}
	rate := request.Kind == policybulk.ItemKindRateBinding
	if request.Kind != policybulk.ItemKindAccessBinding && !rate {
		return managementauthorization.ErrInvalidContext
	}
	policyID := request.PolicyID
	if request.InlinePolicy {
		if !rate || policyID != "" {
			return managementauthorization.ErrInvalidContext
		}
	} else if !canonicalUUID(policyID) {
		return managementauthorization.ErrInvalidContext
	}
	targets, conditions, valid := policyBindingTargets(request.NamespaceID, policyID, request.Subject, rate)
	if !valid {
		return managementauthorization.ErrInvalidContext
	}
	namespaceID := accesscontrol.NamespaceID(request.NamespaceID)
	namespaced, err := namespacedAuthorizationTargets(namespaceID, targets)
	if err != nil {
		return err
	}
	namespaceTarget := accesscontrol.ScopedTarget{Scope: accesscontrol.NamespaceScope(namespaceID)}
	namespaced["request_namespace"] = []accesscontrol.ScopedTarget{namespaceTarget}
	namespaced["path_namespace"] = []accesscontrol.ScopedTarget{namespaceTarget}
	operation := authorizer.access
	if rate {
		operation = authorizer.rate
	}
	_, err = authorizer.runtime.Authorize(ctx, managementauthorization.Request{
		PrincipalID: accesscontrol.ManagementPrincipalID(request.PrincipalID),
		NamespaceID: namespaceID, Permission: operation.Permission,
		Targets: namespaced, Conditions: conditions, Authenticated: true,
	})
	if errors.Is(err, managementauthorization.ErrDenied) {
		return policybulk.ErrExecutionDenied
	}
	return err
}

var _ policybulk.ExecutionAuthorizer = (*PolicyBulkExecutionAuthorizer)(nil)
