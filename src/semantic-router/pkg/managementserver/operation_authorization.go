package managementserver

import (
	"context"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policybulk"
)

var errInvalidStoredOperation = errors.New("stored policy bulk operation is invalid")

func (routes *OperationRoutes) authorizeStoredOperation(
	ctx context.Context,
	session managementauth.AuthenticatedSession,
	operation policybulk.Operation,
	action operationAuthorizationAction,
) error {
	return authorizePolicyBulkStoredOperation(ctx, routes.authorization, routes.contracts, session, operation, action)
}

func authorizePolicyBulkStoredOperation(
	ctx context.Context,
	authorization Authorizer,
	contracts operationRouteContracts,
	session managementauth.AuthenticatedSession,
	operation policybulk.Operation,
	action operationAuthorizationAction,
) error {
	if !validStoredOperation(operation, session.NamespaceID) {
		return errInvalidStoredOperation
	}
	read := action != operationAuthorizationCancel
	operationTargets := make([]accesscontrol.ScopedTarget, 0, len(operation.Targets)*2)
	for _, target := range operation.Targets {
		rate := target.Kind == policybulk.ItemKindRateBinding
		policyID := target.PolicyID
		if target.InlinePolicy {
			policyID = ""
		}
		targets, conditions, valid := policyBindingTargets(operation.NamespaceID, policyID, target.Subject, rate)
		if !valid {
			return errInvalidStoredOperation
		}
		domainContract := contracts.accessMutation
		if read {
			domainContract = contracts.accessRead
		}
		if rate {
			if read {
				domainContract = contracts.rateRead
			} else {
				domainContract = contracts.rateMutation
			}
		}
		if _, err := authorization.Authorize(ctx, AuthorizationRequest{
			Operation: domainContract, Session: session, NamespaceID: operation.NamespaceID,
			Targets: targets, Conditions: conditions,
		}); err != nil {
			return err
		}
		operationTargets = appendUniqueScopedTargets(operationTargets, targets["policy"]...)
		operationTargets = appendUniqueScopedTargets(operationTargets, targets["subject"]...)
	}

	originator := operation.OriginPrincipalID == session.Session.PrincipalID
	genericContract := contracts.detail
	recorded := map[string]bool{"original_domain_read": true}
	if !read {
		genericContract = contracts.cancel
		recorded = map[string]bool{"original_domain_mutation": true}
	}
	_, err := authorization.Authorize(ctx, AuthorizationRequest{
		Operation: genericContract, Session: session, NamespaceID: operation.NamespaceID,
		Targets: map[string][]accesscontrol.ScopedTarget{"operation_targets": operationTargets},
		Conditions: map[string]bool{
			"operation_originator":  originator,
			"cross_actor_operation": !originator,
		},
		Recorded: recorded,
	})
	return err
}

type operationAuthorizationAction uint8

const (
	operationAuthorizationRead operationAuthorizationAction = iota + 1
	operationAuthorizationCancel
)

func validStoredOperation(operation policybulk.Operation, namespaceID string) bool {
	if !canonicalUUID(operation.ID) || operation.NamespaceID != namespaceID || !canonicalUUID(operation.NamespaceID) ||
		!canonicalUUID(operation.OriginPrincipalID) || operation.Version == 0 || !operation.State.Valid() ||
		operation.Total == 0 || operation.Total > policybulk.MaximumItems ||
		operation.Completed > operation.Total || operation.Failed > operation.Completed ||
		len(operation.Targets) != int(operation.Total) || len(operation.TargetIDs) != len(operation.Targets) ||
		operation.CreatedAt.IsZero() || operation.UpdatedAt.IsZero() {
		return false
	}
	for _, principalID := range operation.ActorChain {
		if !canonicalUUID(principalID) {
			return false
		}
	}
	seen := make(map[string]struct{}, len(operation.Targets))
	for index, target := range operation.Targets {
		if !canonicalUUID(target.ItemID) || operation.TargetIDs[index] != target.ItemID ||
			!canonicalUUID(target.Subject.ID) || !target.Subject.Type.Valid() {
			return false
		}
		if _, duplicate := seen[target.ItemID]; duplicate {
			return false
		}
		seen[target.ItemID] = struct{}{}
		switch operation.Kind {
		case policybulk.AccessBindingOperationKind:
			if target.Kind != policybulk.ItemKindAccessBinding || !canonicalUUID(target.PolicyID) ||
				target.InlinePolicy || target.Mode != "" {
				return false
			}
		case policybulk.RateBindingOperationKind:
			if target.Kind != policybulk.ItemKindRateBinding || !target.Mode.Valid() ||
				target.InlinePolicy == (target.PolicyID != "") ||
				(!target.InlinePolicy && !canonicalUUID(target.PolicyID)) {
				return false
			}
		default:
			return false
		}
	}
	return true
}

func appendUniqueScopedTargets(destination []accesscontrol.ScopedTarget, values ...accesscontrol.ScopedTarget) []accesscontrol.ScopedTarget {
	for _, value := range values {
		duplicate := false
		for _, existing := range destination {
			if equalScopedTarget(existing, value) {
				duplicate = true
				break
			}
		}
		if !duplicate {
			destination = append(destination, value)
		}
	}
	return destination
}

func equalScopedTarget(left, right accesscontrol.ScopedTarget) bool {
	if left.Scope != right.Scope || len(left.Ancestors) != len(right.Ancestors) {
		return false
	}
	for index := range left.Ancestors {
		if left.Ancestors[index] != right.Ancestors[index] {
			return false
		}
	}
	return true
}

func operationDenied(err error) bool {
	return errors.Is(err, managementauthorization.ErrDenied)
}
