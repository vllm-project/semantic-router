package managementserver

import (
	"context"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

type AuthorizationRuntime interface {
	Authorize(
		context.Context,
		managementauthorization.Request,
	) (managementauthorization.Decision, error)
}

type resultScopeRuntime interface {
	ResolveResultScope(
		context.Context,
		accesscontrol.ManagementPrincipalID,
		accesscontrol.NamespaceID,
		accesscontrol.Permission,
	) (managementauthorization.ResultScope, error)
}

// RuntimeAuthorizer is the transport adapter between Management routes and the
// shared authorization runtime. It derives scoped targets from authoritative
// route identifiers; callers never submit roles, permissions, or ownership.
type RuntimeAuthorizer struct {
	runtime AuthorizationRuntime
}

func NewRuntimeAuthorizer(runtime AuthorizationRuntime) (*RuntimeAuthorizer, error) {
	if runtime == nil {
		return nil, errors.New("management authorization runtime is required")
	}
	return &RuntimeAuthorizer{runtime: runtime}, nil
}

func (authorizer *RuntimeAuthorizer) Authorize(
	ctx context.Context,
	request AuthorizationRequest,
) (AuthorizationDecision, error) {
	if authorizer == nil || authorizer.runtime == nil {
		return AuthorizationDecision{}, errors.New("management authorization runtime is unavailable")
	}
	namespaceID := accesscontrol.NamespaceID(request.NamespaceID)
	principalID := accesscontrol.ManagementPrincipalID(request.Session.Session.PrincipalID)
	namespaceTarget := accesscontrol.ScopedTarget{Scope: accesscontrol.NamespaceScope(namespaceID)}
	if !canonicalUUID(request.NamespaceID) || !canonicalUUID(string(principalID)) {
		return AuthorizationDecision{}, managementauthorization.ErrInvalidContext
	}
	targets, err := namespacedAuthorizationTargets(namespaceID, request.Targets)
	if err != nil {
		return AuthorizationDecision{}, err
	}
	targets["request_namespace"] = []accesscontrol.ScopedTarget{namespaceTarget}
	targets["path_namespace"] = []accesscontrol.ScopedTarget{namespaceTarget}
	// Cluster-or-Namespace permission expressions are evaluated against one
	// authoritative snapshot containing both cluster and requested-Namespace
	// bindings. Supplying the cluster operand prevents an unresolved first Any
	// branch from turning a valid Namespace grant into an integration error.
	targets["cluster"] = []accesscontrol.ScopedTarget{{Scope: accesscontrol.ClusterScope()}}
	decision, err := authorizer.runtime.Authorize(ctx, managementauthorization.Request{
		PrincipalID:   principalID,
		NamespaceID:   namespaceID,
		Permission:    request.Operation.Permission,
		Targets:       targets,
		Conditions:    request.Conditions,
		SpecialAuth:   request.SpecialAuth,
		Recorded:      request.Recorded,
		Authenticated: true,
	})
	if err != nil {
		return AuthorizationDecision{}, err
	}
	return AuthorizationDecision{AuthorityDigest: decision.AuthorityDigest}, nil
}

func (authorizer *RuntimeAuthorizer) ResolveResultScope(
	ctx context.Context,
	principalID accesscontrol.ManagementPrincipalID,
	namespaceID accesscontrol.NamespaceID,
	permission accesscontrol.Permission,
) (managementauthorization.ResultScope, error) {
	if authorizer == nil || authorizer.runtime == nil {
		return managementauthorization.ResultScope{}, errors.New("management authorization runtime is unavailable")
	}
	runtime, ok := authorizer.runtime.(resultScopeRuntime)
	if !ok {
		return managementauthorization.ResultScope{}, errors.New("management result-scope runtime is unavailable")
	}
	return runtime.ResolveResultScope(ctx, principalID, namespaceID, permission)
}

func namespacedAuthorizationTargets(
	namespaceID accesscontrol.NamespaceID,
	source map[string][]accesscontrol.ScopedTarget,
) (map[string][]accesscontrol.ScopedTarget, error) {
	targets := make(map[string][]accesscontrol.ScopedTarget, len(source)+2)
	for operand, values := range source {
		if operand == "" || operand == "request_namespace" || operand == "path_namespace" || len(values) == 0 {
			return nil, managementauthorization.ErrInvalidContext
		}
		targets[operand] = make([]accesscontrol.ScopedTarget, len(values))
		for index, target := range values {
			if err := target.Validate(); err != nil || target.Scope.Kind == accesscontrol.ScopeKindCluster ||
				target.Scope.NamespaceID != namespaceID {
				return nil, fmt.Errorf("%w: target operand %q", managementauthorization.ErrInvalidContext, operand)
			}
			targets[operand][index] = target
		}
	}
	return targets, nil
}
