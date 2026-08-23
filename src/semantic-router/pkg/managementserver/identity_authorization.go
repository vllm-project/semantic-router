package managementserver

import (
	"context"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

// IdentityRuntimeAuthorizer is the explicit cluster-or-namespace adapter used
// by identity routes. Cluster authorization always loads with an empty
// Namespace ID and accepts only cluster targets. It never derives global
// authority from a header, installation UUID, or sentinel Namespace.
type IdentityRuntimeAuthorizer struct {
	runtime   AuthorizationRuntime
	namespace *RuntimeAuthorizer
}

func NewIdentityRuntimeAuthorizer(runtime AuthorizationRuntime) (*IdentityRuntimeAuthorizer, error) {
	namespace, err := NewRuntimeAuthorizer(runtime)
	if err != nil {
		return nil, err
	}
	return &IdentityRuntimeAuthorizer{runtime: runtime, namespace: namespace}, nil
}

func (authorizer *IdentityRuntimeAuthorizer) Authorize(ctx context.Context, request AuthorizationRequest) (AuthorizationDecision, error) {
	if authorizer == nil || authorizer.runtime == nil || authorizer.namespace == nil {
		return AuthorizationDecision{}, errors.New("management identity authorization runtime is unavailable")
	}
	if request.NamespaceID != "" {
		return authorizer.namespace.Authorize(ctx, request)
	}
	principalID := accesscontrol.ManagementPrincipalID(request.Session.Session.PrincipalID)
	if !canonicalUUID(string(principalID)) || request.Session.NamespaceID != "" {
		return AuthorizationDecision{}, managementauthorization.ErrInvalidContext
	}
	targets, err := clusterAuthorizationTargets(request.Targets)
	if err != nil {
		return AuthorizationDecision{}, err
	}
	cluster := accesscontrol.ScopedTarget{Scope: accesscontrol.ClusterScope()}
	targets["cluster"] = []accesscontrol.ScopedTarget{cluster}
	if _, found := targets["target"]; !found {
		targets["target"] = []accesscontrol.ScopedTarget{cluster}
	}
	decision, err := authorizer.runtime.Authorize(ctx, managementauthorization.Request{
		PrincipalID: principalID, NamespaceID: "", Permission: request.Operation.Permission,
		Targets: targets, Conditions: request.Conditions, SpecialAuth: request.SpecialAuth,
		Recorded: request.Recorded, Authenticated: true,
	})
	if err != nil {
		return AuthorizationDecision{}, err
	}
	return AuthorizationDecision{AuthorityDigest: decision.AuthorityDigest}, nil
}

func (authorizer *IdentityRuntimeAuthorizer) ResolveResultScope(
	ctx context.Context,
	principalID accesscontrol.ManagementPrincipalID,
	namespaceID accesscontrol.NamespaceID,
	permission accesscontrol.Permission,
) (managementauthorization.ResultScope, error) {
	if authorizer == nil || authorizer.namespace == nil {
		return managementauthorization.ResultScope{}, errors.New("management identity authorization runtime is unavailable")
	}
	return authorizer.namespace.ResolveResultScope(ctx, principalID, namespaceID, permission)
}

func clusterAuthorizationTargets(source map[string][]accesscontrol.ScopedTarget) (map[string][]accesscontrol.ScopedTarget, error) {
	targets := make(map[string][]accesscontrol.ScopedTarget, len(source)+2)
	for operand, values := range source {
		if operand == "" || operand == "cluster" || operand == "request_namespace" || operand == "path_namespace" || len(values) == 0 {
			return nil, managementauthorization.ErrInvalidContext
		}
		targets[operand] = make([]accesscontrol.ScopedTarget, len(values))
		for index, target := range values {
			if err := target.Validate(); err != nil || target.Scope.Kind != accesscontrol.ScopeKindCluster {
				return nil, managementauthorization.ErrInvalidContext
			}
			targets[operand][index] = target
		}
	}
	return targets, nil
}
