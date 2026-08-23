package managementserver

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

type resultScopeResolverFunc func(
	context.Context,
	accesscontrol.ManagementPrincipalID,
	accesscontrol.NamespaceID,
	accesscontrol.Permission,
) (managementauthorization.ResultScope, error)

func (function resultScopeResolverFunc) ResolveResultScope(
	ctx context.Context,
	principalID accesscontrol.ManagementPrincipalID,
	namespaceID accesscontrol.NamespaceID,
	permission accesscontrol.Permission,
) (managementauthorization.ResultScope, error) {
	return function(ctx, principalID, namespaceID, permission)
}

func allowAllResultScopes() ResultScopeResolver {
	return resultScopeResolverFunc(func(
		_ context.Context,
		_ accesscontrol.ManagementPrincipalID,
		namespaceID accesscontrol.NamespaceID,
		_ accesscontrol.Permission,
	) (managementauthorization.ResultScope, error) {
		return managementauthorization.ResultScope{NamespaceID: namespaceID, All: true}, nil
	})
}
