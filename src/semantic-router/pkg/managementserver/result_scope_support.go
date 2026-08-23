package managementserver

import (
	"context"
	"errors"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

func configuredResultScopes(explicit ResultScopeResolver, authorization Authorizer) ResultScopeResolver {
	if explicit != nil {
		return explicit
	}
	resolved, _ := authorization.(ResultScopeResolver)
	return resolved
}

func listPermission(operation managementapi.OperationContract) (accesscontrol.Permission, bool) {
	expression := operation.Permission
	if expression.Operator != managementapi.PermissionLeaf {
		return "", false
	}
	permission := accesscontrol.Permission(expression.Permission)
	return permission, permission.Valid() && !permission.Intrinsic()
}

func resolveResultScope(
	ctx context.Context,
	resolver ResultScopeResolver,
	session managementauth.AuthenticatedSession,
	namespaceID string,
	permission accesscontrol.Permission,
) (managementauthorization.ResultScope, error) {
	if resolver == nil {
		return managementauthorization.ResultScope{}, errors.New("management result-scope resolver is unavailable")
	}
	return resolver.ResolveResultScope(ctx,
		accesscontrol.ManagementPrincipalID(session.Session.PrincipalID),
		accesscontrol.NamespaceID(namespaceID), permission)
}

func resolveListResultScope(
	ctx context.Context,
	resolver ResultScopeResolver,
	session managementauth.AuthenticatedSession,
	namespaceID string,
	permission accesscontrol.Permission,
) (managementauthorization.ResultScope, error) {
	scope, err := resolveResultScope(ctx, resolver, session, namespaceID, permission)
	if errors.Is(err, managementauthorization.ErrDenied) {
		return managementauthorization.ResultScope{NamespaceID: accesscontrol.NamespaceID(namespaceID)}, nil
	}
	return scope, err
}

func writeResultScopeError(response http.ResponseWriter, err error, requestID string) {
	if errors.Is(err, managementauthorization.ErrDenied) {
		writeProviderError(response, http.StatusForbidden, "forbidden", "Permission denied.", requestID)
		return
	}
	writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
}
