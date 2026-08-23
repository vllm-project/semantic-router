package managementserver

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

const (
	routingModelsPath      = managementapi.BasePath + "/routing/models"
	routingModelCardsPath  = managementapi.BasePath + "/routing/model-cards"
	routingRecipesPath     = managementapi.BasePath + "/routing/recipes"
	routingEntrypointsPath = managementapi.BasePath + "/routing/entrypoints"
	routingSnapshotsPath   = managementapi.BasePath + "/namespaces/{namespaceId}/routing/snapshots"

	maximumRoutingBodyBytes  = 3 << 20
	maximumRoutingQueryBytes = 16 << 10
)

type RoutingRoutes struct {
	service        RoutingManagementService
	commands       *managementcommand.Codec
	commandResults RoutingCommandResults
	namespaces     NamespaceResolver
	sessions       SessionAuthenticator
	authorization  Authorizer
	scopes         ResultScopeResolver
	idempotencyTTL time.Duration
	now            func() time.Time
	operations     map[string]managementapi.OperationContract
}

func NewRoutingRoutes(options RoutingRoutesOptions) (*RoutingRoutes, error) {
	scopes := configuredResultScopes(options.Scopes, options.Authorization)
	if options.Service == nil || options.Commands == nil || options.CommandResults == nil ||
		options.Namespaces == nil || options.Sessions == nil || options.Authorization == nil || scopes == nil {
		return nil, fmt.Errorf("routing Management routes require service, command, namespace, session, authorization, and result-scope dependencies")
	}
	if options.IdempotencyTTL < time.Minute || options.IdempotencyTTL > 7*24*time.Hour {
		return nil, fmt.Errorf("routing Management idempotency TTL must be between 1m and 7d")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	routes := &RoutingRoutes{
		service: options.Service, commands: options.Commands, commandResults: options.CommandResults,
		namespaces: options.Namespaces, sessions: options.Sessions, authorization: options.Authorization,
		scopes:         scopes,
		idempotencyTTL: options.IdempotencyTTL, now: now,
		operations: make(map[string]managementapi.OperationContract),
	}
	for _, operation := range routingHTTPContracts() {
		contract, found := managementapi.LookupOperation(operation.method, operation.path)
		if !found {
			return nil, fmt.Errorf("routing Management operation contract %s %s is unavailable", operation.method, operation.path)
		}
		routes.operations[routingOperationKey(operation.method, operation.path)] = contract
	}
	return routes, nil
}

func (routes *RoutingRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Routing Management routes and mux are required")
	}
	for _, pattern := range []string{
		"GET " + routingModelCardsPath,
		"GET " + routingModelsPath, "POST " + routingModelsPath,
		"POST " + routingModelsPath + ":bulk-import",
		"GET " + routingModelsPath + "/", "PATCH " + routingModelsPath + "/",
		"DELETE " + routingModelsPath + "/", "POST " + routingModelsPath + "/",
		"GET " + routingRecipesPath, "POST " + routingRecipesPath,
		"GET " + routingRecipesPath + "/", "PATCH " + routingRecipesPath + "/",
		"DELETE " + routingRecipesPath + "/",
		"GET " + routingEntrypointsPath, "POST " + routingEntrypointsPath,
		"GET " + routingEntrypointsPath + "/", "PATCH " + routingEntrypointsPath + "/",
		"DELETE " + routingEntrypointsPath + "/", "POST " + routingEntrypointsPath + "/",
		"GET " + routingSnapshotsPath,
		"GET " + routingSnapshotsPath + "/{routingRevision}",
	} {
		mux.Handle(pattern, routes)
	}
}

func (routes *RoutingRoutes) Ready(ctx context.Context) error {
	if routes == nil || routes.commandResults == nil || routes.commands == nil {
		return errors.New("routing Management routes are unavailable")
	}
	return routes.commandResults.Ready(ctx, routes.commands)
}

func (routes *RoutingRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	switch {
	case request.Method == http.MethodGet && request.URL.Path == routingModelCardsPath:
		routes.listModelCards(response, request, requestID)
	case request.Method == http.MethodGet && request.URL.Path == routingModelsPath:
		routes.listModels(response, request, requestID)
	case request.Method == http.MethodPost && request.URL.Path == routingModelsPath:
		routes.createModel(response, request, requestID)
	case request.Method == http.MethodPost && request.URL.Path == routingModelsPath+":bulk-import":
		routes.bulkImportModels(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, routingModelsPath+"/"):
		routes.modelResource(response, request, requestID)
	case request.Method == http.MethodGet && request.URL.Path == routingRecipesPath:
		routes.listRecipes(response, request, requestID)
	case request.Method == http.MethodPost && request.URL.Path == routingRecipesPath:
		routes.createRecipe(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, routingRecipesPath+"/"):
		routes.recipeResource(response, request, requestID)
	case request.Method == http.MethodGet && request.URL.Path == routingEntrypointsPath:
		routes.listEntrypoints(response, request, requestID)
	case request.Method == http.MethodPost && request.URL.Path == routingEntrypointsPath:
		routes.createEntrypoint(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, routingEntrypointsPath+"/"):
		routes.entrypointResource(response, request, requestID)
	case request.PathValue("namespaceId") != "" &&
		strings.Contains(request.URL.Path, "/routing/snapshots"):
		routes.snapshotResource(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *RoutingRoutes) authenticate(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
) (string, managementauth.AuthenticatedSession, bool) {
	namespaceID, err := routes.namespaces.ResolveNamespace(request.Context(), request)
	if err != nil || !canonicalUUID(namespaceID) {
		writeProviderError(response, http.StatusBadRequest, "invalid_namespace", "A valid namespace is required.", requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	token, ok := bearerToken(request)
	if !ok {
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Authentication is required.", requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	session, err := routes.sessions.Authenticate(request.Context(), token, namespaceID, routes.now().UTC())
	if err != nil {
		status, code, message := http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable."
		if errors.Is(err, managementauth.ErrAuthenticationDenied) {
			status, code, message = http.StatusUnauthorized, "unauthenticated", "Authentication is required."
		}
		writeProviderError(response, status, code, message, requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	if session.NamespaceID != namespaceID || !canonicalUUID(session.Session.PrincipalID) {
		writeProviderError(response, http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable.", requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	return namespaceID, session, true
}

func (routes *RoutingRoutes) authorize(
	ctx context.Context,
	session managementauth.AuthenticatedSession,
	namespaceID string,
	operation managementapi.OperationContract,
	targets map[string][]accesscontrol.ScopedTarget,
	conditions map[string]bool,
) (AuthorizationDecision, error) {
	return routes.authorization.Authorize(ctx, AuthorizationRequest{
		Operation: operation, Session: session, NamespaceID: namespaceID,
		Targets: targets, Conditions: conditions,
	})
}

func writeRoutingAuthorizationError(response http.ResponseWriter, err error, requestID string, nondisclosing bool) {
	if errors.Is(err, managementauthorization.ErrDenied) {
		if nondisclosing {
			writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		} else {
			writeProviderError(response, http.StatusForbidden, "forbidden", "Permission denied.", requestID)
		}
		return
	}
	writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
}

func (routes *RoutingRoutes) operation(method managementapi.HTTPMethod, path string) managementapi.OperationContract {
	return routes.operations[routingOperationKey(method, path)]
}

func routingOperationKey(method managementapi.HTTPMethod, path string) string {
	return string(method) + " " + path
}

type routingHTTPContract struct {
	method managementapi.HTTPMethod
	path   string
}

func routingHTTPContracts() []routingHTTPContract {
	return []routingHTTPContract{
		{managementapi.MethodGET, routingModelCardsPath},
		{managementapi.MethodGET, routingModelsPath},
		{managementapi.MethodPOST, routingModelsPath},
		{managementapi.MethodPOST, routingModelsPath + ":bulk-import"},
		{managementapi.MethodGET, routingModelsPath + "/{modelId}"},
		{managementapi.MethodPATCH, routingModelsPath + "/{modelId}"},
		{managementapi.MethodDELETE, routingModelsPath + "/{modelId}"},
		{managementapi.MethodPOST, routingModelsPath + "/{modelId}:probe"},
		{managementapi.MethodGET, routingRecipesPath},
		{managementapi.MethodPOST, routingRecipesPath},
		{managementapi.MethodGET, routingRecipesPath + "/{recipeId}"},
		{managementapi.MethodPATCH, routingRecipesPath + "/{recipeId}"},
		{managementapi.MethodDELETE, routingRecipesPath + "/{recipeId}"},
		{managementapi.MethodGET, routingEntrypointsPath},
		{managementapi.MethodPOST, routingEntrypointsPath},
		{managementapi.MethodGET, routingEntrypointsPath + "/{entrypointId}"},
		{managementapi.MethodPATCH, routingEntrypointsPath + "/{entrypointId}"},
		{managementapi.MethodDELETE, routingEntrypointsPath + "/{entrypointId}"},
		{managementapi.MethodPOST, routingEntrypointsPath + "/{entrypointId}:publish"},
		{managementapi.MethodPOST, routingEntrypointsPath + "/{entrypointId}:unpublish"},
		{managementapi.MethodPOST, routingEntrypointsPath + "/{entrypointId}:resolve"},
		{managementapi.MethodGET, routingSnapshotsPath},
		{managementapi.MethodGET, routingSnapshotsPath + "/{routingRevision}"},
	}
}

func routingTarget(namespaceID string, resourceType accesscontrol.ScopeResourceType, resourceID string) accesscontrol.ScopedTarget {
	return accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), resourceType, accesscontrol.ResourceID(resourceID),
	)}
}

func routingNamespaceTarget(namespaceID string) accesscontrol.ScopedTarget {
	return accesscontrol.ScopedTarget{Scope: accesscontrol.NamespaceScope(accesscontrol.NamespaceID(namespaceID))}
}

func routingResourcePathValue(basePath, path string) (string, string, bool) {
	value := strings.TrimPrefix(path, basePath+"/")
	if value == path || value == "" || strings.Contains(value, "/") {
		return "", "", false
	}
	resourceID, action, hasAction := strings.Cut(value, ":")
	if routingmanagement.ValidateResourceID(resourceID) != nil || (hasAction && action == "") {
		return "", "", false
	}
	return resourceID, action, true
}
