package managementserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
)

const (
	agentProfilesPath        = managementapi.BasePath + "/agent-profiles"
	agentSkillsPath          = managementapi.BasePath + "/agent-skills"
	agentToolsPath           = managementapi.BasePath + "/agent-tools"
	agentCredentialsPath     = managementapi.BasePath + "/agent-tool-credentials"
	agentSourcesPath         = managementapi.BasePath + "/agent-tool-sources"
	agentSessionsPath        = managementapi.BasePath + "/agent-sessions"
	agentArtifactsPath       = managementapi.BasePath + "/agent-artifacts"
	agentPublicationPlanPath = managementapi.BasePath + "/publication-plans"

	maximumAgentBodyBytes  = 3 << 20
	maximumAgentQueryBytes = 16 << 10
	defaultAgentPageSize   = 50
)

var agentETagPattern = regexp.MustCompile(`^"agent:([1-9][0-9]*)"$`)

type AgentRoutes struct {
	service       *agentmanagement.Service
	defaults      AgentDefaults
	publications  AgentPublicationCommitter
	liveEvents    agentmanagement.LiveEventSubscriber
	namespaces    NamespaceResolver
	sessions      SessionAuthenticator
	authorization Authorizer
	scopes        ResultScopeResolver
	now           func() time.Time
	operations    map[string]managementapi.OperationContract
}

func NewAgentRoutes(options AgentRoutesOptions) (*AgentRoutes, error) {
	scopes := configuredResultScopes(options.Scopes, options.Authorization)
	if options.Service == nil || options.Defaults == nil || options.Publications == nil ||
		options.LiveEvents == nil ||
		options.Namespaces == nil || options.Sessions == nil || options.Authorization == nil || scopes == nil {
		return nil, errors.New("agent Management routes require service, defaults, publication, live events, identity, and authorization dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	routes := &AgentRoutes{
		service: options.Service, defaults: options.Defaults, publications: options.Publications,
		liveEvents: options.LiveEvents,
		namespaces: options.Namespaces, sessions: options.Sessions,
		authorization: options.Authorization, scopes: scopes, now: now,
		operations: make(map[string]managementapi.OperationContract),
	}
	for _, contract := range agentHTTPContracts() {
		operation, found := managementapi.LookupOperation(contract.method, contract.path)
		if !found {
			return nil, fmt.Errorf("agent Management operation contract %s %s is unavailable", contract.method, contract.path)
		}
		routes.operations[agentOperationKey(contract.method, contract.path)] = operation
	}
	return routes, nil
}

func (routes *AgentRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Agent Management routes and mux are required")
	}
	for _, pattern := range []string{
		"GET " + agentProfilesPath, "POST " + agentProfilesPath,
		"GET " + agentProfilesPath + "/", "PATCH " + agentProfilesPath + "/", "DELETE " + agentProfilesPath + "/",
		"GET " + agentSkillsPath, "POST " + agentSkillsPath,
		"GET " + agentSkillsPath + "/", "PATCH " + agentSkillsPath + "/", "DELETE " + agentSkillsPath + "/",
		"GET " + agentToolsPath,
		"GET " + agentCredentialsPath, "POST " + agentCredentialsPath,
		"GET " + agentCredentialsPath + "/", "PATCH " + agentCredentialsPath + "/", "DELETE " + agentCredentialsPath + "/", "POST " + agentCredentialsPath + "/",
		"GET " + agentSourcesPath, "POST " + agentSourcesPath,
		"GET " + agentSourcesPath + "/", "PATCH " + agentSourcesPath + "/", "DELETE " + agentSourcesPath + "/", "POST " + agentSourcesPath + "/",
		"GET " + agentSessionsPath, "POST " + agentSessionsPath,
		"GET " + agentSessionsPath + "/", "PATCH " + agentSessionsPath + "/", "DELETE " + agentSessionsPath + "/", "POST " + agentSessionsPath + "/",
		"GET " + agentArtifactsPath + "/",
		"POST " + agentPublicationPlanPath + "/",
	} {
		mux.Handle(pattern, routes)
	}
}

func (routes *AgentRoutes) Ready(ctx context.Context) error {
	if routes == nil || routes.service == nil || routes.defaults == nil || routes.publications == nil {
		return errors.New("agent Management routes are unavailable")
	}
	if err := routes.service.Ready(ctx); err != nil {
		return err
	}
	if err := routes.defaults.Ready(ctx); err != nil {
		return err
	}
	return routes.publications.Ready(ctx)
}

func (routes *AgentRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	switch {
	case request.URL.Path == agentProfilesPath:
		routes.profiles(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, agentProfilesPath+"/"):
		routes.profile(response, request, requestID)
	case request.URL.Path == agentSkillsPath:
		routes.skills(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, agentSkillsPath+"/"):
		routes.skill(response, request, requestID)
	case request.URL.Path == agentToolsPath:
		routes.tools(response, request, requestID)
	case request.URL.Path == agentCredentialsPath:
		routes.credentials(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, agentCredentialsPath+"/"):
		routes.credential(response, request, requestID)
	case request.URL.Path == agentSourcesPath:
		routes.sources(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, agentSourcesPath+"/"):
		routes.source(response, request, requestID)
	case request.URL.Path == agentSessionsPath:
		routes.sessionsCollection(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, agentSessionsPath+"/"):
		routes.sessionResource(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, agentArtifactsPath+"/"):
		routes.artifact(response, request, requestID)
	case strings.HasPrefix(request.URL.Path, agentPublicationPlanPath+"/"):
		routes.commitPublication(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

type agentHTTPContract struct {
	method managementapi.HTTPMethod
	path   string
}

func agentHTTPContracts() []agentHTTPContract {
	return []agentHTTPContract{
		{managementapi.MethodGET, agentProfilesPath},
		{managementapi.MethodPOST, agentProfilesPath},
		{managementapi.MethodGET, agentProfilesPath + "/{profile}"},
		{managementapi.MethodPATCH, agentProfilesPath + "/{profile}"},
		{managementapi.MethodDELETE, agentProfilesPath + "/{profile}"},
		{managementapi.MethodGET, agentSkillsPath},
		{managementapi.MethodPOST, agentSkillsPath},
		{managementapi.MethodGET, agentSkillsPath + "/{skill}"},
		{managementapi.MethodPATCH, agentSkillsPath + "/{skill}"},
		{managementapi.MethodDELETE, agentSkillsPath + "/{skill}"},
		{managementapi.MethodGET, agentToolsPath},
		{managementapi.MethodGET, agentCredentialsPath},
		{managementapi.MethodPOST, agentCredentialsPath},
		{managementapi.MethodGET, agentCredentialsPath + "/{credential}"},
		{managementapi.MethodPATCH, agentCredentialsPath + "/{credential}"},
		{managementapi.MethodDELETE, agentCredentialsPath + "/{credential}"},
		{managementapi.MethodPOST, agentCredentialsPath + "/{credential}:rotate"},
		{managementapi.MethodGET, agentSourcesPath},
		{managementapi.MethodPOST, agentSourcesPath},
		{managementapi.MethodGET, agentSourcesPath + "/{source}"},
		{managementapi.MethodPATCH, agentSourcesPath + "/{source}"},
		{managementapi.MethodDELETE, agentSourcesPath + "/{source}"},
		{managementapi.MethodPOST, agentSourcesPath + "/{source}:test"},
		{managementapi.MethodPOST, agentSourcesPath + "/{source}:approve"},
		{managementapi.MethodGET, agentSessionsPath},
		{managementapi.MethodPOST, agentSessionsPath},
		{managementapi.MethodGET, agentSessionsPath + "/{session}"},
		{managementapi.MethodPATCH, agentSessionsPath + "/{session}"},
		{managementapi.MethodDELETE, agentSessionsPath + "/{session}"},
		{managementapi.MethodPOST, agentSessionsPath + "/{session}/turns"},
		{managementapi.MethodGET, agentSessionsPath + "/{session}/turns"},
		{managementapi.MethodGET, agentSessionsPath + "/{session}/events"},
		{managementapi.MethodPOST, agentSessionsPath + "/{session}/turns/{turn}:cancel"},
		{managementapi.MethodGET, agentArtifactsPath + "/{artifact}"},
		{managementapi.MethodGET, agentArtifactsPath + "/{artifact}/content"},
		{managementapi.MethodPOST, agentPublicationPlanPath + "/{plan}:commit"},
	}
}

func (routes *AgentRoutes) authenticate(
	response http.ResponseWriter, request *http.Request, requestID string,
) (agentAuthenticatedRequest, bool) {
	namespaceID, err := routes.namespaces.ResolveNamespace(request.Context(), request)
	if err != nil || uuid.Validate(namespaceID) != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_namespace", "A valid namespace is required.", requestID)
		return agentAuthenticatedRequest{}, false
	}
	token, ok := bearerToken(request)
	if !ok {
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Authentication is required.", requestID)
		return agentAuthenticatedRequest{}, false
	}
	session, err := routes.sessions.Authenticate(request.Context(), token, namespaceID, routes.now().UTC())
	if err != nil {
		status, code, message := http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable."
		if errors.Is(err, managementauth.ErrAuthenticationDenied) {
			status, code, message = http.StatusUnauthorized, "unauthenticated", "Authentication is required."
		}
		writeProviderError(response, status, code, message, requestID)
		return agentAuthenticatedRequest{}, false
	}
	if session.NamespaceID != namespaceID || uuid.Validate(session.Session.PrincipalID) != nil {
		writeProviderError(response, http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable.", requestID)
		return agentAuthenticatedRequest{}, false
	}
	return agentAuthenticatedRequest{NamespaceID: namespaceID, Session: session}, true
}

func (routes *AgentRoutes) authorize(
	ctx context.Context, authenticated agentAuthenticatedRequest,
	operation managementapi.OperationContract, targets map[string][]accesscontrol.ScopedTarget,
) (string, error) {
	decision, err := routes.authorization.Authorize(ctx, AuthorizationRequest{
		Operation: operation, Session: authenticated.Session, NamespaceID: authenticated.NamespaceID,
		Targets: targets,
	})
	return decision.AuthorityDigest, err
}

func (routes *AgentRoutes) accessContext(
	ctx context.Context, authenticated agentAuthenticatedRequest,
	operation managementapi.OperationContract,
) (agentmanagement.AccessContext, error) {
	permission, ok := listPermission(operation)
	if !ok {
		return agentmanagement.AccessContext{}, managementauthorization.ErrInvalidContext
	}
	scope, err := resolveListResultScope(
		ctx, routes.scopes, authenticated.Session, authenticated.NamespaceID, permission,
	)
	return agentmanagement.AccessContext{
		PrincipalID: authenticated.Session.Session.PrincipalID, Scope: scope,
	}, err
}

func (routes *AgentRoutes) operation(method managementapi.HTTPMethod, path string) managementapi.OperationContract {
	return routes.operations[agentOperationKey(method, path)]
}

func agentOperationKey(method managementapi.HTTPMethod, path string) string {
	return string(method) + " " + path
}

func agentTarget(namespaceID string, resourceType accesscontrol.ScopeResourceType, id string) map[string][]accesscontrol.ScopedTarget {
	return map[string][]accesscontrol.ScopedTarget{"target": {{Scope: accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), resourceType, accesscontrol.ResourceID(id),
	)}}}
}

func agentNamespaceTarget(namespaceID string) map[string][]accesscontrol.ScopedTarget {
	return map[string][]accesscontrol.ScopedTarget{"target": {{Scope: accesscontrol.NamespaceScope(
		accesscontrol.NamespaceID(namespaceID),
	)}}}
}

func agentMutation(request *http.Request, authenticated agentAuthenticatedRequest, requestID string) agentmanagement.MutationContext {
	return agentmanagement.MutationContext{
		PrincipalID:         authenticated.Session.Session.PrincipalID,
		ManagementSessionID: authenticated.Session.Session.ID,
		ActorChain:          []string{authenticated.Session.Session.PrincipalID},
		RequestID:           requestID, SourceIP: directRequestIP(request),
	}
}

func parseAgentListQuery(request *http.Request) (agentmanagement.PageRequest, error) {
	return parseAgentPageQuery(request, false)
}

func parseAgentSearchListQuery(request *http.Request) (agentmanagement.PageRequest, error) {
	return parseAgentPageQuery(request, true)
}

func parseAgentPageQuery(request *http.Request, searchable bool) (agentmanagement.PageRequest, error) {
	allowed := map[string]bool{"cursor": true, "pageSize": true}
	if searchable {
		allowed["search"] = true
	}
	values, err := strictAgentQuery(request.URL.RawQuery, allowed)
	if err != nil {
		return agentmanagement.PageRequest{}, err
	}
	pageSize, err := parseOptionalPageSize(values.Get("pageSize"))
	if err != nil {
		return agentmanagement.PageRequest{}, err
	}
	if pageSize == 0 {
		pageSize = defaultAgentPageSize
	}
	return agentmanagement.PageRequest{
		Cursor: values.Get("cursor"), PageSize: pageSize, Search: values.Get("search"),
	}, nil
}

func parseAgentToolListQuery(request *http.Request) (agentmanagement.ToolPageRequest, error) {
	values, err := strictAgentQuery(request.URL.RawQuery, map[string]bool{
		"cursor": true, "pageSize": true, "search": true,
	})
	if err != nil {
		return agentmanagement.ToolPageRequest{}, err
	}
	pageSize, err := parseOptionalPageSize(values.Get("pageSize"))
	if err != nil {
		return agentmanagement.ToolPageRequest{}, err
	}
	if pageSize == 0 {
		pageSize = defaultAgentPageSize
	}
	return agentmanagement.ToolPageRequest{
		Cursor: values.Get("cursor"), PageSize: pageSize, Search: values.Get("search"),
	}, nil
}

func strictAgentQuery(raw string, allowed map[string]bool) (url.Values, error) {
	if len(raw) > maximumAgentQueryBytes {
		return nil, agentmanagement.ErrInvalid
	}
	values, err := url.ParseQuery(raw)
	if err != nil {
		return nil, err
	}
	for name, entries := range values {
		if !allowed[name] || len(entries) != 1 {
			return nil, agentmanagement.ErrInvalid
		}
	}
	return values, nil
}

func decodeAgentBody(response http.ResponseWriter, request *http.Request, requestID string, target any) bool {
	if request.ContentLength > maximumAgentBodyBytes {
		writeProviderError(response, http.StatusRequestEntityTooLarge, "invalid_request", "Request body is too large.", requestID)
		return false
	}
	mediaType, parameters, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != managementapi.JSONMediaType ||
		(len(parameters) != 0 && (len(parameters) != 1 || !strings.EqualFold(parameters["charset"], "utf-8"))) {
		writeProviderError(response, http.StatusUnsupportedMediaType, "unsupported_media_type", "Use the Management API media type.", requestID)
		return false
	}
	request.Body = http.MaxBytesReader(response, request.Body, maximumAgentBodyBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request body is invalid.", requestID)
		return false
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request body is invalid.", requestID)
		return false
	}
	return true
}

func requireAgentRevision(response http.ResponseWriter, request *http.Request, requestID string) (int64, bool) {
	match := agentETagPattern.FindStringSubmatch(request.Header.Get(managementapi.HeaderIfMatch))
	if len(match) != 2 {
		writeProviderError(response, http.StatusPreconditionRequired, "precondition_required", "If-Match is required.", requestID)
		return 0, false
	}
	revision, err := strconv.ParseInt(match[1], 10, 64)
	if err != nil || revision < 1 {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "If-Match is invalid.", requestID)
		return 0, false
	}
	return revision, true
}

func setAgentETag(response http.ResponseWriter, revision int64) {
	response.Header().Set(managementapi.HeaderETag, fmt.Sprintf(`"agent:%d"`, revision))
}

func requireAgentIdempotency(response http.ResponseWriter, request *http.Request, requestID string) (string, bool) {
	key, ok := requireIdempotencyKey(response, request, requestID)
	return string(key), ok
}

func newAgentPage[T any](page agentmanagement.Page[T], pageSize int) agentPage[T] {
	return agentPage[T]{Data: page.Items, Page: agentPageInfo{
		NextCursor: page.NextCursor, HasMore: page.HasMore, PageSize: pageSize,
	}}
}

func writeAgentDomainError(response http.ResponseWriter, err error, requestID string) {
	var expired agentmanagement.HistoryExpiredError
	switch {
	case errors.Is(err, agentmanagement.ErrInvalid):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request is invalid.", requestID)
	case errors.Is(err, agentmanagement.ErrNotFound), errors.Is(err, agentmanagement.ErrDenied):
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	case errors.Is(err, agentmanagement.ErrConflict):
		writeProviderError(response, http.StatusConflict, "conflict", "Resource state changed. Refresh and retry.", requestID)
	case errors.Is(err, agentmanagement.ErrApproval):
		writeProviderError(response, http.StatusPreconditionFailed, "approval_invalid", "Publication approval is no longer valid.", requestID)
	case errors.Is(err, agentmanagement.ErrToolUnavailable):
		writeProviderError(response, http.StatusServiceUnavailable, "tool_unavailable", "A required Agent tool is unavailable.", requestID)
	case errors.As(err, &expired):
		writeProviderJSON(response, http.StatusGone, map[string]any{
			"error":    map[string]any{"code": "history_expired", "message": "Event history has expired."},
			"recovery": expired.Recovery,
		}, requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "agent_unavailable", "Agent service is unavailable.", requestID)
	}
}

func writeAgentAuthorizationError(response http.ResponseWriter, err error, requestID string) {
	if errors.Is(err, managementauthorization.ErrDenied) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
}
