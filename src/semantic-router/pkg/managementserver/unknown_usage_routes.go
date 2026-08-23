package managementserver

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotareconciliation"
)

const unknownUsageFencesPath = managementapi.BasePath + "/unknown-usage-fences"

var unknownUsageETagPattern = regexp.MustCompile(`^"unknown-usage-fence:([1-9][0-9]*)"$`)

type UnknownUsageRoutes struct {
	service       UnknownUsageService
	namespaces    NamespaceResolver
	sessions      SessionAuthenticator
	authorization Authorizer
	scopes        ResultScopeResolver
	now           func() time.Time
	operations    map[string]managementapi.OperationContract
}

func NewUnknownUsageRoutes(options UnknownUsageRoutesOptions) (*UnknownUsageRoutes, error) {
	scopes := configuredResultScopes(options.Scopes, options.Authorization)
	if options.Service == nil || options.Namespaces == nil || options.Sessions == nil ||
		options.Authorization == nil || scopes == nil {
		return nil, errors.New("unknown-usage routes require service, namespace, session, authorization, and scope dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	routes := &UnknownUsageRoutes{
		service: options.Service, namespaces: options.Namespaces,
		sessions: options.Sessions, authorization: options.Authorization, scopes: scopes,
		now: now, operations: make(map[string]managementapi.OperationContract),
	}
	for _, contract := range []struct {
		method managementapi.HTTPMethod
		path   string
	}{
		{managementapi.MethodGET, unknownUsageFencesPath},
		{managementapi.MethodGET, unknownUsageFencesPath + "/{fenceId}"},
		{managementapi.MethodPOST, unknownUsageFencesPath + "/{fenceId}:reconcile"},
	} {
		operation, found := managementapi.LookupOperation(contract.method, contract.path)
		if !found {
			return nil, fmt.Errorf("unknown-usage operation contract %s %s is unavailable", contract.method, contract.path)
		}
		routes.operations[string(contract.method)+" "+contract.path] = operation
	}
	return routes, nil
}

func (routes *UnknownUsageRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("unknown-usage routes and mux are required")
	}
	mux.Handle("GET "+unknownUsageFencesPath, routes)
	mux.Handle("GET "+unknownUsageFencesPath+"/", routes)
	mux.Handle("POST "+unknownUsageFencesPath+"/", routes)
}

func (routes *UnknownUsageRoutes) Ready(ctx context.Context) error {
	return routes.service.Ready(ctx)
}

func (routes *UnknownUsageRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if request.URL.Path == unknownUsageFencesPath && request.Method == http.MethodGet {
		routes.list(response, request, requestID)
		return
	}
	identifier, reconcile, ok := unknownUsagePath(request.URL.Path)
	if !ok || (reconcile && request.Method != http.MethodPost) || (!reconcile && request.Method != http.MethodGet) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if reconcile {
		routes.reconcile(response, request, requestID, identifier)
	} else {
		routes.detail(response, request, requestID, identifier)
	}
}

func (routes *UnknownUsageRoutes) list(response http.ResponseWriter, request *http.Request, requestID string) {
	values := request.URL.Query()
	for key := range values {
		if key != "state" && key != "cursor" && key != "pageSize" {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Unknown-usage list query is invalid.", requestID)
			return
		}
	}
	pageSize := 0
	var err error
	if raw := values.Get("pageSize"); raw != "" {
		pageSize, err = strconv.Atoi(raw)
		if err != nil {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Unknown-usage list query is invalid.", requestID)
			return
		}
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	operation := routes.operations[string(managementapi.MethodGET)+" "+unknownUsageFencesPath]
	permission, valid := listPermission(operation)
	if !valid {
		writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
		return
	}
	scope, err := resolveListResultScope(request.Context(), routes.scopes, session, namespaceID, permission)
	if err != nil {
		writeResultScopeError(response, err, requestID)
		return
	}
	page, err := routes.service.List(request.Context(), quotareconciliation.ListRequest{
		NamespaceID: namespaceID, State: quotareconciliation.FenceState(values.Get("state")),
		Scope: scope, Cursor: values.Get("cursor"), PageSize: pageSize,
	})
	if err != nil {
		writeUnknownUsageError(response, err, requestID)
		return
	}
	items := make([]managementapi.UnknownUsageFence, len(page.Items))
	for index := range page.Items {
		items[index] = unknownUsageResponse(page.Items[index], false, false, false)
	}
	writeProviderJSON(response, http.StatusOK, managementapi.UnknownUsageFencePage{
		Data: items, Page: managementapi.PageInfo{NextCursor: page.NextCursor, HasMore: page.HasMore, PageSize: page.PageSize},
	}, requestID)
}

func (routes *UnknownUsageRoutes) detail(response http.ResponseWriter, request *http.Request, requestID, fenceID string) {
	includeInternal, includeEvidence, includeActor, ok := unknownUsageDetailOptions(response, request, requestID)
	if !ok {
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	fence, err := routes.service.Get(request.Context(), namespaceID, fenceID)
	if err != nil {
		writeUnknownUsageError(response, err, requestID)
		return
	}
	targets := unknownUsageTargets(fence)
	conditions := map[string]bool{
		"internal_usage_dimensions_requested":   includeInternal,
		"fence_payload_evidence_requested":      includeEvidence,
		"fence_actor_or_audit_fields_requested": includeActor,
	}
	operation := routes.operations[string(managementapi.MethodGET)+" "+unknownUsageFencesPath+"/{fenceId}"]
	_, err = routes.authorization.Authorize(request.Context(), AuthorizationRequest{
		Operation: operation, Session: session, NamespaceID: namespaceID,
		Targets: targets, Conditions: conditions,
	})
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, true)
		return
	}
	response.Header().Set(managementapi.HeaderETag, unknownUsageETag(fence.Revision))
	writeProviderJSON(response, http.StatusOK, managementapi.UnknownUsageFenceDetail{
		Data: unknownUsageResponse(fence, includeInternal, includeEvidence, includeActor),
	}, requestID)
}

func (routes *UnknownUsageRoutes) reconcile(response http.ResponseWriter, request *http.Request, requestID, fenceID string) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Unknown-usage reconciliation does not accept query parameters.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	idempotencyKey, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	revision, ok := requireUnknownUsageRevision(response, request, requestID)
	if !ok {
		return
	}
	var body managementapi.UnknownUsageReconcileRequest
	if !decodePolicyBody(response, request, requestID, &body) {
		return
	}
	fence, err := routes.service.Get(request.Context(), namespaceID, fenceID)
	if err != nil {
		writeUnknownUsageError(response, err, requestID)
		return
	}
	operation := routes.operations[string(managementapi.MethodPOST)+" "+unknownUsageFencesPath+"/{fenceId}:reconcile"]
	_, err = routes.authorization.Authorize(request.Context(), AuthorizationRequest{
		Operation: operation, Session: session, NamespaceID: namespaceID,
		Targets: unknownUsageTargets(fence), Conditions: map[string]bool{
			"fence_actual_reconciliation":      body.Strategy == string(quotareconciliation.StrategyActual),
			"fence_payload_evidence_requested": false,
		},
	})
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, true)
		return
	}
	result, err := routes.service.Reconcile(request.Context(), quotareconciliation.ReconcileRequest{
		NamespaceID: namespaceID, FenceID: fenceID, ExpectedRevision: revision,
		IdempotencyKey: string(idempotencyKey), Strategy: quotareconciliation.Strategy(body.Strategy),
		Actual: unknownUsageActual(body.Actual), EvidenceReferences: append([]string(nil), body.EvidenceReferences...),
		Reason: body.Reason, Actor: quotareconciliation.Actor{
			PrincipalID: session.Session.PrincipalID, ActorChain: []string{session.Session.PrincipalID},
			RequestID: requestID, SourceIP: directRequestIP(request),
		}, Session: session.Session,
	})
	if err != nil {
		writeUnknownUsageError(response, err, requestID)
		return
	}
	setIdempotencyReplayHeader(response, result.Replayed)
	response.Header().Set("Location", managementapi.BasePath+"/operations/"+result.Operation.ID)
	writeProviderJSON(response, http.StatusAccepted, unknownUsageOperation(result.Operation), requestID)
}

func (routes *UnknownUsageRoutes) authenticate(response http.ResponseWriter, request *http.Request, requestID string) (string, managementauth.AuthenticatedSession, bool) {
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

func unknownUsagePath(path string) (string, bool, bool) {
	value := strings.TrimPrefix(path, unknownUsageFencesPath+"/")
	if value == path || value == "" || strings.Contains(value, "/") {
		return "", false, false
	}
	reconcile := strings.HasSuffix(value, ":reconcile")
	if reconcile {
		value = strings.TrimSuffix(value, ":reconcile")
	}
	return value, reconcile, canonicalUUID(value)
}

func unknownUsageDetailOptions(response http.ResponseWriter, request *http.Request, requestID string) (bool, bool, bool, bool) {
	values := request.URL.Query()
	for key := range values {
		if key != "includeInternalDimensions" && key != "includeEvidence" && key != "includeActor" {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Unknown-usage detail query is invalid.", requestID)
			return false, false, false, false
		}
	}
	parsed := make([]bool, 3)
	for index, key := range []string{"includeInternalDimensions", "includeEvidence", "includeActor"} {
		if raw := values.Get(key); raw != "" {
			value, err := strconv.ParseBool(raw)
			if err != nil {
				writeProviderError(response, http.StatusBadRequest, "invalid_request", "Unknown-usage detail query is invalid.", requestID)
				return false, false, false, false
			}
			parsed[index] = value
		}
	}
	return parsed[0], parsed[1], parsed[2], true
}

func unknownUsageTargets(fence quotareconciliation.Fence) map[string][]accesscontrol.ScopedTarget {
	targets := make([]accesscontrol.ScopedTarget, 0, len(fence.Bindings))
	for _, binding := range fence.Bindings {
		ancestors := []accesscontrol.Scope{}
		switch binding.Subject.Kind {
		case accesscontrol.SubjectKindUser:
			ancestors = append(ancestors, accesscontrol.UserScope(accesscontrol.NamespaceID(fence.NamespaceID), accesscontrol.UserID(binding.Subject.ID)))
		case accesscontrol.SubjectKindTeam:
			ancestors = append(ancestors, accesscontrol.TeamScope(accesscontrol.NamespaceID(fence.NamespaceID), accesscontrol.TeamID(binding.Subject.ID)))
		case accesscontrol.SubjectKindAPIKey:
			ancestors = append(ancestors, accesscontrol.ResourceScope(accesscontrol.NamespaceID(fence.NamespaceID), accesscontrol.ScopeResourceAPIKey, accesscontrol.ResourceID(binding.Subject.ID)))
		}
		targets = append(targets, accesscontrol.ScopedTarget{
			Scope:     accesscontrol.ResourceScope(accesscontrol.NamespaceID(fence.NamespaceID), accesscontrol.ScopeResourceRateLimitBinding, accesscontrol.ResourceID(binding.BindingID)),
			Ancestors: ancestors,
		})
	}
	return map[string][]accesscontrol.ScopedTarget{
		"all_affected_bindings": targets,
		"all_dependencies":      targets,
		"attributed_subject":    targets,
	}
}

func requireUnknownUsageRevision(response http.ResponseWriter, request *http.Request, requestID string) (uint64, bool) {
	values := request.Header.Values(managementapi.HeaderIfMatch)
	if len(values) != 1 {
		writeProviderError(response, http.StatusPreconditionRequired, "precondition_required", "If-Match is required.", requestID)
		return 0, false
	}
	match := unknownUsageETagPattern.FindStringSubmatch(values[0])
	if len(match) != 2 {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return 0, false
	}
	revision, err := strconv.ParseUint(match[1], 10, 64)
	if err != nil || revision == 0 {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return 0, false
	}
	return revision, true
}

func unknownUsageETag(revision uint64) string {
	return `"unknown-usage-fence:` + strconv.FormatUint(revision, 10) + `"`
}

func writeUnknownUsageError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, quotareconciliation.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "Unknown-usage fence not found.", requestID)
	case errors.Is(err, quotareconciliation.ErrRevisionConflict):
		writeProviderError(response, http.StatusPreconditionFailed, "revision_conflict", "The unknown-usage fence changed.", requestID)
	case errors.Is(err, quotareconciliation.ErrResolved):
		writeProviderError(response, http.StatusConflict, "fence_resolved", "The unknown-usage fence is already resolved.", requestID)
	case errors.Is(err, quotareconciliation.ErrReconciliationConflict), errors.Is(err, managementcommand.ErrConflict):
		writeProviderError(response, http.StatusConflict, "reconciliation_conflict", "The unknown-usage fence has another reconciliation.", requestID)
	case errors.Is(err, quotareconciliation.ErrEvidenceConflict):
		writeProviderError(response, http.StatusConflict, "evidence_conflict", "Reconciliation evidence does not match the usage ledger.", requestID)
	case errors.Is(err, quotareconciliation.ErrWaiveDenied):
		writeProviderError(response, http.StatusForbidden, "assurance_required", "Stronger authentication is required.", requestID)
	case errors.Is(err, quotareconciliation.ErrInvalidRequest):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Unknown-usage request is invalid.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "quota_unavailable", "Quota reconciliation is unavailable.", requestID)
	}
}

var _ RouteRegistrar = (*UnknownUsageRoutes)(nil)
