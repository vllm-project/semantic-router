package managementserver

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/auditlog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

const (
	usagePath       = managementapi.BasePath + "/usage"
	requestLogsPath = managementapi.BasePath + "/request-logs"
	auditEventsPath = managementapi.BasePath + "/audit-events"

	defaultObservabilityRange    = 24 * time.Hour
	maximumObservabilityRange    = 5 * 366 * 24 * time.Hour
	defaultObservabilityPageSize = 50
)

type ObservabilityRoutes struct {
	queries                UsageQueryService
	logCursors             *usageledger.LogCursorCodec
	audit                  AuditQueryService
	auditCursors           *auditlog.CursorCodec
	resources              UsageResourceReader
	authorization          Authorizer
	scopes                 ResultScopeResolver
	subjectUsageOperations map[subjectUsageResourceKind]managementapi.OperationContract
	namespaces             NamespaceResolver
	sessions               SessionAuthenticator
	now                    func() time.Time
	maximumRange           time.Duration
	defaultRange           time.Duration
	defaultPageSize        int
}

func NewObservabilityRoutes(options ObservabilityRoutesOptions) (*ObservabilityRoutes, error) {
	if options.Queries == nil || options.LogCursors == nil || options.Audit == nil || options.AuditCursors == nil ||
		options.Resources == nil || options.Authorization == nil || options.Scopes == nil ||
		options.Namespaces == nil || options.Sessions == nil {
		return nil, errors.New("observability Management routes require query, cursor, scope, namespace, and session dependencies")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	maximumRange := options.MaximumRange
	if maximumRange == 0 {
		maximumRange = maximumObservabilityRange
	}
	defaultRange := options.DefaultRange
	if defaultRange == 0 {
		defaultRange = defaultObservabilityRange
	}
	pageSize := options.DefaultPageSize
	if pageSize == 0 {
		pageSize = defaultObservabilityPageSize
	}
	if maximumRange < time.Minute || defaultRange < time.Minute || defaultRange > maximumRange || pageSize < 1 || pageSize > 200 {
		return nil, errors.New("observability Management route bounds are invalid")
	}
	subjectUsageOperations, err := loadSubjectUsageOperations()
	if err != nil {
		return nil, err
	}
	return &ObservabilityRoutes{
		queries: options.Queries, logCursors: options.LogCursors,
		audit: options.Audit, auditCursors: options.AuditCursors,
		resources: options.Resources, authorization: options.Authorization, scopes: options.Scopes,
		subjectUsageOperations: subjectUsageOperations,
		namespaces:             options.Namespaces, sessions: options.Sessions, now: now,
		maximumRange: maximumRange, defaultRange: defaultRange, defaultPageSize: pageSize,
	}, nil
}

func (routes *ObservabilityRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Observability Management routes and mux are required")
	}
	mux.Handle("GET "+usagePath, routes)
	mux.Handle("GET "+usagePath+"/series", routes)
	mux.Handle("GET "+usagePath+"/breakdowns", routes)
	mux.Handle("GET "+requestLogsPath, routes)
	mux.Handle("GET "+auditEventsPath, routes)
	mux.Handle("GET "+managementapi.BasePath+"/namespaces/{namespaceId}/request-logs/{admissionId}", routes)
	mux.Handle("GET "+managementapi.BasePath+"/users/{userId}/usage", routes)
	mux.Handle("GET "+managementapi.BasePath+"/teams/{teamId}/usage", routes)
	mux.Handle("GET "+managementapi.BasePath+"/api-keys/{keyId}/usage", routes)
}

func (routes *ObservabilityRoutes) Ready(context.Context) error {
	if routes == nil || routes.queries == nil || routes.logCursors == nil || routes.audit == nil ||
		routes.auditCursors == nil || routes.resources == nil || routes.authorization == nil || routes.scopes == nil {
		return errors.New("observability Management routes are unavailable")
	}
	return nil
}

func (routes *ObservabilityRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	switch request.URL.Path {
	case usagePath:
		routes.usageSummary(response, request, requestID)
	case usagePath + "/series":
		routes.usageSeries(response, request, requestID)
	case usagePath + "/breakdowns":
		routes.usageBreakdown(response, request, requestID)
	case requestLogsPath:
		routes.requestLogs(response, request, requestID)
	case auditEventsPath:
		routes.auditEvents(response, request, requestID)
	default:
		if resource, ok := parseSubjectUsagePath(request.URL.Path); ok {
			routes.subjectUsageSummary(response, request, requestID, resource)
		} else {
			routes.requestLogDetail(response, request, requestID)
		}
	}
}

func (routes *ObservabilityRoutes) auditEvents(response http.ResponseWriter, request *http.Request, requestID string) {
	values, err := strictProviderQuery(request.URL.RawQuery, map[string]bool{
		"start": true, "end": true, "timeZone": true, "cursor": true, "pageSize": true,
		"actorPrincipalId": true, "action": true, "resourceType": true,
		"resourceId": true, "outcome": true, "requestId": true,
	})
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Audit query is invalid.", requestID)
		return
	}
	namespaceID, _, visibility, ok := routes.authorizedVisibility(
		response, request, requestID, accesscontrol.PermissionAuditRead,
	)
	if !ok {
		return
	}
	if !visibility.All {
		// Audit events are intentionally not reconstructed from mutable ownership
		// joins. Narrow audit authority fails closed until an event carries a
		// durable attribution dimension that can be pushed into this query.
		writeProviderError(response, http.StatusForbidden, "forbidden", "Permission denied.", requestID)
		return
	}
	start, end, ok := routes.timeRange(response, values, requestID)
	if !ok {
		return
	}
	pageSize, err := parseOptionalPageSize(values.Get("pageSize"))
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "pageSize must be between 1 and 200.", requestID)
		return
	}
	if pageSize == 0 {
		pageSize = routes.defaultPageSize
	}
	page, err := routes.audit.List(request.Context(), auditlog.Query{
		NamespaceID: namespaceID, Start: start, End: end, PageSize: pageSize,
		Cursor: values.Get("cursor"), Filters: auditlog.Filters{
			ActorPrincipalID: values.Get("actorPrincipalId"), Action: values.Get("action"),
			ResourceType: values.Get("resourceType"), ResourceID: values.Get("resourceId"),
			Outcome: values.Get("outcome"), RequestID: values.Get("requestId"),
		},
	}, routes.auditCursors)
	if err != nil {
		if errors.Is(err, auditlog.ErrInvalidQuery) {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Audit query is invalid.", requestID)
		} else {
			writeProviderError(response, http.StatusServiceUnavailable, "audit_unavailable", "Audit data is unavailable.", requestID)
		}
		return
	}
	writeProviderJSON(response, http.StatusOK, auditEventPageResponse{
		Data: page.Items,
		Page: managementapi.PageInfo{NextCursor: page.NextCursor, HasMore: page.NextCursor != "", PageSize: pageSize},
	}, requestID)
}

func (routes *ObservabilityRoutes) usageSummary(response http.ResponseWriter, request *http.Request, requestID string) {
	query, _, ok := routes.authorizedUsageQuery(response, request, requestID, false)
	if !ok {
		return
	}
	result, err := routes.queries.Summary(request.Context(), query)
	if err != nil {
		writeObservabilityError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, result, requestID)
}

func (routes *ObservabilityRoutes) usageSeries(response http.ResponseWriter, request *http.Request, requestID string) {
	query, _, ok := routes.authorizedUsageQuery(response, request, requestID, false)
	if !ok {
		return
	}
	result, err := routes.queries.Series(request.Context(), query)
	if err != nil {
		writeObservabilityError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, result, requestID)
}

func (routes *ObservabilityRoutes) usageBreakdown(response http.ResponseWriter, request *http.Request, requestID string) {
	query, values, ok := routes.authorizedUsageQuery(response, request, requestID, true)
	if !ok {
		return
	}
	dimension := usageledger.BreakdownDimension(values.Get("dimension"))
	if dimension == "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "A breakdown dimension is required.", requestID)
		return
	}
	limit, err := parseOptionalPageSize(values.Get("pageSize"))
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "pageSize must be between 1 and 200.", requestID)
		return
	}
	if limit == 0 {
		limit = routes.defaultPageSize
	}
	if isInternalBreakdown(dimension) && !routes.authorizeInternalDimensions(response, request, requestID, query) {
		return
	}
	result, err := routes.queries.Breakdown(request.Context(), usageledger.BreakdownQuery{
		UsageQuery: query, Dimension: dimension, Limit: limit,
	})
	if err != nil {
		writeObservabilityError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, result, requestID)
}

func (routes *ObservabilityRoutes) requestLogs(response http.ResponseWriter, request *http.Request, requestID string) {
	values, err := strictProviderQuery(request.URL.RawQuery, observabilityQueryKeys(true, false))
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request-log query is invalid.", requestID)
		return
	}
	namespaceID, session, visibility, ok := routes.authorizedVisibility(
		response, request, requestID, accesscontrol.PermissionLogRead,
	)
	if !ok {
		return
	}
	start, end, ok := routes.timeRange(response, values, requestID)
	if !ok {
		return
	}
	filters, ok := parseUsageFilters(response, values, requestID)
	if !ok {
		return
	}
	pageSize, err := parseOptionalPageSize(values.Get("pageSize"))
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "pageSize must be between 1 and 200.", requestID)
		return
	}
	if pageSize == 0 {
		pageSize = routes.defaultPageSize
	}
	_ = session
	page, err := routes.queries.ListLogs(request.Context(), usageledger.LogQuery{
		NamespaceID: namespaceID, ExternalRequestID: values.Get("requestId"),
		Start: start, End: end, Filters: filters,
		Visibility: visibility, PageSize: pageSize, Cursor: values.Get("cursor"),
	}, routes.logCursors)
	if err != nil {
		writeObservabilityError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, requestLogPageResponse{
		Data: page.Items,
		Page: managementapi.PageInfo{NextCursor: page.NextCursor, HasMore: page.NextCursor != "", PageSize: pageSize},
	}, requestID)
}

func (routes *ObservabilityRoutes) requestLogDetail(response http.ResponseWriter, request *http.Request, requestID string) {
	namespaceID, admissionID, ok := requestLogDetailPath(request.URL.Path)
	if !ok || request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	resolved, session, visibility, authorized := routes.authorizedVisibilityForNamespace(
		response, request, requestID, namespaceID, accesscontrol.PermissionLogRead,
	)
	if !authorized || resolved != namespaceID {
		return
	}
	detail, err := routes.queries.RequestDetail(request.Context(), namespaceID, admissionID, visibility)
	if errors.Is(err, usageledger.ErrNotFound) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if err != nil {
		writeObservabilityError(response, err, requestID)
		return
	}
	internal, err := routes.scopes.ResolveResultScope(request.Context(),
		accesscontrol.ManagementPrincipalID(session.Session.PrincipalID), accesscontrol.NamespaceID(namespaceID),
		accesscontrol.PermissionUsageInternalDimensionsRead)
	if err != nil || !internal.Covers(scopeFromVisibility(namespaceID, visibility)) {
		detail.Dispatches = nil
	}
	writeProviderJSON(response, http.StatusOK, requestLogDetailResponse{Data: detail}, requestID)
}

func (routes *ObservabilityRoutes) authorizedUsageQuery(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	breakdown bool,
) (usageledger.UsageQuery, url.Values, bool) {
	values, err := strictProviderQuery(request.URL.RawQuery, observabilityQueryKeys(false, breakdown))
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Usage query is invalid.", requestID)
		return usageledger.UsageQuery{}, nil, false
	}
	namespaceID, _, visibility, ok := routes.authorizedVisibility(response, request, requestID, accesscontrol.PermissionUsageRead)
	if !ok {
		return usageledger.UsageQuery{}, nil, false
	}
	start, end, ok := routes.timeRange(response, values, requestID)
	if !ok {
		return usageledger.UsageQuery{}, nil, false
	}
	filters, ok := parseUsageFilters(response, values, requestID)
	if !ok {
		return usageledger.UsageQuery{}, nil, false
	}
	grain := usageledger.Grain(values.Get("grain"))
	if grain == "" {
		grain = usageledger.GrainAuto
	}
	query := usageledger.UsageQuery{
		NamespaceID: namespaceID, Start: start, End: end, Grain: grain,
		TimeZone: observabilityTimeZone(values),
		Filters:  filters, Visibility: visibility,
	}
	if hasInternalUsageFilters(filters) && !routes.authorizeInternalDimensions(response, request, requestID, query) {
		return usageledger.UsageQuery{}, nil, false
	}
	return query, values, true
}

func observabilityTimeZone(values url.Values) string {
	zone := values.Get("timeZone")
	if zone == "" {
		return "UTC"
	}
	return zone
}

func (routes *ObservabilityRoutes) authorizedVisibility(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	permission accesscontrol.Permission,
) (string, managementauth.AuthenticatedSession, usageledger.QueryVisibility, bool) {
	namespaceID, err := routes.namespaces.ResolveNamespace(request.Context(), request)
	if err != nil || !canonicalUUID(namespaceID) {
		writeProviderError(response, http.StatusBadRequest, "invalid_namespace", "A valid namespace is required.", requestID)
		return "", managementauth.AuthenticatedSession{}, usageledger.QueryVisibility{}, false
	}
	return routes.authorizedVisibilityForNamespace(response, request, requestID, namespaceID, permission)
}

func (routes *ObservabilityRoutes) authorizedVisibilityForNamespace(
	response http.ResponseWriter,
	request *http.Request,
	requestID, namespaceID string,
	permission accesscontrol.Permission,
) (string, managementauth.AuthenticatedSession, usageledger.QueryVisibility, bool) {
	if !canonicalUUID(namespaceID) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return "", managementauth.AuthenticatedSession{}, usageledger.QueryVisibility{}, false
	}
	resolved, err := routes.namespaces.ResolveNamespace(request.Context(), request)
	if err != nil || resolved != namespaceID {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return "", managementauth.AuthenticatedSession{}, usageledger.QueryVisibility{}, false
	}
	token, ok := bearerToken(request)
	if !ok {
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Authentication is required.", requestID)
		return "", managementauth.AuthenticatedSession{}, usageledger.QueryVisibility{}, false
	}
	session, err := routes.sessions.Authenticate(request.Context(), token, namespaceID, routes.now().UTC())
	if err != nil {
		status, code, message := http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable."
		if errors.Is(err, managementauth.ErrAuthenticationDenied) {
			status, code, message = http.StatusUnauthorized, "unauthenticated", "Authentication is required."
		}
		writeProviderError(response, status, code, message, requestID)
		return "", managementauth.AuthenticatedSession{}, usageledger.QueryVisibility{}, false
	}
	scope, err := routes.scopes.ResolveResultScope(request.Context(),
		accesscontrol.ManagementPrincipalID(session.Session.PrincipalID), accesscontrol.NamespaceID(namespaceID), permission)
	if err != nil {
		if errors.Is(err, managementauthorization.ErrDenied) {
			writeProviderError(response, http.StatusForbidden, "forbidden", "Permission denied.", requestID)
		} else {
			writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
		}
		return "", managementauth.AuthenticatedSession{}, usageledger.QueryVisibility{}, false
	}
	return namespaceID, session, visibilityFromScope(scope), true
}

func (routes *ObservabilityRoutes) authorizeInternalDimensions(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	query usageledger.UsageQuery,
) bool {
	token, ok := bearerToken(request)
	if !ok {
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Authentication is required.", requestID)
		return false
	}
	session, err := routes.sessions.Authenticate(request.Context(), token, query.NamespaceID, routes.now().UTC())
	if err != nil {
		writeProviderError(response, http.StatusUnauthorized, "unauthenticated", "Authentication is required.", requestID)
		return false
	}
	internal, err := routes.scopes.ResolveResultScope(request.Context(),
		accesscontrol.ManagementPrincipalID(session.Session.PrincipalID), accesscontrol.NamespaceID(query.NamespaceID),
		accesscontrol.PermissionUsageInternalDimensionsRead)
	if err != nil || !internal.Covers(scopeRequiredByQuery(query)) {
		writeProviderError(response, http.StatusForbidden, "forbidden", "Internal usage dimensions are not available.", requestID)
		return false
	}
	return true
}

func (routes *ObservabilityRoutes) timeRange(
	response http.ResponseWriter,
	values url.Values,
	requestID string,
) (time.Time, time.Time, bool) {
	end := routes.now().UTC()
	start := end.Add(-routes.defaultRange)
	var err error
	if value := values.Get("end"); value != "" {
		end, err = time.Parse(time.RFC3339Nano, value)
		if err != nil {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "end must be an RFC 3339 timestamp.", requestID)
			return time.Time{}, time.Time{}, false
		}
	}
	if value := values.Get("start"); value != "" {
		start, err = time.Parse(time.RFC3339Nano, value)
		if err != nil {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "start must be an RFC 3339 timestamp.", requestID)
			return time.Time{}, time.Time{}, false
		}
	} else if values.Get("end") != "" {
		start = end.Add(-routes.defaultRange)
	}
	if !start.Before(end) || end.Sub(start) > routes.maximumRange {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "The requested time range is invalid.", requestID)
		return time.Time{}, time.Time{}, false
	}
	zone := values.Get("timeZone")
	if zone == "" {
		zone = "UTC"
	}
	if _, err := time.LoadLocation(zone); err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "timeZone must be a valid IANA time zone.", requestID)
		return time.Time{}, time.Time{}, false
	}
	return start.UTC(), end.UTC(), true
}

func observabilityQueryKeys(logs, breakdown bool) map[string]bool {
	keys := map[string]bool{
		"start": true, "end": true, "timeZone": true,
		"teamId": true, "userId": true, "apiKeyId": true,
		"entrypointId": true, "recipeId": true, "logicalModelId": true,
		"backendId": true, "providerId": true, "dispatchType": true,
		"protocol": true, "statusCode": true, "errorCode": true,
	}
	if logs {
		keys["cursor"] = true
		keys["pageSize"] = true
		keys["requestId"] = true
	} else {
		keys["grain"] = true
	}
	if breakdown {
		keys["dimension"] = true
		keys["pageSize"] = true
	}
	return keys
}

func parseUsageFilters(response http.ResponseWriter, values url.Values, requestID string) (usageledger.UsageFilters, bool) {
	statusCode := 0
	if value := values.Get("statusCode"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed < 100 || parsed > 599 {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "statusCode is invalid.", requestID)
			return usageledger.UsageFilters{}, false
		}
		statusCode = parsed
	}
	return usageledger.UsageFilters{
		APIKeyID: values.Get("apiKeyId"), UserID: values.Get("userId"), TeamID: values.Get("teamId"),
		EntrypointID: values.Get("entrypointId"), RecipeID: values.Get("recipeId"),
		LogicalModelID: values.Get("logicalModelId"), BackendID: values.Get("backendId"),
		ProviderID: values.Get("providerId"), DispatchType: values.Get("dispatchType"),
		Protocol: values.Get("protocol"), StatusCode: statusCode, ErrorCode: values.Get("errorCode"),
	}, true
}

func hasInternalUsageFilters(filters usageledger.UsageFilters) bool {
	return filters.LogicalModelID != "" || filters.BackendID != "" ||
		filters.ProviderID != "" || filters.DispatchType != ""
}

func isInternalBreakdown(dimension usageledger.BreakdownDimension) bool {
	return dimension == usageledger.BreakdownLogicalModel || dimension == usageledger.BreakdownBackend ||
		dimension == usageledger.BreakdownProvider || dimension == usageledger.BreakdownDispatchType
}

func visibilityFromScope(scope managementauthorization.ResultScope) usageledger.QueryVisibility {
	visibility := usageledger.QueryVisibility{All: scope.All}
	for _, id := range scope.TeamIDs {
		visibility.TeamIDs = append(visibility.TeamIDs, string(id))
	}
	for _, id := range scope.UserIDs {
		visibility.UserIDs = append(visibility.UserIDs, string(id))
	}
	for _, id := range scope.APIKeyIDs {
		visibility.APIKeyIDs = append(visibility.APIKeyIDs, string(id))
	}
	return visibility
}

func scopeFromVisibility(namespaceID string, visibility usageledger.QueryVisibility) managementauthorization.ResultScope {
	scope := managementauthorization.ResultScope{NamespaceID: accesscontrol.NamespaceID(namespaceID), All: visibility.All}
	for _, id := range visibility.TeamIDs {
		scope.TeamIDs = append(scope.TeamIDs, accesscontrol.TeamID(id))
	}
	for _, id := range visibility.UserIDs {
		scope.UserIDs = append(scope.UserIDs, accesscontrol.UserID(id))
	}
	for _, id := range visibility.APIKeyIDs {
		scope.APIKeyIDs = append(scope.APIKeyIDs, accesscontrol.APIKeyID(id))
	}
	return scope
}

func scopeRequiredByQuery(query usageledger.UsageQuery) managementauthorization.ResultScope {
	base := scopeFromVisibility(query.NamespaceID, query.Visibility)
	for _, restricted := range []managementauthorization.ResultScope{
		{NamespaceID: accesscontrol.NamespaceID(query.NamespaceID), APIKeyIDs: []accesscontrol.APIKeyID{accesscontrol.APIKeyID(query.Filters.APIKeyID)}},
		{NamespaceID: accesscontrol.NamespaceID(query.NamespaceID), UserIDs: []accesscontrol.UserID{accesscontrol.UserID(query.Filters.UserID)}},
		{NamespaceID: accesscontrol.NamespaceID(query.NamespaceID), TeamIDs: []accesscontrol.TeamID{accesscontrol.TeamID(query.Filters.TeamID)}},
	} {
		if len(restricted.APIKeyIDs) > 0 && restricted.APIKeyIDs[0] != "" ||
			len(restricted.UserIDs) > 0 && restricted.UserIDs[0] != "" ||
			len(restricted.TeamIDs) > 0 && restricted.TeamIDs[0] != "" {
			return restricted
		}
	}
	return base
}

func requestLogDetailPath(path string) (string, string, bool) {
	prefix := managementapi.BasePath + "/namespaces/"
	if !strings.HasPrefix(path, prefix) {
		return "", "", false
	}
	parts := strings.Split(strings.TrimPrefix(path, prefix), "/")
	if len(parts) != 3 || parts[1] != "request-logs" || !canonicalUUID(parts[0]) || parts[2] == "" || len(parts[2]) > 256 {
		return "", "", false
	}
	return parts[0], parts[2], true
}

func writeObservabilityError(response http.ResponseWriter, err error, requestID string) {
	if errors.Is(err, usageledger.ErrNotFound) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if errors.Is(err, usageledger.ErrLedgerCorrupt) {
		writeProviderError(response, http.StatusServiceUnavailable, "usage_unavailable", "Usage data is unavailable.", requestID)
		return
	}
	if errors.Is(err, usageledger.ErrInvalidQuery) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "The observability query is invalid.", requestID)
		return
	}
	writeProviderError(response, http.StatusServiceUnavailable, "observability_unavailable", "Observability data is unavailable.", requestID)
}

type requestLogPageResponse struct {
	Data []usageledger.RequestLog `json:"data"`
	Page managementapi.PageInfo   `json:"page"`
}

type requestLogDetailResponse struct {
	Data usageledger.RequestDetail `json:"data"`
}

type auditEventPageResponse struct {
	Data []auditlog.Event       `json:"data"`
	Page managementapi.PageInfo `json:"page"`
}

var _ RouteRegistrar = (*ObservabilityRoutes)(nil)
