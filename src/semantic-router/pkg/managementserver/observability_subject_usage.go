package managementserver

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apikeymanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

type subjectUsageResourceKind string

const (
	subjectUsageUser   subjectUsageResourceKind = "user"
	subjectUsageTeam   subjectUsageResourceKind = "team"
	subjectUsageAPIKey subjectUsageResourceKind = "api_key"
)

type subjectUsageResource struct {
	kind subjectUsageResourceKind
	id   string
}

func loadSubjectUsageOperations() (map[subjectUsageResourceKind]managementapi.OperationContract, error) {
	paths := map[subjectUsageResourceKind]string{
		subjectUsageUser:   managementapi.BasePath + "/users/{userId}/usage",
		subjectUsageTeam:   managementapi.BasePath + "/teams/{teamId}/usage",
		subjectUsageAPIKey: managementapi.BasePath + "/api-keys/{keyId}/usage",
	}
	operations := make(map[subjectUsageResourceKind]managementapi.OperationContract, len(paths))
	for kind, path := range paths {
		operation, found := managementapi.LookupOperation(managementapi.MethodGET, path)
		if !found {
			return nil, fmt.Errorf("subject usage operation contract GET %s is unavailable", path)
		}
		operations[kind] = operation
	}
	return operations, nil
}

func parseSubjectUsagePath(path string) (subjectUsageResource, bool) {
	prefix := managementapi.BasePath + "/"
	if !strings.HasPrefix(path, prefix) {
		return subjectUsageResource{}, false
	}
	parts := strings.Split(strings.TrimPrefix(path, prefix), "/")
	if len(parts) != 3 || parts[2] != "usage" || !canonicalUUID(parts[1]) {
		return subjectUsageResource{}, false
	}
	resource := subjectUsageResource{id: parts[1]}
	switch parts[0] {
	case "users":
		resource.kind = subjectUsageUser
	case "teams":
		resource.kind = subjectUsageTeam
	case "api-keys":
		resource.kind = subjectUsageAPIKey
	default:
		return subjectUsageResource{}, false
	}
	return resource, true
}

func (routes *ObservabilityRoutes) subjectUsageSummary(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	resource subjectUsageResource,
) {
	values, err := strictProviderQuery(request.URL.RawQuery, observabilityQueryKeys(false, false))
	if err != nil || subjectUsageFilterConflicts(values, resource) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Usage query is invalid.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticateSubjectUsage(response, request, requestID)
	if !ok {
		return
	}
	targets, ok := routes.subjectUsageTargets(response, request, requestID, namespaceID, resource)
	if !ok || !routes.authorizeSubjectUsage(response, request, requestID, session, namespaceID, resource.kind, targets) {
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
	visibility := subjectUsageVisibility(resource)
	applySubjectUsageFilter(&filters, resource)
	grain := usageledger.Grain(values.Get("grain"))
	if grain == "" {
		grain = usageledger.GrainAuto
	}
	query := usageledger.UsageQuery{
		NamespaceID: namespaceID,
		Start:       start,
		End:         end,
		Grain:       grain,
		TimeZone:    observabilityTimeZone(values),
		Filters:     filters,
		Visibility:  visibility,
	}
	if hasInternalUsageFilters(filters) && !routes.authorizeInternalDimensions(response, request, requestID, query) {
		return
	}
	result, err := routes.queries.Summary(request.Context(), query)
	if err != nil {
		writeObservabilityError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, result, requestID)
}

func subjectUsageFilterConflicts(values url.Values, resource subjectUsageResource) bool {
	switch resource.kind {
	case subjectUsageUser:
		return values.Get("userId") != ""
	case subjectUsageTeam:
		return values.Get("teamId") != ""
	case subjectUsageAPIKey:
		return values.Get("apiKeyId") != ""
	default:
		return true
	}
}

func (routes *ObservabilityRoutes) authenticateSubjectUsage(
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

func (routes *ObservabilityRoutes) subjectUsageTargets(
	response http.ResponseWriter,
	request *http.Request,
	requestID, namespaceID string,
	resource subjectUsageResource,
) (map[string][]accesscontrol.ScopedTarget, bool) {
	switch resource.kind {
	case subjectUsageUser:
		if _, err := routes.resources.GetUser(request.Context(), namespaceID, resource.id); err != nil {
			writeSubjectUsageResourceError(response, err, requestID)
			return nil, false
		}
		return subjectUserTargets(namespaceID, resource.id), true
	case subjectUsageTeam:
		if _, err := routes.resources.GetTeam(request.Context(), namespaceID, resource.id); err != nil {
			writeSubjectUsageResourceError(response, err, requestID)
			return nil, false
		}
		return subjectTeamTargets(namespaceID, resource.id), true
	case subjectUsageAPIKey:
		key, err := routes.resources.GetAPIKey(request.Context(), namespaceID, resource.id)
		if err != nil {
			writeSubjectUsageResourceError(response, err, requestID)
			return nil, false
		}
		return apiKeyTargets(key), true
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return nil, false
	}
}

func (routes *ObservabilityRoutes) authorizeSubjectUsage(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	session managementauth.AuthenticatedSession,
	namespaceID string,
	kind subjectUsageResourceKind,
	targets map[string][]accesscontrol.ScopedTarget,
) bool {
	operation, found := routes.subjectUsageOperations[kind]
	if !found {
		writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
		return false
	}
	_, err := routes.authorization.Authorize(request.Context(), AuthorizationRequest{
		Operation: operation, Session: session, NamespaceID: namespaceID, Targets: targets,
	})
	if err == nil {
		return true
	}
	if errors.Is(err, managementauthorization.ErrDenied) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	} else {
		writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
	}
	return false
}

func subjectUsageVisibility(resource subjectUsageResource) usageledger.QueryVisibility {
	visibility := usageledger.QueryVisibility{}
	switch resource.kind {
	case subjectUsageUser:
		visibility.UserIDs = []string{resource.id}
	case subjectUsageTeam:
		visibility.TeamIDs = []string{resource.id}
	case subjectUsageAPIKey:
		visibility.APIKeyIDs = []string{resource.id}
	}
	return visibility
}

func applySubjectUsageFilter(filters *usageledger.UsageFilters, resource subjectUsageResource) {
	if filters == nil {
		return
	}
	switch resource.kind {
	case subjectUsageUser:
		filters.UserID = resource.id
	case subjectUsageTeam:
		filters.TeamID = resource.id
	case subjectUsageAPIKey:
		filters.APIKeyID = resource.id
	}
}

func writeSubjectUsageResourceError(response http.ResponseWriter, err error, requestID string) {
	if errors.Is(err, subjectmanagement.ErrNotFound) || errors.Is(err, apikeymanagement.ErrNotFound) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	writeProviderError(response, http.StatusServiceUnavailable, "observability_unavailable", "Observability data is unavailable.", requestID)
}
