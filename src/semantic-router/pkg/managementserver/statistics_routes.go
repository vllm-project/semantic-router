package managementserver

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementstatistics"
)

const statisticsPath = managementapi.BasePath + "/statistics"

// StatisticsRoutes exposes one bounded control-plane cardinality snapshot.
// Each optional field is independently projected from that resource's read
// scope, so a missing permission never becomes a misleading zero.
type StatisticsRoutes struct {
	service    StatisticsQueryService
	scopes     ResultScopeResolver
	namespaces NamespaceResolver
	sessions   SessionAuthenticator
	now        func() time.Time
}

func NewStatisticsRoutes(options StatisticsRoutesOptions) (*StatisticsRoutes, error) {
	if options.Service == nil || options.Scopes == nil || options.Namespaces == nil || options.Sessions == nil {
		return nil, errors.New("statistics Management routes require service, scope, namespace, and session dependencies")
	}
	if _, found := managementapi.LookupOperation(managementapi.MethodGET, statisticsPath); !found {
		return nil, errors.New("statistics Management operation contract is unavailable")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &StatisticsRoutes{
		service: options.Service, scopes: options.Scopes,
		namespaces: options.Namespaces, sessions: options.Sessions, now: now,
	}, nil
}

func (routes *StatisticsRoutes) Register(mux *http.ServeMux) {
	if routes == nil || mux == nil {
		panic("Statistics Management routes and mux are required")
	}
	mux.Handle("GET "+statisticsPath, routes)
}

func (routes *StatisticsRoutes) Ready(ctx context.Context) error {
	if routes == nil || routes.service == nil || routes.scopes == nil || routes.namespaces == nil || routes.sessions == nil {
		return managementstatistics.ErrUnavailable
	}
	return routes.service.Ready(ctx)
}

func (routes *StatisticsRoutes) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if routes == nil || request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path ||
		request.Method != http.MethodGet || request.URL.Path != statisticsPath || request.URL.RawQuery != "" || request.ContentLength > 0 {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	namespaceID, session, authenticated := routes.authenticate(response, request, requestID)
	if !authenticated {
		return
	}
	principalID := accesscontrol.ManagementPrincipalID(session.Session.PrincipalID)
	if _, ok := routes.requiredScope(response, request, requestID, principalID, namespaceID, accesscontrol.PermissionUsageRead); !ok {
		return
	}
	scopes, ok := routes.fieldScopes(response, request, requestID, principalID, namespaceID)
	if !ok {
		return
	}
	snapshot, err := routes.service.Snapshot(request.Context(), managementstatistics.Request{
		NamespaceID: namespaceID,
		Scopes:      scopes,
	})
	if err != nil {
		writeProviderError(response, http.StatusServiceUnavailable, "statistics_unavailable", "Statistics are unavailable.", requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, statisticsResponse(snapshot), requestID)
}

func (routes *StatisticsRoutes) authenticate(
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
	if session.NamespaceID != namespaceID || session.Session.PrincipalID == "" {
		writeProviderError(response, http.StatusServiceUnavailable, "authentication_unavailable", "Authentication state is unavailable.", requestID)
		return "", managementauth.AuthenticatedSession{}, false
	}
	return namespaceID, session, true
}

func (routes *StatisticsRoutes) requiredScope(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	principalID accesscontrol.ManagementPrincipalID,
	namespaceID string,
	permission accesscontrol.Permission,
) (accesscontrol.ResultScope, bool) {
	scope, err := routes.scopes.ResolveResultScope(
		request.Context(), principalID, accesscontrol.NamespaceID(namespaceID), permission,
	)
	if err != nil {
		if errors.Is(err, managementauthorization.ErrDenied) {
			writeProviderError(response, http.StatusForbidden, "forbidden", "Permission denied.", requestID)
		} else {
			writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
		}
		return accesscontrol.ResultScope{}, false
	}
	canonical, err := scope.Canonical()
	if err != nil || canonical.NamespaceID != accesscontrol.NamespaceID(namespaceID) || canonical.Empty() {
		writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
		return accesscontrol.ResultScope{}, false
	}
	return canonical, true
}

func (routes *StatisticsRoutes) fieldScopes(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	principalID accesscontrol.ManagementPrincipalID,
	namespaceID string,
) (managementstatistics.Scopes, bool) {
	result := managementstatistics.Scopes{}
	fields := []struct {
		permission accesscontrol.Permission
		target     **accesscontrol.ResultScope
	}{
		{accesscontrol.PermissionUserRead, &result.Users},
		{accesscontrol.PermissionTeamRead, &result.Teams},
		{accesscontrol.PermissionKeyRead, &result.APIKeys},
		{accesscontrol.PermissionAccessPolicyRead, &result.AccessPolicies},
		{accesscontrol.PermissionRatePolicyRead, &result.RatePolicies},
	}
	for _, field := range fields {
		scope, err := routes.scopes.ResolveResultScope(
			request.Context(), principalID, accesscontrol.NamespaceID(namespaceID), field.permission,
		)
		if errors.Is(err, managementauthorization.ErrDenied) {
			continue
		}
		if err != nil {
			writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
			return managementstatistics.Scopes{}, false
		}
		canonical, canonicalErr := scope.Canonical()
		if canonicalErr != nil || canonical.NamespaceID != accesscontrol.NamespaceID(namespaceID) {
			writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
			return managementstatistics.Scopes{}, false
		}
		if canonical.Empty() {
			continue
		}
		*field.target = &canonical
	}
	return result, true
}

func statisticsResponse(snapshot managementstatistics.Snapshot) managementapi.AccessStatistics {
	return managementapi.AccessStatistics{
		AsOf: snapshot.AsOf, ExpiringBefore: snapshot.ExpiringBefore,
		Users: statisticsQuantity(snapshot.Users), Teams: statisticsQuantity(snapshot.Teams),
		ActiveAPIKeys:      statisticsQuantity(snapshot.ActiveAPIKeys),
		ExpiringAPIKeys:    statisticsQuantity(snapshot.ExpiringAPIKeys),
		AccessPolicies:     statisticsQuantity(snapshot.AccessPolicies),
		ActiveRatePolicies: statisticsQuantity(snapshot.ActiveRatePolicies),
	}
}

func statisticsQuantity(count *managementstatistics.Count) *managementapi.WholeQuantity {
	if count == nil {
		return nil
	}
	value := managementapi.WholeQuantity(*count)
	return &value
}

var _ RouteRegistrar = (*StatisticsRoutes)(nil)
