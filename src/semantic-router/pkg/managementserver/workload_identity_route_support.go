package managementserver

import (
	"errors"
	"net/http"
	"regexp"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

type workloadListParameters struct {
	cursor   string
	pageSize int
}

func workloadListQuery(response http.ResponseWriter, request *http.Request, requestID string, additional ...string) (workloadListParameters, bool) {
	allowed := map[string]bool{"cursor": true, "pageSize": true}
	for _, name := range additional {
		allowed[name] = true
	}
	for name, values := range request.URL.Query() {
		if !allowed[name] || len(values) != 1 {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Query parameters are invalid.", requestID)
			return workloadListParameters{}, false
		}
	}
	pageSize := 50
	if raw := request.URL.Query().Get("pageSize"); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value < 1 || value > 200 {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "pageSize must be between 1 and 200.", requestID)
			return workloadListParameters{}, false
		}
		pageSize = value
	}
	return workloadListParameters{cursor: request.URL.Query().Get("cursor"), pageSize: pageSize}, true
}

func workloadRequest(response http.ResponseWriter, request *http.Request) string {
	requestID := managementRequestID(request)
	setProviderResponseHeaders(response, requestID)
	if request == nil || request.URL == nil || request.URL.EscapedPath() != request.URL.Path {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
	return requestID
}

func workloadNoQuery(response http.ResponseWriter, request *http.Request, requestID string) bool {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Query parameters are not accepted.", requestID)
		return false
	}
	return true
}

func workloadActor(request *http.Request, session managementauth.AuthenticatedSession, requestID, reason string) managementidentity.WorkloadActor {
	return managementidentity.WorkloadActor{
		PrincipalID: session.Session.PrincipalID, ActorChain: []string{session.Session.PrincipalID},
		RequestID: requestID, SourceIP: directRequestIP(request), Reason: reason, Session: session.Session,
	}
}

func workloadCreateTarget(namespaceID string) map[string][]accesscontrol.ScopedTarget {
	if namespaceID == "" {
		return nil
	}
	return map[string][]accesscontrol.ScopedTarget{"target": {{
		Scope: accesscontrol.NamespaceScope(accesscontrol.NamespaceID(namespaceID)),
	}}}
}

func serviceAccountTarget(account managementidentity.ServiceAccount) map[string][]accesscontrol.ScopedTarget {
	if account.OwnerScope == managementidentity.ServiceAccountOwnerCluster {
		return nil
	}
	return map[string][]accesscontrol.ScopedTarget{"target": {{
		Scope: accesscontrol.ResourceScope(accesscontrol.NamespaceID(account.NamespaceID),
			accesscontrol.ScopeResourceServiceAccount, accesscontrol.ResourceID(account.ID)),
	}}}
}

func serviceAccountDTO(value managementidentity.ServiceAccount) managementapi.ServiceAccount {
	return managementapi.ServiceAccount{
		ServiceAccountID: value.ID, PrincipalID: value.PrincipalID, DisplayName: value.DisplayName,
		OwnerScope: string(value.OwnerScope), NamespaceID: value.NamespaceID, Status: string(value.Status),
		Revision: value.Revision, CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt,
	}
}

func serviceCredentialDTO(value managementidentity.ServiceCredential) managementapi.ServiceCredential {
	return managementapi.ServiceCredential{
		CredentialID: value.ID, ServiceAccountID: value.ServiceAccountID, PublicID: value.PublicID,
		WorkloadClass: string(value.WorkloadClass), SourceAssuredAt: value.SourceAssuredAt,
		Status: string(value.Status), NotBefore: value.NotBefore, ExpiresAt: value.ExpiresAt,
		RevokedAt: cloneResponseTime(value.RevokedAt), CreatedAt: value.CreatedAt,
	}
}

func mtlsMappingDTO(value managementidentity.MTLSIdentityMapping) managementapi.MTLSIdentityMapping {
	return managementapi.MTLSIdentityMapping{
		MappingID: value.ID, MatcherKind: string(value.MatcherKind), MatcherValue: value.MatcherValue,
		PrincipalID: value.PrincipalID, WorkloadClass: string(value.WorkloadClass),
		SourceAssuredAt: value.SourceAssuredAt, Status: string(value.Status), Revision: value.Revision,
		CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt,
	}
}

func writeServiceCredentialIssue(response http.ResponseWriter, status int, result managementidentity.ServiceCredentialSecretResult, requestID string) {
	response.Header().Set(managementapi.HeaderETag, serviceAccountETag(result.ServiceAccount.Revision))
	setIdempotencyReplayHeader(response, result.Replayed)
	payload := managementapi.ServiceCredentialIssue{
		ServiceAccount: serviceAccountDTO(result.ServiceAccount), Credential: serviceCredentialDTO(result.Credential),
		Secret: result.Secret, DeliveryExpiresAt: result.DeliveryExpiry,
	}
	defer zeroString(&payload.Secret)
	writeProviderJSON(response, status, payload, requestID)
}

func writeWorkloadMutation(response http.ResponseWriter, result managementidentity.WorkloadMutationResult, etag func(uint64) string, base, requestID string) {
	response.Header().Set(managementapi.HeaderETag, etag(result.Revision))
	response.Header().Set("Location", base+"/"+result.ID)
	setIdempotencyReplayHeader(response, result.Replayed)
	var replayed *bool
	if result.HTTPStatus == http.StatusCreated {
		value := result.Replayed
		replayed = &value
	}
	writeProviderJSON(response, result.HTTPStatus, managementapi.NewResourceMutationReceipt(
		result.Kind, result.ID, result.Revision, replayed,
	), requestID)
}

func writeWorkloadIdentityError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, managementidentity.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	case errors.Is(err, managementidentity.ErrInvalidWorkloadRequest):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Workload identity request is invalid.", requestID)
	case errors.Is(err, managementidentity.ErrRevisionConflict):
		writeProviderError(response, http.StatusPreconditionFailed, "revision_conflict", "Resource changed. Refresh and retry.", requestID)
	case errors.Is(err, managementidentity.ErrAlreadyExists), errors.Is(err, managementidentity.ErrWorkloadDependency):
		writeProviderError(response, http.StatusConflict, "conflict", "Request conflicts with current state.", requestID)
	case errors.Is(err, managementcommand.ErrConflict):
		writeProviderError(response, http.StatusConflict, "idempotency_conflict", "Idempotency-Key was already used for a different request.", requestID)
	case errors.Is(err, managementidentity.ErrWorkloadSecretExpired):
		writeProviderError(response, http.StatusGone, "secret_result_expired", "The one-time credential result expired.", requestID)
	case errors.Is(err, managementidentity.ErrServiceCredentialUnavailable):
		writeProviderError(response, http.StatusConflict, "credential_unavailable", "The credential is no longer available.", requestID)
	case errors.Is(err, managementidentity.ErrMTLSListenerUnavailable):
		writeProviderError(response, http.StatusConflict, "mtls_unavailable", "Verified mTLS is not configured on this listener.", requestID)
	case errors.Is(err, managementauth.ErrAuthenticationDenied):
		writeProviderError(response, http.StatusForbidden, "step_up_required", "Stronger authentication is required.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "identity_unavailable", "Workload identity state is unavailable.", requestID)
	}
}

var workloadETagPattern = regexp.MustCompile(`^"(sa|mtls):([1-9][0-9]*)"$`)

func workloadRevision(response http.ResponseWriter, request *http.Request, requestID, kind string) (uint64, bool) {
	values := request.Header.Values(managementapi.HeaderIfMatch)
	if len(values) != 1 {
		writeProviderError(response, http.StatusPreconditionRequired, "precondition_required", "If-Match is required.", requestID)
		return 0, false
	}
	parts := workloadETagPattern.FindStringSubmatch(values[0])
	if len(parts) != 3 || parts[1] != kind {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return 0, false
	}
	revision, _ := strconv.ParseUint(parts[2], 10, 64)
	return revision, revision > 0
}

func serviceAccountETag(revision uint64) string {
	return `"sa:` + strconv.FormatUint(revision, 10) + `"`
}

func mtlsMappingETag(revision uint64) string {
	return `"mtls:` + strconv.FormatUint(revision, 10) + `"`
}

type workloadHTTPContract struct {
	method managementapi.HTTPMethod
	path   string
}

func workloadIdentityHTTPContracts() []workloadHTTPContract {
	return []workloadHTTPContract{
		{managementapi.MethodGET, serviceAccountsPath},
		{managementapi.MethodPOST, serviceAccountsPath},
		{managementapi.MethodGET, serviceAccountsPath + "/{serviceAccountId}"},
		{managementapi.MethodPATCH, serviceAccountsPath + "/{serviceAccountId}"},
		{managementapi.MethodDELETE, serviceAccountsPath + "/{serviceAccountId}"},
		{managementapi.MethodGET, serviceAccountsPath + "/{serviceAccountId}/credentials"},
		{managementapi.MethodPOST, serviceAccountsPath + "/{serviceAccountId}/credentials:rotate"},
		{managementapi.MethodDELETE, serviceAccountsPath + "/{serviceAccountId}/credentials/{credentialId}"},
		{managementapi.MethodGET, mtlsMappingsPath},
		{managementapi.MethodPOST, mtlsMappingsPath},
		{managementapi.MethodGET, mtlsMappingsPath + "/{mappingId}"},
		{managementapi.MethodPATCH, mtlsMappingsPath + "/{mappingId}"},
		{managementapi.MethodDELETE, mtlsMappingsPath + "/{mappingId}"},
	}
}

var _ RouteRegistrar = (*WorkloadIdentityRoutes)(nil)
