package managementserver

import (
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
)

const (
	maximumPolicyBodyBytes  = 512 << 10
	policyETagAccess        = "access-policy"
	policyETagRate          = "rate-limit-policy"
	policyETagAccessBinding = "access-policy-binding"
	policyETagRateBinding   = "rate-limit-binding"
)

var policyETagPattern = regexp.MustCompile(`^"(access-policy|rate-limit-policy|access-policy-binding|rate-limit-binding):([1-9][0-9]*)"$`)

func decodePolicyBody(response http.ResponseWriter, request *http.Request, requestID string, target any) bool {
	if request.ContentLength > maximumPolicyBodyBytes {
		writeProviderError(response, http.StatusRequestEntityTooLarge, "invalid_request", "Request body is too large.", requestID)
		return false
	}
	mediaType, parameters, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != managementapi.JSONMediaType ||
		(len(parameters) != 0 && (len(parameters) != 1 || !strings.EqualFold(parameters["charset"], "utf-8"))) {
		writeProviderError(response, http.StatusUnsupportedMediaType, "unsupported_media_type", "Use the Management API media type.", requestID)
		return false
	}
	request.Body = http.MaxBytesReader(response, request.Body, maximumPolicyBodyBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		var maximum *http.MaxBytesError
		if errors.As(err, &maximum) {
			writeProviderError(response, http.StatusRequestEntityTooLarge, "invalid_request", "Request body is too large.", requestID)
		} else {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request body is invalid.", requestID)
		}
		return false
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request body is invalid.", requestID)
		return false
	}
	return true
}

func requirePolicyRevision(response http.ResponseWriter, request *http.Request, requestID, kind string) (uint64, bool) {
	values := request.Header.Values(managementapi.HeaderIfMatch)
	if len(values) != 1 {
		writeProviderError(response, http.StatusPreconditionRequired, "precondition_required", "If-Match is required.", requestID)
		return 0, false
	}
	match := policyETagPattern.FindStringSubmatch(values[0])
	if len(match) != 3 || match[1] != kind {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return 0, false
	}
	revision, err := strconv.ParseUint(match[2], 10, 64)
	if err != nil || revision == 0 {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return 0, false
	}
	return revision, true
}

func policyResourceETag(kind string, revision uint64) string {
	return `"` + kind + `:` + strconv.FormatUint(revision, 10) + `"`
}

func writePolicyMutation(response http.ResponseWriter, result policymanagement.MutationResult, requestID, etagKind string, idempotent bool) {
	response.Header().Set(managementapi.HeaderETag, policyResourceETag(etagKind, result.Revision))
	if idempotent {
		setIdempotencyReplayHeader(response, result.Replayed)
	}
	if result.HTTPStatus == http.StatusNoContent {
		response.WriteHeader(http.StatusNoContent)
		return
	}
	var replayed *bool
	if idempotent {
		value := result.Replayed
		replayed = &value
	}
	writeProviderJSON(response, result.HTTPStatus, managementapi.NewResourceMutationReceipt(
		result.Kind, result.ID, result.Revision, replayed), requestID)
}

func writePolicyError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, policymanagement.ErrInvalidRequest):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Policy request is invalid.", requestID)
	case errors.Is(err, policymanagement.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "Policy resource not found.", requestID)
	case errors.Is(err, managementcommand.ErrConflict):
		writeProviderError(response, http.StatusConflict, "idempotency_conflict", "Idempotency-Key was already used for a different request.", requestID)
	case errors.Is(err, policymanagement.ErrAlreadyExists), errors.Is(err, accesspostgres.ErrAlreadyExists):
		writeProviderError(response, http.StatusConflict, "already_exists", "Policy resource already exists.", requestID)
	case errors.Is(err, policymanagement.ErrRevisionConflict):
		writeProviderError(response, http.StatusPreconditionFailed, "revision_conflict", "The policy resource changed. Refresh and retry.", requestID)
	case errors.Is(err, policymanagement.ErrResourceInUse):
		writeProviderError(response, http.StatusConflict, "resource_in_use", "Remove policy bindings before deleting this policy.", requestID)
	case errors.Is(err, policymanagement.ErrAllocationConflict):
		writeProviderError(response, http.StatusConflict, "allocation_conflict", "The subject already has an active quota allocation.", requestID)
	case errors.Is(err, policymanagement.ErrCounterSemantics):
		writeProviderError(response, http.StatusConflict, "counter_semantics", "Create a new rule when counter semantics change.", requestID)
	case errors.Is(err, policymanagement.ErrUnknownUsageFence):
		writeProviderError(response, http.StatusConflict, "usage_fenced", "Reconcile unknown usage before changing this quota resource.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "policy_service_unavailable", "Policy service is unavailable.", requestID)
	}
}

func policyListQuery(response http.ResponseWriter, request *http.Request, requestID string) (string, int, string, string, bool) {
	query, err := strictProviderQuery(request.URL.RawQuery, map[string]bool{
		"cursor": true, "pageSize": true, "status": true, "search": true,
	})
	if err != nil || !map[string]bool{"": true, "draft": true, "active": true, "disabled": true}[query.Get("status")] {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Policy list query is invalid.", requestID)
		return "", 0, "", "", false
	}
	pageSize, err := parseOptionalPageSize(query.Get("pageSize"))
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "pageSize must be between 1 and 200.", requestID)
		return "", 0, "", "", false
	}
	return query.Get("cursor"), pageSize, query.Get("status"), query.Get("search"), true
}

func policyBindingListQuery(response http.ResponseWriter, request *http.Request, requestID string, rate bool) (policymanagement.ListBindingsRequest, bool) {
	allowed := map[string]bool{
		"cursor": true, "pageSize": true, "policyId": true,
		"subjectType": true, "subjectId": true, "status": true, "includeTotal": true,
	}
	if rate {
		allowed["mode"] = true
	}
	query, err := strictProviderQuery(request.URL.RawQuery, allowed)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Policy binding list query is invalid.", requestID)
		return policymanagement.ListBindingsRequest{}, false
	}
	pageSize, err := parseOptionalPageSize(query.Get("pageSize"))
	statusValid := map[string]bool{"": true, "active": true, "disabled": true}[query.Get("status")]
	modeValid := !rate || map[string]bool{"": true, "allocation": true, "hard_cap": true}[query.Get("mode")]
	subjectType, subjectID := query.Get("subjectType"), query.Get("subjectId")
	pairedSubject := (subjectType == "") == (subjectID == "")
	includeTotal, totalErr := parseOptionalBoolean(query.Get("includeTotal"))
	if err != nil || totalErr != nil || !statusValid || !modeValid || !pairedSubject ||
		(query.Get("policyId") != "" && !canonicalUUID(query.Get("policyId"))) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Policy binding list query is invalid.", requestID)
		return policymanagement.ListBindingsRequest{}, false
	}
	result := policymanagement.ListBindingsRequest{
		PolicyID: query.Get("policyId"),
		Status:   accesscontrol.BindingStatus(query.Get("status")), Mode: accesscontrol.RateBindingMode(query.Get("mode")),
		Cursor: query.Get("cursor"), PageSize: pageSize, IncludeTotal: includeTotal,
	}
	if subjectType != "" {
		if !map[string]bool{"user": true, "team": true, "api_key": true}[subjectType] || !canonicalUUID(subjectID) {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Policy binding list query is invalid.", requestID)
			return policymanagement.ListBindingsRequest{}, false
		}
		result.Subject = &policymanagement.Subject{Type: accesscontrol.SubjectKind(subjectType), ID: subjectID}
	}
	return result, true
}

func policyTarget(namespaceID string, kind accesscontrol.ScopeResourceType, policyID string) accesscontrol.ScopedTarget {
	if policyID == "" {
		return accesscontrol.ScopedTarget{Scope: accesscontrol.NamespaceScope(accesscontrol.NamespaceID(namespaceID))}
	}
	return subjectResourceTarget(namespaceID, kind, policyID)
}

func policySubjectTarget(namespaceID string, subject policymanagement.Subject) (accesscontrol.ScopedTarget, bool) {
	switch subject.Type {
	case accesscontrol.SubjectKindUser:
		return subjectUserTarget(namespaceID, subject.ID), canonicalUUID(subject.ID)
	case accesscontrol.SubjectKindTeam:
		return subjectTeamTarget(namespaceID, subject.ID), canonicalUUID(subject.ID)
	case accesscontrol.SubjectKindAPIKey:
		return subjectResourceTarget(namespaceID, accesscontrol.ScopeResourceAPIKey, subject.ID), canonicalUUID(subject.ID)
	default:
		return accesscontrol.ScopedTarget{}, false
	}
}

func accessPolicyTargets(namespaceID, policyID string, grants []policymanagement.AccessGrant) map[string][]accesscontrol.ScopedTarget {
	targets := map[string][]accesscontrol.ScopedTarget{"policy": {policyTarget(namespaceID, accesscontrol.ScopeResourceAccessPolicy, policyID)}}
	if len(grants) == 0 {
		return targets
	}
	targets["all_dependencies"] = make([]accesscontrol.ScopedTarget, 0, len(grants))
	for _, grant := range grants {
		kind := accesscontrol.ScopeResourceEntrypoint
		if grant.ResourceType == accesscontrol.GrantResourceModel {
			kind = accesscontrol.ScopeResourceModel
		}
		targets["all_dependencies"] = append(targets["all_dependencies"], subjectResourceTarget(namespaceID, kind, grant.ResourceID))
	}
	return targets
}

func accessPolicyConditions(grants []policymanagement.AccessGrant) map[string]bool {
	return map[string]bool{"access_policy_references_routing_resources": len(grants) > 0}
}

func validPolicyGrantTargets(grants []policymanagement.AccessGrant) bool {
	for _, grant := range grants {
		value := accesscontrol.AccessPolicyGrant{
			PolicyID:   accesscontrol.AccessPolicyID("request-policy"),
			Resource:   accesscontrol.GrantResource{Type: grant.ResourceType, ID: accesscontrol.ResourceID(grant.ResourceID)},
			Permission: grant.Permission, Effect: grant.Effect,
		}
		if value.Validate() != nil {
			return false
		}
	}
	return true
}

func policyBindingTargets(namespaceID, policyID string, subject policymanagement.Subject, rate bool) (map[string][]accesscontrol.ScopedTarget, map[string]bool, bool) {
	subjectTarget, valid := policySubjectTarget(namespaceID, subject)
	policyKind := accesscontrol.ScopeResourceAccessPolicy
	if rate {
		policyKind = accesscontrol.ScopeResourceRateLimitPolicy
	}
	return map[string][]accesscontrol.ScopedTarget{
			"policy": {policyTarget(namespaceID, policyKind, policyID)}, "subject": {subjectTarget},
		}, map[string]bool{
			"user_owner": subject.Type == accesscontrol.SubjectKindUser,
			"team_owner": subject.Type == accesscontrol.SubjectKindTeam,
			"key_owner":  subject.Type == accesscontrol.SubjectKindAPIKey,
		}, valid
}
