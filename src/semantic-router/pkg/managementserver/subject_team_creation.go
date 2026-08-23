package managementserver

import (
	"net/http"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
)

func (routes *SubjectRoutes) resolveTeamPolicySelection(
	response http.ResponseWriter,
	request *http.Request,
	requestID, namespaceID string,
	body managementapi.TeamCreateRequest,
) ([]string, string, *subjectmanagement.TeamDefaults, bool, bool, bool) {
	useDefaultAccess, useDefaultRate := body.AccessPolicyIDs == nil, body.RateLimitPolicyID == nil
	accessPolicyIDs, valid := canonicalSubjectPolicyIDs(body.AccessPolicyIDs)
	if useDefaultAccess {
		valid = true
	}
	rateLimitPolicyID := ""
	if body.RateLimitPolicyID != nil {
		rateLimitPolicyID = *body.RateLimitPolicyID
		valid = valid && canonicalUUID(rateLimitPolicyID)
	}
	if !valid {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Team policy selection is invalid.", requestID)
		return nil, "", nil, false, false, false
	}
	var defaults *subjectmanagement.TeamDefaults
	if useDefaultAccess || useDefaultRate {
		resolved, err := routes.service.ResolveTeamDefaults(request.Context(), namespaceID)
		if err != nil {
			writeSubjectError(response, err, requestID)
			return nil, "", nil, false, false, false
		}
		defaults = &resolved
		if useDefaultAccess {
			accessPolicyIDs = []string{resolved.AccessPolicyID}
		}
		if useDefaultRate {
			rateLimitPolicyID = resolved.RateLimitPolicyID
		}
	}
	return accessPolicyIDs, rateLimitPolicyID, defaults, useDefaultAccess, useDefaultRate, true
}

func canonicalSubjectPolicyIDs(values []string) ([]string, bool) {
	if len(values) == 0 {
		return nil, false
	}
	canonical := append([]string(nil), values...)
	for _, value := range canonical {
		if !canonicalUUID(value) {
			return nil, false
		}
	}
	sort.Strings(canonical)
	for index := 1; index < len(canonical); index++ {
		if canonical[index] == canonical[index-1] {
			return nil, false
		}
	}
	return canonical, true
}
