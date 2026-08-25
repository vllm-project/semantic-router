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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
)

var subjectETagPattern = regexp.MustCompile(`^"(user|team|membership):([1-9][0-9]*)"$`)

func subjectListQuery(response http.ResponseWriter, request *http.Request, requestID string, statuses map[string]bool) (mapValues, int, bool) {
	query, pageSize, _, ok := subjectCollectionQuery(response, request, requestID, statuses, false)
	return query, pageSize, ok
}

func subjectRelationshipListQuery(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	statuses map[string]bool,
) (mapValues, int, bool, bool) {
	return subjectCollectionQuery(response, request, requestID, statuses, true)
}

func subjectCollectionQuery(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	statuses map[string]bool,
	allowTotal bool,
) (mapValues, int, bool, bool) {
	allowed := map[string]bool{
		"cursor": true, "pageSize": true, "status": true, "search": true,
	}
	if allowTotal {
		allowed["includeTotal"] = true
	}
	query, err := strictProviderQuery(request.URL.RawQuery, allowed)
	if err != nil || !statuses[query.Get("status")] {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "List query is invalid.", requestID)
		return nil, 0, false, false
	}
	pageSize, err := parseOptionalPageSize(query.Get("pageSize"))
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "pageSize must be between 1 and 200.", requestID)
		return nil, 0, false, false
	}
	includeTotal, err := parseOptionalBoolean(query.Get("includeTotal"))
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "includeTotal must be true or false.", requestID)
		return nil, 0, false, false
	}
	return mapValues(query), pageSize, includeTotal, true
}

type mapValues map[string][]string

func (values mapValues) Get(key string) string {
	if len(values[key]) == 0 {
		return ""
	}
	return values[key][0]
}

func decodeSubjectBody(response http.ResponseWriter, request *http.Request, requestID string, target any) bool {
	if request.ContentLength > maximumSubjectBodyBytes {
		writeProviderError(response, http.StatusRequestEntityTooLarge, "invalid_request", "Request body is too large.", requestID)
		return false
	}
	mediaType, parameters, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != managementapi.JSONMediaType ||
		(len(parameters) != 0 && (len(parameters) != 1 || !strings.EqualFold(parameters["charset"], "utf-8"))) {
		writeProviderError(response, http.StatusUnsupportedMediaType, "unsupported_media_type", "Use the Management API media type.", requestID)
		return false
	}
	request.Body = http.MaxBytesReader(response, request.Body, maximumSubjectBodyBytes)
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

func requireSubjectRevision(response http.ResponseWriter, request *http.Request, requestID, kind string) (uint64, bool) {
	values := request.Header.Values(managementapi.HeaderIfMatch)
	if len(values) != 1 {
		writeProviderError(response, http.StatusPreconditionRequired, "precondition_required", "If-Match is required.", requestID)
		return 0, false
	}
	match := subjectETagPattern.FindStringSubmatch(values[0])
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

func subjectETag(kind string, revision uint64) string {
	if kind == "team_membership" {
		kind = "membership"
	}
	return `"` + kind + `:` + strconv.FormatUint(revision, 10) + `"`
}

func subjectDeleteRequest(response http.ResponseWriter, request *http.Request, requestID string) bool {
	if request.URL.RawQuery != "" || request.ContentLength > 0 || len(request.TransferEncoding) != 0 {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Delete accepts no query or request body.", requestID)
		return false
	}
	return true
}

func userPathValue(path string) (string, bool, bool) {
	value := strings.TrimPrefix(path, usersPath+"/")
	parts := strings.Split(value, "/")
	if len(parts) < 1 || len(parts) > 2 || !canonicalUUID(parts[0]) {
		return "", false, false
	}
	if len(parts) == 2 && parts[1] != "memberships" {
		return "", false, false
	}
	return parts[0], len(parts) == 2, true
}

func teamPathValue(path string) (string, string, string, bool) {
	value := strings.TrimPrefix(path, teamsPath+"/")
	parts := strings.Split(value, "/")
	if len(parts) < 1 || len(parts) > 3 || !canonicalUUID(parts[0]) {
		return "", "", "", false
	}
	if len(parts) == 1 {
		return parts[0], "", "team", true
	}
	if parts[1] != "members" {
		return "", "", "", false
	}
	if len(parts) == 2 {
		return parts[0], "", "members", true
	}
	if !canonicalUUID(parts[2]) {
		return "", "", "", false
	}
	return parts[0], parts[2], "membership", true
}

func subjectOperationKey(method managementapi.HTTPMethod, path string) string {
	return string(method) + " " + path
}

type subjectHTTPContract struct {
	method managementapi.HTTPMethod
	path   string
}

func subjectHTTPContracts() []subjectHTTPContract {
	return []subjectHTTPContract{
		{managementapi.MethodGET, usersPath},
		{managementapi.MethodPOST, usersPath},
		{managementapi.MethodGET, usersPath + "/{userId}"},
		{managementapi.MethodPATCH, usersPath + "/{userId}"},
		{managementapi.MethodDELETE, usersPath + "/{userId}"},
		{managementapi.MethodGET, usersPath + "/{userId}/memberships"},
		{managementapi.MethodGET, teamsPath},
		{managementapi.MethodPOST, teamsPath},
		{managementapi.MethodGET, teamsPath + "/{teamId}"},
		{managementapi.MethodPATCH, teamsPath + "/{teamId}"},
		{managementapi.MethodDELETE, teamsPath + "/{teamId}"},
		{managementapi.MethodGET, teamsPath + "/{teamId}/members"},
		{managementapi.MethodPUT, teamsPath + "/{teamId}/members/{userId}"},
		{managementapi.MethodPATCH, teamsPath + "/{teamId}/members/{userId}"},
		{managementapi.MethodDELETE, teamsPath + "/{teamId}/members/{userId}"},
	}
}

func writeSubjectError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, subjectmanagement.ErrInvalidRequest):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request is invalid.", requestID)
	case errors.Is(err, subjectmanagement.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	case errors.Is(err, subjectmanagement.ErrAlreadyExists):
		writeProviderError(response, http.StatusConflict, "already_exists", "Resource already exists.", requestID)
	case errors.Is(err, subjectmanagement.ErrRevisionConflict):
		writeProviderError(response, http.StatusPreconditionFailed, "revision_conflict", "Resource changed. Refresh and retry.", requestID)
	case errors.Is(err, subjectmanagement.ErrDefaultsUnavailable):
		writeProviderError(response, http.StatusConflict, "team_defaults_unavailable", "Team defaults are unavailable.", requestID)
	case errors.Is(err, subjectmanagement.ErrPolicySelectionUnavailable):
		writeProviderError(response, http.StatusConflict, "team_policy_unavailable", "A selected Team policy is unavailable.", requestID)
	case errors.Is(err, managementcommand.ErrConflict):
		writeProviderError(response, http.StatusConflict, "idempotency_conflict", "Idempotency-Key was already used for a different request.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "subject_service_unavailable", "Subject Management is unavailable.", requestID)
	}
}
