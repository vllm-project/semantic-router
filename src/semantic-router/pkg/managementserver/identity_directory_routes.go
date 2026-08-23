package managementserver

import (
	"encoding/json"
	"errors"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

const (
	principalDirectoryPath = managementapi.BasePath + "/namespaces/{namespaceId}/principal-directory"
	principalLinksPath     = managementapi.BasePath + "/namespaces/{namespaceId}/principal-user-links"
)

var principalLinkETagPattern = regexp.MustCompile(`^"principal-user-link:([1-9][0-9]*)"$`)

func isIdentityDirectoryOperation(path string) bool {
	return path == principalDirectoryPath || path == principalDirectoryPath+"/{principalId}" ||
		path == principalLinksPath || path == principalLinksPath+"/{principalId}"
}

func registerIdentityDirectoryRoutes(mux *http.ServeMux, routes *IdentityResourceRoutes) {
	for _, pattern := range []string{
		"GET " + principalDirectoryPath,
		"GET " + principalDirectoryPath + "/{principalId}",
		"GET " + principalLinksPath,
		"PUT " + principalLinksPath + "/{principalId}",
		"DELETE " + principalLinksPath + "/{principalId}",
		"GET " + principalPath + "/{principalId}/user-links",
	} {
		mux.Handle(pattern, routes)
	}
}

func (routes *IdentityResourceRoutes) serveIdentityDirectory(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
) bool {
	if namespaceID, resource, principalID, ok := parseNamespaceIdentityPath(request.URL.Path); ok {
		switch resource {
		case "principal-directory":
			if principalID == "" && request.Method == http.MethodGet {
				routes.listPrincipalDirectory(response, request, requestID, namespaceID)
			} else if principalID != "" && request.Method == http.MethodGet {
				routes.getPrincipalDirectoryEntry(response, request, requestID, namespaceID, principalID)
			} else {
				writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
			}
		case "principal-user-links":
			switch {
			case principalID == "" && request.Method == http.MethodGet:
				routes.listNamespacePrincipalLinks(response, request, requestID, namespaceID)
			case principalID != "" && request.Method == http.MethodPut:
				routes.putPrincipalLink(response, request, requestID, namespaceID, principalID)
			case principalID != "" && request.Method == http.MethodDelete:
				routes.deletePrincipalLink(response, request, requestID, namespaceID, principalID)
			default:
				writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
			}
		}
		return true
	}
	if principalID, ok := parseGlobalPrincipalLinksPath(request.URL.Path); ok {
		if request.Method == http.MethodGet {
			routes.listGlobalPrincipalLinks(response, request, requestID, principalID)
		} else {
			writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		}
		return true
	}
	return false
}

func (routes *IdentityResourceRoutes) listPrincipalDirectory(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	namespaceID string,
) {
	pageRequest, ok := identityPageRequest(response, request, requestID, "search")
	if !ok {
		return
	}
	session, ok := routes.authenticate(response, request, requestID, namespaceID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(http.MethodGet, principalDirectoryPath), identityScopeTarget(namespaceID)) {
		return
	}
	includeEmail, ok := routes.directoryEmailVisibility(response, request, requestID, session, namespaceID)
	if !ok {
		return
	}
	page, err := routes.service.ListPrincipalDirectory(request.Context(), managementidentity.PrincipalDirectoryRequest{
		NamespaceID: namespaceID, Search: request.URL.Query().Get("search"),
		AfterID: pageRequest.AfterID, Limit: pageRequest.Limit,
	})
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	data := make([]managementapi.PrincipalDirectoryEntry, len(page.Items))
	for index := range page.Items {
		data[index] = principalDirectoryEntryDTO(page.Items[index], includeEmail)
	}
	writeProviderJSON(response, http.StatusOK, managementapi.PrincipalDirectoryPage{
		Data: data, Page: identityPageInfo(page.NextCursor, pageRequest.Limit),
	}, requestID)
}

func (routes *IdentityResourceRoutes) getPrincipalDirectoryEntry(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	namespaceID string,
	principalID string,
) {
	if request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Directory detail does not accept query parameters.", requestID)
		return
	}
	session, ok := routes.authenticate(response, request, requestID, namespaceID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(http.MethodGet, principalDirectoryPath+"/{principalId}"), identityScopeTarget(namespaceID)) {
		return
	}
	includeEmail, ok := routes.directoryEmailVisibility(response, request, requestID, session, namespaceID)
	if !ok {
		return
	}
	entry, err := routes.service.GetPrincipalDirectoryEntry(request.Context(), namespaceID, principalID)
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, managementapi.PrincipalDirectoryDetail{
		Data: principalDirectoryEntryDTO(entry, includeEmail),
	}, requestID)
}

func (routes *IdentityResourceRoutes) listNamespacePrincipalLinks(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	namespaceID string,
) {
	pageRequest, ok := identityPageRequest(response, request, requestID, "principalId", "userId")
	if !ok {
		return
	}
	session, ok := routes.authenticate(response, request, requestID, namespaceID)
	if !ok || !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(http.MethodGet, principalLinksPath), identityScopeTarget(namespaceID)) {
		return
	}
	page, err := routes.service.ListPrincipalUserLinks(request.Context(), managementidentity.PrincipalUserLinkListRequest{
		NamespaceID: namespaceID, PrincipalID: request.URL.Query().Get("principalId"),
		UserID: request.URL.Query().Get("userId"), AfterID: pageRequest.AfterID, Limit: pageRequest.Limit,
	})
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	writePrincipalUserLinkPage(response, requestID, page, pageRequest.Limit)
}

func (routes *IdentityResourceRoutes) listGlobalPrincipalLinks(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	principalID string,
) {
	pageRequest, ok := identityPageRequest(response, request, requestID)
	if !ok {
		return
	}
	session, ok := routes.authenticate(response, request, requestID, "")
	if !ok || !routes.authorize(response, request, requestID, session, "",
		routes.operation(http.MethodGet, principalPath+"/{principalId}/user-links"), nil) {
		return
	}
	page, err := routes.service.ListPrincipalLinks(request.Context(), principalID, pageRequest)
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	writePrincipalUserLinkPage(response, requestID, page, pageRequest.Limit)
}

func (routes *IdentityResourceRoutes) putPrincipalLink(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	namespaceID string,
	principalID string,
) {
	var body managementapi.PrincipalUserLinkPutRequest
	if !decodeIdentityBody(response, request, requestID, &body) {
		return
	}
	if !canonicalUUID(body.UserID) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "userId is invalid.", requestID)
		return
	}
	session, ok := routes.authenticate(response, request, requestID, namespaceID)
	if !ok {
		return
	}
	current, err := routes.service.GetPrincipalUserLink(request.Context(), principalID, namespaceID)
	currentUserID := body.UserID
	if err == nil {
		currentUserID = string(current.UserID)
	} else if !errors.Is(err, managementidentity.ErrNotFound) {
		writeIdentityError(response, err, requestID)
		return
	}
	targets := map[string][]accesscontrol.ScopedTarget{
		"current_owner": {principalLinkUserTarget(namespaceID, currentUserID)},
		"target_owner":  {principalLinkUserTarget(namespaceID, body.UserID)},
	}
	if !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(http.MethodPut, principalLinksPath+"/{principalId}"), targets) {
		return
	}
	expected, ok := optionalPrincipalLinkRevision(response, request, requestID)
	if !ok {
		return
	}
	key, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return
	}
	canonical, _ := json.Marshal(struct {
		UserID           string  `json:"userId"`
		ExpectedRevision *uint64 `json:"expectedRevision,omitempty"`
	}{body.UserID, expected})
	now := routes.now().UTC()
	command, err := routes.commands.Bind(
		managementcommand.NamespaceCommandScope(namespaceID), session.Session.PrincipalID,
		principalUserLinkResourcePath(namespaceID, principalID), string(key), canonical, now, now.Add(identityCommandTTL),
	)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Request is invalid.", requestID)
		return
	}
	result, err := routes.service.PutPrincipalUserLink(request.Context(), managementidentity.LinkMutation{
		PrincipalID: principalID, NamespaceID: namespaceID, UserID: body.UserID,
		ExpectedRevision: expected, Command: command,
		Actor: identityActor(request, session, requestID, "Link Management principal to User"),
	})
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	link, err := routes.service.GetPrincipalUserLink(request.Context(), principalID, namespaceID)
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, principalLinkETag(uint64(link.Revision)))
	response.Header().Set("Location", principalUserLinkResourcePath(namespaceID, principalID))
	setIdempotencyReplayHeader(response, result.Replayed)
	writeProviderJSON(response, result.ResponseStatus, managementapi.PrincipalUserLinkDetail{
		Data: principalUserLinkDTO(link),
	}, requestID)
}

func (routes *IdentityResourceRoutes) deletePrincipalLink(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	namespaceID string,
	principalID string,
) {
	if !noRequestBody(response, request, requestID) {
		return
	}
	session, ok := routes.authenticate(response, request, requestID, namespaceID)
	if !ok {
		return
	}
	current, err := routes.service.GetPrincipalUserLink(request.Context(), principalID, namespaceID)
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	if !routes.authorize(response, request, requestID, session, namespaceID,
		routes.operation(http.MethodDelete, principalLinksPath+"/{principalId}"),
		map[string][]accesscontrol.ScopedTarget{
			"current_owner": {principalLinkUserTarget(namespaceID, string(current.UserID))},
		}) {
		return
	}
	revision, ok := requirePrincipalLinkRevision(response, request, requestID)
	if !ok {
		return
	}
	result, err := routes.service.DeletePrincipalUserLink(request.Context(), managementidentity.LinkMutation{
		PrincipalID: principalID, NamespaceID: namespaceID, UserID: string(current.UserID),
		ExpectedRevision: &revision,
		Actor:            identityActor(request, session, requestID, "Delete Management principal User link"),
	})
	if err != nil {
		writeIdentityError(response, err, requestID)
		return
	}
	response.Header().Set(managementapi.HeaderETag, principalLinkETag(result.Revision))
	response.WriteHeader(http.StatusNoContent)
}

func (routes *IdentityResourceRoutes) directoryEmailVisibility(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	session managementauth.AuthenticatedSession,
	namespaceID string,
) (bool, bool) {
	_, err := routes.authorization.Authorize(request.Context(), AuthorizationRequest{
		Operation: routes.operation(http.MethodGet, principalLinksPath), Session: session,
		NamespaceID: namespaceID, Targets: identityScopeTarget(namespaceID),
	})
	if err == nil {
		return true, true
	}
	if errors.Is(err, managementauthorization.ErrDenied) {
		return false, true
	}
	writeProviderError(response, http.StatusServiceUnavailable, "authorization_unavailable", "Authorization state is unavailable.", requestID)
	return false, false
}

func writePrincipalUserLinkPage(
	response http.ResponseWriter,
	requestID string,
	page managementidentity.PrincipalUserLinkPage,
	pageSize int,
) {
	data := make([]managementapi.PrincipalUserLink, len(page.Items))
	for index := range page.Items {
		data[index] = principalUserLinkDTO(page.Items[index])
	}
	writeProviderJSON(response, http.StatusOK, managementapi.PrincipalUserLinkPage{
		Data: data, Page: identityPageInfo(page.NextCursor, pageSize),
	}, requestID)
}

func parseNamespaceIdentityPath(path string) (string, string, string, bool) {
	prefix := managementapi.BasePath + "/namespaces/"
	if !strings.HasPrefix(path, prefix) {
		return "", "", "", false
	}
	parts := strings.Split(strings.TrimPrefix(path, prefix), "/")
	if (len(parts) != 2 && len(parts) != 3) || !canonicalUUID(parts[0]) ||
		(parts[1] != "principal-directory" && parts[1] != "principal-user-links") {
		return "", "", "", false
	}
	principalID := ""
	if len(parts) == 3 {
		if !canonicalUUID(parts[2]) {
			return "", "", "", false
		}
		principalID = parts[2]
	}
	return parts[0], parts[1], principalID, true
}

func parseGlobalPrincipalLinksPath(path string) (string, bool) {
	prefix := principalPath + "/"
	if !strings.HasPrefix(path, prefix) {
		return "", false
	}
	parts := strings.Split(strings.TrimPrefix(path, prefix), "/")
	if len(parts) != 2 || !canonicalUUID(parts[0]) || parts[1] != "user-links" {
		return "", false
	}
	return parts[0], true
}

func principalLinkUserTarget(namespaceID, userID string) accesscontrol.ScopedTarget {
	return accesscontrol.ScopedTarget{Scope: accesscontrol.UserScope(
		accesscontrol.NamespaceID(namespaceID), accesscontrol.UserID(userID),
	)}
}

func principalUserLinkResourcePath(namespaceID, principalID string) string {
	return managementapi.BasePath + "/namespaces/" + namespaceID + "/principal-user-links/" + principalID
}

func optionalPrincipalLinkRevision(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
) (*uint64, bool) {
	values := request.Header.Values(managementapi.HeaderIfMatch)
	if len(values) == 0 {
		return nil, true
	}
	if len(values) != 1 {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return nil, false
	}
	match := principalLinkETagPattern.FindStringSubmatch(values[0])
	if len(match) != 2 {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return nil, false
	}
	revision, err := strconv.ParseUint(match[1], 10, 64)
	if err != nil || revision == 0 {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return nil, false
	}
	return &revision, true
}

func requirePrincipalLinkRevision(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
) (uint64, bool) {
	revision, ok := optionalPrincipalLinkRevision(response, request, requestID)
	if !ok {
		return 0, false
	}
	if revision == nil {
		writeProviderError(response, http.StatusPreconditionRequired, "precondition_required", "If-Match is required.", requestID)
		return 0, false
	}
	return *revision, true
}

func principalLinkETag(revision uint64) string {
	return `"principal-user-link:` + strconv.FormatUint(revision, 10) + `"`
}
