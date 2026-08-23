package managementserver

import (
	"errors"
	"net/http"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

func (routes *RoutingRoutes) snapshotResource(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
) {
	if request.Method != http.MethodGet {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	namespaceID := request.PathValue("namespaceId")
	if !canonicalUUID(namespaceID) {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	revisionText := request.PathValue("routingRevision")
	if revisionText == "" {
		routes.listSnapshots(response, request, requestID, namespaceID)
		return
	}
	revision, err := strconv.ParseInt(revisionText, 10, 64)
	if err != nil || revision <= 0 || strconv.FormatInt(revision, 10) != revisionText {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	routes.getSnapshot(response, request, requestID, namespaceID, revision)
}

func (routes *RoutingRoutes) listSnapshots(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	pathNamespaceID string,
) {
	if !rejectRoutingBody(response, request, requestID) {
		return
	}
	pageRequest, pageSize, err := parseRoutingSnapshotListQuery(request.URL.RawQuery)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Routing snapshot query is invalid.", requestID)
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	if namespaceID != pathNamespaceID {
		writeProviderError(response, http.StatusBadRequest, "invalid_namespace", "A valid namespace is required.", requestID)
		return
	}
	_, err = routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodGET, routingSnapshotsPath), nil, nil)
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, false)
		return
	}
	page, err := routes.service.ListSnapshots(request.Context(), namespaceID, pageRequest)
	if err != nil {
		writeRoutingSnapshotError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, routingSnapshotPageDTO(page, pageSize), requestID)
}

func (routes *RoutingRoutes) getSnapshot(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	pathNamespaceID string,
	routingRevision int64,
) {
	if request.URL.RawQuery != "" || !rejectRoutingBody(response, request, requestID) {
		if request.URL.RawQuery != "" {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Routing snapshot detail does not accept query parameters.", requestID)
		}
		return
	}
	namespaceID, session, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	if namespaceID != pathNamespaceID {
		writeProviderError(response, http.StatusBadRequest, "invalid_namespace", "A valid namespace is required.", requestID)
		return
	}
	_, err := routes.authorize(request.Context(), session, namespaceID,
		routes.operation(managementapi.MethodGET, routingSnapshotsPath+"/{routingRevision}"), nil, nil)
	if err != nil {
		writeRoutingAuthorizationError(response, err, requestID, false)
		return
	}
	detail, err := routes.service.GetSnapshot(request.Context(), namespaceID, routingRevision)
	if err != nil {
		writeRoutingSnapshotError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, managementapi.RoutingSnapshotDetail{
		Data: routingSnapshotRecordDTO(detail),
	}, requestID)
}

func parseRoutingSnapshotListQuery(raw string) (routingmanagement.SnapshotPageRequest, int, error) {
	query, err := strictRoutingQuery(raw, map[string]bool{"cursor": true, "pageSize": true})
	if err != nil {
		return routingmanagement.SnapshotPageRequest{}, 0, err
	}
	pageSize := 50
	if value := query.Get("pageSize"); value != "" {
		pageSize, err = strconv.Atoi(value)
		if err != nil || pageSize < 1 || pageSize > 200 {
			return routingmanagement.SnapshotPageRequest{}, 0, errors.New("pageSize is invalid")
		}
	}
	return routingmanagement.SnapshotPageRequest{
		PageSize: pageSize,
		Cursor:   query.Get("cursor"),
	}, pageSize, nil
}

func writeRoutingSnapshotError(response http.ResponseWriter, err error, requestID string) {
	switch {
	case errors.Is(err, routingmanagement.ErrInvalid):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Routing snapshot request is invalid.", requestID)
	case errors.Is(err, routingmanagement.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "Routing snapshot not found.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "routing_snapshot_unavailable", "Routing snapshot is unavailable.", requestID)
	}
}
