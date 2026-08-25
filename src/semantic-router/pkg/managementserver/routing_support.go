package managementserver

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

var routingETagPattern = regexp.MustCompile(`^"(mdl|rcp|ep|routing):([0-9]+)"$`)

func decodeRoutingBody(response http.ResponseWriter, request *http.Request, requestID string, target any) bool {
	if request.ContentLength > maximumRoutingBodyBytes {
		writeProviderError(response, http.StatusRequestEntityTooLarge, "invalid_request", "Routing request body is too large.", requestID)
		return false
	}
	mediaType, parameters, err := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if err != nil || mediaType != managementapi.JSONMediaType ||
		(len(parameters) != 0 && (len(parameters) != 1 || !strings.EqualFold(parameters["charset"], "utf-8"))) {
		writeProviderError(response, http.StatusUnsupportedMediaType, "unsupported_media_type", "Use the Management API media type.", requestID)
		return false
	}
	request.Body = http.MaxBytesReader(response, request.Body, maximumRoutingBodyBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	decoder.UseNumber()
	if err := decoder.Decode(target); err != nil {
		var maximum *http.MaxBytesError
		if errors.As(err, &maximum) {
			writeProviderError(response, http.StatusRequestEntityTooLarge, "invalid_request", "Routing request body is too large.", requestID)
		} else {
			writeProviderError(response, http.StatusBadRequest, "invalid_request", "Routing request body is invalid.", requestID)
		}
		return false
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Routing request body is invalid.", requestID)
		return false
	}
	return true
}

func rejectRoutingBody(response http.ResponseWriter, request *http.Request, requestID string) bool {
	if request.ContentLength > 0 || len(request.TransferEncoding) != 0 {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "This operation does not accept a request body.", requestID)
		return false
	}
	return true
}

func parseRoutingListQuery(raw string) (routingmanagement.PageRequest, int, error) {
	query, err := strictRoutingQuery(raw, map[string]bool{
		"cursor": true, "pageSize": true, "search": true, "status": true,
	})
	if err != nil {
		return routingmanagement.PageRequest{}, 0, err
	}
	pageSize := 50
	if value := query.Get("pageSize"); value != "" {
		pageSize, err = strconv.Atoi(value)
		if err != nil || pageSize < 1 || pageSize > 200 {
			return routingmanagement.PageRequest{}, 0, errors.New("pageSize is invalid")
		}
	}
	status := routingmanagement.Status(query.Get("status"))
	if status != "" && status != routingmanagement.StatusDraft && status != routingmanagement.StatusActive &&
		status != routingmanagement.StatusDisabled {
		return routingmanagement.PageRequest{}, 0, errors.New("status is invalid")
	}
	search := query.Get("search")
	if len(search) > 256 || strings.TrimSpace(search) != search {
		return routingmanagement.PageRequest{}, 0, errors.New("search is invalid")
	}
	return routingmanagement.PageRequest{
		PageSize: pageSize, Cursor: query.Get("cursor"), Search: search, Status: status,
	}, pageSize, nil
}

func strictRoutingQuery(raw string, allowed map[string]bool) (url.Values, error) {
	if len(raw) > maximumRoutingQueryBytes || strings.Contains(raw, ";") {
		return nil, errors.New("query is too large or malformed")
	}
	values, err := url.ParseQuery(raw)
	if err != nil {
		return nil, err
	}
	for name, entries := range values {
		if !allowed[name] || len(entries) != 1 {
			return nil, errors.New("query contains an unknown or repeated parameter")
		}
	}
	return values, nil
}

func requireRoutingRevision(
	response http.ResponseWriter, request *http.Request, requestID, kind string,
) (int64, bool) {
	values := request.Header.Values(managementapi.HeaderIfMatch)
	if len(values) != 1 {
		writeProviderError(response, http.StatusPreconditionRequired, "precondition_required", "If-Match is required.", requestID)
		return 0, false
	}
	match := routingETagPattern.FindStringSubmatch(values[0])
	if len(match) != 3 || match[1] != kind {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return 0, false
	}
	revision, err := strconv.ParseInt(match[2], 10, 64)
	if err != nil || revision < 0 || (revision == 0 && kind != "routing") {
		writeProviderError(response, http.StatusBadRequest, "invalid_precondition", "If-Match is invalid.", requestID)
		return 0, false
	}
	return revision, true
}

func routingETag(kind string, revision int64) string {
	return `"` + kind + `:` + strconv.FormatInt(revision, 10) + `"`
}

func canonicalRoutingRequest(value any) ([]byte, error) {
	wire, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(wire))
	decoder.UseNumber()
	var canonical any
	if err := decoder.Decode(&canonical); err != nil {
		return nil, err
	}
	return json.Marshal(canonical)
}

func (routes *RoutingRoutes) bindCommand(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	namespaceID string,
	session managementauth.AuthenticatedSession,
	endpoint string,
	payload any,
) (managementcommand.Command, bool) {
	key, ok := requireIdempotencyKey(response, request, requestID)
	if !ok {
		return managementcommand.Command{}, false
	}
	canonical, err := canonicalRoutingRequest(payload)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Routing request body is invalid.", requestID)
		return managementcommand.Command{}, false
	}
	now := routes.now().UTC()
	command, err := routes.commands.Bind(
		managementcommand.NamespaceCommandScope(namespaceID), session.Session.PrincipalID,
		endpoint, string(key), canonical, now, now.Add(routes.idempotencyTTL),
	)
	if err != nil {
		writeProviderError(response, http.StatusBadRequest, "invalid_idempotency_key", "A valid Idempotency-Key is required.", requestID)
		return managementcommand.Command{}, false
	}
	return command, true
}

func (routes *RoutingRoutes) lookupCommand(
	ctx context.Context, command managementcommand.Command,
) (managementcommand.StoredResult, bool, error) {
	result, found, err := routes.commandResults.Lookup(ctx, command)
	if err != nil {
		return managementcommand.StoredResult{}, false, fmt.Errorf("lookup Routing Management command: %w", err)
	}
	return result, found, nil
}

func routingResourceReplay(
	stored managementcommand.StoredResult, resourceType, resourceID string,
) (routingmanagement.RevisionReceipt, error) {
	if stored.Resource == nil || stored.Operation != nil || stored.Resource.ResourceType != resourceType ||
		(resourceID != "" && stored.Resource.ResourceID != resourceID) {
		return routingmanagement.RevisionReceipt{}, managementcommand.ErrConflict
	}
	return routingmanagement.RevisionReceipt{
		ResourceRevision: safeRevision(stored.Resource.ResourceRevision), Replayed: true,
	}, nil
}

func routingOperationReplay(
	stored managementcommand.StoredResult, desiredRevision bool,
) (routingmanagement.RevisionReceipt, error) {
	if stored.Operation == nil || stored.Resource != nil ||
		(desiredRevision != (stored.Operation.DesiredRevision != nil)) {
		return routingmanagement.RevisionReceipt{}, managementcommand.ErrConflict
	}
	receipt := routingmanagement.RevisionReceipt{OperationID: stored.Operation.OperationID, Replayed: true}
	if stored.Operation.DesiredRevision != nil {
		receipt.DesiredRevision = safeRevision(*stored.Operation.DesiredRevision)
	}
	return receipt, nil
}

func routingMutationContext(
	session managementauth.AuthenticatedSession,
	requestID, reason string,
	command *managementcommand.Command,
) routingmanagement.MutationContext {
	return routingmanagement.MutationContext{
		PrincipalID: session.Session.PrincipalID, ActorChain: []string{session.Session.PrincipalID},
		RequestID: requestID, Reason: reason, Command: command,
	}
}

func writeRoutingDomainError(response http.ResponseWriter, err error, requestID string, cas bool) {
	switch {
	case errors.Is(err, managementcommand.ErrConflict):
		writeProviderError(response, http.StatusConflict, "idempotency_conflict", "Idempotency-Key was already used for a different request.", requestID)
	case errors.Is(err, routingmanagement.ErrManifest):
		writeProviderError(response, http.StatusBadRequest, "invalid_manifest", "Routing manifest is invalid.", requestID)
	case errors.Is(err, routingmanagement.ErrInvalid):
		writeProviderError(response, http.StatusBadRequest, "invalid_request", "Routing request is invalid.", requestID)
	case errors.Is(err, routingmanagement.ErrNotFound):
		writeProviderError(response, http.StatusNotFound, "not_found", "Routing resource not found.", requestID)
	case errors.Is(err, routingmanagement.ErrImmutable):
		writeProviderError(response, http.StatusConflict, "immutable_resource", "Built-in Recipes cannot be changed. Create a custom Recipe to make edits.", requestID)
	case errors.Is(err, routingmanagement.ErrConflict) && cas:
		writeProviderError(response, http.StatusPreconditionFailed, "revision_conflict", "The Routing resource changed. Refresh and retry.", requestID)
	case errors.Is(err, routingmanagement.ErrConflict):
		writeProviderError(response, http.StatusConflict, "conflict", "Routing resource conflicts with existing state.", requestID)
	case errors.Is(err, routingmanagement.ErrReferenced):
		writeProviderError(response, http.StatusConflict, "resource_in_use", "Routing resource is still referenced.", requestID)
	case errors.Is(err, routingmanagement.ErrClaim):
		writeProviderError(response, http.StatusConflict, "stale_discovery", "The discovery result is stale. Discover Models again.", requestID)
	case errors.Is(err, routingmanagement.ErrPublication):
		writeProviderError(response, http.StatusConflict, "publication_blocked", "Entrypoint cannot be published in its current state.", requestID)
	case errors.Is(err, routingmanagement.ErrProbeUnavailable):
		writeProviderError(response, http.StatusServiceUnavailable, "probe_unavailable", "Model probe is unavailable.", requestID)
	default:
		writeProviderError(response, http.StatusServiceUnavailable, "routing_unavailable", "Routing Management is unavailable.", requestID)
	}
}

func writeRoutingResourceReceipt(
	response http.ResponseWriter,
	status int,
	kind, id string,
	receipt routingmanagement.RevisionReceipt,
	idempotent bool,
	requestID string,
) {
	var replayed *bool
	if idempotent {
		value := receipt.Replayed
		replayed = &value
		setIdempotencyReplayHeader(response, receipt.Replayed)
	}
	writeProviderJSON(response, status, managementapi.NewResourceMutationReceipt(
		kind, id, publicRevision(receipt.ResourceRevision), replayed,
	), requestID)
}

func writeRoutingOperationReceipt(
	response http.ResponseWriter,
	receipt routingmanagement.RevisionReceipt,
	includeDesired bool,
	requestID string,
) {
	var desired *uint64
	if includeDesired {
		value := publicRevision(receipt.DesiredRevision)
		desired = &value
	}
	replayed := receipt.Replayed
	setIdempotencyReplayHeader(response, receipt.Replayed)
	writeProviderJSON(response, http.StatusAccepted, managementapi.NewOperationMutationReceipt(
		receipt.OperationID, desired, &replayed,
	), requestID)
}
