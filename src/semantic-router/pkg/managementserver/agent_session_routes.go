package managementserver

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

const (
	agentSSEBatchSize    = 200
	agentSSEPollInterval = 250 * time.Millisecond
	agentSSEHeartbeat    = 15 * time.Second
)

func (routes *AgentRoutes) sessionsCollection(
	response http.ResponseWriter, request *http.Request, requestID string,
) {
	switch request.Method {
	case http.MethodGet:
		pageRequest, err := parseAgentListQuery(request)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		authenticated, ok := routes.authenticate(response, request, requestID)
		if !ok {
			return
		}
		access, err := routes.accessContext(
			request.Context(), authenticated,
			routes.operation(managementapi.MethodGET, agentSessionsPath),
		)
		if err != nil {
			writeResultScopeError(response, err, requestID)
			return
		}
		pageRequest.Scope = access.Scope
		page, err := routes.service.ListSessions(
			request.Context(), authenticated.NamespaceID, pageRequest, access,
		)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		writeProviderJSON(response, http.StatusOK, newAgentPage(page, pageRequest.PageSize), requestID)
	case http.MethodPost:
		routes.createSession(response, request, requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *AgentRoutes) createSession(
	response http.ResponseWriter, request *http.Request, requestID string,
) {
	if request.URL.RawQuery != "" {
		writeAgentDomainError(response, agentmanagement.ErrInvalid, requestID)
		return
	}
	authenticated, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	idempotencyKey, ok := requireAgentIdempotency(response, request, requestID)
	if !ok {
		return
	}
	var body agentSessionCreateWire
	if !decodeAgentBody(response, request, requestID, &body) {
		return
	}
	input := agentmanagement.SessionInput{
		Mode: body.Mode, ProfileID: body.ProfileID, EffectiveTeamID: body.EffectiveTeamID,
		Target: body.Target, Title: body.Title,
	}
	prepared, createSessionErr := routes.service.PrepareSession(
		request.Context(), authenticated.NamespaceID,
		authenticated.Session.Session.PrincipalID, input,
	)
	if createSessionErr != nil {
		writeAgentDomainError(response, createSessionErr, requestID)
		return
	}
	if _, err := routes.authorize(
		request.Context(), authenticated,
		routes.operation(managementapi.MethodPOST, agentSessionsPath),
		map[string][]accesscontrol.ScopedTarget{
			"attributed_subject": {agentAttributedSubject(authenticated.NamespaceID, prepared)},
		},
	); err != nil {
		writeAgentAuthorizationError(response, err, requestID)
		return
	}
	access := accessForPreparedSession(authenticated, prepared)
	session, replayed, createSessionErr := routes.service.CreateSession(
		request.Context(), authenticated.NamespaceID,
		authenticated.Session.Session.PrincipalID, idempotencyKey, input,
		agentMutation(request, authenticated, requestID), access,
	)
	if createSessionErr != nil {
		writeAgentDomainError(response, createSessionErr, requestID)
		return
	}
	setAgentETag(response, session.Revision)
	response.Header().Set("Location", agentSessionsPath+"/"+session.ID)
	setIdempotencyReplayHeader(response, replayed)
	writeProviderJSON(response, http.StatusCreated, managementapi.NewResourceMutationReceipt(
		"agent_session", session.ID, uint64(session.Revision), &replayed,
	), requestID)
}

func (routes *AgentRoutes) sessionResource(
	response http.ResponseWriter, request *http.Request, requestID string,
) {
	path, ok := parseAgentSessionPath(request.URL.Path)
	if !ok {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	authenticated, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	resolved, err := routes.service.ResolveSessionAccess(
		request.Context(), authenticated.NamespaceID, path.sessionID,
	)
	if err != nil {
		writeAgentDomainError(response, err, requestID)
		return
	}
	contractPath := path.contractPath()
	operation := routes.operation(managementapi.HTTPMethod(request.Method), contractPath)
	if operation.Path == "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	target := agentSessionScopedTarget(authenticated.NamespaceID, resolved)
	if path.turnID != "" {
		target = agentTurnScopedTarget(authenticated.NamespaceID, path.turnID, resolved)
	}
	if _, err := routes.authorize(
		request.Context(), authenticated, operation,
		map[string][]accesscontrol.ScopedTarget{"target": {target}},
	); err != nil {
		writeAgentAuthorizationError(response, err, requestID)
		return
	}
	access := accessForResolvedSession(authenticated, resolved)
	switch path.action {
	case "session":
		routes.sessionDetail(response, request, requestID, authenticated, resolved, access)
	case "turns":
		routes.sessionTurns(response, request, requestID, authenticated, resolved, access)
	case "events":
		routes.sessionEvents(response, request, requestID, resolved, access)
	case "cancel":
		routes.cancelTurn(response, request, requestID, resolved, path.turnID, access)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *AgentRoutes) sessionDetail(
	response http.ResponseWriter, request *http.Request, requestID string,
	authenticated agentAuthenticatedRequest, resolved agentmanagement.SessionAccess,
	access agentmanagement.AccessContext,
) {
	if request.URL.RawQuery != "" {
		writeAgentDomainError(response, agentmanagement.ErrInvalid, requestID)
		return
	}
	switch request.Method {
	case http.MethodGet:
		session, err := routes.service.GetSession(
			request.Context(), authenticated.NamespaceID, resolved.ID, access,
		)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		setAgentETag(response, session.Revision)
		writeProviderJSON(response, http.StatusOK, agentDetail[agentmanagement.Session]{Data: session}, requestID)
	case http.MethodPatch:
		revision, ok := requireAgentRevision(response, request, requestID)
		if !ok {
			return
		}
		var patch agentmanagement.SessionPatch
		if !decodeAgentBody(response, request, requestID, &patch) {
			return
		}
		session, err := routes.service.PatchSession(
			request.Context(), authenticated.NamespaceID, resolved.ID, revision, patch,
			agentMutation(request, authenticated, requestID), access,
		)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		setAgentETag(response, session.Revision)
		writeProviderJSON(response, http.StatusOK, managementapi.NewResourceMutationReceipt(
			"agent_session", session.ID, uint64(session.Revision), nil,
		), requestID)
	case http.MethodDelete:
		revision, ok := requireAgentRevision(response, request, requestID)
		if !ok {
			return
		}
		result, err := routes.service.DeleteSession(
			request.Context(), authenticated.NamespaceID, resolved.ID, revision,
			agentMutation(request, authenticated, requestID), access,
		)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		setAgentETag(response, result)
		response.WriteHeader(http.StatusNoContent)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *AgentRoutes) sessionTurns(
	response http.ResponseWriter, request *http.Request, requestID string,
	authenticated agentAuthenticatedRequest, resolved agentmanagement.SessionAccess,
	access agentmanagement.AccessContext,
) {
	switch request.Method {
	case http.MethodGet:
		pageRequest, err := parseAgentListQuery(request)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		pageRequest.Scope = access.Scope
		page, err := routes.service.ListTurns(
			request.Context(), authenticated.NamespaceID, resolved.ID, pageRequest, access,
		)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		writeProviderJSON(response, http.StatusOK, newAgentPage(page, pageRequest.PageSize), requestID)
	case http.MethodPost:
		if request.URL.RawQuery != "" {
			writeAgentDomainError(response, agentmanagement.ErrInvalid, requestID)
			return
		}
		idempotencyKey, ok := requireAgentIdempotency(response, request, requestID)
		if !ok {
			return
		}
		var body agentTurnCreateWire
		if !decodeAgentBody(response, request, requestID, &body) {
			return
		}
		turn, replayed, err := routes.service.CreateTurn(
			request.Context(), authenticated.NamespaceID, resolved.ID,
			idempotencyKey, body.Input, access,
		)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		response.Header().Set("Location", agentSessionsPath+"/"+resolved.ID+"/turns/"+turn.ID)
		setIdempotencyReplayHeader(response, replayed)
		writeProviderJSON(response, http.StatusCreated, managementapi.NewResourceMutationReceipt(
			"agent_turn", turn.ID, uint64(turn.Revision), &replayed,
		), requestID)
	default:
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
	}
}

func (routes *AgentRoutes) cancelTurn(
	response http.ResponseWriter, request *http.Request, requestID string,
	resolved agentmanagement.SessionAccess, turnID string, access agentmanagement.AccessContext,
) {
	if request.Method != http.MethodPost || request.URL.RawQuery != "" ||
		!emptyAgentBody(response, request, requestID) {
		if request.Method != http.MethodPost {
			writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		}
		return
	}
	if _, ok := requireAgentIdempotency(response, request, requestID); !ok {
		return
	}
	turn, replayed, err := routes.service.RequestCancellation(
		request.Context(), string(access.Scope.NamespaceID), resolved.ID, turnID, access,
	)
	if err != nil {
		writeAgentDomainError(response, err, requestID)
		return
	}
	setIdempotencyReplayHeader(response, replayed)
	writeProviderJSON(response, http.StatusOK, managementapi.NewResourceMutationReceipt(
		"agent_turn", turn.ID, uint64(turn.Revision), &replayed,
	), requestID)
}

func (routes *AgentRoutes) sessionEvents(
	response http.ResponseWriter, request *http.Request, requestID string,
	resolved agentmanagement.SessionAccess, access agentmanagement.AccessContext,
) {
	if request.Method != http.MethodGet {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	if acceptsAgentSSE(request) {
		routes.streamSessionEvents(response, request, requestID, resolved, access)
		return
	}
	values, err := strictAgentQuery(request.URL.RawQuery, map[string]bool{"cursor": true, "pageSize": true})
	if err != nil || request.Header.Get("Last-Event-ID") != "" {
		writeAgentDomainError(response, agentmanagement.ErrInvalid, requestID)
		return
	}
	pageSize, err := parseOptionalPageSize(values.Get("pageSize"))
	if err != nil {
		writeAgentDomainError(response, agentmanagement.ErrInvalid, requestID)
		return
	}
	if pageSize == 0 {
		pageSize = defaultAgentPageSize
	}
	page, err := routes.service.ListEventHistory(
		request.Context(), string(access.Scope.NamespaceID), resolved.ID,
		agentmanagement.EventPageRequest{PageSize: pageSize, Cursor: values.Get("cursor"), Scope: access.Scope},
		access,
	)
	if err != nil {
		writeAgentDomainError(response, err, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, newAgentPage(page, pageSize), requestID)
}

func (routes *AgentRoutes) streamSessionEvents(
	response http.ResponseWriter, request *http.Request, requestID string,
	resolved agentmanagement.SessionAccess, access agentmanagement.AccessContext,
) {
	after, explicit, err := parseAgentResumeSequence(request)
	if err != nil {
		writeAgentDomainError(response, err, requestID)
		return
	}
	liveSubscription, err := routes.liveEvents.SubscribeLiveModelSteps(
		request.Context(), string(access.Scope.NamespaceID), resolved.ID,
	)
	if err != nil {
		writeProviderError(response, http.StatusServiceUnavailable, "stream_unavailable", "Live updates are unavailable.", requestID)
		return
	}
	defer liveSubscription.Close()
	if !explicit {
		latest, historyErr := routes.service.ListEventHistory(
			request.Context(), string(access.Scope.NamespaceID), resolved.ID,
			agentmanagement.EventPageRequest{PageSize: 1, Scope: access.Scope}, access,
		)
		if historyErr != nil {
			writeAgentDomainError(response, historyErr, requestID)
			return
		}
		if len(latest.Items) > 0 {
			after = latest.Items[len(latest.Items)-1].Sequence
		}
	}
	initial, _, err := routes.service.ResumeEvents(
		request.Context(), string(access.Scope.NamespaceID), resolved.ID,
		after, agentSSEBatchSize, access,
	)
	if err != nil {
		writeAgentDomainError(response, err, requestID)
		return
	}
	flusher, ok := response.(http.Flusher)
	if !ok {
		writeProviderError(response, http.StatusNotImplemented, "stream_unavailable", "Streaming is unavailable.", requestID)
		return
	}
	setProviderResponseHeaders(response, requestID)
	response.Header().Set("Content-Type", managementapi.EventStreamMediaType)
	response.Header().Set("Connection", "keep-alive")
	response.WriteHeader(http.StatusOK)
	if err := writeAgentSSEEvents(response, initial); err != nil {
		return
	}
	if len(initial) > 0 {
		after = initial[len(initial)-1].Sequence
	}
	liveState := newAgentLiveStreamState()
	if err := liveState.observeDurable(initial); err != nil {
		return
	}
	flusher.Flush()
	poll := time.NewTicker(agentSSEPollInterval)
	heartbeat := time.NewTicker(agentSSEHeartbeat)
	defer poll.Stop()
	defer heartbeat.Stop()
	liveEvents := liveSubscription.Events()
	catchUpDurable := func() error {
		for {
			events, hasMore, resumeErr := routes.service.ResumeEvents(
				request.Context(), string(access.Scope.NamespaceID), resolved.ID,
				after, agentSSEBatchSize, access,
			)
			if resumeErr != nil {
				return resumeErr
			}
			if len(events) == 0 {
				return nil
			}
			if err := liveState.observeDurable(events); err != nil {
				return err
			}
			if err := writeAgentSSEEvents(response, events); err != nil {
				return err
			}
			after = events[len(events)-1].Sequence
			flusher.Flush()
			if !hasMore {
				return nil
			}
		}
	}
	for {
		select {
		case <-request.Context().Done():
			return
		case live, open := <-liveEvents:
			if !open {
				// Preview transport is acceleration only. Durable polling keeps
				// the stream correct if Pub/Sub is interrupted.
				liveEvents = nil
				continue
			}
			if live.SessionID != resolved.ID {
				continue
			}
			if live.Phase == agentmanagement.LiveModelStepCommitted {
				// CommitModelStep completes before the marker is published. Read
				// its authoritative deltas first so preview text never duplicates
				// or appears after its durable replacement.
				if err := catchUpDurable(); err != nil {
					return
				}
			}
			accepted := liveState.accept(live)
			if accepted == nil {
				continue
			}
			if err := writeAgentSSELiveEvent(response, *accepted); err != nil {
				return
			}
			flusher.Flush()
		case <-heartbeat.C:
			if _, err := response.Write([]byte(": keepalive\n\n")); err != nil {
				return
			}
			flusher.Flush()
		case <-poll.C:
			if err := catchUpDurable(); err != nil {
				return
			}
		}
	}
}

func (routes *AgentRoutes) artifact(
	response http.ResponseWriter, request *http.Request, requestID string,
) {
	id, content, ok := parseAgentArtifactPath(request.URL.Path)
	if !ok || request.Method != http.MethodGet || request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	authenticated, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	resolved, artifactErr := routes.service.ResolveArtifactAccess(request.Context(), authenticated.NamespaceID, id)
	if artifactErr != nil {
		writeAgentDomainError(response, artifactErr, requestID)
		return
	}
	contractPath := agentArtifactsPath + "/{artifact}"
	if content {
		contractPath += "/content"
	}
	operation := routes.operation(managementapi.MethodGET, contractPath)
	target := agentArtifactScopedTarget(authenticated.NamespaceID, resolved)
	if _, err := routes.authorize(
		request.Context(), authenticated, operation,
		map[string][]accesscontrol.ScopedTarget{"target": {target}},
	); err != nil {
		writeAgentAuthorizationError(response, err, requestID)
		return
	}
	access := accessForResolvedSession(authenticated, resolved.Session)
	if content {
		value, err := routes.service.GetArtifactContent(request.Context(), authenticated.NamespaceID, id, access)
		if err != nil {
			writeAgentDomainError(response, err, requestID)
			return
		}
		writeProviderJSON(response, http.StatusOK, agentDetail[agentmanagement.ArtifactContent]{Data: value}, requestID)
		return
	}
	value, artifactErr := routes.service.GetArtifactMetadata(request.Context(), authenticated.NamespaceID, id, access)
	if artifactErr != nil {
		writeAgentDomainError(response, artifactErr, requestID)
		return
	}
	writeProviderJSON(response, http.StatusOK, agentDetail[agentmanagement.Artifact]{Data: value}, requestID)
}

func (routes *AgentRoutes) commitPublication(
	response http.ResponseWriter, request *http.Request, requestID string,
) {
	planID, action, ok := agentResourcePathValue(agentPublicationPlanPath, request.URL.Path)
	if !ok || action != "commit" || request.Method != http.MethodPost || request.URL.RawQuery != "" {
		writeProviderError(response, http.StatusNotFound, "not_found", "Resource not found.", requestID)
		return
	}
	authenticated, ok := routes.authenticate(response, request, requestID)
	if !ok {
		return
	}
	_, ok = requireAgentRevision(response, request, requestID)
	if !ok {
		return
	}
	idempotencyKey, ok := requireAgentIdempotency(response, request, requestID)
	if !ok {
		return
	}
	var body agentPublicationCommitWire
	if !decodeAgentBody(response, request, requestID, &body) {
		return
	}
	resolved, commitPublicationErr := routes.service.ResolvePublicationAccess(request.Context(), authenticated.NamespaceID, planID)
	if commitPublicationErr != nil {
		writeAgentDomainError(response, commitPublicationErr, requestID)
		return
	}
	operation := routes.operation(
		managementapi.MethodPOST, agentPublicationPlanPath+"/{plan}:commit",
	)
	dependencies := []accesscontrol.ScopedTarget{
		agentRoutingResourceTarget(authenticated.NamespaceID, accesscontrol.ScopeResourceRecipe, resolved.RecipeID),
		agentRoutingResourceTarget(authenticated.NamespaceID, accesscontrol.ScopeResourceEntrypoint, resolved.EntrypointID),
	}
	for _, modelID := range resolved.ModelIDs {
		dependencies = append(dependencies, agentRoutingResourceTarget(
			authenticated.NamespaceID, accesscontrol.ScopeResourceModel, modelID,
		))
	}
	if _, err := routes.authorize(
		request.Context(), authenticated, operation,
		map[string][]accesscontrol.ScopedTarget{
			"target": {agentRoutingResourceTarget(
				authenticated.NamespaceID, accesscontrol.ScopeResourceEntrypoint, resolved.EntrypointID,
			)},
			"all_dependencies": dependencies,
		},
	); err != nil {
		writeAgentAuthorizationError(response, err, requestID)
		return
	}
	result, commitPublicationErr := routes.publications.Commit(request.Context(), AgentPublicationCommitRequest{
		NamespaceID: authenticated.NamespaceID, PlanID: planID, PlanDigest: body.PlanDigest,
		ExpectedETag: request.Header.Get(managementapi.HeaderIfMatch), IdempotencyKey: idempotencyKey,
		Mutation: agentMutation(request, authenticated, requestID),
		Access:   accessForResolvedSession(authenticated, resolved.Session),
	})
	if commitPublicationErr != nil {
		writeAgentDomainError(response, commitPublicationErr, requestID)
		return
	}
	desired := uint64(result.Revision)
	setIdempotencyReplayHeader(response, result.Replayed)
	writeProviderJSON(response, http.StatusAccepted, managementapi.NewOperationMutationReceipt(
		result.OperationID, &desired, &result.Replayed,
	), requestID)
}

type parsedAgentSessionPath struct {
	sessionID string
	turnID    string
	action    string
}

func (path parsedAgentSessionPath) contractPath() string {
	switch path.action {
	case "session":
		return agentSessionsPath + "/{session}"
	case "turns":
		return agentSessionsPath + "/{session}/turns"
	case "events":
		return agentSessionsPath + "/{session}/events"
	case "cancel":
		return agentSessionsPath + "/{session}/turns/{turn}:cancel"
	default:
		return ""
	}
}

func parseAgentSessionPath(path string) (parsedAgentSessionPath, bool) {
	value := strings.TrimPrefix(path, agentSessionsPath+"/")
	if value == path || value == "" {
		return parsedAgentSessionPath{}, false
	}
	parts := strings.Split(value, "/")
	if uuid.Validate(parts[0]) != nil {
		return parsedAgentSessionPath{}, false
	}
	parsed := parsedAgentSessionPath{sessionID: parts[0]}
	switch {
	case len(parts) == 1:
		parsed.action = "session"
	case len(parts) == 2 && parts[1] == "turns":
		parsed.action = "turns"
	case len(parts) == 2 && parts[1] == "events":
		parsed.action = "events"
	case len(parts) == 3 && parts[1] == "turns" && strings.HasSuffix(parts[2], ":cancel"):
		parsed.turnID = strings.TrimSuffix(parts[2], ":cancel")
		if uuid.Validate(parsed.turnID) != nil {
			return parsedAgentSessionPath{}, false
		}
		parsed.action = "cancel"
	default:
		return parsedAgentSessionPath{}, false
	}
	return parsed, true
}

func parseAgentArtifactPath(path string) (string, bool, bool) {
	value := strings.TrimPrefix(path, agentArtifactsPath+"/")
	if value == path || value == "" {
		return "", false, false
	}
	parts := strings.Split(value, "/")
	if uuid.Validate(parts[0]) != nil || (len(parts) != 1 && (len(parts) != 2 || parts[1] != "content")) {
		return "", false, false
	}
	return parts[0], len(parts) == 2, true
}

func parseAgentResumeSequence(request *http.Request) (int64, bool, error) {
	values, err := strictAgentQuery(request.URL.RawQuery, map[string]bool{"afterSequence": true})
	if err != nil {
		return 0, false, err
	}
	queryValue := values.Get("afterSequence")
	headerValue := request.Header.Get("Last-Event-ID")
	if queryValue != "" && headerValue != "" {
		return 0, false, agentmanagement.ErrInvalid
	}
	value := queryValue
	if value == "" {
		value = headerValue
	}
	if value == "" {
		return 0, false, nil
	}
	sequence, err := strconv.ParseInt(value, 10, 64)
	if err != nil || sequence < 0 {
		return 0, false, agentmanagement.ErrInvalid
	}
	return sequence, true, nil
}

func acceptsAgentSSE(request *http.Request) bool {
	if selected := negotiatedManagementMedia(request); selected != "" {
		return selected == managementapi.EventStreamMediaType
	}
	selected, ok := selectManagementResponseMedia(request.Header.Values("Accept"), []string{
		managementapi.JSONMediaType, managementapi.EventStreamMediaType,
	})
	return ok && selected == managementapi.EventStreamMediaType
}

func writeAgentSSEEvents(response http.ResponseWriter, events []agentmanagement.Event) error {
	for _, event := range events {
		encoded, err := json.Marshal(event)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(
			response, "id: %d\nevent: %s\ndata: %s\n\n", event.Sequence, event.Type, encoded,
		); err != nil {
			return err
		}
	}
	return nil
}

func agentAttributedSubject(
	namespaceID string, prepared agentmanagement.SessionAuthorization,
) accesscontrol.ScopedTarget {
	if prepared.EffectiveTeamID != "" {
		return accesscontrol.ScopedTarget{Scope: accesscontrol.TeamScope(
			accesscontrol.NamespaceID(namespaceID), accesscontrol.TeamID(prepared.EffectiveTeamID),
		)}
	}
	return accesscontrol.ScopedTarget{Scope: accesscontrol.UserScope(
		accesscontrol.NamespaceID(namespaceID), accesscontrol.UserID(prepared.EffectiveUserID),
	)}
}

func agentSessionScopedTarget(
	namespaceID string, session agentmanagement.SessionAccess,
) accesscontrol.ScopedTarget {
	target := accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), accesscontrol.ScopeResourceAgentSession,
		accesscontrol.ResourceID(session.ID),
	)}
	return withAgentSessionAncestors(namespaceID, target, session)
}

func agentTurnScopedTarget(
	namespaceID, turnID string, session agentmanagement.SessionAccess,
) accesscontrol.ScopedTarget {
	target := accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), accesscontrol.ScopeResourceAgentTurn,
		accesscontrol.ResourceID(turnID),
	), Ancestors: []accesscontrol.Scope{accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), accesscontrol.ScopeResourceAgentSession,
		accesscontrol.ResourceID(session.ID),
	)}}
	return withAgentSessionAncestors(namespaceID, target, session)
}

func agentArtifactScopedTarget(
	namespaceID string, artifact agentmanagement.ArtifactAccess,
) accesscontrol.ScopedTarget {
	target := accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), accesscontrol.ScopeResourceAgentArtifact,
		accesscontrol.ResourceID(artifact.ID),
	), Ancestors: []accesscontrol.Scope{accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), accesscontrol.ScopeResourceAgentSession,
		accesscontrol.ResourceID(artifact.Session.ID),
	)}}
	return withAgentSessionAncestors(namespaceID, target, artifact.Session)
}

func withAgentSessionAncestors(
	namespaceID string, target accesscontrol.ScopedTarget, session agentmanagement.SessionAccess,
) accesscontrol.ScopedTarget {
	if session.EffectiveUserID != "" {
		target.Ancestors = append(target.Ancestors, accesscontrol.UserScope(
			accesscontrol.NamespaceID(namespaceID), accesscontrol.UserID(session.EffectiveUserID),
		))
	}
	if session.EffectiveTeamID != "" {
		target.Ancestors = append(target.Ancestors, accesscontrol.TeamScope(
			accesscontrol.NamespaceID(namespaceID), accesscontrol.TeamID(session.EffectiveTeamID),
		))
	}
	return target
}

func agentRoutingResourceTarget(
	namespaceID string, kind accesscontrol.ScopeResourceType, id string,
) accesscontrol.ScopedTarget {
	return accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), kind, accesscontrol.ResourceID(id),
	)}
}

func accessForPreparedSession(
	authenticated agentAuthenticatedRequest, prepared agentmanagement.SessionAuthorization,
) agentmanagement.AccessContext {
	scope := accesscontrol.ResultScope{NamespaceID: accesscontrol.NamespaceID(authenticated.NamespaceID)}
	if prepared.EffectiveUserID != "" {
		scope.UserIDs = []accesscontrol.UserID{accesscontrol.UserID(prepared.EffectiveUserID)}
	}
	if prepared.EffectiveTeamID != "" {
		scope.TeamIDs = []accesscontrol.TeamID{accesscontrol.TeamID(prepared.EffectiveTeamID)}
	}
	return agentmanagement.AccessContext{
		PrincipalID: authenticated.Session.Session.PrincipalID, Scope: mustCanonicalAgentScope(scope),
	}
}

func accessForResolvedSession(
	authenticated agentAuthenticatedRequest, session agentmanagement.SessionAccess,
) agentmanagement.AccessContext {
	scope := accesscontrol.ResultScope{
		NamespaceID: accesscontrol.NamespaceID(authenticated.NamespaceID),
		ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
			accesscontrol.ScopeResourceAgentSession: {accesscontrol.ResourceID(session.ID)},
		},
	}
	if session.EffectiveUserID != "" {
		scope.UserIDs = []accesscontrol.UserID{accesscontrol.UserID(session.EffectiveUserID)}
	}
	if session.EffectiveTeamID != "" {
		scope.TeamIDs = []accesscontrol.TeamID{accesscontrol.TeamID(session.EffectiveTeamID)}
	}
	return agentmanagement.AccessContext{
		PrincipalID: authenticated.Session.Session.PrincipalID, Scope: mustCanonicalAgentScope(scope),
	}
}

func mustCanonicalAgentScope(scope accesscontrol.ResultScope) accesscontrol.ResultScope {
	canonical, err := scope.Canonical()
	if err != nil {
		panic("server constructed an invalid Agent result scope")
	}
	return canonical
}
