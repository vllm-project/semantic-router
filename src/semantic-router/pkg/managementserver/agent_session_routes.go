package managementserver

import (
	"net/http"
	"time"

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
		Mode: body.Mode, ProfileID: body.ProfileID, KeyID: body.KeyID, EffectiveTeamID: body.EffectiveTeamID,
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
		"agent_session", session.ID, publicRevision(session.Revision), &replayed,
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
			"agent_session", session.ID, publicRevision(session.Revision), nil,
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
			"agent_turn", turn.ID, publicRevision(turn.Revision), &replayed,
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
		"agent_turn", turn.ID, publicRevision(turn.Revision), &replayed,
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
	stream, ok := routes.openAgentEventStream(response, request, requestID, resolved, access)
	if !ok {
		return
	}
	defer stream.subscription.Close()
	poll := time.NewTicker(agentSSEPollInterval)
	heartbeat := time.NewTicker(agentSSEHeartbeat)
	defer poll.Stop()
	defer heartbeat.Stop()
	liveEvents := stream.subscription.Events()
	catchUpDurable := func() error {
		for {
			events, hasMore, resumeErr := routes.service.ResumeEvents(
				request.Context(), string(access.Scope.NamespaceID), resolved.ID,
				stream.after, agentSSEBatchSize, access,
			)
			if resumeErr != nil {
				return resumeErr
			}
			if len(events) == 0 {
				return nil
			}
			if err := stream.state.observeDurable(events); err != nil {
				return err
			}
			if err := writeAgentSSEEvents(response, events); err != nil {
				return err
			}
			stream.after = events[len(events)-1].Sequence
			stream.flusher.Flush()
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
			accepted := stream.state.accept(live)
			if accepted == nil {
				continue
			}
			if err := writeAgentSSELiveEvent(response, *accepted); err != nil {
				return
			}
			stream.flusher.Flush()
		case <-heartbeat.C:
			if _, err := response.Write([]byte(": keepalive\n\n")); err != nil {
				return
			}
			stream.flusher.Flush()
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
	desired := publicRevision(result.Revision)
	setIdempotencyReplayHeader(response, result.Replayed)
	writeProviderJSON(response, http.StatusAccepted, managementapi.NewOperationMutationReceipt(
		result.OperationID, &desired, &result.Replayed,
	), requestID)
}
