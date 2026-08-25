package managementserver

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

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
		if _, err := fmt.Fprintf(response, "id: %d\nevent: %s\ndata: %s\n\n", event.Sequence, event.Type, encoded); err != nil {
			return err
		}
	}
	return nil
}

func agentAttributedSubject(namespaceID string, prepared agentmanagement.SessionAuthorization) accesscontrol.ScopedTarget {
	if prepared.EffectiveTeamID != "" {
		return accesscontrol.ScopedTarget{Scope: accesscontrol.TeamScope(
			accesscontrol.NamespaceID(namespaceID), accesscontrol.TeamID(prepared.EffectiveTeamID),
		)}
	}
	return accesscontrol.ScopedTarget{Scope: accesscontrol.UserScope(
		accesscontrol.NamespaceID(namespaceID), accesscontrol.UserID(prepared.EffectiveUserID),
	)}
}

func agentSessionScopedTarget(namespaceID string, session agentmanagement.SessionAccess) accesscontrol.ScopedTarget {
	target := accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), accesscontrol.ScopeResourceAgentSession,
		accesscontrol.ResourceID(session.ID),
	)}
	return withAgentSessionAncestors(namespaceID, target, session)
}

func agentTurnScopedTarget(namespaceID, turnID string, session agentmanagement.SessionAccess) accesscontrol.ScopedTarget {
	target := accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), accesscontrol.ScopeResourceAgentTurn,
		accesscontrol.ResourceID(turnID),
	), Ancestors: []accesscontrol.Scope{accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), accesscontrol.ScopeResourceAgentSession,
		accesscontrol.ResourceID(session.ID),
	)}}
	return withAgentSessionAncestors(namespaceID, target, session)
}

func agentArtifactScopedTarget(namespaceID string, artifact agentmanagement.ArtifactAccess) accesscontrol.ScopedTarget {
	target := accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), accesscontrol.ScopeResourceAgentArtifact,
		accesscontrol.ResourceID(artifact.ID),
	), Ancestors: []accesscontrol.Scope{accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), accesscontrol.ScopeResourceAgentSession,
		accesscontrol.ResourceID(artifact.Session.ID),
	)}}
	return withAgentSessionAncestors(namespaceID, target, artifact.Session)
}

func withAgentSessionAncestors(namespaceID string, target accesscontrol.ScopedTarget, session agentmanagement.SessionAccess) accesscontrol.ScopedTarget {
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

func agentRoutingResourceTarget(namespaceID string, kind accesscontrol.ScopeResourceType, id string) accesscontrol.ScopedTarget {
	return accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(
		accesscontrol.NamespaceID(namespaceID), kind, accesscontrol.ResourceID(id),
	)}
}

func accessForPreparedSession(authenticated agentAuthenticatedRequest, prepared agentmanagement.SessionAuthorization) agentmanagement.AccessContext {
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

func accessForResolvedSession(authenticated agentAuthenticatedRequest, session agentmanagement.SessionAccess) agentmanagement.AccessContext {
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
