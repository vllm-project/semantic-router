package managementserver

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

type agentEventStream struct {
	after        int64
	subscription agentmanagement.LiveEventSubscription
	flusher      http.Flusher
	state        *agentLiveStreamState
}

func (routes *AgentRoutes) openAgentEventStream(
	response http.ResponseWriter,
	request *http.Request,
	requestID string,
	resolved agentmanagement.SessionAccess,
	access agentmanagement.AccessContext,
) (agentEventStream, bool) {
	after, explicit, err := parseAgentResumeSequence(request)
	if err != nil {
		writeAgentDomainError(response, err, requestID)
		return agentEventStream{}, false
	}
	subscription, err := routes.liveEvents.SubscribeLiveModelSteps(
		request.Context(), string(access.Scope.NamespaceID), resolved.ID,
	)
	if err != nil {
		writeProviderError(response, http.StatusServiceUnavailable, "stream_unavailable", "Live updates are unavailable.", requestID)
		return agentEventStream{}, false
	}
	if !explicit {
		latest, historyErr := routes.service.ListEventHistory(
			request.Context(), string(access.Scope.NamespaceID), resolved.ID,
			agentmanagement.EventPageRequest{PageSize: 1, Scope: access.Scope}, access,
		)
		if historyErr != nil {
			_ = subscription.Close()
			writeAgentDomainError(response, historyErr, requestID)
			return agentEventStream{}, false
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
		_ = subscription.Close()
		writeAgentDomainError(response, err, requestID)
		return agentEventStream{}, false
	}
	flusher, ok := response.(http.Flusher)
	if !ok {
		_ = subscription.Close()
		writeProviderError(response, http.StatusNotImplemented, "stream_unavailable", "Streaming is unavailable.", requestID)
		return agentEventStream{}, false
	}
	setProviderResponseHeaders(response, requestID)
	response.Header().Set("Content-Type", managementapi.EventStreamMediaType)
	response.Header().Set("Connection", "keep-alive")
	response.WriteHeader(http.StatusOK)
	if err := writeAgentSSEEvents(response, initial); err != nil {
		_ = subscription.Close()
		return agentEventStream{}, false
	}
	if len(initial) > 0 {
		after = initial[len(initial)-1].Sequence
	}
	state := newAgentLiveStreamState()
	if err := state.observeDurable(initial); err != nil {
		_ = subscription.Close()
		return agentEventStream{}, false
	}
	flusher.Flush()
	return agentEventStream{after: after, subscription: subscription, flusher: flusher, state: state}, true
}
