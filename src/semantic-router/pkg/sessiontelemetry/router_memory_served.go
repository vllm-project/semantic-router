package sessiontelemetry

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// RecordSessionServedModel hands a session's ownership from the model a
// request was routed to over to the fallback candidate that answered it, so
// the next request is held on the model whose cache it can read. A session
// whose current model has since changed, by a newer dispatch, keeps it.
func RecordSessionServedModel(sessionID, routed, served string, candidate *config.ModelRef) {
	if sessionID == "" || routed == "" || served == "" || routed == served {
		return
	}
	s := globalRouterSessionMemory
	s.mu.Lock()
	st := s.sessions[sessionID]
	if st == nil || st.currentModel != routed {
		s.mu.Unlock()
		return
	}
	st.currentModel = served
	st.currentCandidate = cloneSessionCandidate(candidate)
	if st.modelTurns[routed]--; st.modelTurns[routed] <= 0 {
		delete(st.modelTurns, routed)
	}
	st.modelTurns[served]++
	s.mu.Unlock()
	RecordLastModel(sessionID, served)
	persistRouterSessionState(sessionID)
}
