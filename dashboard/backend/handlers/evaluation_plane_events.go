package handlers

import (
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

func (h *EvaluationPlaneHandler) events(w http.ResponseWriter, r *http.Request, runID string) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeEvaluationError(w, evaluationplane.ErrConflict)
		return
	}
	live, unsubscribe, err := h.service.Subscribe(runID)
	if err != nil {
		writeEvaluationError(w, err)
		return
	}
	defer unsubscribe()
	replay, err := h.service.EventsAfter(runID, r.Header.Get("Last-Event-ID"))
	if err != nil {
		writeEvaluationError(w, err)
		return
	}
	run, err := h.service.GetRun(runID)
	if err != nil {
		writeEvaluationError(w, err)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "private, no-store")
	w.Header().Set("Pragma", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	lastID := uint64(0)
	terminalReplayed := false
	for _, event := range replay {
		if _, err := w.Write(encodeSSE(event)); err != nil {
			return
		}
		lastID = decodeEventID(event.ID)
		terminalReplayed = terminalReplayed || terminalEventType(event.Type)
	}
	flusher.Flush()
	if terminalReplayed || terminalRunStatus(run.Status) {
		return
	}
	keepalive := time.NewTicker(15 * time.Second)
	defer keepalive.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case event, open := <-live:
			if !open {
				return
			}
			if decodeEventID(event.ID) <= lastID {
				continue
			}
			if _, err := w.Write(encodeSSE(event)); err != nil {
				return
			}
			lastID = decodeEventID(event.ID)
			flusher.Flush()
			if terminalEventType(event.Type) {
				return
			}
		case <-keepalive.C:
			if _, err := w.Write([]byte(": keepalive\n\n")); err != nil {
				return
			}
			flusher.Flush()
		}
	}
}

func terminalEventType(eventType string) bool {
	return eventType == "completed" || eventType == "failed" || eventType == "cancelled"
}

func terminalRunStatus(status evaluationplane.RunStatus) bool {
	return status == evaluationplane.StatusCompleted || status == evaluationplane.StatusFailed || status == evaluationplane.StatusCancelled
}
