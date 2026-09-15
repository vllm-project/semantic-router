package handlers

import (
	"net/http"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

func writeEvaluationEventReplay(
	writer *evaluationDeadlineWriter,
	events []evaluationplane.Event,
	lastID uint64,
) (uint64, bool, error) {
	writer.Header().Set("Content-Type", "text/event-stream")
	writer.Header().Set("Cache-Control", "private, no-store")
	writer.Header().Set("Pragma", "no-cache")
	writer.Header().Set("Connection", "keep-alive")
	writer.Header().Set("X-Accel-Buffering", "no")
	terminalReplayed := false
	for _, event := range events {
		if _, err := writer.Write(encodeSSE(event)); err != nil {
			return lastID, terminalReplayed, err
		}
		lastID = decodeEventID(event.ID)
		terminalReplayed = terminalReplayed || terminalEventType(event.Type)
	}
	return lastID, terminalReplayed, writer.flush()
}

func (h *EvaluationPlaneHandler) events(w http.ResponseWriter, r *http.Request, runID string) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	actor, ok := h.evaluationActor(w, r)
	if !ok {
		return
	}
	releaseStream, streamErr := h.responseStreams.acquire(actor)
	if streamErr != nil {
		writeEvaluationError(w, streamErr)
		return
	}
	defer releaseStream()
	if _, ok := w.(http.Flusher); !ok {
		writeEvaluationError(w, evaluationplane.ErrConflict)
		return
	}
	live, unsubscribe, subscribeErr := h.service.SubscribeAs(actor, runID)
	if subscribeErr != nil {
		writeEvaluationError(w, subscribeErr)
		return
	}
	defer unsubscribe()
	deadlineWriter := newEvaluationDeadlineWriter(w, h.streamWriteTimeout)
	defer deadlineWriter.clearDeadline()
	replay, replayErr := h.service.EventsAfterAs(actor, runID, r.Header.Get("Last-Event-ID"))
	if replayErr != nil {
		writeEvaluationError(w, replayErr)
		return
	}
	lastID, terminalReplayed, replayWriteErr := writeEvaluationEventReplay(
		deadlineWriter, replay, decodeEventID(r.Header.Get("Last-Event-ID")),
	)
	if replayWriteErr != nil || terminalReplayed {
		return
	}
	// A terminal transition can commit after the first replay snapshot but
	// before this status read. Re-read strictly after the last emitted ID so
	// that window closes with the derived terminal event instead of silently
	// ending the stream. A client already at the terminal ID receives no
	// duplicate and the stream closes immediately.
	run, err := h.service.GetRunAs(actor, runID)
	if err != nil {
		return
	}
	if terminalRunStatus(run.Status) {
		catchup, catchupErr := h.service.EventsAfterAs(actor, runID, strconv.FormatUint(lastID, 10))
		if catchupErr != nil {
			return
		}
		for _, event := range catchup {
			if decodeEventID(event.ID) <= lastID {
				continue
			}
			if _, err := deadlineWriter.Write(encodeSSE(event)); err != nil {
				return
			}
			lastID = decodeEventID(event.ID)
		}
		_ = deadlineWriter.flush()
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
			if _, err := deadlineWriter.Write(encodeSSE(event)); err != nil {
				return
			}
			lastID = decodeEventID(event.ID)
			if err := deadlineWriter.flush(); err != nil {
				return
			}
			if terminalEventType(event.Type) {
				return
			}
		case <-keepalive.C:
			if _, err := deadlineWriter.Write([]byte(": keepalive\n\n")); err != nil {
				return
			}
			if err := deadlineWriter.flush(); err != nil {
				return
			}
		}
	}
}

func terminalEventType(eventType string) bool {
	return eventType == "completed" || eventType == "failed" || eventType == "cancelled"
}

func terminalRunStatus(status evaluationplane.RunStatus) bool {
	return status == evaluationplane.StatusCompleted || status == evaluationplane.StatusFailed || status == evaluationplane.StatusCancelled
}
