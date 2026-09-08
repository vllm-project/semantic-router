package evaluationplane

import (
	"fmt"
	"strconv"
)

// Terminal events are a deterministic projection of status.json. They are
// never appended to the control event log, so terminal state has one durable
// authority while SSE retains a stable run-local numeric identity.
func terminalEventForRun(run Run, sequence uint64) (Event, error) {
	if !terminalStatus(run.Status) || run.CompletedAt == nil || sequence == 0 || sequence > maxEventsPerRun {
		return Event{}, fmt.Errorf("%w: cannot derive an evaluation terminal event", ErrInvalid)
	}
	eventType, message := terminalEventIdentity(run)
	event := Event{
		ID: strconv.FormatUint(sequence, 10), RunID: run.ID, Type: eventType,
		Timestamp: run.CompletedAt.UTC(), Message: message, Progress: &run.Progress,
	}
	if err := validateStoredEvent(event); err != nil {
		return Event{}, err
	}
	return event, nil
}

func terminalEventIdentity(run Run) (string, string) {
	switch run.Status {
	case StatusCompleted:
		return "completed", "Evaluation completed"
	case StatusCancelled:
		return "cancelled", "Run cancelled by user"
	default:
		return "failed", run.Progress.Message
	}
}

func terminalWorkerEventType(eventType string) bool {
	return eventType == "completed" || eventType == "failed" || eventType == "cancelled"
}
