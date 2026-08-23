package agentmanagement

import (
	"fmt"

	"github.com/google/uuid"
)

// NormalizeLiveModelStepEvent validates the transient preview vocabulary. It
// deliberately does not return Event: live previews have no durable sequence
// and must never enter the transcript by accident.
func NormalizeLiveModelStepEvent(value LiveModelStepEvent) (LiveModelStepEvent, error) {
	if uuid.Validate(value.SessionID) != nil || uuid.Validate(value.TurnID) != nil ||
		uuid.Validate(value.ModelStepID) != nil || value.CreatedAt.IsZero() || value.Ordinal < 0 {
		return LiveModelStepEvent{}, fmt.Errorf("%w: Agent live model event is invalid", ErrInvalid)
	}
	switch value.Phase {
	case LiveModelStepDelta:
		if value.Ordinal < 1 || value.Delta == nil || validateAssistantDelta(*value.Delta) != nil {
			return LiveModelStepEvent{}, fmt.Errorf("%w: Agent live model delta is invalid", ErrInvalid)
		}
		delta := *value.Delta
		value.Delta = &delta
	case LiveModelStepCommitted, LiveModelStepDiscarded:
		if value.Delta != nil {
			return LiveModelStepEvent{}, fmt.Errorf("%w: Agent live model terminal is invalid", ErrInvalid)
		}
	default:
		return LiveModelStepEvent{}, fmt.Errorf("%w: Agent live model phase is invalid", ErrInvalid)
	}
	return value, nil
}
