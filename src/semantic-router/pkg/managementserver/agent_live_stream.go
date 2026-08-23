package managementserver

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

const (
	agentSSELiveDelta          = "assistant_delta.provisional"
	agentSSELiveCommitted      = "model_step.committed"
	agentSSELiveDiscarded      = "model_step.discarded"
	agentSSETerminalTurnWindow = 256
)

type agentLiveStreamState struct {
	durable       map[string]struct{}
	ordinals      map[string]int
	suppressed    map[string]struct{}
	stepTurns     map[string]string
	terminalTurns map[string]struct{}
	terminalOrder []string
}

func newAgentLiveStreamState() *agentLiveStreamState {
	return &agentLiveStreamState{
		durable: make(map[string]struct{}), ordinals: make(map[string]int),
		suppressed: make(map[string]struct{}),
		stepTurns:  make(map[string]string), terminalTurns: make(map[string]struct{}),
	}
}

func (state *agentLiveStreamState) observeDurable(events []agentmanagement.Event) error {
	for _, event := range events {
		if event.Type == agentmanagement.EventAssistantDelta {
			var payload agentmanagement.AssistantDeltaEvent
			if err := json.Unmarshal(event.Payload, &payload); err != nil || payload.ModelStepID == "" {
				return fmt.Errorf("decode durable Agent model step: %w", agentmanagement.ErrConflict)
			}
			state.durable[payload.ModelStepID] = struct{}{}
			if event.TurnID != "" {
				state.stepTurns[payload.ModelStepID] = event.TurnID
			}
			delete(state.ordinals, payload.ModelStepID)
			delete(state.suppressed, payload.ModelStepID)
		}
		if event.Type == agentmanagement.EventTerminal && event.TurnID != "" {
			state.observeTerminalTurn(event.TurnID)
		}
	}
	return nil
}

func (state *agentLiveStreamState) observeTerminalTurn(turnID string) {
	if _, exists := state.terminalTurns[turnID]; !exists {
		state.terminalTurns[turnID] = struct{}{}
		state.terminalOrder = append(state.terminalOrder, turnID)
	}
	for stepID, stepTurnID := range state.stepTurns {
		if stepTurnID != turnID {
			continue
		}
		delete(state.durable, stepID)
		delete(state.ordinals, stepID)
		delete(state.suppressed, stepID)
		delete(state.stepTurns, stepID)
	}
	for len(state.terminalOrder) > agentSSETerminalTurnWindow {
		oldest := state.terminalOrder[0]
		state.terminalOrder = state.terminalOrder[1:]
		delete(state.terminalTurns, oldest)
	}
}

// accept returns the event to write. A nil result means the preview is either
// already durable or a duplicate. An ordinal gap emits one discarded marker
// and suppresses the incomplete preview until its durable replacement arrives.
func (state *agentLiveStreamState) accept(
	value agentmanagement.LiveModelStepEvent,
) *agentmanagement.LiveModelStepEvent {
	if _, terminal := state.terminalTurns[value.TurnID]; terminal {
		return nil
	}
	if _, durable := state.durable[value.ModelStepID]; durable {
		return nil
	}
	switch value.Phase {
	case agentmanagement.LiveModelStepDelta:
		if _, suppressed := state.suppressed[value.ModelStepID]; suppressed {
			return nil
		}
		previous := state.ordinals[value.ModelStepID]
		if value.Ordinal <= previous {
			return nil
		}
		if value.Ordinal != previous+1 {
			state.suppressed[value.ModelStepID] = struct{}{}
			delete(state.ordinals, value.ModelStepID)
			discarded := value
			discarded.Phase = agentmanagement.LiveModelStepDiscarded
			discarded.Delta = nil
			return &discarded
		}
		state.ordinals[value.ModelStepID] = value.Ordinal
		state.stepTurns[value.ModelStepID] = value.TurnID
		return &value
	case agentmanagement.LiveModelStepCommitted, agentmanagement.LiveModelStepDiscarded:
		delete(state.ordinals, value.ModelStepID)
		delete(state.suppressed, value.ModelStepID)
		delete(state.stepTurns, value.ModelStepID)
		return &value
	default:
		return nil
	}
}

func writeAgentSSELiveEvent(
	response io.Writer, value agentmanagement.LiveModelStepEvent,
) error {
	eventName := ""
	switch value.Phase {
	case agentmanagement.LiveModelStepDelta:
		eventName = agentSSELiveDelta
	case agentmanagement.LiveModelStepCommitted:
		eventName = agentSSELiveCommitted
	case agentmanagement.LiveModelStepDiscarded:
		eventName = agentSSELiveDiscarded
	default:
		return agentmanagement.ErrInvalid
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	// There is intentionally no SSE id: Last-Event-ID belongs exclusively to
	// the durable PostgreSQL sequence.
	_, err = fmt.Fprintf(response, "event: %s\ndata: %s\n\n", eventName, encoded)
	return err
}
