package managementserver

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

func TestAgentLiveSSEHasNoResumeID(t *testing.T) {
	t.Parallel()
	value := agentmanagement.LiveModelStepEvent{
		SessionID:   "c16eb857-42f7-47d4-8916-1d7a5d01df50",
		TurnID:      "bf2afaa4-0791-493b-89b3-25bfc8791282",
		ModelStepID: "93f95f29-29ab-4562-9e1a-fb3279a72553",
		Phase:       agentmanagement.LiveModelStepDelta,
		Ordinal:     1,
		Delta:       &agentmanagement.AssistantDelta{Kind: agentmanagement.AssistantTextDelta, Text: "first"},
		CreatedAt:   time.Date(2026, 8, 23, 0, 0, 0, 0, time.UTC),
	}
	var output bytes.Buffer
	if err := writeAgentSSELiveEvent(&output, value); err != nil {
		t.Fatalf("write live event: %v", err)
	}
	if strings.Contains(output.String(), "id:") {
		t.Fatalf("live preview became resumable: %q", output.String())
	}
	if !strings.Contains(output.String(), "event: "+agentSSELiveDelta) {
		t.Fatalf("live event name missing: %q", output.String())
	}
}

func TestAgentLiveStreamStateFailsClosedAndReconcilesDurableStep(t *testing.T) {
	t.Parallel()
	const stepID = "93f95f29-29ab-4562-9e1a-fb3279a72553"
	base := agentmanagement.LiveModelStepEvent{
		SessionID:   "c16eb857-42f7-47d4-8916-1d7a5d01df50",
		TurnID:      "bf2afaa4-0791-493b-89b3-25bfc8791282",
		ModelStepID: stepID,
		Phase:       agentmanagement.LiveModelStepDelta,
		Ordinal:     1,
		Delta:       &agentmanagement.AssistantDelta{Kind: agentmanagement.AssistantTextDelta, Text: "one"},
		CreatedAt:   time.Date(2026, 8, 23, 0, 0, 0, 0, time.UTC),
	}
	state := newAgentLiveStreamState()
	if accepted := state.accept(base); accepted == nil || accepted.Ordinal != 1 {
		t.Fatalf("first preview was not accepted: %#v", accepted)
	}
	if accepted := state.accept(base); accepted != nil {
		t.Fatalf("duplicate preview was accepted: %#v", accepted)
	}

	gap := base
	gap.Ordinal = 3
	gap.Delta = &agentmanagement.AssistantDelta{Kind: agentmanagement.AssistantTextDelta, Text: "three"}
	accepted := state.accept(gap)
	if accepted == nil || accepted.Phase != agentmanagement.LiveModelStepDiscarded || accepted.Delta != nil {
		t.Fatalf("ordinal gap did not discard the preview: %#v", accepted)
	}
	gap.Ordinal = 4
	if accepted := state.accept(gap); accepted != nil {
		t.Fatalf("suppressed preview resumed before durable reconciliation: %#v", accepted)
	}

	payload, err := json.Marshal(agentmanagement.AssistantDeltaEvent{
		ModelStepID: stepID,
		ChunkIndex:  0,
		Delta:       agentmanagement.AssistantDelta{Kind: agentmanagement.AssistantTextDelta, Text: "authoritative"},
	})
	if err != nil {
		t.Fatalf("marshal durable payload: %v", err)
	}
	if err := state.observeDurable([]agentmanagement.Event{{
		Type:    agentmanagement.EventAssistantDelta,
		Payload: payload,
	}}); err != nil {
		t.Fatalf("observe durable event: %v", err)
	}
	committed := base
	committed.Phase = agentmanagement.LiveModelStepCommitted
	committed.Ordinal = 0
	committed.Delta = nil
	if accepted := state.accept(committed); accepted != nil {
		t.Fatalf("commit marker duplicated an authoritative step: %#v", accepted)
	}
	if err := state.observeDurable([]agentmanagement.Event{{
		TurnID: base.TurnID, Type: agentmanagement.EventTerminal,
	}}); err != nil {
		t.Fatalf("observe terminal event: %v", err)
	}
	late := base
	late.ModelStepID = "f58987d7-9577-475d-b3ba-169590394b8a"
	if accepted := state.accept(late); accepted != nil {
		t.Fatalf("preview arrived after authoritative terminal: %#v", accepted)
	}
	if len(state.durable) != 0 || len(state.stepTurns) != 0 {
		t.Fatalf("completed turn retained step state: %#v", state)
	}
}
