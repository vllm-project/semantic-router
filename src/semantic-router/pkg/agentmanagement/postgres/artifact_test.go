package postgres

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

func TestEncodeCheckpointNormalizesEmptyCollectionsToArrays(t *testing.T) {
	checkpoint := agentmanagement.Checkpoint{
		SessionID: uuid.NewString(), TurnID: uuid.NewString(), ThroughSequence: 3,
		Summary: "Conversation state through event 3.",
		State:   json.RawMessage(`{"version":2,"messages":[]}`),
	}

	goals, resources, toolResults, decisions, _, canonical, err := encodeCheckpoint(checkpoint)
	if err != nil {
		t.Fatalf("encodeCheckpoint() error = %v", err)
	}
	for name, encoded := range map[string][]byte{
		"unresolved_goals":       goals,
		"resource_references":    resources,
		"tool_result_references": toolResults,
		"decisions":              decisions,
	} {
		if !bytes.Equal(encoded, []byte(`[]`)) {
			t.Errorf("%s = %s, want []", name, encoded)
		}
	}

	var digestDocument struct {
		Goals       []string                            `json:"goals"`
		Resources   []agentmanagement.ResourceReference `json:"resources"`
		ToolResults []string                            `json:"toolResults"`
		Decisions   []string                            `json:"decisions"`
	}
	if err := json.Unmarshal(canonical, &digestDocument); err != nil {
		t.Fatalf("decode canonical checkpoint: %v", err)
	}
	if digestDocument.Goals == nil || digestDocument.Resources == nil ||
		digestDocument.ToolResults == nil || digestDocument.Decisions == nil {
		t.Fatalf("canonical checkpoint contains a null collection: %s", canonical)
	}
}

func TestEncodeCheckpointPreservesResourceReferenceArray(t *testing.T) {
	reference := agentmanagement.ResourceReference{
		Kind: "routing_model", ID: "local/fast", Revision: "7",
	}
	checkpoint := agentmanagement.Checkpoint{
		SessionID: uuid.NewString(), TurnID: uuid.NewString(), ThroughSequence: 9,
		Summary:            "Conversation state through event 9.",
		ResourceReferences: []agentmanagement.ResourceReference{reference},
		State:              json.RawMessage(`{"version":2,"messages":[]}`),
	}

	_, resources, _, _, _, canonical, err := encodeCheckpoint(checkpoint)
	if err != nil {
		t.Fatalf("encodeCheckpoint() error = %v", err)
	}
	var stored []agentmanagement.ResourceReference
	if err := json.Unmarshal(resources, &stored); err != nil {
		t.Fatalf("decode resource_references: %v", err)
	}
	if len(stored) != 1 || stored[0] != reference {
		t.Fatalf("resource_references = %#v, want %#v", stored, []agentmanagement.ResourceReference{reference})
	}
	var digestDocument struct {
		Resources []agentmanagement.ResourceReference `json:"resources"`
	}
	if err := json.Unmarshal(canonical, &digestDocument); err != nil {
		t.Fatalf("decode canonical checkpoint: %v", err)
	}
	if len(digestDocument.Resources) != 1 || digestDocument.Resources[0] != reference {
		t.Fatalf("canonical resources = %#v, want %#v", digestDocument.Resources, stored)
	}
}
