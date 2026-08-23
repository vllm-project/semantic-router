package agentruntime

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestFitExecutionContextCompactsWithoutLosingContinuityState(t *testing.T) {
	oldConstraint := "Always keep the multilingual probe and require an explicit publication review."
	pendingID := uuid.NewString()
	messages := []llmprotocol.Message{{
		Role:    llmprotocol.RoleUser,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: oldConstraint}},
	}}
	for index := 0; index < 24; index++ {
		messages = append(messages, llmprotocol.Message{
			Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentText, Text: strings.Repeat("historical explanation ", 32),
			}},
		})
	}
	messages = append(messages,
		llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentText, Text: "Tune the current recipe now.",
		}}},
		llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
			Kind:     llmprotocol.ContentToolCall,
			ToolCall: &llmprotocol.ToolCall{ID: pendingID, Name: "router.recipe.validate", Arguments: `{}`},
		}}},
	)
	approval := &agentmanagement.ApprovalRequestEvent{
		PlanID: uuid.NewString(), PlanDigest: "sha256:" + strings.Repeat("a", 64),
		PlanRevision: 4, PlanETag: `"agent:4"`, ExpiresAt: time.Now().UTC().Add(time.Hour),
	}
	projection := executionContext{
		Instructions: []llmprotocol.InstructionBlock{{
			Role:    llmprotocol.RoleSystem,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "stable"}},
		}},
		Messages: messages, Pending: []pendingToolCall{{
			InvocationID: pendingID, Name: "router.recipe.validate", Arguments: json.RawMessage(`{}`),
		}},
		ThroughSequence: 800,
		Memory: executionMemory{
			PendingApproval: approval,
			ResourceReferences: []agentmanagement.ResourceReference{{
				Kind: "recipe", ID: "draft", Revision: `"rcp:12"`,
			}},
			ToolFailures: []toolFailureMemory{{
				InvocationID: uuid.NewString(), ToolName: "router.recipe.probe",
				Failure: agentmanagement.Failure{Code: "probe_failed", Message: "Probe failed.", Retryable: true},
			}},
		},
	}

	compacted, changed, err := fitExecutionContext(projection, 1800, nil)
	if err != nil {
		t.Fatalf("fitExecutionContext() error = %v", err)
	}
	if !changed || compacted.Memory.CompactedThroughSequence != projection.ThroughSequence {
		t.Fatalf("compaction was not recorded: %#v", compacted.Memory)
	}
	if !containsString(compacted.Memory.UserConstraints, oldConstraint) {
		t.Fatalf("old user constraint was lost: %#v", compacted.Memory.UserConstraints)
	}
	if compacted.Memory.PendingApproval == nil || len(compacted.Memory.ResourceReferences) != 1 ||
		len(compacted.Memory.ToolFailures) != 1 {
		t.Fatalf("continuity state was lost: %#v", compacted.Memory)
	}
	if !messageContainsPendingCall(compacted.Messages[len(compacted.Messages)-1], compacted.Pending) {
		t.Fatal("pending tool request was compacted away")
	}
	tokens, err := estimateContextTokens(compacted.Instructions, compacted.Messages, nil)
	if err != nil || tokens > 1800 {
		t.Fatalf("compacted tokens = %d, error = %v", tokens, err)
	}
}

func TestExecutionCheckpointRoundTripKeepsCompactedState(t *testing.T) {
	turnID := uuid.NewString()
	projection := executionContext{
		ThroughSequence: 42, ToolSteps: 3, ModelSteps: 4,
		Memory: executionMemory{
			UserConstraints: []string{"Keep the image decision."},
			ResourceReferences: []agentmanagement.ResourceReference{{
				Kind: "entrypoint", ID: "image-router", Revision: `"ep:7"`,
			}},
			CompactedThroughSequence: 40,
		},
	}
	checkpoint, err := encodeExecutionCheckpoint(agentmanagement.TurnLease{
		SessionID: uuid.NewString(), TurnID: turnID,
	}, projection)
	if err != nil {
		t.Fatalf("encodeExecutionCheckpoint() error = %v", err)
	}
	restored, err := decodeExecutionCheckpoint(checkpoint)
	if err != nil {
		t.Fatalf("decodeExecutionCheckpoint() error = %v", err)
	}
	if restored.ThroughSequence != 42 || restored.Memory.CompactedThroughSequence != 40 ||
		!containsString(restored.Memory.UserConstraints, "Keep the image decision.") ||
		len(restored.Memory.ResourceReferences) != 1 {
		t.Fatalf("checkpoint continuity changed: %#v", restored)
	}
}

func TestContextCompactionKeepsCompletedToolExchangeAtomic(t *testing.T) {
	t.Parallel()
	invocationID := uuid.NewString()
	projection := executionContext{Messages: []llmprotocol.Message{
		{
			Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolCall,
				ToolCall: &llmprotocol.ToolCall{
					ID: invocationID, Name: "router.recipe.get", Arguments: `{}`,
				},
			}},
		},
		{
			Role: llmprotocol.RoleTool,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolResult,
				ToolResult: &llmprotocol.ToolResult{
					CallID:  invocationID,
					Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: `{}`}},
				},
			}},
		},
		{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Continue."}},
		},
	}}
	indices := oldestCompactableMessages(projection, 2)
	if len(indices) != 2 || indices[0] != 0 || indices[1] != 1 {
		t.Fatalf("completed Tool exchange split across compaction: %v", indices)
	}
}

func containsString(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}
