package agentmanagement

import (
	"testing"
	"time"

	"github.com/google/uuid"
)

func TestNormalizeLiveModelStepEventSeparatesPreviewFromTerminal(t *testing.T) {
	base := LiveModelStepEvent{
		SessionID: uuid.NewString(), TurnID: uuid.NewString(), ModelStepID: uuid.NewString(),
		CreatedAt: time.Now().UTC(),
	}
	delta := AssistantDelta{Kind: AssistantTextDelta, Text: "hello"}
	preview := base
	preview.Phase, preview.Ordinal, preview.Delta = LiveModelStepDelta, 1, &delta
	if _, err := NormalizeLiveModelStepEvent(preview); err != nil {
		t.Fatalf("valid preview rejected: %v", err)
	}
	terminal := base
	terminal.Phase, terminal.Ordinal = LiveModelStepCommitted, 1
	if _, err := NormalizeLiveModelStepEvent(terminal); err != nil {
		t.Fatalf("valid terminal rejected: %v", err)
	}
	terminal.Delta = &delta
	if _, err := NormalizeLiveModelStepEvent(terminal); err == nil {
		t.Fatal("terminal live event accepted provisional text")
	}
}
