package agentruntime

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type liveEventCapture struct {
	events chan agentmanagement.LiveModelStepEvent
}

func (capture *liveEventCapture) PublishLiveModelStep(
	_ context.Context, _ string, event agentmanagement.LiveModelStepEvent,
) error {
	capture.events <- event
	return nil
}

func TestModelStepPublishesFirstDeltaBeforeTerminalWithoutDurableWrite(t *testing.T) {
	now := time.Now().UTC()
	capture := &liveEventCapture{events: make(chan agentmanagement.LiveModelStepEvent, 4)}
	worker := &Worker{liveEvents: capture, now: func() time.Time { return now }}
	lease := agentmanagement.TurnLease{
		NamespaceID: uuid.NewString(), SessionID: uuid.NewString(), TurnID: uuid.NewString(), Fence: 3,
	}
	stepID := uuid.NewString()
	collector := newModelStepCollector(
		context.Background(), worker, lease, nil, agentmanagement.ToolPolicy{}, stepID, 0,
	)
	if err := collector.consume(llmprotocol.Event{Type: llmprotocol.EventResponseStarted}); err != nil {
		t.Fatal(err)
	}
	if err := collector.consume(llmprotocol.Event{
		Type: llmprotocol.EventOutputTextDelta, Delta: "first token",
	}); err != nil {
		t.Fatal(err)
	}
	select {
	case live := <-capture.events:
		if live.Phase != agentmanagement.LiveModelStepDelta || live.ModelStepID != stepID ||
			live.Ordinal != 1 || live.Delta == nil || live.Delta.Text != "first token" {
			t.Fatalf("live event = %#v", live)
		}
	default:
		t.Fatal("first assistant delta was not observable before model completion")
	}
	if _, err := collector.finish(); err == nil {
		t.Fatal("unfinished inference produced a durable model-step output")
	}
	if err := collector.consume(llmprotocol.Event{
		Type: llmprotocol.EventResponseCompleted, StopReason: llmprotocol.StopEndTurn,
	}); err != nil {
		t.Fatal(err)
	}
	if err := collector.observe(PublicInferenceObservation{
		RequestID: "request-1", ResponsePath: "upstream", LatencyMilliseconds: 25,
	}); err != nil {
		t.Fatal(err)
	}
	output, err := collector.finish()
	if err != nil || len(output.Events) != 2 {
		t.Fatalf("finish() events = %d, error = %v", len(output.Events), err)
	}
	var durable agentmanagement.AssistantDeltaEvent
	if err := json.Unmarshal(output.Events[0].Payload, &durable); err != nil {
		t.Fatal(err)
	}
	if durable.ModelStepID != stepID || durable.ChunkIndex != 0 || durable.Delta.Text != "first token" {
		t.Fatalf("durable reconciliation event = %#v", durable)
	}
	var summary agentmanagement.ModelStepSummaryEvent
	if err := json.Unmarshal(output.Events[1].Payload, &summary); err != nil {
		t.Fatal(err)
	}
	if output.Events[1].Type != agentmanagement.EventModelStepSummary ||
		summary.ModelStepID != stepID || summary.RequestID != "request-1" {
		t.Fatalf("durable model-step summary = %#v", summary)
	}
}

func TestModelStepSummaryPersistsOnlyAuthoritativeUsage(t *testing.T) {
	collector := newModelStepCollector(
		context.Background(), &Worker{}, agentmanagement.TurnLease{Fence: 1}, nil,
		agentmanagement.ToolPolicy{}, uuid.NewString(), 0,
	)
	input, output, total, cached := int64(8), int64(3), int64(11), int64(2)
	if err := collector.consume(llmprotocol.Event{Type: llmprotocol.EventResponseStarted}); err != nil {
		t.Fatal(err)
	}
	if err := collector.consume(llmprotocol.Event{
		Type: llmprotocol.EventUsageUpdated,
		Usage: &llmprotocol.Usage{
			State:          llmprotocol.UsageAvailable,
			InputTotal:     llmprotocol.TokenCount{Value: &input, Provenance: llmprotocol.UsageAuthoritative},
			OutputTotal:    llmprotocol.TokenCount{Value: &output, Provenance: llmprotocol.UsageAuthoritative},
			Total:          llmprotocol.TokenCount{Value: &total, Provenance: llmprotocol.UsageAuthoritative},
			InputCacheRead: llmprotocol.TokenCount{Value: &cached, Provenance: llmprotocol.UsageAuthoritative},
			InputUncached:  llmprotocol.TokenCount{Value: &input, Provenance: llmprotocol.UsageEstimated},
		},
	}); err != nil {
		t.Fatal(err)
	}
	if err := collector.consume(llmprotocol.Event{
		Type: llmprotocol.EventResponseCompleted, StopReason: llmprotocol.StopEndTurn,
	}); err != nil {
		t.Fatal(err)
	}
	if err := collector.observe(PublicInferenceObservation{
		RequestID: "request-2", LatencyMilliseconds: 10,
	}); err != nil {
		t.Fatal(err)
	}
	result, err := collector.finish()
	if err != nil || len(result.Events) != 1 {
		t.Fatalf("finish() events = %d, error = %v", len(result.Events), err)
	}
	var summary agentmanagement.ModelStepSummaryEvent
	if err := json.Unmarshal(result.Events[0].Payload, &summary); err != nil {
		t.Fatal(err)
	}
	if summary.Usage == nil || summary.Usage.TotalTokens != 11 ||
		summary.Usage.InputCacheReadTokens == nil || *summary.Usage.InputCacheReadTokens != 2 {
		t.Fatalf("authoritative usage summary = %#v", summary.Usage)
	}
	if summary.Usage.InputUncachedTokens != nil {
		t.Fatalf("estimated input breakdown was persisted: %#v", summary.Usage)
	}
}
