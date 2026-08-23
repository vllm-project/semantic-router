package llmprotocol

import "context"

type EventType string

const (
	EventResponseStarted     EventType = "response.started"
	EventOutputItemStarted   EventType = "output.item.started"
	EventOutputTextDelta     EventType = "output.text.delta"
	EventReasoningDelta      EventType = "output.reasoning.delta"
	EventToolCallDelta       EventType = "tool.call.delta"
	EventOutputItemCompleted EventType = "output.item.completed"
	EventUsageUpdated        EventType = "usage.updated"
	EventResponseCompleted   EventType = "response.completed"
	EventResponseFailed      EventType = "response.failed"
	EventProviderOpaque      EventType = "provider.opaque"
)

type Event struct {
	Sequence   uint64
	Type       EventType
	ResponseID string
	Model      string
	ItemIndex  int
	ItemID     string
	Role       Role
	Delta      string
	ToolCall   *ToolCall
	Content    *Content
	StopReason StopReason
	Usage      *Usage
	Error      *ProtocolError
	Opaque     []byte
}

type StreamContext struct {
	Context       context.Context
	Source        WireFormat
	Target        WireFormat
	PublicModel   string
	ProviderModel string
	ResponseID    string
}

type StreamDecoder interface {
	Push([]byte) ([]Event, Diagnostics, error)
	Finalize(error) ([]Event, Diagnostics, error)
}

type StreamEncoder interface {
	Push(Event) ([][]byte, Diagnostics, error)
	Finalize(error) ([][]byte, Diagnostics, error)
}
