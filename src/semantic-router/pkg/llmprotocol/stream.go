package llmprotocol

import "context"

type EventType string

// FailureScope distinguishes a transport-level stream error from a model
// response that reached a terminal failed state. Formats that expose only one
// error event may render both scopes identically.
type FailureScope string

const (
	FailureTransport FailureScope = "transport"
	FailureResponse  FailureScope = "response"
)

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
	Failure    FailureScope
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
