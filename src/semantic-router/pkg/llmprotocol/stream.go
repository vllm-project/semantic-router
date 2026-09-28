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
	EventResponseStarted         EventType = "response.started"
	EventOutputItemStarted       EventType = "output.item.started"
	EventOutputTextDelta         EventType = "output.text.delta"
	EventReasoningDelta          EventType = "output.reasoning.delta"
	EventToolCallDelta           EventType = "tool.call.delta"
	EventImageGenerationProgress EventType = "image_generation.progress"
	EventOutputItemCompleted     EventType = "output.item.completed"
	EventUsageUpdated            EventType = "usage.updated"
	EventResponseCompleted       EventType = "response.completed"
	EventResponseFailed          EventType = "response.failed"
	EventProviderOpaque          EventType = "provider.opaque"
)

type Event struct {
	Sequence   uint64
	Type       EventType
	ResponseID string
	Model      string
	ItemIndex  int
	// ContentIndex identifies one ordered content block within an output item.
	// It is independent from ItemIndex: Responses can stream several text or
	// refusal parts inside one message, while other wire formats may flatten
	// those blocks on the wire.
	ContentIndex        int
	ItemID              string
	Role                Role
	Delta               string
	ToolCall            *ToolCall
	Content             *Content
	GeneratedImage      *GeneratedImage
	StopReason          StopReason
	MatchedStopSequence string
	Usage               *Usage
	Error               *ProtocolError
	Failure             FailureScope
	Opaque              []byte
}

type StreamContext struct {
	Context            context.Context
	Source             WireFormat
	Target             WireFormat
	Options            StreamOptions
	PublicModel        string
	ProviderModel      string
	ResponseID         string
	PreviousResponseID string
}

type StreamDecoder interface {
	Push([]byte) ([]Event, Diagnostics, error)
	Finalize(error) ([]Event, Diagnostics, error)
}

type StreamEncoder interface {
	Push(Event) ([][]byte, Diagnostics, error)
	Finalize(error) ([][]byte, Diagnostics, error)
}
