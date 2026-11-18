package llmprotocol

import (
	"context"
	"time"
)

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
	// CreatedAt is the response's creation time, on the first event of a
	// stream whose source reports one.
	CreatedAt time.Time
}

type StreamContext struct {
	Context        context.Context
	Source         WireFormat
	Target         WireFormat
	Options        StreamOptions
	PublicModel    string
	ProviderModel  string
	ResponseID     string
	ResponseRender ResponseRenderContext
	// PreferProviderModel names events with the model the provider's stream
	// reports, once it reports one, and with PublicModel until then.
	PreferProviderModel bool
	// EstimatedInputTokens is the Router's own count of the request's input
	// tokens, which an encoder that reports input usage before the provider
	// does reports instead, as Messages message_start does. 0 reports none.
	EstimatedInputTokens int64
}

type StreamDecoder interface {
	Push([]byte) ([]Event, Diagnostics, error)
	Finalize(error) ([]Event, Diagnostics, error)
}

type StreamEncoder interface {
	Push(Event) ([][]byte, Diagnostics, error)
	Finalize(error) ([][]byte, Diagnostics, error)
}
