// Package llmprotocol defines the protocol-neutral semantic contract used by
// inference ingress, routing, backend dispatch, streaming, and accounting.
// Wire JSON belongs to codecs; it must not leak into this package.
package llmprotocol

import (
	"encoding/json"
	"time"
)

// WireFormat is a stable wire contract identifier, not a provider product.
type WireFormat string

const (
	OpenAIChatV1        WireFormat = "openai.chat.v1"
	OpenAIResponsesV1   WireFormat = "openai.responses.v1"
	AnthropicMessagesV1 WireFormat = "anthropic.messages.v1"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type ContentKind string

const (
	ContentText       ContentKind = "text"
	ContentRefusal    ContentKind = "refusal"
	ContentImage      ContentKind = "image"
	ContentAudio      ContentKind = "audio"
	ContentVideo      ContentKind = "video"
	ContentFile       ContentKind = "file"
	ContentToolCall   ContentKind = "tool_call"
	ContentToolResult ContentKind = "tool_result"
	ContentReasoning  ContentKind = "reasoning"
)

// Content is one ordered semantic block. Fields are closed by Kind. Data and
// references are never fetched by a codec.
type Content struct {
	Kind       ContentKind
	Text       string
	Citations  []Citation
	MediaType  string
	URL        string
	Data       string
	FileID     string
	Filename   string
	Detail     string
	ToolCall   *ToolCall
	ToolResult *ToolResult
	Signature  string
}

// Citation is bounded, protocol-neutral attribution attached to a text block.
// Offsets are Unicode code-point indexes into Content.Text.
type Citation struct {
	URL        string
	Title      string
	StartIndex int64
	EndIndex   int64
}

type Message struct {
	ID      string
	Role    Role
	Content []Content
}

type InstructionBlock struct {
	Role    Role
	Content []Content
}

type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

type ToolResult struct {
	CallID  string
	Content []Content
	IsError *bool
	// DeferredLink means the referenced call belongs to retained conversation
	// state identified by PreviousResponseID instead of this request body. It is
	// semantic validation state and is never accepted without that continuation
	// reference.
	DeferredLink bool
}

type Tool struct {
	Name        string
	Description string
	Strict      *bool
	InputSchema json.RawMessage
}

type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceNamed    ToolChoiceMode = "named"
)

type ToolChoice struct {
	Mode ToolChoiceMode
	Name string
}

type OutputFormatKind string

const (
	OutputText       OutputFormatKind = "text"
	OutputJSONObject OutputFormatKind = "json_object"
	OutputJSONSchema OutputFormatKind = "json_schema"
)

type OutputFormat struct {
	Kind        OutputFormatKind
	Name        string
	Description string
	Strict      *bool
	Schema      json.RawMessage
}

type Sampling struct {
	Temperature      *float64
	TopP             *float64
	TopK             *int64
	MaxOutputTokens  *int64
	Seed             *int64
	FrequencyPenalty *float64
	PresencePenalty  *float64
	Stop             []string
}

// TrustedMetadata is populated by the Router after transport authentication.
// Codecs never populate trusted fields from client headers.
type TrustedMetadata struct {
	NamespaceID   string
	ActorID       string
	SubjectID     string
	SessionID     string
	AgentID       string
	TaskID        string
	TurnID        string
	CorrelationID string
	SourceFormat  WireFormat
}

type Request struct {
	Generation            uint64
	Model                 string
	Instructions          []InstructionBlock
	Messages              []Message
	Tools                 []Tool
	ToolChoice            ToolChoice
	ParallelToolCalls     *bool
	CandidateCount        *int64
	Sampling              Sampling
	OutputFormat          OutputFormat
	ReasoningEffort       string
	ReasoningBudgetTokens *int64
	Stream                bool
	Metadata              map[string]string
	PreviousResponseID    string
	ConversationID        string
	Store                 *bool
	AutoStore             *bool
	Trusted               TrustedMetadata
}

type StopReason string

const (
	StopEndTurn       StopReason = "end_turn"
	StopMaxTokens     StopReason = "max_tokens"
	StopSequence      StopReason = "stop_sequence"
	StopToolCall      StopReason = "tool_call"
	StopContentFilter StopReason = "content_filter"
	StopCanceled      StopReason = "canceled"
	StopError         StopReason = "error"
	StopUnknown       StopReason = "unknown"
)

type UsageProvenance string

const (
	UsageAuthoritative UsageProvenance = "authoritative"
	UsageDerived       UsageProvenance = "derived"
	UsageEstimated     UsageProvenance = "estimated"
	UsageUnknown       UsageProvenance = "unknown"
)

// TokenCount uses a pointer so absent and an authoritative zero remain
// distinguishable.
type TokenCount struct {
	Value      *int64
	Provenance UsageProvenance
}

type Usage struct {
	State           UsageState
	InputUncached   TokenCount
	InputCacheRead  TokenCount
	InputCacheWrite TokenCount
	OutputReasoning TokenCount
	OutputOther     TokenCount
	InputTotal      TokenCount
	OutputTotal     TokenCount
	Total           TokenCount
}

type UsageState string

const (
	UsageAvailable   UsageState = "available"
	UsageUnavailable UsageState = "unknown"
)

type Response struct {
	Generation uint64
	ID         string
	CreatedAt  time.Time
	Model      string
	Output     []OutputItem
	// Alternatives preserves additional, ordered model choices when a source
	// format supports them. A target that cannot represent alternatives must
	// apply the configured lossy policy; it may never silently pick one.
	Alternatives      [][]OutputItem
	StopReason        StopReason
	SourceStopReason  string
	Usage             Usage
	ProviderRequestID string
	// Evidence is bounded, protocol-neutral model evidence for Router
	// algorithms. It is never usage evidence and codecs do not publish it unless
	// the target protocol explicitly represents the same semantic field.
	Evidence ResponseEvidence
	Error    *ProtocolError
}

type ResponseEvidence struct {
	TokenLogprobs []TokenLogprob
}

type TokenLogprob struct {
	Token        string
	Logprob      float64
	Alternatives []TokenLogprobAlternative
}

type TokenLogprobAlternative struct {
	Token   string
	Logprob float64
}

type OutputItem struct {
	ID      string
	Role    Role
	Content []Content
}

func Int64(value int64) *int64       { return &value }
func Bool(value bool) *bool          { return &value }
func Float64(value float64) *float64 { return &value }
