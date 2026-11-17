package llmprotocol

import "slices"

type UnknownFieldPolicy string

const (
	UnknownReject             UnknownFieldPolicy = "reject"
	UnknownPreserveSameFormat UnknownFieldPolicy = "preserve_same_format"
)

type LossyPolicy string

const (
	LossyReject              LossyPolicy = "reject"
	LossyAllowWithDiagnostic LossyPolicy = "allow_with_diagnostic"
)

// LossyFieldToolResultError names the Anthropic tool-result failure flag, which
// OpenAI request formats cannot carry.
const LossyFieldToolResultError = "tool_result.is_error"

// LossyFieldSystemMessageRole names the role of Anthropic system messages
// inside the conversation, which ProjectAnthropicSystemMessages sends to Chat
// and Responses backends as user text.
const LossyFieldSystemMessageRole = "messages[].role"

// LossyFieldUnreadableFile names files in the conversation when the selected
// model declares no file input. Accepting the loss sends each such file as a
// short note, which ProjectUnreadableFiles writes.
const LossyFieldUnreadableFile = "content.file"

// LossyFieldToolDeferLoading names a tool's request to be loaded on demand
// through the provider's tool search. Accepting the loss sends the tool with its
// full definition, as a backend without tool search would see it.
const LossyFieldToolDeferLoading = "tools.defer_loading"

// LossyFieldReasoningSignature names the signature an Anthropic backend puts
// on its thinking, which Chat Completions and Responses clients cannot carry.
const LossyFieldReasoningSignature = "reasoning.signature"

// LossyFieldStore names a request's store: true, which asks the provider to
// keep the completion. Accepting the loss sends the request without it to a
// backend whose wire format keeps no completions, which ProjectUnstoredRequest
// writes.
const LossyFieldStore = "store"

// OptInLossyFields lists the losses an operator may accept while every other
// lossy translation keeps failing closed.
func OptInLossyFields() []string {
	return []string{LossyFieldToolResultError, LossyFieldSystemMessageRole, LossyFieldUnreadableFile, LossyFieldToolDeferLoading, LossyFieldReasoningSignature, LossyFieldStore}
}

// OptInHint is what a refusal of field adds when an operator may accept that
// loss: the setting that lists it. It is empty for every other field.
func OptInHint(field string) string {
	if !slices.Contains(OptInLossyFields(), field) {
		return ""
	}
	return "; an operator can accept this loss by listing " + field + " in global.router.allow_lossy_translation"
}

type MissingIDPolicy string

const (
	MissingIDReject         MissingIDPolicy = "reject"
	MissingIDGenerateStable MissingIDPolicy = "generate_stable"
)

type SourcePreservationPolicy string

const (
	SourceDisabled          SourcePreservationPolicy = "disabled"
	SourceBoundedSameFormat SourcePreservationPolicy = "bounded_same_format"
)

// ResponseVendor identifies a provider whose failures arrive in an envelope
// other than its wire format's canonical error object. The backend's provider
// type selects it; members a wire contract does not declare are tolerated for
// every backend and need no vendor.
type ResponseVendor string

// ResponseVendorCloudflare decodes Cloudflare Workers AI failures: a top-level
// errors[] array with Workers AI's own integral codes instead of the canonical
// OpenAI error object.
const ResponseVendorCloudflare ResponseVendor = "cloudflare"

// ResponseVendorSnowflake decodes Snowflake Cortex AI failures: a flat object
// carrying the vendor's own string code instead of the canonical OpenAI error
// object. Accepted responses use the canonical contract.
const ResponseVendorSnowflake ResponseVendor = "snowflake"

// ResponseVendorOpenRouter decodes OpenRouter failures: the canonical OpenAI
// error object without a type, plus a provider metadata object and a
// top-level user_id. Accepted responses use the canonical contract.
const ResponseVendorOpenRouter ResponseVendor = "openrouter"

type Policy struct {
	UnknownFields      UnknownFieldPolicy
	LossyFeatures      LossyPolicy
	MissingStableIDs   MissingIDPolicy
	SourcePreservation SourcePreservationPolicy
	Limits             Limits
	// ResponseVendor selects a provider's own failure envelope at the response
	// boundary. Empty decodes the wire format's canonical error object.
	ResponseVendor ResponseVendor
	// RejectUnknownUpstreamFields fails an upstream response, stream frame or
	// error envelope that carries a member its wire contract does not declare,
	// instead of dropping and reporting the member. Conformance tests set it so
	// the Router's own fixtures and provider simulator cannot drift from the
	// modelled contracts. The Router never sets it, and configuration cannot
	// reach it.
	RejectUnknownUpstreamFields bool
	// ProviderStopSequences lets a decoder report a provider's non-standard
	// matched stop sequence (vLLM's choices[].stop_reason) as a stop_sequence
	// terminal. TranslateResponse and NewStreamWithMutation set it only when
	// the target wire format can carry a matched stop sequence.
	ProviderStopSequences bool
	// AllowedLossyFields are translated with a diagnostic even under
	// LossyReject. Entries come from OptInLossyFields.
	AllowedLossyFields []string
}

type Limits struct {
	BodyBytes int
	// SourceEnvelopeBytes is deliberately independent from BodyBytes. Source
	// preservation is optional fidelity state and must never cause a second
	// copy of a large accepted request or response.
	SourceEnvelopeBytes  int
	ModelBytes           int
	Instructions         int
	Messages             int
	Candidates           int
	Alternatives         int
	OutputItems          int
	ContentBlocks        int
	Citations            int
	CitationURLBytes     int
	CitationTitleBytes   int
	JSONDepth            int
	ToolResultDepth      int
	Tools                int
	ToolNameBytes        int
	ToolDescriptionBytes int
	IdentifierBytes      int
	SchemaBytes          int
	MetadataBytes        int
	MetadataEntries      int
	MetadataKeyBytes     int
	MetadataValueBytes   int
	ToolArgumentsBytes   int
	ReasoningEffortBytes int
	StopSequences        int
	StopBytes            int
	LogitBiasEntries     int
	TextBytes            int
	MediaReferenceBytes  int
	MediaDataBytes       int
	SSEFrameBytes        int
	UnfinishedArguments  int
	Events               int
	Diagnostics          int
}

func DefaultPolicy() Policy {
	return Policy{
		UnknownFields: UnknownReject, LossyFeatures: LossyReject,
		MissingStableIDs:   MissingIDGenerateStable,
		SourcePreservation: SourceBoundedSameFormat,
		Limits: Limits{
			BodyBytes: 64 << 20, SourceEnvelopeBytes: 256 << 10,
			ModelBytes: 1024, Instructions: 256, Messages: 4096, Candidates: 32,
			Alternatives: 32, OutputItems: 4096, ContentBlocks: 16_384,
			Citations: 256, CitationURLBytes: 16 << 10, CitationTitleBytes: 16 << 10,
			JSONDepth: 128, ToolResultDepth: 8,
			Tools: 256, ToolNameBytes: 256, ToolDescriptionBytes: 256 << 10,
			IdentifierBytes: 1024, SchemaBytes: 4 << 20, MetadataBytes: 64 << 10,
			MetadataEntries: 256, MetadataKeyBytes: 256, MetadataValueBytes: 8 << 10,
			ToolArgumentsBytes: 4 << 20, ReasoningEffortBytes: 32,
			StopSequences: 16, StopBytes: 64 << 10, LogitBiasEntries: 16_384, TextBytes: 16 << 20,
			MediaReferenceBytes: 16 << 10, MediaDataBytes: 48 << 20,
			SSEFrameBytes: 1 << 20, UnfinishedArguments: 4 << 20,
			Events: 1_000_000, Diagnostics: 64,
		},
	}
}

type DiagnosticAction string

const (
	DiagnosticDropped      DiagnosticAction = "dropped"
	DiagnosticApproximated DiagnosticAction = "approximated"
	DiagnosticGenerated    DiagnosticAction = "generated"
	DiagnosticTruncated    DiagnosticAction = "truncated"
)

type Diagnostic struct {
	Source WireFormat
	Target WireFormat
	Field  string
	Action DiagnosticAction
	Reason string
}

type Diagnostics []Diagnostic

// ResponseRenderContext is bounded request context that a response wire
// contract may echo. It is not model output and must never be retained as
// provider state.
type ResponseRenderContext struct {
	PreviousResponseID string
}

// Envelope is bounded, ephemeral wire fidelity and rendering state. It must
// never be serialized into logs, snapshots, YAML, or usage records.
type Envelope struct {
	Format                   WireFormat
	Generation               uint64
	Request                  []byte
	Response                 []byte
	ResponseReencodeRequired bool // Provider decorations must not be replayed to a same-format client.
	SourceStop               string
	ResponseRender           ResponseRenderContext
}

func (envelope Envelope) CanReplay(format WireFormat, generation uint64, policy Policy, response bool) bool {
	if policy.SourcePreservation != SourceBoundedSameFormat || envelope.Format != format ||
		envelope.Generation == 0 || envelope.Generation != generation {
		return false
	}
	if response {
		if envelope.ResponseReencodeRequired {
			return false
		}
		if envelope.ResponseRender.PreviousResponseID != "" {
			return false
		}
		return len(envelope.Response) > 0
	}
	return len(envelope.Request) > 0
}
