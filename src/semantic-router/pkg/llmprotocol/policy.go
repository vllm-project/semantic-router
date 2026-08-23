package llmprotocol

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

type Policy struct {
	UnknownFields      UnknownFieldPolicy
	LossyFeatures      LossyPolicy
	MissingStableIDs   MissingIDPolicy
	SourcePreservation SourcePreservationPolicy
	Limits             Limits
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
			Tools: 256, ToolNameBytes: 256, ToolDescriptionBytes: 16 << 10,
			IdentifierBytes: 1024, SchemaBytes: 4 << 20, MetadataBytes: 64 << 10,
			MetadataEntries: 256, MetadataKeyBytes: 256, MetadataValueBytes: 8 << 10,
			ToolArgumentsBytes: 4 << 20, ReasoningEffortBytes: 32,
			StopSequences: 16, StopBytes: 64 << 10, TextBytes: 16 << 20,
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

// Envelope is bounded, ephemeral source fidelity state. It must never be
// serialized into logs, snapshots, YAML, or usage records.
type Envelope struct {
	Format     WireFormat
	Generation uint64
	Request    []byte
	Response   []byte
	SourceStop string
}

func (envelope Envelope) CanReplay(format WireFormat, generation uint64, policy Policy, response bool) bool {
	if policy.SourcePreservation != SourceBoundedSameFormat || envelope.Format != format ||
		envelope.Generation == 0 || envelope.Generation != generation {
		return false
	}
	if response {
		return len(envelope.Response) > 0
	}
	return len(envelope.Request) > 0
}
