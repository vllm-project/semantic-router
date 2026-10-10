package topiccontinuity

const (
	// SchemaVersion identifies the result shape and its safety invariants.
	// Consumers gate on it; heuristic changes never bump it.
	SchemaVersion = "v1"
	// EvaluatorVersion identifies lexicons, extraction rules, formulas,
	// constants, and precedence. Any behavior change bumps it.
	EvaluatorVersion = "lexical.1"

	maxScanMessages            = 512
	maxContentBlocksPerPrepare = 4096
	maxToolNamesPerTurn        = 16
	maxToolNameBytes           = 64
	maxEntitiesPerSegment      = 1024
	maxEntityBytes             = 128

	// Hard ranges for configurable limits. Config validation enforces them,
	// and EvaluateAll rejects an out-of-range rule at the package boundary.
	MinPriorTurns = 1
	MaxPriorTurns = 32
	MinTurnBytes  = 256
	MaxTurnBytes  = 65536
	MinInputBytes = 1024
	MaxInputBytes = 1 << 20
)

// Limits bounds the evidence one rule may read.
type Limits struct {
	MaxPriorTurns int
	MaxTurnBytes  int
	MaxInputBytes int
}

// HistoryPolicy is the comparable preparation key: rules sharing a policy
// share one Prepare and one Extract.
type HistoryPolicy struct {
	Limits           Limits
	IncludeAssistant bool
}

// EvalConfig holds one rule's effective (defaulted, validated) settings.
type EvalConfig struct {
	Name         string
	Policy       HistoryPolicy
	Continuation float64
	Change       float64
}

// textSegment is one contiguous slice of original text. Features never cross a
// segment boundary, so truncation can never invent a token or phrase.
type textSegment string

// evidenceTurn holds one conversation turn's retained evidence, already windowed
// under its policy. User holds one or more segments: each retained user text
// block, or its head and tail when truncated.
type evidenceTurn struct {
	User      []textSegment
	Assistant []textSegment
	ToolNames []string
	HasOpaque bool
	Truncated bool

	toolCallCapReached bool
}

// liveKind describes the final message of the request.
type liveKind uint8

const (
	liveNone liveKind = iota
	liveUser
	liveToolContinuation
)

// Coverage is relative to the configured evidence policy. It states whether
// everything the policy selects was processed, not whether every content
// block in the request was evaluated.
type Coverage string

const (
	// CoverageFull means all policy-selected evidence was processed, no cap
	// was reached, and no turn exists beyond the window.
	CoverageFull Coverage = "full"
	// CoverageWindow means evidence inside the window is complete, but older
	// turns exist beyond MaxPriorTurns.
	CoverageWindow Coverage = "window"
	// CoveragePartial means truncation, a byte-cap drop, an incomplete scan,
	// or an evidence-phase count cap (entity, tool-name, or content-block).
	CoveragePartial Coverage = "partial"
)

// EvidenceScope makes deliberate policy exclusions observable. It never
// blocks change by itself; consumers decide what to trust.
type EvidenceScope struct {
	AssistantIncluded      bool
	ExcludedContentPresent bool
	FeatureCapReached      bool
}

// HistorySource records provenance. It does not bind a result to particular
// snapshot bytes.
type HistorySource string

// SourceOriginalSnapshot is the original history captured before any RAG or
// Memory enrichment.
const SourceOriginalSnapshot HistorySource = "original_snapshot"

// Class is continuation, change, or unknown. Change always implies full
// coverage; unknown always has confidence 0.
type Class string

const (
	ClassContinuation Class = "continuation"
	ClassChange       Class = "change"
	ClassUnknown      Class = "unknown"
)

// Reason is a bounded, content-free explanation. Its prefix names its class.
type Reason string

const (
	ReasonToolExchange        Reason = "continuation_tool_exchange"
	ReasonReference           Reason = "continuation_reference"
	ReasonAcknowledgement     Reason = "continuation_acknowledgement"
	ReasonEntityOverlap       Reason = "continuation_entity_overlap"
	ReasonLexicalOverlap      Reason = "continuation_lexical"
	ReasonExplicitChange      Reason = "change_explicit_marker"
	ReasonDisjoint            Reason = "change_disjoint"
	ReasonHistoryUnavailable  Reason = "unknown_history_unavailable"
	ReasonNoLiveUserTurn      Reason = "unknown_no_live_user_turn"
	ReasonOrphanToolResult    Reason = "unknown_orphan_tool_result"
	ReasonNoPriorTurn         Reason = "unknown_no_prior_turn"
	ReasonOpaqueOnly          Reason = "unknown_opaque_only"
	ReasonInsufficientText    Reason = "unknown_insufficient_text"
	ReasonOverBudget          Reason = "unknown_over_budget"
	ReasonHistoryBeyondWindow Reason = "unknown_history_beyond_window"
	ReasonIncompleteHistory   Reason = "unknown_incomplete_history"
	ReasonConflicting         Reason = "unknown_conflicting"
	ReasonAmbiguousMarker     Reason = "unknown_ambiguous_marker"
	ReasonInconclusive        Reason = "unknown_inconclusive"
	ReasonCancelled           Reason = "unknown_cancelled"
	ReasonInternalError       Reason = "unknown_internal_error"
)

var reasonClass = map[Reason]Class{
	ReasonToolExchange:        ClassContinuation,
	ReasonReference:           ClassContinuation,
	ReasonAcknowledgement:     ClassContinuation,
	ReasonEntityOverlap:       ClassContinuation,
	ReasonLexicalOverlap:      ClassContinuation,
	ReasonExplicitChange:      ClassChange,
	ReasonDisjoint:            ClassChange,
	ReasonHistoryUnavailable:  ClassUnknown,
	ReasonNoLiveUserTurn:      ClassUnknown,
	ReasonOrphanToolResult:    ClassUnknown,
	ReasonNoPriorTurn:         ClassUnknown,
	ReasonOpaqueOnly:          ClassUnknown,
	ReasonInsufficientText:    ClassUnknown,
	ReasonOverBudget:          ClassUnknown,
	ReasonHistoryBeyondWindow: ClassUnknown,
	ReasonIncompleteHistory:   ClassUnknown,
	ReasonConflicting:         ClassUnknown,
	ReasonAmbiguousMarker:     ClassUnknown,
	ReasonInconclusive:        ClassUnknown,
	ReasonCancelled:           ClassUnknown,
	ReasonInternalError:       ClassUnknown,
}

// fallbackReason reports the reasons whose result is a degraded fallback.
func fallbackReason(reason Reason) bool {
	return reason == ReasonHistoryUnavailable || reason == ReasonCancelled || reason == ReasonInternalError
}

// Features are content-free numbers and flags. All fields are plain values,
// so a struct copy is detached.
type Features struct {
	PriorTurnsExamined int
	InputBytes         int
	LiveTerms          int
	// LiveProseTerms counts live terms outside code and quoted regions. A
	// turn must have enough prose to count as a self-contained new request.
	LiveProseTerms  int
	LexicalScore    float64
	EntityScore     float64
	CombinedScore   float64
	MaxRawScore     float64
	MaxEntityScore  float64
	StrongReference bool
	WeakReference   bool
	ChangeMarker    bool
	MarkerAmbiguous bool
	Acknowledgement bool
}

// Result is one rule's typed evidence.
type Result struct {
	SchemaVersion    string
	EvaluatorVersion string
	Signal           string
	Class            Class
	// Confidence is a fixed-formula heuristic strength in [0,1], not a
	// calibrated probability.
	Confidence    float64
	Reason        Reason
	Fallback      bool
	Coverage      Coverage
	Scope         EvidenceScope
	HistorySource HistorySource
	Features      Features
}

// preparation is the bounded, segmented evidence for one HistoryPolicy.
type preparation struct {
	Policy         HistoryPolicy
	liveKind       liveKind
	Live           evidenceTurn
	Prior          []evidenceTurn // newest first
	LiveOpaqueOnly bool
	Coverage       Coverage
	Scope          EvidenceScope
	InputBytes     int
	// Status is empty when the evidence is usable; otherwise it is the
	// terminal unknown reason.
	Status Reason
}

// extraction holds every threshold-independent feature of one prepared group.
type extraction struct {
	Policy   HistoryPolicy
	Features Features
	// Coverage is preparation.Coverage, lowered to partial when a feature cap is
	// reached during extraction.
	Coverage Coverage
	Scope    EvidenceScope
	Status   Reason
}
