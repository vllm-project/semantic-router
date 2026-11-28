package turnselection

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Strategy chooses how eligible prior turns compete for the history budget.
type Strategy string

const (
	// StrategyRelevance combines minimum recent-turn retention with bounded
	// lexical/external relevance, recency decay, and continuity scoring.
	StrategyRelevance Strategy = "relevance"
	// StrategyRecency retains only the most recent eligible prior turns that
	// fit within the budget and turn bounds.
	StrategyRecency Strategy = "recency"
	// StrategyFull keeps all prior turns unchanged as a control baseline.
	StrategyFull Strategy = "full"
)

// BudgetConflictPolicy controls how selection responds when protected content
// or MinRecentTurns cannot fit inside the effective pre-compression budget.
type BudgetConflictPolicy string

const (
	// ConflictStrict rejects the selection with an explicit error rather than
	// relaxing MinRecentTurns or pretending an overflowing request fits.
	ConflictStrict BudgetConflictPolicy = "strict"
	// ConflictRelaxMinRecency relaxes MinRecentTurns from oldest to newest when
	// protected content fits but the guaranteed recent window exceeds the
	// available history budget. Protected overflow still fails explicitly.
	ConflictRelaxMinRecency BudgetConflictPolicy = "relax_min_recency"
	// ConflictDropEligible relaxes MinRecentTurns and, if protected content
	// alone exceeds the budget, removes all eligible prior turns to minimize
	// overflow while explicitly reporting BudgetProtectedOverflow.
	ConflictDropEligible BudgetConflictPolicy = "drop_eligible"
)

// ContinuityMode controls deterministic conversation-continuity signals.
type ContinuityMode string

const (
	// ContinuityEnabled enables bounded referential and forward-revision
	// continuity scoring alongside relevance and recency.
	ContinuityEnabled ContinuityMode = "enabled"
	// ContinuityDisabled disables continuity boosts for pure lexical/recency
	// ablation and comparison.
	ContinuityDisabled ContinuityMode = "disabled"
)

// ScorerFallbackMode controls behavior when an injected RelevanceScorer fails,
// times out, or returns invalid scores.
type ScorerFallbackMode string

const (
	// ScorerFallbackLexical deterministically falls back to bounded local BM25
	// lexical scoring and records the fallback reason in Diagnostics.
	ScorerFallbackLexical ScorerFallbackMode = "lexical"
	// ScorerFallbackFail fails the selection step with ReasonScoringFailed.
	ScorerFallbackFail ScorerFallbackMode = "fail"
)

// Outcome describes the reconciled outcome of one turn-selection step.
type Outcome string

// Outcome values reported on reconciled Diagnostics.
const (
	OutcomeApplied Outcome = "applied"
	OutcomeSkipped Outcome = "skipped"
	OutcomeFailed  Outcome = "failed"
)

// BudgetStatus describes whether the selected pre-compression request satisfied
// the configured selection-stage budget.
type BudgetStatus string

// BudgetStatus values reported on Diagnostics.
const (
	BudgetUnbounded         BudgetStatus = "unbounded"
	BudgetSatisfied         BudgetStatus = "satisfied"
	BudgetMinRecencyRelaxed BudgetStatus = "min_recency_relaxed"
	BudgetProtectedOverflow BudgetStatus = "protected_overflow"
	BudgetUnsatisfiable     BudgetStatus = "unsatisfiable"
)

// ScoringSource records which scoring path produced relevance scores.
type ScoringSource string

// ScoringSource values reported on Diagnostics.
const (
	ScoringNone            ScoringSource = "none"
	ScoringLexical         ScoringSource = "lexical"
	ScoringExternal        ScoringSource = "external"
	ScoringLexicalFallback ScoringSource = "lexical_fallback"
)

// RecoveryStatus records the content-minimized recovery state for a selection.
type RecoveryStatus string

// RecoveryStatus values reported on Diagnostics.
const (
	RecoveryNotRequired   RecoveryStatus = "not_required"
	RecoveryNotConfigured RecoveryStatus = "not_configured"
)

// Bounded fixed reason values recorded in content-minimized Diagnostics.
const (
	ReasonApplied             = "applied"
	ReasonNoEligibleHistory   = "no_eligible_history"
	ReasonWithinBudget        = "within_budget"
	ReasonFullHistoryBaseline = "full_history_baseline"
	ReasonUnboundedPolicy     = "unbounded_policy"
	ReasonProtectedOverflow   = "protected_overflow"
	ReasonMinRecencyRelaxed   = "min_recency_relaxed"
	ReasonBudgetUnsatisfiable = "budget_unsatisfiable"
	ReasonMinRecencyConflict  = "min_recency_conflict"
	ReasonInvalidPolicy       = "invalid_policy"
	ReasonScanLimitExceeded   = "scan_limit_exceeded"
	ReasonScoringFailed       = "scoring_failed"
	ReasonCancelled           = "cancelled"
	ReasonNotEvaluated        = "not_evaluated"
)

// Bounded fallback reason values for external RelevanceScorer failures.
const (
	FallbackNone           = ""
	FallbackScorerError    = "scorer_error"
	FallbackLengthMismatch = "scorer_length_mismatch"
	FallbackInvalidScore   = "scorer_invalid_score"
	FallbackScorerTimeout  = "scorer_timeout"
	FallbackCandidateCap   = "scorer_candidate_cap"
)

// Bounded pin reasons explaining why a turn is not eligible for removal.
const (
	PinLiveTurn          = "live_turn"
	PinInstructions      = "instructions"
	PinAuthorization     = "authorization"
	PinSafety            = "safety"
	PinMultimodal        = "multimodal"
	PinContinuation      = "continuation"
	PinNonHistorySource  = "non_history_source"
	PinCrossTurnExchange = "cross_turn_exchange"
	PinUnknown           = "unknown"
)

// Bounded per-turn decision outcomes and reasons for content-minimized receipts.
const (
	TurnOutcomePinned         = "pinned"
	TurnOutcomeRetainedRecent = "retained_recent"
	TurnOutcomeRetainedRanked = "retained_ranked"
	TurnOutcomeRetainedFull   = "retained_full"
	TurnOutcomeRemoved        = "removed"

	TurnReasonProtected         = "protected"
	TurnReasonMinRecency        = "min_recency"
	TurnReasonRelevance         = "relevance"
	TurnReasonContinuity        = "continuity"
	TurnReasonRecency           = "recency"
	TurnReasonFullBaseline      = "full_baseline"
	TurnReasonOverTokenBudget   = "over_token_budget"
	TurnReasonOverTurnLimit     = "over_turn_limit"
	TurnReasonProtectedOverflow = "protected_overflow"
)

// Default bounded operational limits.
const (
	DefaultRelevanceWeight  = 0.55
	DefaultContinuityWeight = 0.25
	DefaultRecencyWeight    = 0.20

	DefaultMaxHistoryTurns      = 256
	DefaultMaxCandidateTurns    = 64
	DefaultMaxTurnTextBytes     = 4096
	DefaultMaxQueryBytes        = 2048
	DefaultMaxScoredTurns       = 32
	DefaultMaxScoringBytes      = 65536
	DefaultMaxContinuityAnchors = 3
	DefaultScoringTimeout       = 50 * time.Millisecond
	DefaultPlanTimeout          = 100 * time.Millisecond

	// MaxReceiptTurns caps the per-turn decision slice carried on Diagnostics.
	MaxReceiptTurns = 64
)

// Sentinel errors returned for explicit selection failures.
var (
	ErrInvalidPolicy       = errors.New("invalid turn selection policy")
	ErrBudgetUnsatisfiable = errors.New("protected context exceeds turn selection budget")
	ErrMinRecencyConflict  = errors.New("minimum recent turns exceed turn selection budget")
	ErrScanLimitExceeded   = errors.New("history exceeds turn selection scan limit")
	ErrScoringFailed       = errors.New("turn relevance scoring failed")
)

// BudgetPolicy configures pre-compression token and turn bounds for selection.
type BudgetPolicy struct {
	MaxInputTokens      int
	MaxHistoryTokens    int
	MaxPriorTurns       int
	MinRecentTurns      int
	OutputReserveTokens int
	ConflictPolicy      BudgetConflictPolicy
}

// ScoringPolicy configures ranking weights, continuity, and strict work bounds.
type ScoringPolicy struct {
	RelevanceWeight      float64
	RecencyWeight        float64
	ContinuityWeight     float64
	Continuity           ContinuityMode
	ScorerFallback       ScorerFallbackMode
	MaxHistoryTurns      int
	MaxCandidateTurns    int
	MaxTurnTextBytes     int
	MaxQueryBytes        int
	MaxScoredTurns       int
	MaxScoringBytes      int
	MaxContinuityAnchors int
	ScoringTimeout       time.Duration
	Timeout              time.Duration
}

// Policy defines the complete pure configuration for turn selection.
type Policy struct {
	Strategy    Strategy
	Budget      BudgetPolicy
	Scoring     ScoringPolicy
	FailureMode contextcompression.FailureMode
}

// DefaultPolicy returns conservative, bounded defaults suitable for deterministic
// relevance-and-continuity turn selection.
func DefaultPolicy() Policy {
	return Policy{
		Strategy: StrategyRelevance,
		Budget: BudgetPolicy{
			MinRecentTurns: 1,
			ConflictPolicy: ConflictStrict,
		},
		Scoring: ScoringPolicy{
			RelevanceWeight:      DefaultRelevanceWeight,
			RecencyWeight:        DefaultRecencyWeight,
			ContinuityWeight:     DefaultContinuityWeight,
			Continuity:           ContinuityEnabled,
			ScorerFallback:       ScorerFallbackLexical,
			MaxHistoryTurns:      DefaultMaxHistoryTurns,
			MaxCandidateTurns:    DefaultMaxCandidateTurns,
			MaxTurnTextBytes:     DefaultMaxTurnTextBytes,
			MaxQueryBytes:        DefaultMaxQueryBytes,
			MaxScoredTurns:       DefaultMaxScoredTurns,
			MaxScoringBytes:      DefaultMaxScoringBytes,
			MaxContinuityAnchors: DefaultMaxContinuityAnchors,
			ScoringTimeout:       DefaultScoringTimeout,
			Timeout:              DefaultPlanTimeout,
		},
		FailureMode: contextcompression.FailureOpen,
	}
}

// normalized fills zero-value fields with safe bounded defaults without mutating
// explicit caller overrides.
func (p Policy) normalized() Policy {
	out := p
	if out.Strategy == "" {
		out.Strategy = StrategyRelevance
	}
	if out.Budget.ConflictPolicy == "" {
		out.Budget.ConflictPolicy = ConflictStrict
	}
	if out.FailureMode == "" {
		out.FailureMode = contextcompression.FailureOpen
	}
	if out.Scoring.Continuity == "" {
		out.Scoring.Continuity = ContinuityEnabled
	}
	if out.Scoring.ScorerFallback == "" {
		out.Scoring.ScorerFallback = ScorerFallbackLexical
	}
	if out.Scoring.RelevanceWeight == 0 && out.Scoring.RecencyWeight == 0 && out.Scoring.ContinuityWeight == 0 {
		out.Scoring.RelevanceWeight = DefaultRelevanceWeight
		out.Scoring.RecencyWeight = DefaultRecencyWeight
		out.Scoring.ContinuityWeight = DefaultContinuityWeight
	}
	if out.Scoring.MaxHistoryTurns == 0 {
		out.Scoring.MaxHistoryTurns = DefaultMaxHistoryTurns
	}
	if out.Scoring.MaxCandidateTurns == 0 {
		out.Scoring.MaxCandidateTurns = DefaultMaxCandidateTurns
	}
	if out.Scoring.MaxTurnTextBytes == 0 {
		out.Scoring.MaxTurnTextBytes = DefaultMaxTurnTextBytes
	}
	if out.Scoring.MaxQueryBytes == 0 {
		out.Scoring.MaxQueryBytes = DefaultMaxQueryBytes
	}
	if out.Scoring.MaxScoredTurns == 0 {
		out.Scoring.MaxScoredTurns = DefaultMaxScoredTurns
	}
	if out.Scoring.MaxScoringBytes == 0 {
		out.Scoring.MaxScoringBytes = DefaultMaxScoringBytes
	}
	if out.Scoring.MaxContinuityAnchors == 0 {
		out.Scoring.MaxContinuityAnchors = DefaultMaxContinuityAnchors
	}
	if out.Scoring.ScoringTimeout == 0 {
		out.Scoring.ScoringTimeout = DefaultScoringTimeout
	}
	if out.Scoring.Timeout == 0 {
		out.Scoring.Timeout = DefaultPlanTimeout
	}
	return out
}

// Validate checks policy invariants before planning executes.
func (p Policy) Validate() error {
	n := p.normalized()
	switch n.Strategy {
	case StrategyRelevance, StrategyRecency, StrategyFull:
	default:
		return fmt.Errorf("%w: unsupported strategy %q", ErrInvalidPolicy, n.Strategy)
	}
	switch n.Budget.ConflictPolicy {
	case ConflictStrict, ConflictRelaxMinRecency, ConflictDropEligible:
	default:
		return fmt.Errorf("%w: unsupported conflict policy %q", ErrInvalidPolicy, n.Budget.ConflictPolicy)
	}
	switch n.FailureMode {
	case contextcompression.FailureOpen, contextcompression.FailureClosed:
	default:
		return fmt.Errorf("%w: unsupported failure mode %q", ErrInvalidPolicy, n.FailureMode)
	}
	switch n.Scoring.Continuity {
	case ContinuityEnabled, ContinuityDisabled:
	default:
		return fmt.Errorf("%w: unsupported continuity mode %q", ErrInvalidPolicy, n.Scoring.Continuity)
	}
	switch n.Scoring.ScorerFallback {
	case ScorerFallbackLexical, ScorerFallbackFail:
	default:
		return fmt.Errorf("%w: unsupported scorer fallback %q", ErrInvalidPolicy, n.Scoring.ScorerFallback)
	}
	if n.Budget.MaxInputTokens < 0 ||
		n.Budget.MaxHistoryTokens < 0 ||
		n.Budget.MaxPriorTurns < 0 ||
		n.Budget.MinRecentTurns < 0 ||
		n.Budget.OutputReserveTokens < 0 {
		return fmt.Errorf("%w: budget bounds must be non-negative", ErrInvalidPolicy)
	}
	if n.Budget.MaxPriorTurns > 0 && n.Budget.MinRecentTurns > n.Budget.MaxPriorTurns {
		return fmt.Errorf("%w: min_recent_turns (%d) exceeds max_prior_turns (%d)",
			ErrInvalidPolicy, n.Budget.MinRecentTurns, n.Budget.MaxPriorTurns)
	}
	for _, w := range []float64{n.Scoring.RelevanceWeight, n.Scoring.RecencyWeight, n.Scoring.ContinuityWeight} {
		if math.IsNaN(w) || math.IsInf(w, 0) || w < 0 {
			return fmt.Errorf("%w: scoring weights must be finite and non-negative", ErrInvalidPolicy)
		}
	}
	if n.Scoring.MaxHistoryTurns < 0 ||
		n.Scoring.MaxCandidateTurns < 0 ||
		n.Scoring.MaxTurnTextBytes < 0 ||
		n.Scoring.MaxQueryBytes < 0 ||
		n.Scoring.MaxScoredTurns < 0 ||
		n.Scoring.MaxScoringBytes < 0 ||
		n.Scoring.MaxContinuityAnchors < 0 ||
		n.Scoring.ScoringTimeout < 0 ||
		n.Scoring.Timeout < 0 {
		return fmt.Errorf("%w: scoring limits must be non-negative", ErrInvalidPolicy)
	}
	return nil
}

// EffectiveFailureMode returns the normalized FailureMode for TransformationStep.
func (p Policy) EffectiveFailureMode() contextcompression.FailureMode {
	return p.normalized().FailureMode
}

// ModelBudget provides selected-model capacity and output demand for one request.
type ModelBudget struct {
	Model               string
	Capabilities        contextcompression.ModelContextCapabilities
	ReserveOutputTokens int
}

// TurnCostEstimator supplies conservative pre-compression token counts for
// non-message request overhead (instructions, tool schemas, output format) and
// individual messages keyed by their stable MessageView.ID.
//
// Implementations must be read-only and deterministic. Model-tokenizer-aware
// counters (#3050) can implement this interface directly.
type TurnCostEstimator interface {
	FixedOverheadTokens(model string) (tokens int, source string)
	MessageTokens(model string, message contextcompression.MessageView) (tokens int, source string)
}

// RequestInput bundles the narrow per-request dependencies injected into
// selection planning. All fields are optional; nil counters and scorers fall
// back to deterministic local heuristics.
type RequestInput struct {
	ModelBudget   ModelBudget
	TokenCounter  contextcompression.TokenCounter
	CostEstimator TurnCostEstimator
	Scorer        contextcompression.RelevanceScorer
}

// ViewCostEstimator counts MessageView blocks using a contextcompression.TokenCounter
// (defaulting to HeuristicTokenCounter) plus conservative message/exchange framing.
type ViewCostEstimator struct {
	Counter contextcompression.TokenCounter
}

// FixedOverheadTokens returns zero because TransformationView does not expose
// top-level instructions or tool schemas.
func (e ViewCostEstimator) FixedOverheadTokens(model string) (int, string) {
	_, source := e.counter().CountText(model, "")
	return 0, sanitizeCounterSource(source)
}

// MessageTokens estimates conservative tokens for one detached MessageView.
func (e ViewCostEstimator) MessageTokens(model string, message contextcompression.MessageView) (int, string) {
	counter := e.counter()
	total := llmprotocol.InputMessageFramingTokens
	source := "heuristic"
	for _, block := range message.Blocks {
		tokens, blockSource := counter.CountText(model, block.Text)
		total = llmprotocol.SaturatingTokenSum(total, max(0, tokens))
		if blockSource != "" {
			source = blockSource
		}
	}
	if len(message.ExchangeIDs) > 0 {
		total = llmprotocol.SaturatingTokenSum(total, len(message.ExchangeIDs)*llmprotocol.InputToolCallFramingTokens)
	}
	return total, sanitizeCounterSource(source)
}

func (e ViewCostEstimator) counter() contextcompression.TokenCounter {
	if e.Counter != nil {
		return e.Counter
	}
	return contextcompression.HeuristicTokenCounter{}
}

// semanticCostEstimator snapshots read-only per-message and fixed-overhead costs
// from a neutral llmprotocol.Request before history removal runs. Because
// MessageView.ID preserves original message indices across upstream reset and
// deduplication, snapshot lookups remain stable without retaining a mutable
// request pointer.
type semanticCostEstimator struct {
	fixedOverhead int
	messageCosts  map[int]int
	source        string
	fallback      ViewCostEstimator
}

// NewSemanticCostEstimator captures immutable, conservative pre-compression
// token costs from request, including top-level instructions, tool definitions,
// output schemas, tool-call arguments, reasoning payloads, and media reserves.
// When counter is non-nil, prose text is counted through counter.CountText
// while non-text, tool-structure, media, and framing reserves remain preserved.
// It never mutates or retains request.
func NewSemanticCostEstimator(
	request *llmprotocol.Request,
	counter contextcompression.TokenCounter,
) TurnCostEstimator {
	fallback := ViewCostEstimator{Counter: counter}
	if request == nil {
		return fallback
	}
	fixedReq := &llmprotocol.Request{
		Instructions: request.Instructions,
		Tools:        request.Tools,
		OutputFormat: request.OutputFormat,
	}
	fixedTokens, source := estimateSemanticSliceTokens(request.Model, fixedReq, counter)
	costs := make(map[int]int, len(request.Messages))
	for idx := range request.Messages {
		single := &llmprotocol.Request{Messages: request.Messages[idx : idx+1]}
		msgTokens, msgSource := estimateSemanticSliceTokens(request.Model, single, counter)
		costs[idx] = msgTokens
		if msgSource != "" {
			source = msgSource
		}
	}
	return &semanticCostEstimator{
		fixedOverhead: fixedTokens,
		messageCosts:  costs,
		source:        sanitizeCounterSource(source),
		fallback:      fallback,
	}
}

func estimateSemanticSliceTokens(
	model string,
	req *llmprotocol.Request,
	counter contextcompression.TokenCounter,
) (int, string) {
	neutral := llmprotocol.EstimateInput(req)
	if counter == nil {
		return neutral.Tokens, "neutral_estimate"
	}
	// Separate prose text from structural/media/framing overhead so custom or
	// model-calibrated TokenCounters accurately govern prose while preserving
	// non-text reserves (tool schemas, tool-call args, images, framing).
	neutralProseTokens := 0
	if neutral.TextBytes > 0 {
		neutralProseTokens = (neutral.TextBytes + llmprotocol.InputTextBytesPerToken - 1) / llmprotocol.InputTextBytesPerToken
	}
	nonProseTokens := max(0, neutral.Tokens-neutralProseTokens)
	proseText := collectSemanticProse(req)
	proseTokens := 0
	source := "heuristic"
	if proseText != "" {
		counted, countedSource := counter.CountText(model, proseText)
		proseTokens = max(0, counted)
		if countedSource != "" {
			source = countedSource
		}
	} else {
		_, countedSource := counter.CountText(model, "")
		if countedSource != "" {
			source = countedSource
		}
	}
	return llmprotocol.SaturatingTokenSum(proseTokens, nonProseTokens), source
}

func collectSemanticProse(req *llmprotocol.Request) string {
	if req == nil {
		return ""
	}
	var b strings.Builder
	appendText := func(s string) {
		if s == "" {
			return
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(s)
	}
	var walk func([]llmprotocol.Content)
	walk = func(contents []llmprotocol.Content) {
		for _, c := range contents {
			switch c.Kind {
			case llmprotocol.ContentText, llmprotocol.ContentRefusal, llmprotocol.ContentReasoning:
				appendText(c.Text)
				if c.ReasoningDetail != nil {
					appendText(c.ReasoningDetail.Text.Value)
					appendText(c.ReasoningDetail.Summary.Value)
				}
			case llmprotocol.ContentToolResult:
				if c.ToolResult != nil {
					walk(c.ToolResult.Content)
				}
			}
		}
	}
	for _, instr := range req.Instructions {
		walk(instr.Content)
	}
	for _, msg := range req.Messages {
		walk(msg.Content)
	}
	for _, tool := range req.Tools {
		appendText(tool.Name)
		appendText(tool.Description)
	}
	return b.String()
}

func (e *semanticCostEstimator) FixedOverheadTokens(_ string) (int, string) {
	return e.fixedOverhead, e.source
}

func (e *semanticCostEstimator) MessageTokens(model string, message contextcompression.MessageView) (int, string) {
	if cost, ok := e.messageCosts[message.ID]; ok {
		return cost, e.source
	}
	return e.fallback.MessageTokens(model, message)
}

func sanitizeCounterSource(source string) string {
	switch strings.TrimSpace(source) {
	case "heuristic", "model_heuristic", "provider_calibrated", "tokenizer", "neutral_estimate":
		return strings.TrimSpace(source)
	case "":
		return "heuristic"
	default:
		return "custom"
	}
}

// TurnDecision records a single turn's content-minimized selection outcome.
// It stores only numeric identities, token counts, and fixed enum tags.
type TurnDecision struct {
	TurnID         int    `json:"turn_id"`
	FirstMessageID int    `json:"first_message_id"`
	MessageCount   int    `json:"message_count"`
	Tokens         int    `json:"tokens"`
	Outcome        string `json:"outcome"`
	Reason         string `json:"reason"`
}

// Diagnostics is the bounded, content-minimized selection receipt. It never
// contains conversation text, tool arguments, content hashes, recovery keys, or
// raw callback error strings.
type Diagnostics struct {
	Strategy               Strategy       `json:"strategy"`
	Outcome                Outcome        `json:"outcome"`
	Reason                 string         `json:"reason"`
	BudgetStatus           BudgetStatus   `json:"budget_status"`
	ScoringSource          ScoringSource  `json:"scoring_source"`
	FallbackReason         string         `json:"fallback_reason,omitempty"`
	TokenCounterSource     string         `json:"token_counter_source"`
	RecoveryStatus         RecoveryStatus `json:"recovery_status"`
	ExaminedMessages       int            `json:"examined_messages"`
	ExaminedTurns          int            `json:"examined_turns"`
	ProtectedMessages      int            `json:"protected_messages"`
	ProtectedTurns         int            `json:"protected_turns"`
	EligibleTurns          int            `json:"eligible_turns"`
	ScoredTurns            int            `json:"scored_turns"`
	CandidatesTruncated    bool           `json:"candidates_truncated,omitempty"`
	ContinuityBoostedTurns int            `json:"continuity_boosted_turns"`
	MinRecentRequested     int            `json:"min_recent_requested"`
	MinRecentRetained      int            `json:"min_recent_retained"`
	RetainedMessages       int            `json:"retained_messages"`
	RetainedTurns          int            `json:"retained_turns"`
	RemovedMessages        int            `json:"removed_messages"`
	RemovedTurns           int            `json:"removed_turns"`
	EffectiveInputBudget   int            `json:"effective_input_budget"`
	ReservedOutputTokens   int            `json:"reserved_output_tokens"`
	FixedOverheadTokens    int            `json:"fixed_overhead_tokens"`
	AvailableHistoryBudget int            `json:"available_history_budget"`
	RetainedHistoryTokens  int            `json:"retained_history_tokens"`
	RemovedHistoryTokens   int            `json:"removed_history_tokens"`
	Decisions              []TurnDecision `json:"decisions,omitempty"`
	DecisionsTruncated     bool           `json:"decisions_truncated,omitempty"`
}
