package contextcompression

import (
	"context"
	"time"
)

type Mode string

const (
	ModeAuto   Mode = "auto"
	ModeAlways Mode = "always"
)

type TargetMode string

const (
	TargetPreserve    TargetMode = "preserve"
	TargetExtractive  TargetMode = "extractive"
	TargetRecoverable TargetMode = "recoverable"
)

type TargetKind string

const (
	TargetToolOutput TargetKind = "tool_output"
	TargetHistory    TargetKind = "history"
	TargetRAG        TargetKind = "rag"
	TargetMemory     TargetKind = "memory"
)

type FailureMode string

const (
	FailureOpen   FailureMode = "fail_open"
	FailureClosed FailureMode = "fail_closed"
)

type TokenCounter interface {
	CountText(model string, text string) (tokens int, source string)
	CountRequest(model string, request *RequestIR) (tokens int, source string)
}

type RelevanceScorer interface {
	Score(ctx context.Context, modelRef string, query string, texts []string) ([]float64, error)
}

type RecoveryStore interface {
	Put(ctx context.Context, entry RecoveryEntry, ttl time.Duration) error
	Get(ctx context.Context, scope string, key string) (RecoveryEntry, error)
	InvalidateScope(ctx context.Context, scope string) (int64, error)
	Health(ctx context.Context) error
	Close() error
}

type RecoveryEntry struct {
	Key       string    `json:"key"`
	Scope     string    `json:"scope"`
	Content   string    `json:"content"`
	CreatedAt time.Time `json:"created_at"`
	ExpiresAt time.Time `json:"expires_at"`
}

type ModelContextCapabilities struct {
	ContextWindow   int
	MaxOutputTokens int
	RequestedOutput int
}

type Budget struct {
	TriggerTokens       int
	TargetTokens        int
	ReserveOutputTokens int
	TriggerAuto         bool
	TargetAuto          bool
	ReserveAuto         bool
}

type TargetPolicy struct {
	Mode         TargetMode
	MinTokens    int
	TargetTokens int
}

type Targets struct {
	ToolOutputs TargetPolicy
	History     TargetPolicy
	RAG         TargetPolicy
	Memory      TargetPolicy
}

type Scoring struct {
	Method            string
	EmbeddingModelRef string
}

type RecoveryPolicy struct {
	Enabled            bool
	TTL                time.Duration
	MaxBytesPerRequest int
	MaxTotalBytes      int
	MaxRetrievals      int
}

type Policy struct {
	Mode        Mode
	Budget      Budget
	Targets     Targets
	Scoring     Scoring
	Recovery    RecoveryPolicy
	FailureMode FailureMode
}

type Provenance struct {
	RAGToolCallIDs       map[string]struct{}
	MemoryMessageIndexes map[int]struct{}
}

type Request struct {
	Model        string
	Scope        string
	Request      *RequestIR
	Policy       Policy
	Capabilities ModelContextCapabilities
	TokenCounter TokenCounter
	Scorer       RelevanceScorer
	Recovery     RecoveryStore
	Provenance   Provenance
}

type SkipReason string

const (
	SkipNone                SkipReason = ""
	SkipDisabled            SkipReason = "disabled"
	SkipBelowTrigger        SkipReason = "below_trigger"
	SkipNoTargets           SkipReason = "no_targets"
	SkipBudgetUnavailable   SkipReason = "budget_unavailable"
	SkipBudgetNotReduced    SkipReason = "budget_not_reduced"
	SkipRecoveryUnavailable SkipReason = "recovery_unavailable"
	SkipScoringUnavailable  SkipReason = "scoring_unavailable"
)

type TargetPlan struct {
	MessageIndex   int
	BlockIndex     int
	Kind           TargetKind
	Mode           TargetMode
	OriginalTokens int
	TargetTokens   int
	Query          string
	Score          float64
}

type Plan struct {
	Model                string
	OriginalTokens       int
	TargetRequestTokens  int
	AvailableInputTokens int
	ReservedOutputTokens int
	TokenCounterSource   string
	TriggerReason        string
	SkipReason           SkipReason
	Quality              string
	FallbackReason       string
	Targets              []TargetPlan
}

type ServiceResult struct {
	Request            *RequestIR
	Plan               Plan
	Applied            bool
	MessagesCompressed int
	BlocksCompressed   int
	TokensBefore       int
	TokensAfter        int
	OmittedChunks      int
	JSONBlocks         int
	RecoveryKeys       []string
	Failure            error
}

type Capabilities struct {
	Strategies             []string `json:"strategies"`
	ScoringMethods         []string `json:"scoring_methods"`
	Targets                []string `json:"targets"`
	Recovery               bool     `json:"recovery"`
	JSONStructurePreserved bool     `json:"json_structure_preserved"`
	MultimodalPreserved    bool     `json:"multimodal_preserved"`
	ProtocolMetadataKept   bool     `json:"protocol_metadata_kept"`
}

type Stats struct {
	Requests              int64   `json:"requests"`
	Applied               int64   `json:"applied"`
	Skipped               int64   `json:"skipped"`
	Failures              int64   `json:"failures"`
	RecoveryWrites        int64   `json:"recovery_writes"`
	RecoveryReads         int64   `json:"recovery_reads"`
	RecoveryFailures      int64   `json:"recovery_failures"`
	TokensBefore          int64   `json:"tokens_before"`
	TokensAfter           int64   `json:"tokens_after"`
	TokensSaved           int64   `json:"tokens_saved"`
	EstimatedCostSavedUSD float64 `json:"estimated_cost_saved_usd"`
}
