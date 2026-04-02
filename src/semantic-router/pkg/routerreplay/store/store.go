package store

import (
	"context"
	"io"
	"time"
)

// Signal represents various routing signals captured during a request.
type Signal struct {
	Keyword      []string `json:"keyword,omitempty"`
	Embedding    []string `json:"embedding,omitempty"`
	Domain       []string `json:"domain,omitempty"`
	FactCheck    []string `json:"fact_check,omitempty"`
	UserFeedback []string `json:"user_feedback,omitempty"`
	Reask        []string `json:"reask,omitempty"`
	Preference   []string `json:"preference,omitempty"`
	Language     []string `json:"language,omitempty"`
	Context      []string `json:"context,omitempty"`
	Structure    []string `json:"structure,omitempty"`
	Complexity   []string `json:"complexity,omitempty"`
	Modality     []string `json:"modality,omitempty"`
	Authz        []string `json:"authz,omitempty"`
	Jailbreak    []string `json:"jailbreak,omitempty"`
	PII          []string `json:"pii,omitempty"`
	KB           []string `json:"kb,omitempty"`
}

// UsageCost captures token usage and pricing-derived cost details for a record.
type UsageCost struct {
	PromptTokens     *int     `json:"prompt_tokens,omitempty"`
	CompletionTokens *int     `json:"completion_tokens,omitempty"`
	TotalTokens      *int     `json:"total_tokens,omitempty"`
	ActualCost       *float64 `json:"actual_cost,omitempty"`
	BaselineCost     *float64 `json:"baseline_cost,omitempty"`
	CostSavings      *float64 `json:"cost_savings,omitempty"`
	Currency         *string  `json:"currency,omitempty"`
	BaselineModel    *string  `json:"baseline_model,omitempty"`
}

// ToolTrace captures the request-local assistant/tool exchange timeline.
type ToolTrace struct {
	Flow      string          `json:"flow,omitempty"`
	Stage     string          `json:"stage,omitempty"`
	ToolNames []string        `json:"tool_names,omitempty"`
	Steps     []ToolTraceStep `json:"steps,omitempty"`
}

// ToolTraceStep represents a single step in a request-local tool-calling flow.
type ToolTraceStep struct {
	Type       string `json:"type"`
	Source     string `json:"source,omitempty"`
	Role       string `json:"role,omitempty"`
	Text       string `json:"text,omitempty"`
	ToolName   string `json:"tool_name,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`
	Arguments  string `json:"arguments,omitempty"`
}

// Record represents a routing decision record with metadata and captured payloads.
type Record struct {
	ID                    string             `json:"id"`
	Timestamp             time.Time          `json:"timestamp"`
	RequestID             string             `json:"request_id,omitempty"`
	Decision              string             `json:"decision,omitempty"`
	DecisionTier          int                `json:"decision_tier"`
	DecisionPriority      int                `json:"decision_priority"`
	Category              string             `json:"category,omitempty"`
	OriginalModel         string             `json:"original_model,omitempty"`
	SelectedModel         string             `json:"selected_model,omitempty"`
	ReasoningMode         string             `json:"reasoning_mode,omitempty"`
	ConfidenceScore       float64            `json:"confidence_score,omitempty"`
	SelectionMethod       string             `json:"selection_method,omitempty"`
	Signals               Signal             `json:"signals"`
	Projections           []string           `json:"projections,omitempty"`
	ProjectionScores      map[string]float64 `json:"projection_scores,omitempty"`
	SignalConfidences     map[string]float64 `json:"signal_confidences,omitempty"`
	SignalValues          map[string]float64 `json:"signal_values,omitempty"`
	ToolTrace             *ToolTrace         `json:"tool_trace,omitempty"`
	RequestBody           string             `json:"request_body,omitempty"`
	ResponseBody          string             `json:"response_body,omitempty"`
	ResponseStatus        int                `json:"response_status,omitempty"`
	FromCache             bool               `json:"from_cache,omitempty"`
	Streaming             bool               `json:"streaming,omitempty"`
	RequestBodyTruncated  bool               `json:"request_body_truncated,omitempty"`
	ResponseBodyTruncated bool               `json:"response_body_truncated,omitempty"`

	// Guardrails
	GuardrailsEnabled bool `json:"guardrails_enabled,omitempty"`
	JailbreakEnabled  bool `json:"jailbreak_enabled,omitempty"`
	PIIEnabled        bool `json:"pii_enabled,omitempty"`

	// Jailbreak Detection Results (request-level)
	JailbreakDetected   bool    `json:"jailbreak_detected,omitempty"`
	JailbreakType       string  `json:"jailbreak_type,omitempty"`
	JailbreakConfidence float32 `json:"jailbreak_confidence,omitempty"`

	// Response Jailbreak Detection Results
	ResponseJailbreakDetected   bool    `json:"response_jailbreak_detected,omitempty"`
	ResponseJailbreakType       string  `json:"response_jailbreak_type,omitempty"`
	ResponseJailbreakConfidence float32 `json:"response_jailbreak_confidence,omitempty"`

	// PII Detection Results
	PIIDetected bool     `json:"pii_detected,omitempty"`
	PIIEntities []string `json:"pii_entities,omitempty"`
	PIIBlocked  bool     `json:"pii_blocked,omitempty"`

	// RAG (Retrieval-Augmented Generation)
	RAGEnabled         bool    `json:"rag_enabled,omitempty"`
	RAGBackend         string  `json:"rag_backend,omitempty"`
	RAGContextLength   int     `json:"rag_context_length,omitempty"`
	RAGSimilarityScore float32 `json:"rag_similarity_score,omitempty"`

	// Hallucination Detection
	HallucinationEnabled    bool     `json:"hallucination_enabled,omitempty"`
	HallucinationDetected   bool     `json:"hallucination_detected,omitempty"`
	HallucinationConfidence float32  `json:"hallucination_confidence,omitempty"`
	HallucinationSpans      []string `json:"hallucination_spans,omitempty"`

	// Usage & Cost
	PromptTokens     *int     `json:"prompt_tokens,omitempty"`
	CompletionTokens *int     `json:"completion_tokens,omitempty"`
	TotalTokens      *int     `json:"total_tokens,omitempty"`
	ActualCost       *float64 `json:"actual_cost,omitempty"`
	BaselineCost     *float64 `json:"baseline_cost,omitempty"`
	CostSavings      *float64 `json:"cost_savings,omitempty"`
	Currency         *string  `json:"currency,omitempty"`
	BaselineModel    *string  `json:"baseline_model,omitempty"`
}

// Writer mutates router replay records.
type Writer interface {
	// Add inserts a new record. Returns the record ID.
	Add(ctx context.Context, record Record) (string, error)

	// UpdateStatus updates the response status and flags for an existing record.
	UpdateStatus(ctx context.Context, id string, status int, fromCache bool, streaming bool) error

	// AttachRequest updates the request body for an existing record.
	AttachRequest(ctx context.Context, id string, body string, truncated bool) error

	// AttachResponse updates the response body for an existing record.
	AttachResponse(ctx context.Context, id string, body string, truncated bool) error
}

// Reader retrieves router replay records.
type Reader interface {
	// Get retrieves a record by ID. Returns false if not found.
	Get(ctx context.Context, id string) (Record, bool, error)

	// List retrieves all records, ordered by timestamp descending.
	List(ctx context.Context) ([]Record, error)
}

// Enricher updates derived signal analysis fields after the initial record write.
type Enricher interface {
	// UpdateHallucinationStatus updates hallucination detection results for an existing record.
	UpdateHallucinationStatus(ctx context.Context, id string, detected bool, confidence float32, spans []string) error

	// UpdateUsageCost updates token usage and pricing-derived cost fields for an existing record.
	UpdateUsageCost(ctx context.Context, id string, usage UsageCost) error

	// UpdateToolTrace updates the request-local tool-calling timeline for an existing record.
	UpdateToolTrace(ctx context.Context, id string, trace ToolTrace) error
}

// Storage is the interface that all storage backends must implement.
type Storage interface {
	Writer
	Reader
	Enricher
	io.Closer
}

func cloneStringSlice(values []string) []string {
	if values == nil {
		return nil
	}
	return append([]string(nil), values...)
}

func cloneFloat64Map(values map[string]float64) map[string]float64 {
	if values == nil {
		return nil
	}
	cloned := make(map[string]float64, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}

func cloneToolTraceSteps(steps []ToolTraceStep) []ToolTraceStep {
	if steps == nil {
		return nil
	}
	return append([]ToolTraceStep(nil), steps...)
}

func cloneToolTrace(trace *ToolTrace) *ToolTrace {
	if trace == nil {
		return nil
	}
	cloned := *trace
	cloned.ToolNames = cloneStringSlice(trace.ToolNames)
	cloned.Steps = cloneToolTraceSteps(trace.Steps)
	return &cloned
}

func cloneSignal(signal Signal) Signal {
	return Signal{
		Keyword:      cloneStringSlice(signal.Keyword),
		Embedding:    cloneStringSlice(signal.Embedding),
		Domain:       cloneStringSlice(signal.Domain),
		FactCheck:    cloneStringSlice(signal.FactCheck),
		UserFeedback: cloneStringSlice(signal.UserFeedback),
		Reask:        cloneStringSlice(signal.Reask),
		Preference:   cloneStringSlice(signal.Preference),
		Language:     cloneStringSlice(signal.Language),
		Context:      cloneStringSlice(signal.Context),
		Structure:    cloneStringSlice(signal.Structure),
		Complexity:   cloneStringSlice(signal.Complexity),
		Modality:     cloneStringSlice(signal.Modality),
		Authz:        cloneStringSlice(signal.Authz),
		Jailbreak:    cloneStringSlice(signal.Jailbreak),
		PII:          cloneStringSlice(signal.PII),
		KB:           cloneStringSlice(signal.KB),
	}
}

func cloneRecord(record Record) Record {
	cloned := record
	cloned.Signals = cloneSignal(record.Signals)
	cloned.Projections = cloneStringSlice(record.Projections)
	cloned.ProjectionScores = cloneFloat64Map(record.ProjectionScores)
	cloned.SignalConfidences = cloneFloat64Map(record.SignalConfidences)
	cloned.SignalValues = cloneFloat64Map(record.SignalValues)
	cloned.ToolTrace = cloneToolTrace(record.ToolTrace)
	cloned.PIIEntities = cloneStringSlice(record.PIIEntities)
	cloned.HallucinationSpans = cloneStringSlice(record.HallucinationSpans)
	cloned.PromptTokens = cloneIntPtr(record.PromptTokens)
	cloned.CompletionTokens = cloneIntPtr(record.CompletionTokens)
	cloned.TotalTokens = cloneIntPtr(record.TotalTokens)
	cloned.ActualCost = cloneFloat64Ptr(record.ActualCost)
	cloned.BaselineCost = cloneFloat64Ptr(record.BaselineCost)
	cloned.CostSavings = cloneFloat64Ptr(record.CostSavings)
	cloned.Currency = cloneStringPtr(record.Currency)
	cloned.BaselineModel = cloneStringPtr(record.BaselineModel)
	return cloned
}

func cloneIntPtr(value *int) *int {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func cloneFloat64Ptr(value *float64) *float64 {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func cloneStringPtr(value *string) *string {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

// Config holds common configuration options for all storage backends.
type Config struct {
	Backend      string // "memory", "redis", "postgres", "milvus"
	TTLSeconds   int    // Time-to-live for records (0 = no expiration)
	AsyncWrites  bool   // Enable asynchronous writes
	MaxBodyBytes int    // Maximum bytes to store for request/response bodies

	// Backend-specific configurations
	Redis    *RedisConfig
	Postgres *PostgresConfig
	Milvus   *MilvusConfig
}

// RedisConfig holds Redis-specific configuration.
type RedisConfig struct {
	Address  string `json:"address" yaml:"address"`
	DB       int    `json:"db" yaml:"db"`
	Password string `json:"password" yaml:"password"`
	// Optional TLS configuration
	UseTLS        bool   `json:"use_tls,omitempty" yaml:"use_tls,omitempty"`
	TLSSkipVerify bool   `json:"tls_skip_verify,omitempty" yaml:"tls_skip_verify,omitempty"`
	MaxRetries    int    `json:"max_retries,omitempty" yaml:"max_retries,omitempty"`
	PoolSize      int    `json:"pool_size,omitempty" yaml:"pool_size,omitempty"`
	KeyPrefix     string `json:"key_prefix,omitempty" yaml:"key_prefix,omitempty"`
}

// PostgresConfig holds PostgreSQL-specific configuration.
type PostgresConfig struct {
	Host     string `json:"host" yaml:"host"`
	Port     int    `json:"port" yaml:"port"`
	Database string `json:"database" yaml:"database"`
	User     string `json:"user" yaml:"user"`
	Password string `json:"password" yaml:"password"`
	SSLMode  string `json:"ssl_mode,omitempty" yaml:"ssl_mode,omitempty"` // disable, require, verify-ca, verify-full
	// Connection pool settings
	MaxOpenConns    int    `json:"max_open_conns,omitempty" yaml:"max_open_conns,omitempty"`
	MaxIdleConns    int    `json:"max_idle_conns,omitempty" yaml:"max_idle_conns,omitempty"`
	ConnMaxLifetime int    `json:"conn_max_lifetime,omitempty" yaml:"conn_max_lifetime,omitempty"` // seconds
	TableName       string `json:"table_name,omitempty" yaml:"table_name,omitempty"`
}

// MilvusConfig holds Milvus-specific configuration.
type MilvusConfig struct {
	Address        string `json:"address" yaml:"address"`
	Username       string `json:"username,omitempty" yaml:"username,omitempty"`
	Password       string `json:"password,omitempty" yaml:"password,omitempty"`
	CollectionName string `json:"collection_name,omitempty" yaml:"collection_name,omitempty"`
	// Milvus specific settings
	ConsistencyLevel string `json:"consistency_level,omitempty" yaml:"consistency_level,omitempty"` // Strong, Session, Bounded, Eventually
	ShardNum         int    `json:"shard_num,omitempty" yaml:"shard_num,omitempty"`
}
