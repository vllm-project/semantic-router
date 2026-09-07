package config

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

const (
	ContextCompressionModeAuto   = "auto"
	ContextCompressionModeAlways = "always"

	ContextCompressionTargetPreserve    = "preserve"
	ContextCompressionTargetExtractive  = "extractive"
	ContextCompressionTargetRecoverable = "recoverable"

	ContextCompressionScoringBM25      = "bm25"
	ContextCompressionScoringEmbedding = "embedding"
	ContextCompressionScoringHybrid    = "hybrid"

	ContextCompressionFailureOpen   = "fail_open"
	ContextCompressionFailureClosed = "fail_closed"
)

// CompressionTokenLimit is either a non-negative token count or the literal
// "auto". The zero value is auto.
type CompressionTokenLimit struct {
	Auto  bool
	Value int
}

func AutoCompressionTokenLimit() *CompressionTokenLimit {
	return &CompressionTokenLimit{Auto: true}
}

func (limit CompressionTokenLimit) MarshalJSON() ([]byte, error) {
	if limit.Auto {
		return json.Marshal("auto")
	}
	return json.Marshal(limit.Value)
}

func (limit *CompressionTokenLimit) UnmarshalJSON(data []byte) error {
	if limit == nil {
		return fmt.Errorf("compression token limit is nil")
	}
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		return limit.setText(text)
	}
	var value int
	if err := json.Unmarshal(data, &value); err != nil {
		return fmt.Errorf("compression token limit must be a non-negative integer or \"auto\"")
	}
	return limit.setValue(value)
}

func (limit CompressionTokenLimit) MarshalYAML() (interface{}, error) {
	if limit.Auto {
		return "auto", nil
	}
	return limit.Value, nil
}

func (limit *CompressionTokenLimit) UnmarshalYAML(
	unmarshal func(interface{}) error,
) error {
	var raw interface{}
	if err := unmarshal(&raw); err != nil {
		return err
	}
	switch typed := raw.(type) {
	case string:
		return limit.setText(typed)
	case int:
		return limit.setValue(typed)
	case int64:
		return limit.setValue(int(typed))
	case uint64:
		if typed > uint64(^uint(0)>>1) {
			return fmt.Errorf("compression token limit is too large")
		}
		return limit.setValue(int(typed))
	default:
		return fmt.Errorf("compression token limit must be a non-negative integer or \"auto\"")
	}
}

func (limit *CompressionTokenLimit) setText(value string) error {
	if strings.EqualFold(strings.TrimSpace(value), "auto") {
		limit.Auto = true
		limit.Value = 0
		return nil
	}
	parsed, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil {
		return fmt.Errorf("compression token limit must be a non-negative integer or \"auto\"")
	}
	return limit.setValue(parsed)
}

func (limit *CompressionTokenLimit) setValue(value int) error {
	if value < 0 {
		return fmt.Errorf("compression token limit cannot be negative")
	}
	limit.Auto = false
	limit.Value = value
	return nil
}

func (limit *CompressionTokenLimit) Resolve(fallback int) int {
	if limit == nil || limit.Auto {
		return fallback
	}
	return limit.Value
}

type ContextCompressionBudgetConfig struct {
	TriggerTokens       *CompressionTokenLimit `json:"trigger_tokens,omitempty" yaml:"trigger_tokens,omitempty"`
	TargetTokens        *CompressionTokenLimit `json:"target_tokens,omitempty" yaml:"target_tokens,omitempty"`
	ReserveOutputTokens *CompressionTokenLimit `json:"reserve_output_tokens,omitempty" yaml:"reserve_output_tokens,omitempty"`
}

type ContextCompressionTargetConfig struct {
	Mode         string `json:"mode,omitempty" yaml:"mode,omitempty"`
	MinTokens    int    `json:"min_tokens,omitempty" yaml:"min_tokens,omitempty"`
	TargetTokens int    `json:"target_tokens,omitempty" yaml:"target_tokens,omitempty"`
}

type ContextCompressionTargetsConfig struct {
	ToolOutputs ContextCompressionTargetConfig `json:"tool_outputs,omitempty" yaml:"tool_outputs,omitempty"`
	History     ContextCompressionTargetConfig `json:"history,omitempty" yaml:"history,omitempty"`
	RAG         ContextCompressionTargetConfig `json:"rag,omitempty" yaml:"rag,omitempty"`
	Memory      ContextCompressionTargetConfig `json:"memory,omitempty" yaml:"memory,omitempty"`
}

type ContextCompressionScoringConfig struct {
	Method            string `json:"method,omitempty" yaml:"method,omitempty"`
	EmbeddingModelRef string `json:"embedding_model_ref,omitempty" yaml:"embedding_model_ref,omitempty"`
}

type ContextCompressionRecoveryConfig struct {
	Enabled            bool   `json:"enabled,omitempty" yaml:"enabled,omitempty"`
	Store              string `json:"store,omitempty" yaml:"store,omitempty"`
	TTLSeconds         int    `json:"ttl_seconds,omitempty" yaml:"ttl_seconds,omitempty"`
	MaxBytesPerRequest int    `json:"max_bytes_per_request,omitempty" yaml:"max_bytes_per_request,omitempty"`
	MaxTotalBytes      int    `json:"max_total_bytes,omitempty" yaml:"max_total_bytes,omitempty"`
	MaxRetrievals      int    `json:"max_retrievals,omitempty" yaml:"max_retrievals,omitempty"`
}

type ContextCompressionRequestControlsConfig struct {
	Enabled         bool     `json:"enabled,omitempty" yaml:"enabled,omitempty"`
	Header          string   `json:"header,omitempty" yaml:"header,omitempty"`
	Allowed         []string `json:"allowed,omitempty" yaml:"allowed,omitempty"`
	MaxTargetTokens int      `json:"max_target_tokens,omitempty" yaml:"max_target_tokens,omitempty"`
}
