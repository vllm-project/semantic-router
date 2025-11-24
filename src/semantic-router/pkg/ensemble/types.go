package ensemble

import (
	"context"
	"time"
)

// Strategy defines the aggregation strategy for combining model outputs
type Strategy string

const (
	// StrategyVoting uses majority consensus for classification
	StrategyVoting Strategy = "voting"

	// StrategyWeighted uses confidence-weighted combination
	StrategyWeighted Strategy = "weighted"

	// StrategyFirstSuccess returns first valid response (latency optimization)
	StrategyFirstSuccess Strategy = "first_success"

	// StrategyScoreAveraging averages numerical outputs or probabilities
	StrategyScoreAveraging Strategy = "score_averaging"

	// StrategyReranking uses a separate model to rank and select best output
	StrategyReranking Strategy = "reranking"
)

// Config represents configuration for ensemble orchestration
type Config struct {
	// Enabled controls whether ensemble mode is available
	Enabled bool `yaml:"enabled"`

	// DefaultStrategy is the default aggregation strategy
	DefaultStrategy Strategy `yaml:"default_strategy,omitempty"`

	// DefaultMinResponses is the default minimum number of responses required
	DefaultMinResponses int `yaml:"default_min_responses,omitempty"`

	// TimeoutSeconds is the maximum time to wait for model responses
	TimeoutSeconds int `yaml:"timeout_seconds,omitempty"`

	// MaxConcurrentRequests limits parallel model queries
	MaxConcurrentRequests int `yaml:"max_concurrent_requests,omitempty"`
}

// Request represents an ensemble orchestration request
type Request struct {
	// Models is the list of model names to query
	Models []string

	// Strategy is the aggregation strategy to use
	Strategy Strategy

	// MinResponses is the minimum number of successful responses required
	MinResponses int

	// OriginalRequest is the original OpenAI API request body
	OriginalRequest []byte

	// Context for cancellation and timeout
	Context context.Context
}

// Response represents the result of ensemble orchestration
type Response struct {
	// FinalResponse is the aggregated response body
	FinalResponse []byte

	// ModelsQueried is the number of models that were queried
	ModelsQueried int

	// ResponsesReceived is the number of successful responses
	ResponsesReceived int

	// Strategy is the strategy that was used
	Strategy Strategy

	// Metadata contains additional information about the ensemble process
	Metadata Metadata

	// Error is set if the ensemble process failed
	Error error
}

// Metadata contains information about the ensemble process
type Metadata struct {
	// TotalLatencyMs is the total time taken for the ensemble process
	TotalLatencyMs int64

	// ModelLatenciesMs contains latency for each model response
	ModelLatenciesMs map[string]int64

	// ConfidenceScores contains confidence scores from each model
	ConfidenceScores map[string]float64

	// SelectedModel is the model whose response was selected (if applicable)
	SelectedModel string

	// AggregationDetails contains strategy-specific details
	AggregationDetails map[string]interface{}
}

// ModelResponse represents a response from a single model
type ModelResponse struct {
	// ModelName is the name of the model
	ModelName string

	// Response is the response body from the model
	Response []byte

	// Latency is the time taken for this model to respond
	Latency time.Duration

	// Error is set if the model request failed
	Error error

	// Confidence is the confidence score (if available)
	Confidence float64
}
