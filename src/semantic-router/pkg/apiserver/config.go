//go:build !windows && cgo

package apiserver

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// ClassificationAPIServer holds the server state and dependencies
type ClassificationAPIServer struct {
	classificationSvc     *services.ClassificationService
	config                *config.RouterConfig
	enableSystemPromptAPI bool
}

// ModelsInfoResponse represents the response for models info endpoint
type ModelsInfoResponse struct {
	Models []ModelInfo `json:"models"`
	System SystemInfo  `json:"system"`
}

// ModelInfo represents information about a loaded model
type ModelInfo struct {
	Name        string            `json:"name"`
	Type        string            `json:"type"`
	Loaded      bool              `json:"loaded"`
	ModelPath   string            `json:"model_path,omitempty"`
	Categories  []string          `json:"categories,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
	LoadTime    string            `json:"load_time,omitempty"`
	MemoryUsage string            `json:"memory_usage,omitempty"`
}

// SystemInfo represents system information
type SystemInfo struct {
	GoVersion    string `json:"go_version"`
	Architecture string `json:"architecture"`
	OS           string `json:"os"`
	MemoryUsage  string `json:"memory_usage"`
	GPUAvailable bool   `json:"gpu_available"`
}

// OpenAIModel represents a single model in the OpenAI /v1/models response
type OpenAIModel struct {
	ID          string `json:"id"`
	Object      string `json:"object"`
	Created     int64  `json:"created"`
	OwnedBy     string `json:"owned_by"`
	Description string `json:"description,omitempty"` // Optional description for Chat UI
	LogoURL     string `json:"logo_url,omitempty"`    // Optional logo URL for Chat UI
	// Keeping the structure minimal; additional fields like permissions can be added later
}

// OpenAIModelList is the container for the models list response
type OpenAIModelList struct {
	Object string        `json:"object"`
	Data   []OpenAIModel `json:"data"`
}

// BatchClassificationRequest represents a batch classification request
type BatchClassificationRequest struct {
	Texts    []string               `json:"texts"`
	TaskType string                 `json:"task_type,omitempty"` // "intent", "pii", "security", or "all"
	Options  *ClassificationOptions `json:"options,omitempty"`
}

// BatchClassificationResult represents a single classification result with optional probabilities
type BatchClassificationResult struct {
	Category         string             `json:"category"`
	Confidence       float64            `json:"confidence"`
	ProcessingTimeMs int64              `json:"processing_time_ms"`
	Probabilities    map[string]float64 `json:"probabilities,omitempty"`
}

// BatchClassificationResponse represents the response from batch classification
type BatchClassificationResponse struct {
	Results          []BatchClassificationResult      `json:"results"`
	TotalCount       int                              `json:"total_count"`
	ProcessingTimeMs int64                            `json:"processing_time_ms"`
	Statistics       CategoryClassificationStatistics `json:"statistics"`
}

// CategoryClassificationStatistics provides batch processing statistics
type CategoryClassificationStatistics struct {
	CategoryDistribution map[string]int `json:"category_distribution"`
	AvgConfidence        float64        `json:"avg_confidence"`
	LowConfidenceCount   int            `json:"low_confidence_count"`
}

// ClassificationOptions mirrors services.IntentOptions for API layer
type ClassificationOptions struct {
	ReturnProbabilities bool    `json:"return_probabilities,omitempty"`
	ConfidenceThreshold float64 `json:"confidence_threshold,omitempty"`
	IncludeExplanation  bool    `json:"include_explanation,omitempty"`
}

// EmbeddingRequest represents a request for embedding generation
type EmbeddingRequest struct {
	Texts           []string `json:"texts"`
	Model           string   `json:"model,omitempty"`            // "auto" (default), "qwen3", "gemma", "mmbert"
	Dimension       int      `json:"dimension,omitempty"`        // Target dimension: 768 (default), 512, 256, 128, 64
	TargetLayer     int      `json:"target_layer,omitempty"`     // Target layer for early exit (mmbert only): 3, 6, 11, 22 (0=full)
	QualityPriority float32  `json:"quality_priority,omitempty"` // 0.0-1.0, default 0.5 (only used when model="auto")
	LatencyPriority float32  `json:"latency_priority,omitempty"` // 0.0-1.0, default 0.5 (only used when model="auto")
	SequenceLength  int      `json:"sequence_length,omitempty"`  // Optional, auto-detected if not provided
}

// EmbeddingResult represents a single embedding result
type EmbeddingResult struct {
	Text             string    `json:"text"`
	Embedding        []float32 `json:"embedding"`
	Dimension        int       `json:"dimension"`
	ModelUsed        string    `json:"model_used"`
	ProcessingTimeMs int64     `json:"processing_time_ms"`
}

// EmbeddingResponse represents the response from embedding generation
type EmbeddingResponse struct {
	Embeddings            []EmbeddingResult `json:"embeddings"`
	TotalCount            int               `json:"total_count"`
	TotalProcessingTimeMs int64             `json:"total_processing_time_ms"`
	AvgProcessingTimeMs   float64           `json:"avg_processing_time_ms"`
}

// SimilarityRequest represents a request to calculate similarity between two texts
type SimilarityRequest struct {
	Text1           string  `json:"text1"`
	Text2           string  `json:"text2"`
	Model           string  `json:"model,omitempty"`            // "auto" (default), "qwen3", "gemma", "mmbert"
	Dimension       int     `json:"dimension,omitempty"`        // Target dimension: 768 (default), 512, 256, 128, 64
	TargetLayer     int     `json:"target_layer,omitempty"`     // Target layer for early exit (mmbert only): 3, 6, 11, 22 (0=full)
	QualityPriority float32 `json:"quality_priority,omitempty"` // 0.0-1.0, only for "auto" model
	LatencyPriority float32 `json:"latency_priority,omitempty"` // 0.0-1.0, only for "auto" model
}

// SimilarityResponse represents the response of a similarity calculation
type SimilarityResponse struct {
	ModelUsed        string  `json:"model_used"`         // "qwen3", "gemma", or "unknown"
	Similarity       float32 `json:"similarity"`         // Cosine similarity score (-1.0 to 1.0)
	ProcessingTimeMs float32 `json:"processing_time_ms"` // Processing time in milliseconds
}

// BatchSimilarityRequest represents a request to find top-k similar candidates for a query
type BatchSimilarityRequest struct {
	Query           string   `json:"query"`                      // Query text
	Candidates      []string `json:"candidates"`                 // Array of candidate texts
	TopK            int      `json:"top_k,omitempty"`            // Max number of matches to return (0 = return all)
	Model           string   `json:"model,omitempty"`            // "auto" (default), "qwen3", "gemma", "mmbert"
	Dimension       int      `json:"dimension,omitempty"`        // Target dimension: 768 (default), 512, 256, 128, 64
	TargetLayer     int      `json:"target_layer,omitempty"`     // Target layer for early exit (mmbert only): 3, 6, 11, 22 (0=full)
	QualityPriority float32  `json:"quality_priority,omitempty"` // 0.0-1.0, only for "auto" model
	LatencyPriority float32  `json:"latency_priority,omitempty"` // 0.0-1.0, only for "auto" model
}

// BatchSimilarityMatch represents a single match in batch similarity matching
type BatchSimilarityMatch struct {
	Index      int     `json:"index"`      // Index of the candidate in the input array
	Similarity float32 `json:"similarity"` // Cosine similarity score
	Text       string  `json:"text"`       // The matched candidate text
}

// BatchSimilarityResponse represents the response of batch similarity matching
type BatchSimilarityResponse struct {
	Matches          []BatchSimilarityMatch `json:"matches"`            // Top-k matches, sorted by similarity (descending)
	TotalCandidates  int                    `json:"total_candidates"`   // Total number of candidates processed
	ModelUsed        string                 `json:"model_used"`         // "qwen3", "gemma", or "unknown"
	ProcessingTimeMs float32                `json:"processing_time_ms"` // Processing time in milliseconds
}

// EndpointInfo represents information about an API endpoint
type EndpointInfo struct {
	Path        string `json:"path"`
	Method      string `json:"method"`
	Description string `json:"description"`
}

// TaskTypeInfo represents information about a task type
type TaskTypeInfo struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// EndpointMetadata stores metadata about an endpoint for API documentation
type EndpointMetadata struct {
	Path        string
	Method      string
	Description string
}
