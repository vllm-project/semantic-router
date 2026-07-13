//go:build !windows && cgo

package apiserver

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelinventory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

// ClassificationAPIServer holds the server state and dependencies
type ClassificationAPIServer struct {
	classificationSvc     classificationService
	config                *config.RouterConfig
	runtimeConfig         *liveRuntimeConfig
	runtimeRegistry       *routerruntime.Registry
	configPath            string // path to the router config file (for read/update/rollback)
	memoryStore           memory.Store
	knowledgeBaseMapCache *knowledgeBaseMapCache
	startupStateLoader    func() *startupstatus.State
	embeddingAdmission    *embedding.ProcessAdmission
}

type (
	ModelsInfoResponse = modelinventory.ModelsInfoResponse
	ModelsInfoSummary  = modelinventory.ModelsInfoSummary
	ModelInfo          = modelinventory.ModelInfo
	ModelRegistryInfo  = config.ModelRegistryInfo
	SystemInfo         = modelinventory.SystemInfo
)

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
	Images          []string `json:"images,omitempty"`           // Inline base64 image data URIs (data:image/...;base64,...); encoded via the multi-modal model
	Model           string   `json:"model,omitempty"`            // Backend-aware text model; mixed text+image requests require "auto"
	Dimension       int      `json:"dimension,omitempty"`        // Model-compatible target dimension; defaults: text 768, image 384, mixed 256
	TargetLayer     int      `json:"target_layer,omitempty"`     // Text-only mmBERT early-exit layer from the loaded model manifest (0=full)
	QualityPriority float32  `json:"quality_priority,omitempty"` // 0.0-1.0, text-only Candle auto-routing; default 0.5
	LatencyPriority float32  `json:"latency_priority,omitempty"` // 0.0-1.0, text-only Candle auto-routing; default 0.5
	SequenceLength  int      `json:"sequence_length,omitempty"`  // Optional, auto-detected if not provided
}

// EmbeddingResult represents a single embedding result
type EmbeddingResult struct {
	Text             string    `json:"text"`
	Modality         string    `json:"modality,omitempty"` // "text"/"image" in mixed mode; empty for text-only (backward compatible)
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
	Model           string  `json:"model,omitempty"`            // Backend-aware: "auto" (default), "mmbert", and Candle-only "qwen3"/"gemma"
	Dimension       int     `json:"dimension,omitempty"`        // Model-compatible target dimension; default 768
	TargetLayer     int     `json:"target_layer,omitempty"`     // mmBERT early-exit layer from the loaded model manifest (0=full)
	QualityPriority float32 `json:"quality_priority,omitempty"` // 0.0-1.0, Candle auto-routing only
	LatencyPriority float32 `json:"latency_priority,omitempty"` // 0.0-1.0, Candle auto-routing only
}

// SimilarityResponse represents the response of a similarity calculation
type SimilarityResponse struct {
	ModelUsed        string  `json:"model_used"`         // "qwen3", "gemma", "mmbert", "multimodal", or "unknown"
	Similarity       float32 `json:"similarity"`         // Cosine similarity score (-1.0 to 1.0)
	ProcessingTimeMs float32 `json:"processing_time_ms"` // Processing time in milliseconds
}

// BatchSimilarityRequest represents a request to find top-k similar candidates for a query
type BatchSimilarityRequest struct {
	Query           string   `json:"query"`                      // Query text
	Candidates      []string `json:"candidates"`                 // Array of candidate texts
	TopK            int      `json:"top_k,omitempty"`            // Max number of matches to return (0 = return all)
	Model           string   `json:"model,omitempty"`            // Backend-aware: "auto" (default), "mmbert", and Candle-only "qwen3"/"gemma"
	Dimension       int      `json:"dimension,omitempty"`        // Model-compatible target dimension; default 768
	TargetLayer     int      `json:"target_layer,omitempty"`     // mmBERT early-exit layer from the loaded model manifest (0=full)
	QualityPriority float32  `json:"quality_priority,omitempty"` // 0.0-1.0, Candle auto-routing only
	LatencyPriority float32  `json:"latency_priority,omitempty"` // 0.0-1.0, Candle auto-routing only
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
	ModelUsed        string                 `json:"model_used"`         // "qwen3", "gemma", "mmbert", "multimodal", or "unknown"
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
