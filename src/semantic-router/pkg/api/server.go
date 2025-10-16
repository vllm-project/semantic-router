//go:build !windows && cgo
// +build !windows,cgo

package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// ClassificationAPIServer holds the server state and dependencies
type ClassificationAPIServer struct {
	classificationSvc *services.ClassificationService
	config            *config.RouterConfig
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
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
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
	Model           string   `json:"model,omitempty"`            // "auto" (default), "qwen3", "gemma"
	Dimension       int      `json:"dimension,omitempty"`        // Target dimension: 768 (default), 512, 256, 128
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
	Model           string  `json:"model,omitempty"`            // "auto" (default), "qwen3", "gemma"
	Dimension       int     `json:"dimension,omitempty"`        // Target dimension: 768 (default), 512, 256, 128
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
	Model           string   `json:"model,omitempty"`            // "auto" (default), "qwen3", "gemma"
	Dimension       int      `json:"dimension,omitempty"`        // Target dimension: 768 (default), 512, 256, 128
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

// StartClassificationAPI starts the Classification API server
func StartClassificationAPI(configPath string, port int) error {
	// Load configuration
	cfg, err := config.LoadConfig(configPath)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	// Create classification service - try to get global service with retry
	classificationSvc := getClassificationServiceWithRetry(5, 500*time.Millisecond)
	if classificationSvc == nil {
		// If no global service exists, try auto-discovery unified classifier
		observability.Infof("No global classification service found, attempting auto-discovery...")
		autoSvc, err := services.NewClassificationServiceWithAutoDiscovery(cfg)
		if err != nil {
			observability.Warnf("Auto-discovery failed: %v, using placeholder service", err)
			classificationSvc = services.NewPlaceholderClassificationService()
		} else {
			observability.Infof("Auto-discovery successful, using unified classifier service")
			classificationSvc = autoSvc
		}
	}

	// Initialize batch metrics configuration
	if cfg != nil && cfg.API.BatchClassification.Metrics.Enabled {
		metricsConfig := metrics.BatchMetricsConfig{
			Enabled:                   cfg.API.BatchClassification.Metrics.Enabled,
			DetailedGoroutineTracking: cfg.API.BatchClassification.Metrics.DetailedGoroutineTracking,
			DurationBuckets:           cfg.API.BatchClassification.Metrics.DurationBuckets,
			SizeBuckets:               cfg.API.BatchClassification.Metrics.SizeBuckets,
			BatchSizeRanges:           cfg.API.BatchClassification.Metrics.BatchSizeRanges,
			HighResolutionTiming:      cfg.API.BatchClassification.Metrics.HighResolutionTiming,
			SampleRate:                cfg.API.BatchClassification.Metrics.SampleRate,
		}
		metrics.SetBatchMetricsConfig(metricsConfig)
	}

	// Create server instance
	apiServer := &ClassificationAPIServer{
		classificationSvc: classificationSvc,
		config:            cfg,
	}

	// Create HTTP server with routes
	mux := apiServer.setupRoutes()
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", port),
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	observability.Infof("Classification API server listening on port %d", port)
	return server.ListenAndServe()
}

// getClassificationServiceWithRetry attempts to get the global classification service with retry logic
func getClassificationServiceWithRetry(maxRetries int, retryInterval time.Duration) *services.ClassificationService {
	for i := 0; i < maxRetries; i++ {
		if svc := services.GetGlobalClassificationService(); svc != nil {
			observability.Infof("Found global classification service on attempt %d/%d", i+1, maxRetries)
			return svc
		}

		if i < maxRetries-1 { // Don't sleep on the last attempt
			observability.Infof("Global classification service not ready, retrying in %v (attempt %d/%d)", retryInterval, i+1, maxRetries)
			time.Sleep(retryInterval)
		}
	}

	observability.Warnf("Failed to find global classification service after %d attempts", maxRetries)
	return nil
}

// setupRoutes configures all API routes
func (s *ClassificationAPIServer) setupRoutes() *http.ServeMux {
	mux := http.NewServeMux()

	// Health check endpoint
	mux.HandleFunc("GET /health", s.handleHealth)

	// Classification endpoints
	mux.HandleFunc("POST /api/v1/classify/intent", s.handleIntentClassification)
	mux.HandleFunc("POST /api/v1/classify/pii", s.handlePIIDetection)
	mux.HandleFunc("POST /api/v1/classify/security", s.handleSecurityDetection)
	mux.HandleFunc("POST /api/v1/classify/combined", s.handleCombinedClassification)
	mux.HandleFunc("POST /api/v1/classify/batch", s.handleBatchClassification)

	// Embedding endpoints
	mux.HandleFunc("POST /api/v1/embeddings", s.handleEmbeddings)
	mux.HandleFunc("POST /api/v1/similarity", s.handleSimilarity)
	mux.HandleFunc("POST /api/v1/similarity/batch", s.handleBatchSimilarity)
	mux.HandleFunc("GET /api/v1/embeddings/models", s.handleEmbeddingModelsInfo) // Only embedding models

	// Information endpoints
	mux.HandleFunc("GET /info/models", s.handleModelsInfo) // All models (classification + embedding)
	mux.HandleFunc("GET /info/classifier", s.handleClassifierInfo)

	// OpenAI-compatible endpoints
	mux.HandleFunc("GET /v1/models", s.handleOpenAIModels)

	// Metrics endpoints
	mux.HandleFunc("GET /metrics/classification", s.handleClassificationMetrics)

	// Configuration endpoints
	mux.HandleFunc("GET /config/classification", s.handleGetConfig)
	mux.HandleFunc("PUT /config/classification", s.handleUpdateConfig)

	return mux
}

// handleHealth handles health check requests
func (s *ClassificationAPIServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status": "healthy", "service": "classification-api"}`))
}

// handleIntentClassification handles intent classification requests
func (s *ClassificationAPIServer) handleIntentClassification(w http.ResponseWriter, r *http.Request) {
	var req services.IntentRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	response, err := s.classificationSvc.ClassifyIntent(req)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFICATION_ERROR", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handlePIIDetection handles PII detection requests
func (s *ClassificationAPIServer) handlePIIDetection(w http.ResponseWriter, r *http.Request) {
	var req services.PIIRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	response, err := s.classificationSvc.DetectPII(req)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFICATION_ERROR", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleSecurityDetection handles security detection requests
func (s *ClassificationAPIServer) handleSecurityDetection(w http.ResponseWriter, r *http.Request) {
	var req services.SecurityRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	response, err := s.classificationSvc.CheckSecurity(req)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFICATION_ERROR", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// Placeholder handlers for remaining endpoints
func (s *ClassificationAPIServer) handleCombinedClassification(w http.ResponseWriter, r *http.Request) {
	s.writeErrorResponse(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Combined classification not implemented yet")
}

func (s *ClassificationAPIServer) handleBatchClassification(w http.ResponseWriter, r *http.Request) {
	// Record batch classification request
	metrics.RecordBatchClassificationRequest("unified")

	// Start timing for duration metrics
	start := time.Now()

	// First, read the raw body to check if texts field exists
	body, err := io.ReadAll(r.Body)
	if err != nil {
		metrics.RecordBatchClassificationError("unified", "read_body_failed")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "Failed to read request body")
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(body))

	// Check if texts field exists in JSON
	var rawReq map[string]interface{}
	if err := json.Unmarshal(body, &rawReq); err != nil {
		metrics.RecordBatchClassificationError("unified", "invalid_json")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "Invalid JSON format")
		return
	}

	// Check if texts field is present
	if _, exists := rawReq["texts"]; !exists {
		metrics.RecordBatchClassificationError("unified", "missing_texts_field")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "texts field is required")
		return
	}

	var req BatchClassificationRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		metrics.RecordBatchClassificationError("unified", "parse_request_failed")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	// Input validation - now we know texts field exists, check if it's empty
	if len(req.Texts) == 0 {
		// Record validation error in metrics
		metrics.RecordBatchClassificationError("unified", "empty_texts")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "texts array cannot be empty")
		return
	}

	// Record the number of texts being processed
	metrics.RecordBatchClassificationTexts("unified", len(req.Texts))

	// Batch classification requires unified classifier
	if !s.classificationSvc.HasUnifiedClassifier() {
		metrics.RecordBatchClassificationError("unified", "classifier_unavailable")
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "UNIFIED_CLASSIFIER_UNAVAILABLE",
			"Batch classification requires unified classifier. Please ensure models are available in ./models/ directory.")
		return
	}

	// Use unified classifier for true batch processing with options support
	unifiedResults, err := s.classificationSvc.ClassifyBatchUnifiedWithOptions(req.Texts, req.Options)
	if err != nil {
		metrics.RecordBatchClassificationError("unified", "classification_failed")
		s.writeErrorResponse(w, http.StatusInternalServerError, "UNIFIED_CLASSIFICATION_ERROR", err.Error())
		return
	}

	// Convert unified results to legacy format based on requested task type
	results := s.extractRequestedResults(unifiedResults, req.TaskType, req.Options)
	statistics := s.calculateUnifiedStatistics(unifiedResults)

	// Record successful processing duration
	duration := time.Since(start).Seconds()
	metrics.RecordBatchClassificationDuration("unified", len(req.Texts), duration)

	response := BatchClassificationResponse{
		Results:          results,
		TotalCount:       len(req.Texts),
		ProcessingTimeMs: unifiedResults.ProcessingTimeMs,
		Statistics:       statistics,
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleModelsInfo(w http.ResponseWriter, r *http.Request) {
	response := s.buildModelsInfoResponse()
	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleEmbeddingModelsInfo handles GET /api/v1/embeddings/models
// Returns ONLY embedding models information
func (s *ClassificationAPIServer) handleEmbeddingModelsInfo(w http.ResponseWriter, r *http.Request) {
	embeddingModels := s.getEmbeddingModelsInfo()

	response := map[string]interface{}{
		"models": embeddingModels,
		"count":  len(embeddingModels),
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleClassifierInfo(w http.ResponseWriter, r *http.Request) {
	if s.config == nil {
		s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
			"status": "no_config",
			"config": nil,
		})
		return
	}

	// Return the config directly
	s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"status": "config_loaded",
		"config": s.config,
	})
}

// handleOpenAIModels handles OpenAI-compatible model listing at /v1/models
// It returns all models discoverable from the router configuration plus a synthetic "auto" model.
func (s *ClassificationAPIServer) handleOpenAIModels(w http.ResponseWriter, r *http.Request) {
	now := time.Now().Unix()

	// Start with the special "auto" model always available from the router
	models := []OpenAIModel{
		{
			ID:      "auto",
			Object:  "model",
			Created: now,
			OwnedBy: "semantic-router",
		},
	}

	// Append underlying models from config (if available)
	if s.config != nil {
		for _, m := range s.config.GetAllModels() {
			// Skip if already added as "auto" (or avoid duplicates in general)
			if m == "auto" {
				continue
			}
			models = append(models, OpenAIModel{
				ID:      m,
				Object:  "model",
				Created: now,
				OwnedBy: "upstream-endpoint",
			})
		}
	}

	resp := OpenAIModelList{
		Object: "list",
		Data:   models,
	}

	s.writeJSONResponse(w, http.StatusOK, resp)
}

func (s *ClassificationAPIServer) handleClassificationMetrics(w http.ResponseWriter, r *http.Request) {
	s.writeErrorResponse(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Classification metrics not implemented yet")
}

func (s *ClassificationAPIServer) handleGetConfig(w http.ResponseWriter, r *http.Request) {
	s.writeErrorResponse(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Get config not implemented yet")
}

func (s *ClassificationAPIServer) handleUpdateConfig(w http.ResponseWriter, r *http.Request) {
	s.writeErrorResponse(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Update config not implemented yet")
}

// Helper methods for JSON handling
func (s *ClassificationAPIServer) parseJSONRequest(r *http.Request, v interface{}) error {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return fmt.Errorf("failed to read request body: %w", err)
	}
	defer r.Body.Close()

	if err := json.Unmarshal(body, v); err != nil {
		return fmt.Errorf("failed to parse JSON: %w", err)
	}

	return nil
}

func (s *ClassificationAPIServer) writeJSONResponse(w http.ResponseWriter, statusCode int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	if err := json.NewEncoder(w).Encode(data); err != nil {
		observability.Errorf("Failed to encode JSON response: %v", err)
	}
}

func (s *ClassificationAPIServer) writeErrorResponse(w http.ResponseWriter, statusCode int, errorCode, message string) {
	errorResponse := map[string]interface{}{
		"error": map[string]interface{}{
			"code":      errorCode,
			"message":   message,
			"timestamp": time.Now().UTC().Format(time.RFC3339),
		},
	}

	s.writeJSONResponse(w, statusCode, errorResponse)
}

// buildModelsInfoResponse builds the models info response
func (s *ClassificationAPIServer) buildModelsInfoResponse() ModelsInfoResponse {
	var models []ModelInfo

	// Check if we have a real classification service with classifier
	if s.classificationSvc != nil && s.classificationSvc.HasClassifier() {
		// Get model information from the classifier
		models = s.getLoadedModelsInfo()
	} else {
		// Return placeholder model info
		models = s.getPlaceholderModelsInfo()
	}

	// Add embedding models information
	embeddingModels := s.getEmbeddingModelsInfo()
	models = append(models, embeddingModels...)

	// Get system information
	systemInfo := s.getSystemInfo()

	return ModelsInfoResponse{
		Models: models,
		System: systemInfo,
	}
}

// getLoadedModelsInfo returns information about actually loaded models
func (s *ClassificationAPIServer) getLoadedModelsInfo() []ModelInfo {
	var models []ModelInfo

	if s.config == nil {
		return models
	}

	// Category classifier model
	if s.config.Classifier.CategoryModel.CategoryMappingPath != "" {
		categories := []string{}
		// Extract category names from config.Categories
		for _, cat := range s.config.Categories {
			categories = append(categories, cat.Name)
		}

		models = append(models, ModelInfo{
			Name:       "category_classifier",
			Type:       "intent_classification",
			Loaded:     true,
			ModelPath:  s.config.Classifier.CategoryModel.ModelID,
			Categories: categories,
			Metadata: map[string]string{
				"mapping_path": s.config.Classifier.CategoryModel.CategoryMappingPath,
				"model_type":   "modernbert",
				"threshold":    fmt.Sprintf("%.2f", s.config.Classifier.CategoryModel.Threshold),
			},
		})
	}

	// PII classifier model
	if s.config.Classifier.PIIModel.PIIMappingPath != "" {
		models = append(models, ModelInfo{
			Name:      "pii_classifier",
			Type:      "pii_detection",
			Loaded:    true,
			ModelPath: s.config.Classifier.PIIModel.ModelID,
			Metadata: map[string]string{
				"mapping_path": s.config.Classifier.PIIModel.PIIMappingPath,
				"model_type":   "modernbert_token",
				"threshold":    fmt.Sprintf("%.2f", s.config.Classifier.PIIModel.Threshold),
			},
		})
	}

	// Jailbreak classifier model
	if s.config.PromptGuard.Enabled {
		models = append(models, ModelInfo{
			Name:      "jailbreak_classifier",
			Type:      "security_detection",
			Loaded:    true,
			ModelPath: s.config.PromptGuard.JailbreakMappingPath,
			Metadata: map[string]string{
				"enabled": "true",
			},
		})
	}

	// BERT similarity model
	if s.config.BertModel.ModelID != "" {
		models = append(models, ModelInfo{
			Name:      "bert_similarity_model",
			Type:      "similarity",
			Loaded:    true,
			ModelPath: s.config.BertModel.ModelID,
			Metadata: map[string]string{
				"model_type": "sentence_transformer",
				"threshold":  fmt.Sprintf("%.2f", s.config.BertModel.Threshold),
				"use_cpu":    fmt.Sprintf("%t", s.config.BertModel.UseCPU),
			},
		})
	}

	return models
}

// getPlaceholderModelsInfo returns placeholder model information
func (s *ClassificationAPIServer) getPlaceholderModelsInfo() []ModelInfo {
	return []ModelInfo{
		{
			Name:   "category_classifier",
			Type:   "intent_classification",
			Loaded: false,
			Metadata: map[string]string{
				"status": "not_initialized",
			},
		},
		{
			Name:   "pii_classifier",
			Type:   "pii_detection",
			Loaded: false,
			Metadata: map[string]string{
				"status": "not_initialized",
			},
		},
		{
			Name:   "jailbreak_classifier",
			Type:   "security_detection",
			Loaded: false,
			Metadata: map[string]string{
				"status": "not_initialized",
			},
		},
	}
}

// getSystemInfo returns system information
func (s *ClassificationAPIServer) getSystemInfo() SystemInfo {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return SystemInfo{
		GoVersion:    runtime.Version(),
		Architecture: runtime.GOARCH,
		OS:           runtime.GOOS,
		MemoryUsage:  fmt.Sprintf("%.2f MB", float64(m.Alloc)/1024/1024),
		GPUAvailable: false, // TODO: Implement GPU detection
	}
}

// getEmbeddingModelsInfo returns information about loaded embedding models
func (s *ClassificationAPIServer) getEmbeddingModelsInfo() []ModelInfo {
	var models []ModelInfo

	// Query embedding models info from Rust FFI
	embeddingInfo, err := candle_binding.GetEmbeddingModelsInfo()
	if err != nil {
		observability.Warnf("Failed to get embedding models info: %v", err)
		return models
	}

	// Convert to ModelInfo format
	for _, model := range embeddingInfo.Models {
		models = append(models, ModelInfo{
			Name:      fmt.Sprintf("%s_embedding_model", model.ModelName),
			Type:      "embedding",
			Loaded:    model.IsLoaded,
			ModelPath: model.ModelPath,
			Metadata: map[string]string{
				"model_type":           model.ModelName,
				"max_sequence_length":  fmt.Sprintf("%d", model.MaxSequenceLength),
				"default_dimension":    fmt.Sprintf("%d", model.DefaultDimension),
				"matryoshka_supported": "true",
			},
		})
	}

	return models
}

// extractRequestedResults converts unified results to batch format based on task type
func (s *ClassificationAPIServer) extractRequestedResults(unifiedResults *services.UnifiedBatchResponse, taskType string, options *ClassificationOptions) []BatchClassificationResult {
	// Determine the correct batch size based on task type
	var batchSize int
	switch taskType {
	case "pii":
		batchSize = len(unifiedResults.PIIResults)
	case "security":
		batchSize = len(unifiedResults.SecurityResults)
	default:
		batchSize = len(unifiedResults.IntentResults)
	}

	results := make([]BatchClassificationResult, batchSize)

	switch taskType {
	case "pii":
		// Convert PII results to batch format
		for i, piiResult := range unifiedResults.PIIResults {
			category := "no_pii"
			if piiResult.HasPII {
				if len(piiResult.PIITypes) > 0 {
					category = piiResult.PIITypes[0] // Use first PII type
				} else {
					category = "pii_detected"
				}
			}
			results[i] = BatchClassificationResult{
				Category:         category,
				Confidence:       float64(piiResult.Confidence),
				ProcessingTimeMs: unifiedResults.ProcessingTimeMs / int64(len(unifiedResults.PIIResults)),
			}
		}
	case "security":
		// Convert security results to batch format
		for i, securityResult := range unifiedResults.SecurityResults {
			category := "safe"
			if securityResult.IsJailbreak {
				category = securityResult.ThreatType
			}
			results[i] = BatchClassificationResult{
				Category:         category,
				Confidence:       float64(securityResult.Confidence),
				ProcessingTimeMs: unifiedResults.ProcessingTimeMs / int64(len(unifiedResults.SecurityResults)),
			}
		}
	case "intent":
		fallthrough
	default:
		// Convert intent results to batch format with probabilities support (default)
		for i, intentResult := range unifiedResults.IntentResults {
			result := BatchClassificationResult{
				Category:         intentResult.Category,
				Confidence:       float64(intentResult.Confidence),
				ProcessingTimeMs: unifiedResults.ProcessingTimeMs / int64(len(unifiedResults.IntentResults)),
			}

			// Add probabilities if requested and available
			if options != nil && options.ReturnProbabilities && len(intentResult.Probabilities) > 0 {
				result.Probabilities = make(map[string]float64)
				// Convert probabilities array to map (assuming they match category order)
				// For now, just include the main category probability
				result.Probabilities[intentResult.Category] = float64(intentResult.Confidence)
			}

			results[i] = result
		}
	}

	return results
}

// calculateUnifiedStatistics calculates statistics from unified batch results
func (s *ClassificationAPIServer) calculateUnifiedStatistics(unifiedResults *services.UnifiedBatchResponse) CategoryClassificationStatistics {
	// For now, calculate statistics based on intent results
	// This maintains compatibility with existing API expectations

	categoryDistribution := make(map[string]int)
	totalConfidence := 0.0
	lowConfidenceCount := 0
	lowConfidenceThreshold := 0.7

	for _, intentResult := range unifiedResults.IntentResults {
		categoryDistribution[intentResult.Category]++
		confidence := float64(intentResult.Confidence)
		totalConfidence += confidence

		if confidence < lowConfidenceThreshold {
			lowConfidenceCount++
		}
	}

	avgConfidence := 0.0
	if len(unifiedResults.IntentResults) > 0 {
		avgConfidence = totalConfidence / float64(len(unifiedResults.IntentResults))
	}

	return CategoryClassificationStatistics{
		CategoryDistribution: categoryDistribution,
		AvgConfidence:        avgConfidence,
		LowConfidenceCount:   lowConfidenceCount,
	}
}

// handleEmbeddings handles embedding generation requests
func (s *ClassificationAPIServer) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var req EmbeddingRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	// Validate input
	if len(req.Texts) == 0 {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "texts array cannot be empty")
		return
	}

	// Set defaults
	if req.Model == "" {
		req.Model = "auto"
	}
	if req.Dimension == 0 {
		req.Dimension = 768 // Default to full dimension
	}
	if req.QualityPriority == 0 && req.LatencyPriority == 0 {
		req.QualityPriority = 0.5
		req.LatencyPriority = 0.5
	}

	// Validate dimension
	validDimensions := map[int]bool{128: true, 256: true, 512: true, 768: true, 1024: true}
	if !validDimensions[req.Dimension] {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_DIMENSION",
			fmt.Sprintf("dimension must be one of: 128, 256, 512, 768, 1024 (got %d)", req.Dimension))
		return
	}

	// Generate embeddings for each text
	results := make([]EmbeddingResult, 0, len(req.Texts))
	var totalProcessingTime int64

	for _, text := range req.Texts {
		var output *candle_binding.EmbeddingOutput
		var err error

		// Choose between manual model selection or automatic routing
		if req.Model == "auto" || req.Model == "" {
			// Automatic routing based on quality/latency priorities
			output, err = candle_binding.GetEmbeddingWithMetadata(
				text,
				req.QualityPriority,
				req.LatencyPriority,
				req.Dimension,
			)
		} else {
			// Manual model selection ("qwen3" or "gemma")
			output, err = candle_binding.GetEmbeddingWithModelType(
				text,
				req.Model,
				req.Dimension,
			)
		}

		if err != nil {
			s.writeErrorResponse(w, http.StatusInternalServerError, "EMBEDDING_GENERATION_FAILED",
				fmt.Sprintf("failed to generate embedding: %v", err))
			return
		}

		// Use metadata directly from Rust layer
		processingTime := int64(output.ProcessingTimeMs)

		results = append(results, EmbeddingResult{
			Text:             text,
			Embedding:        output.Embedding,
			Dimension:        len(output.Embedding),
			ModelUsed:        output.ModelType,
			ProcessingTimeMs: processingTime,
		})

		totalProcessingTime += processingTime
	}

	// Calculate statistics
	avgProcessingTime := float64(totalProcessingTime) / float64(len(req.Texts))

	response := EmbeddingResponse{
		Embeddings:            results,
		TotalCount:            len(results),
		TotalProcessingTimeMs: totalProcessingTime,
		AvgProcessingTimeMs:   avgProcessingTime,
	}

	observability.Infof("Generated %d embeddings in %dms (avg: %.2fms)",
		len(results), totalProcessingTime, avgProcessingTime)

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleSimilarity handles text similarity calculation requests
func (s *ClassificationAPIServer) handleSimilarity(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var req SimilarityRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	// Validate input
	if req.Text1 == "" || req.Text2 == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "both text1 and text2 must be provided")
		return
	}

	// Set defaults
	if req.Model == "" {
		req.Model = "auto"
	}
	if req.Dimension == 0 {
		req.Dimension = 768 // Default to full dimension
	}
	if req.Model == "auto" && req.QualityPriority == 0 && req.LatencyPriority == 0 {
		req.QualityPriority = 0.5
		req.LatencyPriority = 0.5
	}

	// Validate dimension
	validDimensions := map[int]bool{128: true, 256: true, 512: true, 768: true, 1024: true}
	if !validDimensions[req.Dimension] {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_DIMENSION",
			fmt.Sprintf("dimension must be one of: 128, 256, 512, 768, 1024 (got %d)", req.Dimension))
		return
	}

	// Calculate similarity
	result, err := candle_binding.CalculateEmbeddingSimilarity(
		req.Text1,
		req.Text2,
		req.Model,
		req.Dimension,
	)

	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "SIMILARITY_CALCULATION_FAILED",
			fmt.Sprintf("failed to calculate similarity: %v", err))
		return
	}

	response := SimilarityResponse{
		Similarity:       result.Similarity,
		ModelUsed:        result.ModelType,
		ProcessingTimeMs: result.ProcessingTimeMs,
	}

	observability.Infof("Calculated similarity: %.4f (model: %s, took: %.2fms)",
		result.Similarity, result.ModelType, result.ProcessingTimeMs)

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleBatchSimilarity handles batch similarity matching requests
func (s *ClassificationAPIServer) handleBatchSimilarity(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var req BatchSimilarityRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	// Validate input
	if req.Query == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "query must be provided")
		return
	}
	if len(req.Candidates) == 0 {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "candidates array cannot be empty")
		return
	}

	// Set defaults
	if req.Model == "" {
		req.Model = "auto"
	}
	if req.Dimension == 0 {
		req.Dimension = 768 // Default to full dimension
	}
	if req.TopK == 0 {
		req.TopK = len(req.Candidates) // Default to all candidates
	}
	if req.Model == "auto" && req.QualityPriority == 0 && req.LatencyPriority == 0 {
		req.QualityPriority = 0.5
		req.LatencyPriority = 0.5
	}

	// Validate dimension
	validDimensions := map[int]bool{128: true, 256: true, 512: true, 768: true, 1024: true}
	if !validDimensions[req.Dimension] {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_DIMENSION",
			fmt.Sprintf("dimension must be one of: 128, 256, 512, 768, 1024 (got %d)", req.Dimension))
		return
	}

	// Calculate batch similarity
	result, err := candle_binding.CalculateSimilarityBatch(
		req.Query,
		req.Candidates,
		req.TopK,
		req.Model,
		req.Dimension,
	)

	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "BATCH_SIMILARITY_FAILED",
			fmt.Sprintf("failed to calculate batch similarity: %v", err))
		return
	}

	// Build response with matched text included
	matches := make([]BatchSimilarityMatch, len(result.Matches))
	for i, match := range result.Matches {
		matches[i] = BatchSimilarityMatch{
			Index:      match.Index,
			Similarity: match.Similarity,
			Text:       req.Candidates[match.Index],
		}
	}

	response := BatchSimilarityResponse{
		Matches:          matches,
		TotalCandidates:  len(req.Candidates),
		ModelUsed:        result.ModelType,
		ProcessingTimeMs: result.ProcessingTimeMs,
	}

	observability.Infof("Calculated batch similarity: query='%s', %d candidates, top-%d matches (model: %s, took: %.2fms)",
		req.Query, len(req.Candidates), len(matches), result.ModelType, result.ProcessingTimeMs)

	s.writeJSONResponse(w, http.StatusOK, response)
}
