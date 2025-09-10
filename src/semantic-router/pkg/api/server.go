package api

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"runtime"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
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

// BatchClassificationRequest represents a batch classification request
type BatchClassificationRequest struct {
	Texts   []string               `json:"texts"`
	Options *ClassificationOptions `json:"options,omitempty"`
}

// BatchClassificationResponse represents the response from batch classification
type BatchClassificationResponse struct {
	Results          []services.Classification        `json:"results"`
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
		// If no global service exists after retries, create a placeholder service
		log.Printf("No global classification service found after retries, using placeholder service")
		classificationSvc = services.NewPlaceholderClassificationService()
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

	log.Printf("Classification API server listening on port %d", port)
	return server.ListenAndServe()
}

// getClassificationServiceWithRetry attempts to get the global classification service with retry logic
func getClassificationServiceWithRetry(maxRetries int, retryInterval time.Duration) *services.ClassificationService {
	for i := 0; i < maxRetries; i++ {
		if svc := services.GetGlobalClassificationService(); svc != nil {
			log.Printf("Found global classification service on attempt %d/%d", i+1, maxRetries)
			return svc
		}

		if i < maxRetries-1 { // Don't sleep on the last attempt
			log.Printf("Global classification service not ready, retrying in %v (attempt %d/%d)", retryInterval, i+1, maxRetries)
			time.Sleep(retryInterval)
		}
	}

	log.Printf("Failed to find global classification service after %d attempts", maxRetries)
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

	// Information endpoints
	mux.HandleFunc("GET /info/models", s.handleModelsInfo)
	mux.HandleFunc("GET /info/classifier", s.handleClassifierInfo)

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
	start := time.Now()

	var req BatchClassificationRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	// Input validation
	if len(req.Texts) == 0 {
		// Record validation error in metrics
		metrics.RecordBatchClassificationError("validation", "empty_texts")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "texts array cannot be empty")
		return
	}

	// Get max batch size from config, default to 100
	maxBatchSize := 100
	if s.config != nil && s.config.API.BatchClassification.MaxBatchSize > 0 {
		maxBatchSize = s.config.API.BatchClassification.MaxBatchSize
	}

	if len(req.Texts) > maxBatchSize {
		// Record validation error in metrics
		metrics.RecordBatchClassificationError("validation", "batch_too_large")
		s.writeErrorResponse(w, http.StatusBadRequest, "BATCH_TOO_LARGE",
			fmt.Sprintf("batch size cannot exceed %d texts", maxBatchSize))
		return
	}

	// Get concurrency threshold from config, default to 5
	concurrencyThreshold := 5
	if s.config != nil && s.config.API.BatchClassification.ConcurrencyThreshold > 0 {
		concurrencyThreshold = s.config.API.BatchClassification.ConcurrencyThreshold
	}

	// Process texts based on batch size
	var results []services.Classification
	var err error

	if len(req.Texts) <= concurrencyThreshold {
		results, err = s.processSequentially(req.Texts, req.Options)
	} else {
		results, err = s.processConcurrently(req.Texts, req.Options)
	}

	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFICATION_ERROR", err.Error())
		return
	}

	// Calculate statistics
	statistics := s.calculateStatistics(results)

	response := BatchClassificationResponse{
		Results:          results,
		TotalCount:       len(req.Texts),
		ProcessingTimeMs: time.Since(start).Milliseconds(),
		Statistics:       statistics,
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleModelsInfo(w http.ResponseWriter, r *http.Request) {
	response := s.buildModelsInfoResponse()
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
		log.Printf("Failed to encode JSON response: %v", err)
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

// processSequentially handles small batches with sequential processing
func (s *ClassificationAPIServer) processSequentially(texts []string, options *ClassificationOptions) ([]services.Classification, error) {
	start := time.Now()
	processingType := "sequential"
	batchSize := len(texts)

	// Record request and batch size metrics
	metrics.RecordBatchClassificationRequest(processingType)
	metrics.RecordBatchSizeDistribution(processingType, batchSize)

	// Defer recording processing time and text count
	defer func() {
		duration := time.Since(start).Seconds()
		metrics.RecordBatchClassificationDuration(processingType, batchSize, duration)
		metrics.RecordBatchClassificationTexts(processingType, batchSize)
	}()

	results := make([]services.Classification, len(texts))
	for i, text := range texts {
		result, err := s.classifySingleText(text, options)
		if err != nil {
			metrics.RecordBatchClassificationError(processingType, "classification_failed")
			return nil, fmt.Errorf("failed to classify text at index %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// processConcurrently handles large batches with concurrent processing
func (s *ClassificationAPIServer) processConcurrently(texts []string, options *ClassificationOptions) ([]services.Classification, error) {
	start := time.Now()
	processingType := "concurrent"
	batchSize := len(texts)
	batchID := fmt.Sprintf("batch_%d", time.Now().UnixNano())

	// Record request and batch size metrics
	metrics.RecordBatchClassificationRequest(processingType)
	metrics.RecordBatchSizeDistribution(processingType, batchSize)

	// Defer recording processing time and text count
	defer func() {
		duration := time.Since(start).Seconds()
		metrics.RecordBatchClassificationDuration(processingType, batchSize, duration)
		metrics.RecordBatchClassificationTexts(processingType, batchSize)
	}()

	// Get max concurrency from config, default to 8
	maxConcurrency := 8
	if s.config != nil && s.config.API.BatchClassification.MaxConcurrency > 0 {
		maxConcurrency = s.config.API.BatchClassification.MaxConcurrency
	}

	results := make([]services.Classification, len(texts))
	errors := make([]error, len(texts))

	semaphore := make(chan struct{}, maxConcurrency)
	var wg sync.WaitGroup

	for i, text := range texts {
		wg.Add(1)
		go func(index int, txt string) {
			defer wg.Done()

			// Record goroutine start (if detailed tracking is enabled)
			metricsConfig := metrics.GetBatchMetricsConfig()
			if metricsConfig.DetailedGoroutineTracking {
				metrics.ConcurrentGoroutines.WithLabelValues(batchID).Inc()

				defer func() {
					// Record goroutine end
					metrics.ConcurrentGoroutines.WithLabelValues(batchID).Dec()
				}()
			}

			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			// TODO: Refactor candle-binding to support batch mode for better performance
			// This would allow processing multiple texts in a single model inference call
			// instead of individual calls, significantly improving throughput
			result, err := s.classifySingleText(txt, options)
			if err != nil {
				errors[index] = err
				metrics.RecordBatchClassificationError(processingType, "classification_failed")
				return
			}
			results[index] = result
		}(i, text)
	}

	wg.Wait()

	// Check for errors
	for i, err := range errors {
		if err != nil {
			return nil, fmt.Errorf("failed to classify text at index %d: %w", i, err)
		}
	}

	return results, nil
}

// classifySingleText processes a single text using existing service
func (s *ClassificationAPIServer) classifySingleText(text string, options *ClassificationOptions) (services.Classification, error) {
	// Convert API options to service options
	var serviceOptions *services.IntentOptions
	if options != nil {
		serviceOptions = &services.IntentOptions{
			ReturnProbabilities: options.ReturnProbabilities,
			ConfidenceThreshold: options.ConfidenceThreshold,
			IncludeExplanation:  options.IncludeExplanation,
		}
	}

	individualReq := services.IntentRequest{
		Text:    text,
		Options: serviceOptions,
	}

	response, err := s.classificationSvc.ClassifyIntent(individualReq)
	if err != nil {
		return services.Classification{}, err
	}

	return response.Classification, nil
}

// calculateStatistics computes batch processing statistics
func (s *ClassificationAPIServer) calculateStatistics(results []services.Classification) CategoryClassificationStatistics {
	categoryDistribution := make(map[string]int)
	var totalConfidence float64
	lowConfidenceCount := 0

	for _, result := range results {
		if result.Category != "" {
			categoryDistribution[result.Category]++
		}
		totalConfidence += result.Confidence
		if result.Confidence < 0.7 {
			lowConfidenceCount++
		}
	}

	avgConfidence := 0.0
	if len(results) > 0 {
		avgConfidence = totalConfidence / float64(len(results))
	}

	return CategoryClassificationStatistics{
		CategoryDistribution: categoryDistribution,
		AvgConfidence:        avgConfidence,
		LowConfidenceCount:   lowConfidenceCount,
	}
}
