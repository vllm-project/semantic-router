//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// Init starts the API server
func Init(configPath string, port int, enableSystemPromptAPI bool) error {
	// Get the global configuration instead of loading from file
	// This ensures we use the same config as the rest of the application
	cfg := config.Get()
	if cfg == nil {
		return fmt.Errorf("configuration not initialized")
	}

	// Create classification service - try to get global service with retry
	classificationSvc := initClassify(5, 500*time.Millisecond)
	if classificationSvc == nil {
		// If no global service exists, try auto-discovery unified classifier
		logging.Infof("No global classification service found, attempting auto-discovery...")
		autoSvc, err := services.NewClassificationServiceWithAutoDiscovery(cfg)
		if err != nil {
			logging.Warnf("Auto-discovery failed: %v, using placeholder service", err)
			classificationSvc = services.NewPlaceholderClassificationService()
		} else {
			logging.Infof("Auto-discovery successful, using unified classifier service")
			classificationSvc = autoSvc
		}
	}

	// Initialize batch metrics configuration
	if cfg.API.BatchClassification.Metrics.Enabled {
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
		classificationSvc:     classificationSvc,
		config:                cfg,
		enableSystemPromptAPI: enableSystemPromptAPI,
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

	logging.Infof("Classification API server listening on port %d", port)
	return server.ListenAndServe()
}

// initClassify attempts to get the global classification service with retry logic
func initClassify(maxRetries int, retryInterval time.Duration) *services.ClassificationService {
	for i := 0; i < maxRetries; i++ {
		if svc := services.GetGlobalClassificationService(); svc != nil {
			return svc
		}

		if i < maxRetries-1 { // Don't sleep on the last attempt
			logging.Infof("Global classification service not ready, retrying in %v (attempt %d/%d)", retryInterval, i+1, maxRetries)
			time.Sleep(retryInterval)
		}
	}

	logging.Warnf("Failed to find global classification service after %d attempts", maxRetries)
	return nil
}

// setupRoutes configures all API routes
func (s *ClassificationAPIServer) setupRoutes() *http.ServeMux {
	mux := http.NewServeMux()

	// Health check endpoint
	mux.HandleFunc("GET /health", s.handleHealth)

	// API discovery endpoint
	mux.HandleFunc("GET /api/v1", s.handleAPIOverview)

	// OpenAPI and documentation endpoints
	mux.HandleFunc("GET /openapi.json", s.handleOpenAPISpec)
	mux.HandleFunc("GET /docs", s.handleSwaggerUI)

	// Classification endpoints
	mux.HandleFunc("POST /api/v1/classify/intent", s.handleIntentClassification)
	mux.HandleFunc("POST /api/v1/classify/pii", s.handlePIIDetection)
	mux.HandleFunc("POST /api/v1/classify/security", s.handleSecurityDetection)
	mux.HandleFunc("POST /api/v1/classify/fact-check", s.handleFactCheckClassification)
	mux.HandleFunc("POST /api/v1/classify/user-feedback", s.handleUserFeedbackClassification)
	mux.HandleFunc("POST /api/v1/classify/combined", s.handleCombinedClassification)
	mux.HandleFunc("POST /api/v1/classify/batch", s.handleBatchClassification)

	// Evaluation endpoint - evaluates all configured signals regardless of decision usage
	mux.HandleFunc("POST /api/v1/eval", s.handleEvalClassification)

	// Embedding endpoints
	mux.HandleFunc("POST /api/v1/embeddings", s.handleEmbeddings)
	mux.HandleFunc("POST /api/v1/similarity", s.handleSimilarity)
	mux.HandleFunc("POST /api/v1/similarity/batch", s.handleBatchSimilarity)

	// Information endpoints
	mux.HandleFunc("GET /info/models", s.handleModelsInfo) // All models (classification + embedding)
	mux.HandleFunc("GET /info/classifier", s.handleClassifierInfo)
	mux.HandleFunc("GET /api/v1/embeddings/models", s.handleEmbeddingModelsInfo) // Only embedding models

	// OpenAI-compatible endpoints
	mux.HandleFunc("GET /v1/models", s.handleOpenAIModels)

	// Metrics endpoints
	mux.HandleFunc("GET /metrics/classification", s.handleClassificationMetrics)

	// Model selection feedback endpoints
	mux.HandleFunc("POST /api/v1/feedback", s.handleFeedback)
	mux.HandleFunc("GET /api/v1/ratings", s.handleGetRatings)
	mux.HandleFunc("GET /api/v1/rl-state", s.handleRLState)

	// Configuration endpoints
	mux.HandleFunc("GET /config/classification", s.handleGetConfig)
	mux.HandleFunc("PUT /config/classification", s.handleUpdateConfig)

	// System prompt configuration endpoints (only if explicitly enabled)
	if s.enableSystemPromptAPI {
		logging.Infof("System prompt configuration endpoints enabled")
		mux.HandleFunc("GET /config/system-prompts", s.handleGetSystemPrompts)
		mux.HandleFunc("PUT /config/system-prompts", s.handleUpdateSystemPrompts)
	} else {
		logging.Infof("System prompt configuration endpoints disabled for security")
	}

	return mux
}

// handleHealth handles health check requests
func (s *ClassificationAPIServer) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status": "healthy", "service": "classification-api"}`))
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
		logging.Errorf("Failed to encode JSON response: %v", err)
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

// handleRLState returns the current state of RL-based selectors for debugging
func (s *ClassificationAPIServer) handleRLState(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Query().Get("user_id")

	state := map[string]interface{}{
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}

	// Get RLDriven selector state
	if rlSelector, ok := selection.GlobalRegistry.Get(selection.MethodRLDriven); ok {
		if rlDriven, ok := rlSelector.(*selection.RLDrivenSelector); ok {
			state["rl_driven"] = rlDriven.GetDebugState(userID)
		}
	}

	// Get GMTRouter selector state
	if gmtSelector, ok := selection.GlobalRegistry.Get(selection.MethodGMTRouter); ok {
		if gmtRouter, ok := gmtSelector.(*selection.GMTRouterSelector); ok {
			state["gmtrouter"] = gmtRouter.GetDebugState(userID)
		}
	}

	s.writeJSONResponse(w, http.StatusOK, state)
}
