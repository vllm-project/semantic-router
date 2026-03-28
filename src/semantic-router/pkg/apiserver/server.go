//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

// Init starts the API server.
func Init(configPath string, port int) error {
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

	// Get memory store if available (set by ExtProc router during init)
	var memoryStore memory.Store
	if shouldInitMemoryStore(cfg) {
		memoryStore = initMemoryStore(5, 500*time.Millisecond)
		if memoryStore != nil {
			logging.Infof("Memory management API enabled")
		} else {
			logging.Infof("Memory store not available, memory management API will return 503")
		}
	} else {
		logging.Infof("Memory disabled in config, skipping memory store initialization")
	}

	liveClassificationSvc := newLiveClassificationService(
		classificationSvc,
		func() classificationService { return services.GetGlobalClassificationService() },
	)

	// Create server instance
	apiServer := &ClassificationAPIServer{
		classificationSvc:     liveClassificationSvc,
		config:                cfg,
		runtimeConfig:         newLiveRuntimeConfig(cfg, config.Get, liveClassificationSvc.RefreshRuntimeConfig),
		configPath:            configPath,
		memoryStore:           memoryStore,
		knowledgeBaseMapCache: newKnowledgeBaseMapCache(),
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

	logging.Infof("Router apiserver listening on port %d", port)
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

// initMemoryStore attempts to get the global memory store with retry logic.
// The memory store is created by the ExtProc router which may start concurrently.
func initMemoryStore(maxRetries int, retryInterval time.Duration) memory.Store {
	for i := 0; i < maxRetries; i++ {
		if store := memory.GetGlobalMemoryStore(); store != nil {
			return store
		}

		if i < maxRetries-1 {
			logging.Infof("Global memory store not ready, retrying in %v (attempt %d/%d)", retryInterval, i+1, maxRetries)
			time.Sleep(retryInterval)
		}
	}

	logging.Warnf("Memory store not available after %d attempts", maxRetries)
	return nil
}

func shouldInitMemoryStore(cfg *config.RouterConfig) bool {
	if cfg == nil {
		return false
	}
	if cfg.Memory.Enabled {
		return true
	}
	for _, decision := range cfg.Decisions {
		if decision.HasPlugin("memory") {
			return true
		}
	}
	return false
}

// setupRoutes configures all API routes
func (s *ClassificationAPIServer) setupRoutes() *http.ServeMux {
	mux := http.NewServeMux()
	s.registerCoreRoutes(mux)
	s.registerClassificationRoutes(mux)
	s.registerEmbeddingRoutes(mux)
	s.registerInfoRoutes(mux)
	s.registerConfigRoutes(mux)
	s.registerMemoryRoutes(mux)
	registerVectorStoreRoutes(mux, s)
	registerFileRoutes(mux, s)
	return mux
}

func (s *ClassificationAPIServer) registerCoreRoutes(mux *http.ServeMux) {
	// Health check endpoint
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /ready", s.handleReady)

	// API discovery endpoint
	mux.HandleFunc("GET /api/v1", s.handleAPIOverview)

	// OpenAPI and documentation endpoints
	mux.HandleFunc("GET /openapi.json", s.handleOpenAPISpec)
	mux.HandleFunc("GET /docs", s.handleSwaggerUI)
}

func (s *ClassificationAPIServer) registerClassificationRoutes(mux *http.ServeMux) {
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
}

func (s *ClassificationAPIServer) registerEmbeddingRoutes(mux *http.ServeMux) {
	// Embedding endpoints
	mux.HandleFunc("POST /api/v1/embeddings", s.handleEmbeddings)
	mux.HandleFunc("POST /api/v1/similarity", s.handleSimilarity)
	mux.HandleFunc("POST /api/v1/similarity/batch", s.handleBatchSimilarity)
}

func (s *ClassificationAPIServer) registerInfoRoutes(mux *http.ServeMux) {
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
}

func (s *ClassificationAPIServer) registerConfigRoutes(mux *http.ServeMux) {
	// Configuration endpoints
	mux.HandleFunc("GET /config/kbs", s.handleListKnowledgeBases)
	mux.HandleFunc("POST /config/kbs", s.handleCreateKnowledgeBase)
	mux.HandleFunc("GET /config/kbs/{name}", s.handleGetKnowledgeBase)
	mux.HandleFunc("GET /config/kbs/{name}/map/metadata", s.handleGetKnowledgeBaseMapMetadata)
	mux.HandleFunc("GET /config/kbs/{name}/map/data.ndjson", s.handleGetKnowledgeBaseMapData)
	mux.HandleFunc("PUT /config/kbs/{name}", s.handleUpdateKnowledgeBase)
	mux.HandleFunc("DELETE /config/kbs/{name}", s.handleDeleteKnowledgeBase)
	mux.HandleFunc("GET /config/router", s.handleConfigGet)
	mux.HandleFunc("PATCH /config/router", s.handleConfigPatch)
	mux.HandleFunc("PUT /config/router", s.handleConfigPut)
	mux.HandleFunc("POST /config/router/rollback", s.handleConfigRollback)
	mux.HandleFunc("GET /config/router/versions", s.handleConfigVersions)
	mux.HandleFunc("GET /config/hash", s.handleConfigHash)
}

func (s *ClassificationAPIServer) registerMemoryRoutes(mux *http.ServeMux) {
	// Memory management endpoints
	mux.HandleFunc("GET /v1/memory/{id}", s.handleGetMemory)
	mux.HandleFunc("GET /v1/memory", s.handleListMemories)
	mux.HandleFunc("DELETE /v1/memory/{id}", s.handleDeleteMemory)
	mux.HandleFunc("DELETE /v1/memory", s.handleDeleteMemoriesByScope)
}

// handleHealth handles health check requests
func (s *ClassificationAPIServer) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status": "healthy", "service": "classification-api"}`))
}

// handleReady reports whether router startup has completed enough for traffic.
func (s *ClassificationAPIServer) handleReady(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	state, err := startupstatus.Load(startupstatus.StatusPathFromConfigPath(s.configPath))
	if err != nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"status":"starting","service":"classification-api","ready":false}`))
		return
	}

	if !state.Ready {
		s.writeJSONResponse(w, http.StatusServiceUnavailable, map[string]interface{}{
			"status":            "starting",
			"service":           "classification-api",
			"ready":             false,
			"phase":             state.Phase,
			"message":           state.Message,
			"downloading_model": state.DownloadingModel,
			"pending_models":    state.PendingModels,
			"ready_models":      state.ReadyModels,
			"total_models":      state.TotalModels,
		})
		return
	}

	s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"status":            "ready",
		"service":           "classification-api",
		"ready":             true,
		"phase":             state.Phase,
		"message":           state.Message,
		"downloading_model": state.DownloadingModel,
		"pending_models":    state.PendingModels,
		"ready_models":      state.ReadyModels,
		"total_models":      state.TotalModels,
	})
}

// Helper methods for JSON handling
func (s *ClassificationAPIServer) parseJSONRequest(r *http.Request, v interface{}) error {
	defer func() {
		_ = r.Body.Close()
	}()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return fmt.Errorf("failed to read request body: %w", err)
	}

	if err := json.Unmarshal(body, v); err != nil {
		return fmt.Errorf("failed to parse JSON: %w", err)
	}

	return nil
}

func (s *ClassificationAPIServer) writeJSONResponse(w http.ResponseWriter, statusCode int, data interface{}) {
	payload, err := json.Marshal(data)
	if err != nil {
		logging.Errorf("Failed to encode JSON response: %v", err)
		s.writeJSONEncodingError(w)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	if _, err := w.Write(append(payload, '\n')); err != nil {
		logging.Errorf("Failed to write JSON response: %v", err)
	}
}

func (s *ClassificationAPIServer) writeJSONEncodingError(w http.ResponseWriter) {
	payload, err := json.Marshal(map[string]interface{}{
		"error": map[string]interface{}{
			"code":      "JSON_ENCODE_ERROR",
			"message":   "failed to encode response",
			"timestamp": time.Now().UTC().Format(time.RFC3339),
		},
	})
	if err != nil {
		logging.Errorf("Failed to encode JSON error response: %v", err)
		http.Error(w, "failed to encode response", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusInternalServerError)
	if _, err := w.Write(append(payload, '\n')); err != nil {
		logging.Errorf("Failed to write JSON error response: %v", err)
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
