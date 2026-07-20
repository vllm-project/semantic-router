//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// Init starts the API server.
func Init(configPath string, port int) error {
	return InitWithRuntime(configPath, port, nil)
}

// InitWithRuntime starts the API server using the shared runtime registry when
// one is available. Legacy callers can continue using Init and fall back to the
// compatibility globals.
func InitWithRuntime(configPath string, port int, runtimeRegistry *routerruntime.Registry) error {
	// Get the global configuration instead of loading from file
	// This ensures we use the same config as the rest of the application
	cfg := resolveAPIServerConfig(runtimeRegistry)
	if cfg == nil {
		return fmt.Errorf("configuration not initialized")
	}

	classificationSvc := resolveClassificationService(cfg, runtimeRegistry)
	classificationSvc = ensureClassificationService(cfg, runtimeRegistry, classificationSvc)

	// The cross-encoder reranker is loaded on the model-runtime path
	// (see modelruntime.PrepareRouterRuntime). Here we only resolve the public
	// model id used to dispatch /v1/rerank requests to that reranker.
	configureCrossEncoderServedName()

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
		memoryStore = resolveMemoryStore(cfg, runtimeRegistry)
		if memoryStore != nil {
			logging.ComponentEvent("apiserver", "memory_api_enabled", map[string]interface{}{})
		} else {
			logging.ComponentWarnEvent("apiserver", "memory_api_degraded", map[string]interface{}{
				"reason": "memory_store_unavailable",
				"status": 503,
			})
		}
	} else {
		logging.ComponentEvent("apiserver", "memory_api_disabled", map[string]interface{}{
			"reason": "config_disabled",
		})
	}

	liveClassificationSvc := newLiveClassificationService(
		classificationSvc,
		buildClassificationResolver(runtimeRegistry),
	)

	// Create server instance
	apiServer := &ClassificationAPIServer{
		classificationSvc:     liveClassificationSvc,
		config:                cfg,
		runtimeConfig:         newLiveRuntimeConfig(cfg, buildConfigResolver(runtimeRegistry), buildConfigUpdater(runtimeRegistry, liveClassificationSvc)),
		runtimeRegistry:       runtimeRegistry,
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

	logging.ComponentEvent("apiserver", "server_listening", map[string]interface{}{
		"port": port,
	})
	return server.ListenAndServe()
}

// configureCrossEncoderServedName resolves the public model id that maps to the
// loaded cross-encoder reranker for /v1/rerank dispatch. The reranker itself is
// loaded on the model-runtime path (see modelruntime.PrepareRouterRuntime);
// SR_CROSS_ENCODER_MODEL_NAME optionally sets the served id clients use to
// request it (e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"); the "cross-encoder"
// alias always works regardless. Cleared first so a missing value never leaves
// stale dispatch state.
//
// INTERIM: name resolution is still env-based; it moves into model_catalog with
// the rest of the rerank config in the planned follow-up. See the PR/issue.
func configureCrossEncoderServedName() {
	crossEncoderServedName = strings.TrimSpace(os.Getenv("SR_CROSS_ENCODER_MODEL_NAME"))
}

func resolveAPIServerConfig(runtimeRegistry *routerruntime.Registry) *config.RouterConfig {
	if runtimeRegistry != nil {
		return runtimeRegistry.CurrentConfig()
	}
	return config.Get()
}

func resolveClassificationService(
	cfg *config.RouterConfig,
	runtimeRegistry *routerruntime.Registry,
) *services.ClassificationService {
	if runtimeRegistry != nil {
		return runtimeRegistry.ClassificationService()
	}
	return initClassify(5, 500*time.Millisecond)
}

func ensureClassificationService(
	cfg *config.RouterConfig,
	runtimeRegistry *routerruntime.Registry,
	svc *services.ClassificationService,
) *services.ClassificationService {
	if svc != nil {
		return svc
	}

	if runtimeRegistry != nil {
		logging.ComponentEvent("apiserver", "classification_service_waiting_for_runtime", map[string]interface{}{
			"using_placeholder": true,
		})
		return services.NewPlaceholderClassificationService()
	}

	// If no global service exists, try auto-discovery unified classifier.
	logging.ComponentEvent("apiserver", "classification_service_autodiscovery_started", map[string]interface{}{})
	autoSvc, err := services.NewClassificationServiceWithAutoDiscovery(cfg)
	if err != nil {
		logging.ComponentWarnEvent("apiserver", "classification_service_autodiscovery_failed", map[string]interface{}{
			"error":             err.Error(),
			"using_placeholder": true,
		})
		return services.NewPlaceholderClassificationService()
	}

	logging.ComponentEvent("apiserver", "classification_service_autodiscovery_succeeded", map[string]interface{}{})
	return autoSvc
}

func resolveMemoryStore(cfg *config.RouterConfig, runtimeRegistry *routerruntime.Registry) memory.Store {
	if runtimeRegistry != nil {
		return runtimeRegistry.MemoryStore()
	}
	return initMemoryStore(5, 500*time.Millisecond)
}

// buildClassificationResolver returns the resolver used by the live
// classification service. Both sources return a concrete
// *services.ClassificationService which is nil until the runtime finishes
// initializing; returning that nil pointer directly would wrap it into a
// non-nil interface value, defeat the nil check in
// liveClassificationService.current(), and panic with a nil receiver on the
// first request. Return an untyped nil instead so current() falls back to
// the placeholder service.
func buildClassificationResolver(runtimeRegistry *routerruntime.Registry) func() classificationService {
	return func() classificationService {
		if runtimeRegistry != nil {
			if svc := runtimeRegistry.ClassificationService(); svc != nil {
				return svc
			}
			return nil
		}
		if svc := services.GetGlobalClassificationService(); svc != nil {
			return svc
		}
		return nil
	}
}

func buildConfigResolver(runtimeRegistry *routerruntime.Registry) func() *config.RouterConfig {
	if runtimeRegistry == nil {
		return config.Get
	}
	return runtimeRegistry.CurrentConfig
}

func buildConfigUpdater(
	runtimeRegistry *routerruntime.Registry,
	liveClassificationSvc classificationService,
) func(*config.RouterConfig) {
	if runtimeRegistry == nil {
		return func(newCfg *config.RouterConfig) {
			if liveClassificationSvc != nil {
				liveClassificationSvc.RefreshRuntimeConfig(newCfg)
			}
			config.Replace(newCfg)
		}
	}
	return runtimeRegistry.RefreshRuntimeConfig
}

// initClassify attempts to get the global classification service with retry logic
func initClassify(maxRetries int, retryInterval time.Duration) *services.ClassificationService {
	for i := 0; i < maxRetries; i++ {
		if svc := services.GetGlobalClassificationService(); svc != nil {
			return svc
		}

		if i < maxRetries-1 { // Don't sleep on the last attempt
			logging.ComponentDebugEvent("apiserver", "classification_service_retry_pending", map[string]interface{}{
				"retry_interval_ms": retryInterval.Milliseconds(),
				"attempt":           i + 1,
				"max_retries":       maxRetries,
			})
			time.Sleep(retryInterval)
		}
	}

	logging.ComponentWarnEvent("apiserver", "classification_service_unavailable", map[string]interface{}{
		"max_retries": maxRetries,
	})
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
			logging.ComponentDebugEvent("apiserver", "memory_store_retry_pending", map[string]interface{}{
				"retry_interval_ms": retryInterval.Milliseconds(),
				"attempt":           i + 1,
				"max_retries":       maxRetries,
			})
			time.Sleep(retryInterval)
		}
	}

	logging.ComponentWarnEvent("apiserver", "memory_store_unavailable", map[string]interface{}{
		"max_retries": maxRetries,
	})
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
	for _, route := range apiRoutes() {
		mux.HandleFunc(route.pattern(), route.bind(s))
	}
	return mux
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

	state := s.loadStartupState()
	if state == nil {
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
