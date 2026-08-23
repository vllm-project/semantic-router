//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

const (
	apiReadTimeout  = 30 * time.Second
	apiWriteTimeout = 2 * time.Minute
	apiIdleTimeout  = 60 * time.Second
)

// Init starts the API server.
func Init(configPath string, port int) error {
	return InitWithOptions(InitOptions{
		ConfigPath: configPath,
		Port:       port,
	})
}

// InitOptions carries management listener startup overrides.
type InitOptions struct {
	Context         context.Context
	OnListenerStart func(error)
	ConfigPath      string
	Port            int
	BindAddress     string
	RemoteExposure  *bool
	AuthMode        string
	RuntimeRegistry *routerruntime.Registry
	ManagedAPI      ManagedAPI
}

// InitWithRuntime starts the API server using the shared runtime registry when
// one is available.
func InitWithRuntime(configPath string, port int, runtimeRegistry *routerruntime.Registry) error {
	return InitWithOptions(InitOptions{
		ConfigPath:      configPath,
		Port:            port,
		RuntimeRegistry: runtimeRegistry,
	})
}

// InitWithOptions starts the API server with explicit management listener policy.
func InitWithOptions(opts InitOptions) (resultErr error) {
	listenerStarted := false
	defer func() {
		if !listenerStarted && opts.OnListenerStart != nil {
			opts.OnListenerStart(resultErr)
		}
	}()
	// Get the global configuration instead of loading from file
	// This ensures we use the same config as the rest of the application
	cfg := resolveAPIServerConfig(opts.RuntimeRegistry)
	if cfg == nil {
		return fmt.Errorf("configuration not initialized")
	}

	managementCfg, err := cfg.ManagementAPI.ResolvedManagementAPI(config.ManagementAPIRuntimeOptions{
		ControlPlaneMode: cfg.ControlPlane.Mode,
		Port:             opts.Port,
		BindAddress:      opts.BindAddress,
		RemoteExposure:   opts.RemoteExposure,
		AuthMode:         opts.AuthMode,
	})
	if err != nil {
		return fmt.Errorf("invalid management API configuration: %w", err)
	}
	cfg.ManagementAPI = managementCfg
	managedMode := cfg.ControlPlane.Mode == config.ControlPlaneModeManaged
	if managedMode && opts.ManagedAPI == nil {
		return fmt.Errorf("managed control-plane mode requires a Router-native Management API")
	}
	if !managedMode && opts.ManagedAPI != nil {
		return fmt.Errorf("standalone control-plane mode rejects a Router-native Management API")
	}
	var listenerTLS *managementListenerTLS
	if managedMode {
		listenerTLS, err = loadManagementListenerTLS(managementCfg.TLS, time.Now())
		if err != nil {
			return fmt.Errorf("initialize managed Management listener TLS: %w", err)
		}
	}

	classificationSvc := resolveClassificationService(cfg, opts.RuntimeRegistry)
	classificationSvc = ensureClassificationService(opts.RuntimeRegistry, classificationSvc)

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
		memoryStore = resolveMemoryStore(cfg, opts.RuntimeRegistry)
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
		buildClassificationResolver(opts.RuntimeRegistry),
	)

	// Create server instance
	apiServer := &ClassificationAPIServer{
		classificationSvc:   liveClassificationSvc,
		config:              cfg,
		runtimeConfig:       newLiveRuntimeConfig(cfg, buildConfigResolver(opts.RuntimeRegistry)),
		runtimeRegistry:     opts.RuntimeRegistry,
		configPath:          opts.ConfigPath,
		memoryStore:         memoryStore,
		startupStatusConfig: &cfg.StartupStatus,
		managedAPI:          opts.ManagedAPI,
		managementTLS:       listenerTLS,
	}

	// Create HTTP server with routes
	mux := apiServer.setupRoutes()
	server := &http.Server{
		Addr:         managementCfg.ListenAddress(),
		Handler:      mux,
		ReadTimeout:  apiReadTimeout,
		WriteTimeout: apiWriteTimeout,
		IdleTimeout:  apiIdleTimeout,
	}
	if listenerTLS != nil {
		server.TLSConfig = listenerTLS.config
	}

	logging.ComponentEvent("apiserver", "server_listening", map[string]interface{}{
		"address":         managementCfg.ListenAddress(),
		"bind_address":    managementCfg.BindAddress,
		"port":            managementCfg.Port,
		"remote_exposure": managementCfg.RemoteExposure,
		"auth_mode":       managementCfg.Auth.Mode,
		"transport":       managementListenerTransport(managedMode),
	})
	runtimeContext := opts.Context
	if runtimeContext == nil {
		runtimeContext = context.Background()
	}
	listenerContext, stopListener := context.WithCancel(runtimeContext)
	var tlsReloadDone <-chan struct{}
	if listenerTLS != nil {
		tlsReloadDone = listenerTLS.Watch(listenerContext)
	}
	err = serveManagementListener(listenerContext, server, managedMode, func() {
		listenerStarted = true
		if opts.OnListenerStart != nil {
			opts.OnListenerStart(nil)
		}
	})
	stopListener()
	if tlsReloadDone != nil {
		<-tlsReloadDone
	}
	return err
}

func managementListenerTransport(managed bool) string {
	if managed {
		return "tls"
	}
	return "plaintext"
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
	service, err := services.NewStandaloneClassificationService(cfg)
	if err != nil {
		logging.ComponentWarnEvent("apiserver", "standalone_classification_service_failed", map[string]interface{}{
			"error": err.Error(),
		})
		return nil
	}
	return service
}

func ensureClassificationService(
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

	logging.ComponentWarnEvent("apiserver", "standalone_classification_service_unavailable", map[string]interface{}{
		"using_placeholder": true,
	})
	return services.NewPlaceholderClassificationService()
}

func resolveMemoryStore(cfg *config.RouterConfig, runtimeRegistry *routerruntime.Registry) memory.Store {
	if runtimeRegistry != nil {
		return runtimeRegistry.MemoryStore()
	}
	return initMemoryStore(5, 500*time.Millisecond)
}

// buildClassificationResolver follows published runtime snapshots when a
// Registry exists. Standalone composition is immutable and remains the live
// service's initial value.
func buildClassificationResolver(runtimeRegistry *routerruntime.Registry) func() classificationService {
	return func() classificationService {
		if runtimeRegistry != nil {
			if svc := runtimeRegistry.ClassificationService(); svc != nil {
				return svc
			}
			return nil
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
	return s.setupListenerRoutes()
}

// handleHealth handles health check requests
func (s *ClassificationAPIServer) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status": "healthy", "service": "classification-api"}`))
}

// handleReady reports whether router startup has completed enough for traffic.
func (s *ClassificationAPIServer) handleReady(w http.ResponseWriter, request *http.Request) {
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
	if s.controlPlaneMode() == config.ControlPlaneModeManaged {
		if err := s.managementTLS.Ready(time.Now()); err != nil {
			s.writeJSONResponse(w, http.StatusServiceUnavailable, map[string]interface{}{
				"status": "starting", "service": "classification-api", "ready": false,
				"phase": "management_tls", "message": "Management listener TLS is not ready.",
			})
			return
		}
	}
	if s.managedAPI != nil {
		if err := s.managedAPI.Ready(request.Context()); err != nil {
			s.writeJSONResponse(w, http.StatusServiceUnavailable, map[string]interface{}{
				"status": "starting", "service": "classification-api", "ready": false,
				"phase": "managed_control_plane", "message": "Managed control plane is not ready.",
			})
			return
		}
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
			"message":   scrubSecretsInErrorMessage(message),
			"timestamp": time.Now().UTC().Format(time.RFC3339),
		},
	}

	s.writeJSONResponse(w, statusCode, errorResponse)
}
