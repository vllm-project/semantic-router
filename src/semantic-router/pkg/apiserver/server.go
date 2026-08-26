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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimecapabilities"
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
	Context          context.Context
	OnListenerStart  func(error)
	ConfigPath       string
	Port             int
	BindAddress      string
	RemoteExposure   *bool
	AuthMode         string
	RuntimeRegistry  *routerruntime.Registry
	ManagementAPI    ManagementAPI
	RuntimeReadiness RuntimeReadiness
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
	apiServer, managementCfg, err := prepareClassificationAPIServer(opts)
	if err != nil {
		return err
	}
	server := &http.Server{
		Addr:         managementCfg.ListenAddress(),
		Handler:      apiServer.setupRoutes(),
		ReadTimeout:  apiReadTimeout,
		WriteTimeout: apiWriteTimeout,
		IdleTimeout:  apiIdleTimeout,
	}
	if apiServer.managementTLS != nil {
		server.TLSConfig = apiServer.managementTLS.config
	}

	logging.ComponentEvent("apiserver", "server_listening", map[string]interface{}{
		"address":         managementCfg.ListenAddress(),
		"bind_address":    managementCfg.BindAddress,
		"port":            managementCfg.Port,
		"remote_exposure": managementCfg.RemoteExposure,
		"auth_mode":       managementCfg.Auth.Mode,
		"transport":       managementListenerTransport(apiServer.managementTLS),
	})
	runtimeContext := opts.Context
	if runtimeContext == nil {
		runtimeContext = context.Background()
	}
	listenerContext, stopListener := context.WithCancel(runtimeContext)
	var tlsReloadDone <-chan struct{}
	if apiServer.managementTLS != nil {
		tlsReloadDone = apiServer.managementTLS.Watch(listenerContext)
	}
	err = serveManagementListener(listenerContext, server, func() {
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

func prepareClassificationAPIServer(
	opts InitOptions,
) (*ClassificationAPIServer, config.ManagementAPIConfig, error) {
	// Resolve the live Router generation so the API and data plane share one
	// configuration authority.
	cfg := resolveAPIServerConfig(opts.RuntimeRegistry)
	if cfg == nil {
		return nil, config.ManagementAPIConfig{}, fmt.Errorf("configuration not initialized")
	}
	capabilities, err := runtimecapabilities.Derive(cfg)
	if err != nil {
		return nil, config.ManagementAPIConfig{}, fmt.Errorf("derive runtime capabilities: %w", err)
	}

	managementCfg, err := cfg.ManagementAPI.ResolvedManagementAPI(config.ManagementAPIRuntimeOptions{
		DurableRouting: capabilities.DurableRouting,
		Port:           opts.Port,
		BindAddress:    opts.BindAddress,
		RemoteExposure: opts.RemoteExposure,
		AuthMode:       opts.AuthMode,
	})
	if err != nil {
		return nil, config.ManagementAPIConfig{}, fmt.Errorf("invalid management API configuration: %w", err)
	}
	cfg.ManagementAPI = managementCfg
	if capabilities.ManagementAPI && opts.ManagementAPI == nil {
		return nil, config.ManagementAPIConfig{}, fmt.Errorf("enabled Management API requires a Router-native Management application")
	}
	if !capabilities.ManagementAPI && opts.ManagementAPI != nil {
		return nil, config.ManagementAPIConfig{}, fmt.Errorf("disabled Management API rejects a Router-native Management application")
	}
	runtimeReadiness := opts.RuntimeReadiness
	if runtimeReadiness == nil && opts.ManagementAPI != nil {
		// A production ManagementAPI is already the aggregate runtime wrapper.
		runtimeReadiness = opts.ManagementAPI
	}
	if capabilities.DurableRouting && runtimeReadiness == nil {
		return nil, config.ManagementAPIConfig{}, fmt.Errorf("durable routing requires process runtime readiness")
	}
	var listenerTLS *managementListenerTLS
	if capabilities.ManagementAPI {
		listenerTLS, err = loadManagementListenerTLS(managementCfg.TLS, time.Now())
		if err != nil {
			return nil, config.ManagementAPIConfig{}, fmt.Errorf("initialize Management API listener TLS: %w", err)
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

	apiServer := &ClassificationAPIServer{
		classificationSvc:   liveClassificationSvc,
		config:              cfg,
		runtimeConfig:       newLiveRuntimeConfig(cfg, buildConfigResolver(opts.RuntimeRegistry)),
		runtimeRegistry:     opts.RuntimeRegistry,
		capabilities:        capabilities,
		configPath:          opts.ConfigPath,
		memoryStore:         memoryStore,
		startupStatusConfig: &cfg.StartupStatus,
		managementAPI:       opts.ManagementAPI,
		runtimeReadiness:    runtimeReadiness,
		managementTLS:       listenerTLS,
	}
	return apiServer, managementCfg, nil
}

func managementListenerTransport(listenerTLS *managementListenerTLS) string {
	if listenerTLS != nil {
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
	service, err := services.NewFileClassificationService(cfg)
	if err != nil {
		logging.ComponentWarnEvent("apiserver", "file_classification_service_failed", map[string]interface{}{
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

	logging.ComponentWarnEvent("apiserver", "file_classification_service_unavailable", map[string]interface{}{
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
// Registry exists. File-authoritative composition is immutable and remains the live
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
	if s.capabilities.ManagementAPI {
		if s.managementTLS == nil {
			s.writeJSONResponse(w, http.StatusServiceUnavailable, map[string]interface{}{
				"status": "starting", "service": "classification-api", "ready": false,
				"phase": "management_tls", "message": "Management listener TLS is unavailable.",
			})
			return
		}
		if err := s.managementTLS.Ready(time.Now()); err != nil {
			s.writeJSONResponse(w, http.StatusServiceUnavailable, map[string]interface{}{
				"status": "starting", "service": "classification-api", "ready": false,
				"phase": "management_tls", "message": "Management listener TLS is not ready.",
			})
			return
		}
	}
	readiness := s.runtimeReadiness
	if readiness == nil && s.managementAPI != nil {
		readiness = s.managementAPI
	}
	if s.capabilities.DurableRouting && readiness == nil {
		s.writeJSONResponse(w, http.StatusServiceUnavailable, map[string]interface{}{
			"status": "starting", "service": "classification-api", "ready": false,
			"phase": "routing_runtime", "message": "Routing runtime is not ready.",
		})
		return
	}
	if readiness != nil {
		if err := readiness.Ready(request.Context()); err != nil {
			s.writeJSONResponse(w, http.StatusServiceUnavailable, map[string]interface{}{
				"status": "starting", "service": "classification-api", "ready": false,
				"phase": "routing_runtime", "message": "Routing runtime is not ready.",
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
