package ensembleserver

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ensemble"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// EnsembleServer handles OpenAI-compatible ensemble requests
type EnsembleServer struct {
	factory *ensemble.Factory
	config  *config.RouterConfig
}

// Init starts the ensemble API server
func Init(cfg *config.RouterConfig, port int) error {
	if cfg == nil {
		return fmt.Errorf("configuration not initialized")
	}

	if !cfg.Ensemble.Enabled {
		logging.Infof("Ensemble service is disabled in configuration")
		return nil
	}

	// Initialize ensemble factory
	ensembleConfig := &ensemble.Config{
		Enabled:               cfg.Ensemble.Enabled,
		DefaultStrategy:       ensemble.Strategy(cfg.Ensemble.DefaultStrategy),
		DefaultMinResponses:   cfg.Ensemble.DefaultMinResponses,
		TimeoutSeconds:        cfg.Ensemble.TimeoutSeconds,
		MaxConcurrentRequests: cfg.Ensemble.MaxConcurrentRequests,
	}
	factory := ensemble.NewFactory(ensembleConfig)

	// Register endpoint mappings from config
	for modelName, endpoint := range cfg.Ensemble.EndpointMappings {
		factory.RegisterEndpoint(modelName, endpoint)
	}

	server := &EnsembleServer{
		factory: factory,
		config:  cfg,
	}

	// Create HTTP server
	mux := server.setupRoutes()
	httpServer := &http.Server{
		Addr:         fmt.Sprintf(":%d", port),
		Handler:      mux,
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	logging.Infof("Ensemble API server listening on port %d", port)
	return httpServer.ListenAndServe()
}

// setupRoutes configures HTTP routes
func (s *EnsembleServer) setupRoutes() *http.ServeMux {
	mux := http.NewServeMux()
	
	// OpenAI-compatible endpoints
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/health", s.handleHealth)
	
	return mux
}

// handleHealth returns service health status
func (s *EnsembleServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "healthy",
		"service": "ensemble",
	})
}

// handleChatCompletions processes OpenAI chat completion requests with ensemble
func (s *EnsembleServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		logging.Errorf("Failed to read request body: %v", err)
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	// Parse ensemble headers
	ensembleEnabled := strings.ToLower(r.Header.Get(headers.EnsembleEnable)) == "true"
	if !ensembleEnabled {
		http.Error(w, "Ensemble not enabled in request headers", http.StatusBadRequest)
		return
	}

	// Parse models list
	modelsHeader := r.Header.Get(headers.EnsembleModels)
	if modelsHeader == "" {
		http.Error(w, "No models specified in ensemble header", http.StatusBadRequest)
		return
	}
	
	var models []string
	for _, model := range strings.Split(modelsHeader, ",") {
		trimmedModel := strings.TrimSpace(model)
		if trimmedModel != "" {
			models = append(models, trimmedModel)
		}
	}

	if len(models) == 0 {
		http.Error(w, "No valid models specified", http.StatusBadRequest)
		return
	}

	// Parse strategy
	strategy := ensemble.Strategy(r.Header.Get(headers.EnsembleStrategy))
	if strategy == "" {
		strategy = s.factory.GetDefaultStrategy()
	}

	// Parse min responses
	minResponses := s.factory.GetDefaultMinResponses()
	if minRespHeader := r.Header.Get(headers.EnsembleMinResponses); minRespHeader != "" {
		if parsed, err := strconv.Atoi(minRespHeader); err == nil && parsed > 0 {
			minResponses = parsed
		}
	}

	logging.Infof("Ensemble request: models=%v, strategy=%s, minResponses=%d", models, strategy, minResponses)

	// Forward headers for authentication
	headerMap := make(map[string]string)
	for key, values := range r.Header {
		if len(values) > 0 {
			headerMap[key] = values[0]
		}
	}

	// Build ensemble request
	ensembleReq := &ensemble.Request{
		Models:           models,
		Strategy:         strategy,
		MinResponses:     minResponses,
		OriginalRequest:  body,
		Headers:          headerMap,
		Context:          r.Context(),
	}

	// Execute ensemble orchestration
	ensembleResp := s.factory.Execute(ensembleReq)

	// Check for errors
	if ensembleResp.Error != nil {
		logging.Errorf("Ensemble execution failed: %v", ensembleResp.Error)
		http.Error(w, fmt.Sprintf("Ensemble orchestration failed: %v", ensembleResp.Error), http.StatusInternalServerError)
		return
	}

	// Add ensemble metadata headers
	w.Header().Set(headers.VSREnsembleUsed, "true")
	w.Header().Set(headers.VSREnsembleModelsQueried, strconv.Itoa(ensembleResp.ModelsQueried))
	w.Header().Set(headers.VSREnsembleResponsesReceived, strconv.Itoa(ensembleResp.ResponsesReceived))
	w.Header().Set("Content-Type", "application/json")

	// Return the aggregated response
	logging.Infof("Ensemble execution successful: queried=%d, received=%d, strategy=%s",
		ensembleResp.ModelsQueried, ensembleResp.ResponsesReceived, ensembleResp.Strategy)

	w.WriteHeader(http.StatusOK)
	w.Write(ensembleResp.FinalResponse)
}
