package http

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/router/engine"
)

type Adapter struct {
	engine *engine.RouterEngine
	server *http.Server
	port   int
}

// NewAdapter creates a new HTTP adapter
func NewAdapter(eng *engine.RouterEngine, port int) (*Adapter, error) {
	adapter := &Adapter{
		engine: eng,
		port:   port,
	}

	mux := http.NewServeMux()
	adapter.registerRoutes(mux)

	adapter.server = &http.Server{
		Addr:         fmt.Sprintf(":%d", port),
		Handler:      mux,
		ReadTimeout:  300 * time.Second,
		WriteTimeout: 300 * time.Second,
		IdleTimeout:  300 * time.Second,
	}

	return adapter, nil
}

func (a *Adapter) registerRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/chat/completions", a.handleChatCompletions)
	mux.HandleFunc("/v1/completions", a.handleCompletions)
	mux.HandleFunc("/v1/models", a.handleModels)
	mux.HandleFunc("/v1/router_replay", a.handleRouterReplay)
	mux.HandleFunc("/v1/router_replay/", a.handleRouterReplayDetail)
	mux.HandleFunc("/v1/responses", a.handleResponsesCreate)
	mux.HandleFunc("/v1/responses/", a.handleResponsesDetail)
	mux.HandleFunc("/health", a.handleHealth)
	mux.HandleFunc("/ready", a.handleReady)
	mux.HandleFunc("/v1/classify", a.handleClassify)
	mux.HandleFunc("/v1/route", a.handleRoute)
}

// ChatCompletionRequest represents an OpenAI chat completion request
type ChatCompletionRequest struct {
	Model    string           `json:"model"`
	Messages []engine.Message `json:"messages"`
	Stream   bool             `json:"stream,omitempty"`
	User     string           `json:"user,omitempty"`
}

// ChatCompletionResponse represents an OpenAI chat completion response
type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
}

// ChatCompletionChoice represents a completion choice
type ChatCompletionChoice struct {
	Index        int            `json:"index"`
	Message      engine.Message `json:"message"`
	FinishReason string         `json:"finish_reason"`
}

// ErrorResponse represents an API error response
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail contains error details
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

// handleChatCompletions handles POST /v1/chat/completions
// This is now a thin translation layer that delegates to RouterEngine.Route()
func (a *Adapter) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		a.writeError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	// 1. Parse HTTP request
	body, err := io.ReadAll(r.Body)
	if err != nil {
		a.writeError(w, http.StatusBadRequest, "Failed to read request body")
		return
	}
	defer r.Body.Close()

	var req ChatCompletionRequest
	if unmarshalErr := json.Unmarshal(body, &req); unmarshalErr != nil {
		a.writeError(w, http.StatusBadRequest, "Invalid JSON request")
		return
	}

	// 2. Convert to RouteRequest
	routeReq := &engine.RouteRequest{
		Model:    req.Model,
		Messages: req.Messages,
		User:     req.User,
		Headers:  make(map[string]string),
		Context:  r.Context(),
	}

	// Copy relevant headers
	for key := range r.Header {
		routeReq.Headers[key] = r.Header.Get(key)
	}

	// 3. Call RouterEngine (single source of truth for routing)
	routeResp, err := a.engine.Route(r.Context(), routeReq)
	if err != nil {
		logging.Errorf("Routing failed: %v", err)
		a.writeError(w, http.StatusInternalServerError, fmt.Sprintf("Routing failed: %v", err))
		return
	}

	// 4. Handle cache hit
	if routeResp.CacheHit {
		w.Header().Set("X-Cache", "HIT")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if _, err := w.Write([]byte(routeResp.CachedResponse)); err != nil {
			logging.Errorf("Failed to write cached response: %v", err)
		}
		return
	}

	// 5. Handle blocked request
	if routeResp.Blocked {
		a.writeError(w, http.StatusForbidden, routeResp.BlockReason)
		return
	}

	// 6. Set routing metadata headers
	w.Header().Set("X-Router-Decision", routeResp.DecisionName)
	w.Header().Set("X-Router-Model", routeResp.SelectedModel)
	w.Header().Set("X-Router-Confidence", fmt.Sprintf("%.2f", routeResp.Confidence))
	if routeResp.ReplayID != "" {
		w.Header().Set("X-Replay-ID", routeResp.ReplayID)
	}

	// 7. Return backend response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(routeResp.StatusCode)
	if _, err := w.Write(routeResp.ResponseBody); err != nil {
		logging.Errorf("Failed to write response body: %v", err)
	}
}

// handleCompletions handles POST /v1/completions
func (a *Adapter) handleCompletions(w http.ResponseWriter, r *http.Request) {
	a.writeError(w, http.StatusNotImplemented, "Completions endpoint not yet implemented")
}

// handleModels handles GET /v1/models
func (a *Adapter) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		a.writeError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	now := time.Now().Unix()
	models := make([]map[string]interface{}, 0)

	// Add the configured auto model name (default "MoM")
	effectiveAutoModelName := a.engine.Config.GetEffectiveAutoModelName()
	models = append(models, map[string]interface{}{
		"id":          effectiveAutoModelName,
		"object":      "model",
		"created":     now,
		"owned_by":    "vllm-semantic-router",
		"description": "Intelligent Router for Mixture-of-Models",
		"logo_url":    "https://github.com/vllm-project/semantic-router/blob/main/website/static/img/vllm.png",
	})

	// Optionally include underlying models from config (if configured)
	if a.engine.Config.IncludeConfigModelsInList {
		seen := make(map[string]bool)
		seen[effectiveAutoModelName] = true // Avoid duplicates

		for _, model := range a.engine.Config.GetAllModels() {
			if !seen[model] {
				seen[model] = true
				models = append(models, map[string]interface{}{
					"id":       model,
					"object":   "model",
					"created":  now,
					"owned_by": "vllm-semantic-router",
				})
			}
		}
	}

	response := map[string]interface{}{
		"object": "list",
		"data":   models,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		logging.Errorf("Failed to encode models response: %v", err)
	}
}

// handleHealth handles GET /health
func (a *Adapter) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]string{
		"status": "healthy",
	}); err != nil {
		logging.Errorf("Failed to encode health response: %v", err)
	}
}

// handleReady handles GET /ready
func (a *Adapter) handleReady(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]string{
		"status": "ready",
	}); err != nil {
		logging.Errorf("Failed to encode ready response: %v", err)
	}
}

// handleClassify handles POST /v1/classify
func (a *Adapter) handleClassify(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		a.writeError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	var req struct {
		Text string `json:"text"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		a.writeError(w, http.StatusBadRequest, "Invalid JSON request")
		return
	}

	// Convert text to messages format
	messages := []engine.Message{
		{Role: "user", Content: req.Text},
	}

	result, err := a.engine.ClassifyRequest(r.Context(), messages)
	if err != nil {
		logging.Errorf("Classification error: %v", err)
		a.writeError(w, http.StatusInternalServerError, "Classification failed")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err = json.NewEncoder(w).Encode(result); err != nil {
		logging.Errorf("Failed to encode classification result: %v", err)
	}
}

// handleRoute handles POST /v1/route
func (a *Adapter) handleRoute(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		a.writeError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		a.writeError(w, http.StatusBadRequest, "Invalid JSON request")
		return
	}

	routeReq := &engine.RouteRequest{
		Model:    req.Model,
		Messages: req.Messages,
		User:     req.User,
		Headers:  convertHeaders(r.Header),
		Context:  r.Context(),
	}

	routeResp, err := a.engine.Route(r.Context(), routeReq)
	if err != nil {
		a.writeError(w, http.StatusInternalServerError, "Routing failed")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err = json.NewEncoder(w).Encode(routeResp); err != nil {
		logging.Errorf("Failed to encode route response: %v", err)
	}
}

// handleRouterReplay handles GET /v1/router_replay
func (a *Adapter) handleRouterReplay(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		a.writeError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	// Check if any recorders are initialized
	hasRecorders := len(a.engine.ReplayRecorders) > 0
	if !hasRecorders {
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"object":  "router_replay.list",
			"count":   0,
			"data":    []interface{}{},
			"message": "Router replay not configured",
		}); err != nil {
			logging.Errorf("Failed to encode empty replay response: %v", err)
		}
		return
	}

	// Aggregate records from all recorders
	var allRecords []interface{}
	for _, recorder := range a.engine.ReplayRecorders {
		if recorder != nil {
			records := recorder.ListAllRecords()
			for _, r := range records {
				allRecords = append(allRecords, r)
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "router_replay.list",
		"count":  len(allRecords),
		"data":   allRecords,
	}); err != nil {
		logging.Errorf("Failed to encode replay list: %v", err)
	}
}

// handleRouterReplayDetail handles GET /v1/router_replay/{id}
func (a *Adapter) handleRouterReplayDetail(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		a.writeError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	// Extract ID from path
	path := r.URL.Path
	replayID := strings.TrimPrefix(path, "/v1/router_replay/")
	if replayID == "" || replayID == "/" {
		a.writeError(w, http.StatusBadRequest, "replay id is required")
		return
	}

	// Search in all recorders
	for _, recorder := range a.engine.ReplayRecorders {
		if recorder != nil {
			if rec, ok := recorder.GetRecord(replayID); ok {
				w.Header().Set("Content-Type", "application/json")
				if err := json.NewEncoder(w).Encode(rec); err != nil {
					logging.Errorf("Failed to encode replay record: %v", err)
				}
				return
			}
		}
	}

	a.writeError(w, http.StatusNotFound, "replay record not found")
}

// handleResponsesCreate handles POST /v1/responses
func (a *Adapter) handleResponsesCreate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		a.writeError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	a.writeError(w, http.StatusNotImplemented, "Response API not yet implemented in HTTP adapter")
}

// handleResponsesDetail handles GET/DELETE /v1/responses/{id} and GET /v1/responses/{id}/input_items
func (a *Adapter) handleResponsesDetail(w http.ResponseWriter, r *http.Request) {
	a.writeError(w, http.StatusNotImplemented, "Response API not yet implemented in HTTP adapter")
}

// writeError writes an error response
func (a *Adapter) writeError(w http.ResponseWriter, statusCode int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	if err := json.NewEncoder(w).Encode(ErrorResponse{
		Error: ErrorDetail{
			Message: message,
			Type:    "invalid_request_error",
		},
	}); err != nil {
		logging.Errorf("Failed to encode error response: %v", err)
	}
}

// convertHeaders converts http.Header to map[string]string
func convertHeaders(headers http.Header) map[string]string {
	result := make(map[string]string)
	for k, v := range headers {
		if len(v) > 0 {
			result[k] = v[0]
		}
	}
	return result
}

func (a *Adapter) Start() error {
	logging.Infof("Starting HTTP adapter on port %d", a.port)
	if err := a.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("HTTP server error: %w", err)
	}
	return nil
}

// Stop gracefully stops the HTTP server
func (a *Adapter) Stop() error {
	logging.Infof("Stopping HTTP adapter")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return a.server.Shutdown(ctx)
}

// GetEngine returns the underlying router engine
func (a *Adapter) GetEngine() *engine.RouterEngine {
	return a.engine
}
