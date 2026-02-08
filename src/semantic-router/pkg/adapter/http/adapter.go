package http

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/router/engine"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
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
func (a *Adapter) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		a.writeError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	// Parse request body
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

	originalModel := req.Model
	classificationSvc := services.GetGlobalClassificationService()
	if classificationSvc == nil || !classificationSvc.HasClassifier() {
		a.writeError(w, http.StatusInternalServerError, "Classification service not initialized")
		return
	}

	var userMessage string
	for _, msg := range req.Messages {
		if msg.Role == "user" {
			userMessage = msg.Content
			break
		}
	}

	classifyReq := services.IntentRequest{
		Text: userMessage,
		Options: &services.IntentOptions{
			ReturnProbabilities: false,
		},
	}

	classifyResp, err := classificationSvc.ClassifyIntent(classifyReq)
	if err != nil {
		logging.Errorf("Classification error: %v", err)
		a.writeError(w, http.StatusInternalServerError, "Classification failed")
		return
	}
	decisionName := classifyResp.RoutingDecision
	if decisionName == "" {
		decisionName = "general_decision"
	}

	decision := a.findDecision(decisionName)
	if decision == nil {
		req.Model = a.engine.Config.DefaultModel
	} else {
		// Check PII if enabled for this decision
		if piiConfig := decision.GetPIIConfig(); piiConfig != nil && piiConfig.Enabled {
			detectedPII := a.engine.Classifier.DetectPIIInContent([]string{userMessage})
			if len(detectedPII) > 0 {
				allowed, deniedPII, piiErr := a.engine.PIIChecker.CheckPolicy(decisionName, detectedPII)
				if piiErr != nil {
					logging.Errorf("Error checking PII policy: %v", piiErr)
				} else if !allowed {
					logging.Warnf("PII policy violation for decision %s, blocking request", decisionName)
					a.writeError(w, http.StatusForbidden, fmt.Sprintf("Request blocked due to PII policy violation: %v", deniedPII))
					return
				}
			}
		}

		// Check jailbreak if enabled for this decision
		if jailbreakConfig := decision.GetJailbreakConfig(); jailbreakConfig != nil && jailbreakConfig.Enabled {
			isJailbreak, jailbreakType, confidence, jailbreakErr := a.engine.Classifier.CheckForJailbreak(userMessage)
			if jailbreakErr != nil {
				logging.Errorf("Error checking for jailbreak: %v", jailbreakErr)
			} else if isJailbreak {
				logging.Warnf("Jailbreak detected for decision %s: %s (confidence: %.2f)",
					decisionName, jailbreakType, confidence)
				a.writeError(w, http.StatusForbidden, "Request blocked due to jailbreak detection")
				return
			}
		}

		// Get selected model from decision
		if len(decision.ModelRefs) > 0 {
			modelRef := decision.ModelRefs[0]
			if modelRef.LoRAName != "" {
				req.Model = modelRef.LoRAName
			} else {
				req.Model = modelRef.Model
			}

			// Inject system prompt if configured
			if systemPromptConfig := decision.GetSystemPromptConfig(); systemPromptConfig != nil && systemPromptConfig.SystemPrompt != "" {
				req.Messages = a.injectSystemPrompt(req.Messages, systemPromptConfig.SystemPrompt)
			}
		}
	}

	// Get endpoint for selected model
	selectedEndpoint := a.getEndpointForModel(req.Model)
	if selectedEndpoint == "" {
		logging.Errorf("No endpoint found for model: %s", req.Model)
		a.writeError(w, http.StatusInternalServerError, "No backend endpoint configured")
		return
	}

	// Start router replay if plugin is enabled for this decision (like ExtProc does)
	var replayID string
	var recorder *routerreplay.Recorder
	if decision != nil {
		if replayConfig := decision.GetRouterReplayConfig(); replayConfig != nil && replayConfig.Enabled {
			recorder = a.engine.ReplayRecorders[decisionName]
			if recorder != nil {
				// Set capture policy like ExtProc does
				maxBodyBytes := replayConfig.MaxBodyBytes
				if maxBodyBytes <= 0 {
					maxBodyBytes = routerreplay.DefaultMaxBodyBytes
				}
				recorder.SetCapturePolicy(
					replayConfig.CaptureRequestBody,
					replayConfig.CaptureResponseBody,
					maxBodyBytes,
				)

				// Create initial record like ExtProc startRouterReplay
				reqBytes, _ := json.Marshal(req)
				record := routerreplay.RoutingRecord{
					RequestID:       fmt.Sprintf("http_%d", time.Now().UnixNano()),
					Decision:        decisionName,
					Category:        classifyResp.Classification.Category,
					OriginalModel:   originalModel,
					SelectedModel:   req.Model,
					ConfidenceScore: classifyResp.Classification.Confidence,
					RequestBody:     string(reqBytes),
					Streaming:       req.Stream,
				}

				id, addErr := recorder.AddRecord(record)
				if addErr != nil {
					logging.Warnf("Failed to start replay recording: %v", addErr)
				} else {
					replayID = id
					logging.Infof("Started replay recording for decision %s: %s", decisionName, replayID)
				}
			}
		}
	}

	// Proxy to backend
	backendResp, err := a.proxyToBackend(r.Context(), selectedEndpoint, req)
	if err != nil {
		logging.Errorf("Backend proxy error: %v", err)
		a.writeError(w, http.StatusBadGateway, "Backend request failed")

		if replayID != "" && recorder != nil {
			if updateErr := recorder.UpdateStatus(replayID, 502, false, req.Stream); updateErr != nil {
				logging.Warnf("Failed to update replay status: %v", updateErr)
			}
		}
		return
	}

	if replayID != "" && recorder != nil {
		if attachErr := recorder.AttachResponse(replayID, backendResp); attachErr != nil {
			logging.Warnf("Failed to attach replay response: %v", attachErr)
		}
		if updateErr := recorder.UpdateStatus(replayID, 200, false, req.Stream); updateErr != nil {
			logging.Warnf("Failed to update replay status: %v", updateErr)
		}
		logging.Infof("Updated replay record %s with response", replayID)
	}

	// Set routing headers
	w.Header().Set("X-Router-Decision", decisionName)
	w.Header().Set("X-Router-Model", req.Model)
	w.Header().Set("X-Router-Category", classifyResp.Classification.Category)
	w.Header().Set("X-Router-Confidence", fmt.Sprintf("%.2f", classifyResp.Classification.Confidence))

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if _, writeErr := w.Write(backendResp); writeErr != nil {
		logging.Errorf("Failed to write response: %v", writeErr)
	}
}

// findDecision finds a decision by name in the config
func (a *Adapter) findDecision(name string) *config.Decision {
	for _, decision := range a.engine.Config.IntelligentRouting.Decisions {
		if decision.Name == name {
			return &decision
		}
	}
	return nil
}

// proxyToBackend proxies the request to the selected vLLM backend
func (a *Adapter) proxyToBackend(ctx context.Context, endpoint string, req ChatCompletionRequest) ([]byte, error) {
	// Marshal request
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create backend request
	backendURL := fmt.Sprintf("http://%s/v1/chat/completions", endpoint)
	backendReq, err := http.NewRequestWithContext(ctx, "POST", backendURL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create backend request: %w", err)
	}

	backendReq.Header.Set("Content-Type", "application/json")

	// Execute request
	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(backendReq)
	if err != nil {
		return nil, fmt.Errorf("backend request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read backend response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("backend returned status %d: %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

// getEndpointForModel returns the endpoint address for a given model
func (a *Adapter) getEndpointForModel(model string) string {
	// Get model config from engine config
	modelConfig, exists := a.engine.Config.ModelConfig[model]
	if !exists {
		// Try to find any available endpoint
		if len(a.engine.Config.VLLMEndpoints) > 0 {
			endpoint := a.engine.Config.VLLMEndpoints[0]
			return fmt.Sprintf("%s:%d", endpoint.Address, endpoint.Port)
		}
		return ""
	}

	// Get preferred endpoint
	if len(modelConfig.PreferredEndpoints) > 0 {
		endpointName := modelConfig.PreferredEndpoints[0]
		for _, ep := range a.engine.Config.VLLMEndpoints {
			if ep.Name == endpointName {
				return fmt.Sprintf("%s:%d", ep.Address, ep.Port)
			}
		}
	}

	// Fallback to first endpoint
	if len(a.engine.Config.VLLMEndpoints) > 0 {
		endpoint := a.engine.Config.VLLMEndpoints[0]
		return fmt.Sprintf("%s:%d", endpoint.Address, endpoint.Port)
	}

	return ""
}

// injectSystemPrompt adds or replaces system prompt in messages
func (a *Adapter) injectSystemPrompt(messages []engine.Message, systemPrompt string) []engine.Message {
	// Check if first message is system prompt
	if len(messages) > 0 && messages[0].Role == "system" {
		// Replace existing system prompt
		messages[0].Content = systemPrompt
		return messages
	}

	// Prepend system prompt
	return append([]engine.Message{{Role: "system", Content: systemPrompt}}, messages...)
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
