package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/pii"
)

type RouterEngine struct {
	Config               *config.RouterConfig
	CategoryDescriptions []string
	Classifier           *classification.Classifier
	PIIChecker           *pii.PolicyChecker
	Cache                cache.CacheBackend
	ToolsDatabase        *tools.ToolsDatabase
	ModelSelector        *selection.Registry
	ReplayRecorders      map[string]*routerreplay.Recorder
}

// RouteRequest represents a protocol-agnostic routing request
type RouteRequest struct {
	Model    string
	Messages []Message
	User     string
	Headers  map[string]string
	Context  context.Context
}

// Message represents a chat message in OpenAI format
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// RouteResponse represents the routing decision and any modifications
type RouteResponse struct {
	SelectedModel    string
	SelectedEndpoint string
	DecisionName     string
	Confidence       float64
	Blocked          bool
	BlockReason      string
	ModifiedMessages []Message
	Headers          map[string]string
	CacheHit         bool
	CacheKey         string
	CachedResponse   string
	SelectedTools    []openai.ChatCompletionToolParam
	ReplayID         string
	// Backend response data
	ResponseBody []byte
	StatusCode   int
}

// Route performs the core routing logic for a request
func (e *RouterEngine) Route(ctx context.Context, req *RouteRequest) (*RouteResponse, error) {
	response := &RouteResponse{
		SelectedModel: req.Model,
		Headers:       make(map[string]string),
	}

	classResult, err := e.ClassifyRequest(ctx, req.Messages)
	if err != nil {
		return nil, fmt.Errorf("classification failed: %w", err)
	}

	decisionName := classResult.DecisionName
	selectedModel := req.Model

	if classResult.Decision != nil && len(classResult.Decision.ModelRefs) > 0 {
		selectedModel = classResult.Decision.ModelRefs[0].Model
	}

	response.SelectedModel = selectedModel
	response.DecisionName = decisionName
	response.Confidence = classResult.Confidence

	if e.Config.SemanticCache.Enabled {
		userContent := e.extractUserContent(req.Messages)
		if userContent != "" {
			cacheResp, found, cacheErr := e.CheckCache(ctx, selectedModel, userContent, decisionName)
			if cacheErr == nil && found {
				logging.Infof("Cache hit for query: %s", userContent)
				response.CacheHit = true
				response.CachedResponse = cacheResp
				return response, nil
			}
		}
	}

	if e.Config.Tools.Enabled && e.ToolsDatabase != nil {
		userContent := e.extractUserContent(req.Messages)
		if userContent != "" {
			topK := e.Config.Tools.TopK
			if topK <= 0 {
				topK = 3
			}
			tools, toolsErr := e.SelectTools(ctx, userContent, topK)
			if toolsErr != nil {
				logging.Warnf("Tool selection failed: %v", toolsErr)
			} else if len(tools) > 0 {
				response.SelectedTools = tools
				logging.Infof("Selected %d tools for request", len(tools))
			}
		}
	}

	// Get endpoint for selected model
	selectedEndpoint := e.getEndpointForModel(selectedModel)
	if selectedEndpoint == "" {
		// Try fallback to default_model if configured
		if e.Config.DefaultModel != "" {
			logging.Infof("No endpoint for model %s, trying default_model: %s", selectedModel, e.Config.DefaultModel)
			selectedEndpoint = e.getEndpointForModel(e.Config.DefaultModel)
			if selectedEndpoint != "" {
				selectedModel = e.Config.DefaultModel
				response.SelectedModel = selectedModel
			}
		}

		if selectedEndpoint == "" {
			return nil, fmt.Errorf("no endpoint found for model: %s", selectedModel)
		}
	}
	response.SelectedEndpoint = selectedEndpoint

	// Proxy to backend
	backendResp, statusCode, err := e.proxyToBackend(ctx, selectedEndpoint, req, response.SelectedTools)
	if err != nil {
		logging.Errorf("Backend proxy error: %v", err)
		return nil, fmt.Errorf("backend proxy failed: %w", err)
	}

	response.ResponseBody = backendResp
	response.StatusCode = statusCode

	// Update cache if enabled
	if e.Config.SemanticCache.Enabled && !response.CacheHit {
		userContent := e.extractUserContent(req.Messages)
		if userContent != "" {
			if err := e.UpdateCache(ctx, selectedModel, userContent, string(backendResp), decisionName); err != nil {
				logging.Warnf("Failed to update cache: %v", err)
			}
		}
	}

	if decisionName != "" {
		reqBody, _ := json.Marshal(map[string]interface{}{
			"model":    req.Model,
			"messages": req.Messages,
		})

		record := &routerreplay.RoutingRecord{
			RequestID:       fmt.Sprintf("route_%d", time.Now().UnixNano()),
			OriginalModel:   req.Model,
			SelectedModel:   selectedModel,
			Decision:        decisionName,
			ConfidenceScore: classResult.Confidence,
			RequestBody:     string(reqBody),
			ResponseBody:    string(backendResp),
			ResponseStatus:  statusCode,
		}
		if err := e.RecordReplay(ctx, decisionName, record, classResult); err != nil {
			logging.Warnf("Failed to record replay: %v", err)
		} else {
			response.ReplayID = record.RequestID
		}
	}

	return response, nil
}

func (e *RouterEngine) ClassifyRequest(ctx context.Context, messages []Message) (*ClassificationResult, error) {
	if e.Classifier == nil {
		return nil, fmt.Errorf("classifier not initialized")
	}

	// Extract user content
	userContent, nonUserMessages := e.extractUserAndNonUserContent(messages)

	// Check if there's content to evaluate
	if len(nonUserMessages) == 0 && userContent == "" {
		return &ClassificationResult{}, nil
	}

	if len(e.Config.Decisions) == 0 {
		logging.Warnf("No decisions configured")
		return &ClassificationResult{}, nil
	}

	evaluationText := userContent
	if evaluationText == "" && len(nonUserMessages) > 0 {
		evaluationText = strings.Join(nonUserMessages, " ")
	}

	if evaluationText == "" {
		return &ClassificationResult{}, nil
	}

	// For context token counting, we need to include ALL messages
	var allMessagesText string
	if userContent != "" && len(nonUserMessages) > 0 {
		allMessages := make([]string, 0, len(nonUserMessages)+1)
		allMessages = append(allMessages, nonUserMessages...)
		allMessages = append(allMessages, userContent)
		allMessagesText = strings.Join(allMessages, " ")
	} else if userContent != "" {
		allMessagesText = userContent
	} else {
		allMessagesText = strings.Join(nonUserMessages, " ")
	}

	signals := e.Classifier.EvaluateAllSignalsWithContext(evaluationText, allMessagesText, true)

	result, err := e.Classifier.EvaluateDecisionWithEngine(signals)
	if err != nil {
		logging.Errorf("Decision evaluation error: %v", err)
		return nil, err
	}

	if result == nil || result.Decision == nil {
		return &ClassificationResult{}, nil
	}

	// Extract category from matched rules
	categoryName := ""
	for _, rule := range result.MatchedRules {
		if strings.HasPrefix(rule, "domain:") {
			categoryName = strings.TrimPrefix(rule, "domain:")
			break
		}
	}

	return &ClassificationResult{
		DecisionName: result.Decision.Name,
		Decision:     result.Decision,
		Confidence:   result.Confidence,
		CategoryName: categoryName,
		MatchedRules: result.MatchedRules,
	}, nil
}

// ClassificationResult holds the results of classification
type ClassificationResult struct {
	DecisionName string
	Decision     *config.Decision
	Confidence   float64
	CategoryName string
	MatchedRules []string
}

// extractUserContent extracts just the user content from messages
func (e *RouterEngine) extractUserContent(messages []Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return messages[i].Content
		}
	}
	return ""
}

// extractUserAndNonUserContent extracts user and non-user messages
func (e *RouterEngine) extractUserAndNonUserContent(messages []Message) (string, []string) {
	var userContents []string
	var nonUserContents []string

	for _, msg := range messages {
		if msg.Role == "user" {
			userContents = append(userContents, msg.Content)
		} else {
			nonUserContents = append(nonUserContents, msg.Content)
		}
	}

	userContent := strings.Join(userContents, " ")
	return userContent, nonUserContents
}

func (e *RouterEngine) CheckPII(ctx context.Context, content string, decisionName string, detectedPII []string) (bool, []string, error) {
	if e.PIIChecker == nil {
		return false, nil, nil
	}
	return e.PIIChecker.CheckPolicy(decisionName, detectedPII)
}

// CheckCache checks if request exists in cache
func (e *RouterEngine) CheckCache(ctx context.Context, model, query, decisionName string) (string, bool, error) {
	if e.Cache == nil || !e.Cache.IsEnabled() {
		return "", false, nil
	}

	// Check if caching is enabled for this decision
	cacheEnabled := e.Config.SemanticCache.Enabled
	if decisionName != "" {
		cacheEnabled = e.Config.IsCacheEnabledForDecision(decisionName)
	}

	if !cacheEnabled || query == "" {
		return "", false, nil
	}

	// Get decision-specific threshold
	threshold := e.Config.GetCacheSimilarityThreshold()
	if decisionName != "" {
		threshold = e.Config.GetCacheSimilarityThresholdForDecision(decisionName)
	}

	// Try to find a similar cached response
	cachedResponse, found, err := e.Cache.FindSimilarWithThreshold(model, query, threshold)
	if err != nil {
		logging.Errorf("Error searching cache: %v", err)
		return "", false, err
	}

	return string(cachedResponse), found, nil
}

// UpdateCache updates the cache with a new entry
func (e *RouterEngine) UpdateCache(ctx context.Context, model, query, response, decisionName string) error {
	if e.Cache == nil || !e.Cache.IsEnabled() {
		return nil
	}

	// Check if caching is enabled for this decision
	cacheEnabled := e.Config.SemanticCache.Enabled
	if decisionName != "" {
		cacheEnabled = e.Config.IsCacheEnabledForDecision(decisionName)
	}

	if !cacheEnabled {
		return nil
	}

	// Use AddEntry to store complete request-response pair
	// Generate a request ID for the cache entry
	requestID := fmt.Sprintf("cache_%s_%d", model, time.Now().UnixNano())
	return e.Cache.AddEntry(requestID, model, query, []byte(query), []byte(response), e.Config.SemanticCache.TTLSeconds)
}

// SelectTools selects relevant tools for the request using semantic similarity
func (e *RouterEngine) SelectTools(ctx context.Context, query string, topK int) ([]openai.ChatCompletionToolParam, error) {
	if e.ToolsDatabase == nil || !e.ToolsDatabase.IsEnabled() {
		return nil, nil
	}

	// Check if advanced filtering is enabled
	advanced := e.Config.Tools.AdvancedFiltering
	if advanced != nil && advanced.Enabled {
		// Use advanced filtering with candidate pool
		candidatePoolSize := topK
		if advanced.CandidatePoolSize != nil && *advanced.CandidatePoolSize > 0 {
			candidatePoolSize = *advanced.CandidatePoolSize
		} else if advanced.CandidatePoolSize == nil {
			candidatePoolSize = max(topK*5, 20) // candidatePoolMultiplier * topK, candidatePoolMinSize
		}
		if candidatePoolSize < topK {
			candidatePoolSize = topK
		}

		// Get candidates with scores
		candidates, err := e.ToolsDatabase.FindSimilarToolsWithScores(query, candidatePoolSize)
		if err != nil {
			if e.Config.Tools.FallbackToEmpty {
				return nil, nil
			}
			return nil, fmt.Errorf("tool selection failed: %w", err)
		}

		// Apply advanced filtering and ranking
		selectedTools := tools.FilterAndRankTools(query, candidates, topK, advanced, "")
		return selectedTools, nil
	}

	// Standard tool selection
	selectedTools, err := e.ToolsDatabase.FindSimilarTools(query, topK)
	if err != nil {
		if e.Config.Tools.FallbackToEmpty {
			return nil, nil
		}
		return nil, fmt.Errorf("tool selection failed: %w", err)
	}

	return selectedTools, nil
}

// RecordReplay records the routing decision for replay
func (e *RouterEngine) RecordReplay(ctx context.Context, decisionName string, record *routerreplay.RoutingRecord, classResult *ClassificationResult) error {
	recorder, ok := e.ReplayRecorders[decisionName]
	if !ok || recorder == nil {
		return nil
	}

	// Find decision config for category and reasoning info
	var decision *config.Decision
	for _, d := range e.Config.Decisions {
		if d.Name == decisionName {
			decision = &d
			break
		}
	}

	category := ""
	reasoningMode := "off"
	if decision != nil {
		if len(decision.Rules.Conditions) > 0 && decision.Rules.Conditions[0].Type == "domain" {
			category = decision.Rules.Conditions[0].Name
		}
		// Check if reasoning is enabled for selected model
		for _, modelRef := range decision.ModelRefs {
			if modelRef.Model == record.SelectedModel && modelRef.UseReasoning != nil && *modelRef.UseReasoning {
				reasoningMode = "on"
				break
			}
		}
	}

	// Build signals map - use matched rules from classification
	domainSignals := []string{}
	for _, rule := range classResult.MatchedRules {
		// Extract domain from rule format "domain:math"
		if strings.HasPrefix(rule, "domain:") {
			domainSignals = append(domainSignals, strings.TrimPrefix(rule, "domain:"))
		}
	}

	signals := map[string]interface{}{
		"domain":        domainSignals,
		"embedding":     nil,
		"fact_check":    nil,
		"keyword":       nil,
		"preference":    nil,
		"user_feedback": nil,
	}

	// Log router_replay_start event with structured fields
	timestamp := time.Now().UTC().Format(time.RFC3339)
	logging.Infof("router_replay_start: replay_id=%s, request_id=%s, decision=%s, original_model=%s, selected_model=%s, category=%s, reasoning_mode=%s, from_cache=%v, streaming=%v, response_status=%d, timestamp=%s, request_body=%s, request_body_truncated=%v, signals=%v",
		record.RequestID, record.RequestID, decisionName, record.OriginalModel, record.SelectedModel, category, reasoningMode, false, false, 0, timestamp, record.RequestBody, recorder.ShouldCaptureRequest() && len(record.RequestBody) > 10240, signals)

	// AddRecord stores the routing record and returns the assigned ID
	replayID, err := recorder.AddRecord(*record)
	if err != nil {
		return fmt.Errorf("failed to record replay: %w", err)
	}

	// Log router_replay_complete event with structured fields
	logging.Infof("router_replay_complete: replay_id=%s, request_id=%s, decision=%s, original_model=%s, selected_model=%s, category=%s, reasoning_mode=%s, from_cache=%v, streaming=%v, response_status=%d, timestamp=%s, request_body=%s, request_body_truncated=%v, response_body=%s, response_body_truncated=%v, signals=%v",
		replayID, record.RequestID, decisionName, record.OriginalModel, record.SelectedModel, category, reasoningMode, false, false, record.ResponseStatus, timestamp, record.RequestBody, recorder.ShouldCaptureRequest() && len(record.RequestBody) > 10240, record.ResponseBody, recorder.ShouldCaptureResponse() && len(record.ResponseBody) > 10240, signals)

	return nil
}

// proxyToBackend proxies the request to the selected vLLM backend
func (e *RouterEngine) proxyToBackend(ctx context.Context, endpoint string, req *RouteRequest, tools []openai.ChatCompletionToolParam) ([]byte, int, error) {
	// Build backend request body
	backendReq := map[string]interface{}{
		"model":    req.Model,
		"messages": req.Messages,
	}

	// Add tools if selected
	if len(tools) > 0 {
		backendReq["tools"] = tools
	}

	// Marshal request
	reqBody, err := json.Marshal(backendReq)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create backend HTTP request
	backendURL := fmt.Sprintf("http://%s/v1/chat/completions", endpoint)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", backendURL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, 0, fmt.Errorf("failed to create backend request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Add custom headers from request, filtering out HTTP/2 pseudo-headers
	for k, v := range req.Headers {
		// Skip HTTP/2 pseudo-headers (they start with ":")
		if len(k) > 0 && k[0] == ':' {
			continue
		}
		httpReq.Header.Set(k, v)
	}

	// Execute request
	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, 0, fmt.Errorf("backend request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, fmt.Errorf("failed to read backend response: %w", err)
	}

	return respBody, resp.StatusCode, nil
}

// getEndpointForModel returns the endpoint address for a given model
func (e *RouterEngine) getEndpointForModel(model string) string {
	// Get model config
	modelConfig, exists := e.Config.ModelConfig[model]
	if !exists {
		// Try to find any available endpoint
		if len(e.Config.VLLMEndpoints) > 0 {
			endpoint := e.Config.VLLMEndpoints[0]
			return fmt.Sprintf("%s:%d", endpoint.Address, endpoint.Port)
		}
		return ""
	}

	// Get preferred endpoint
	if len(modelConfig.PreferredEndpoints) > 0 {
		endpointName := modelConfig.PreferredEndpoints[0]
		for _, ep := range e.Config.VLLMEndpoints {
			if ep.Name == endpointName {
				return fmt.Sprintf("%s:%d", ep.Address, ep.Port)
			}
		}
	}

	// Fallback to first endpoint
	if len(e.Config.VLLMEndpoints) > 0 {
		endpoint := e.Config.VLLMEndpoints[0]
		return fmt.Sprintf("%s:%d", endpoint.Address, endpoint.Port)
	}

	return ""
}

// Reload reloads the configuration
func (e *RouterEngine) Reload(configPath string) error {
	cfg, err := config.Parse(configPath)
	if err != nil {
		return err
	}
	e.Config = cfg
	return nil
}
