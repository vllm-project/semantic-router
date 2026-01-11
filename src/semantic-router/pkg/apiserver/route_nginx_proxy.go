//go:build !windows && cgo

// Package apiserver provides the Classification API server.
// This file implements the nginx proxy mode for full classification and blocking.
// See: https://github.com/vllm-project/semantic-router/issues/557
//
// nginx proxy mode uses the FULL vSR pipeline (same as ExtProc/Envoy):
// - Classification (domain/category detection)
// - Decision Engine (routing rules)
// - Semantic Cache (response caching)
// - System Prompt Injection
// - PII Detection and Blocking
// - Jailbreak Detection and Blocking
// - Model Routing
//
// This is NOT a simplified version - it's the real vSR with all features!
package apiserver

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// Default LLM backend configuration
const (
	defaultLLMBackendURL = "http://localhost:8000"
	proxyTimeout         = 5 * time.Minute // LLM requests can be slow
)

// ProxyConfig holds configuration for the nginx proxy mode
type ProxyConfig struct {
	LLMBackendURL    string
	BlockOnPII       bool
	BlockOnJailbreak bool
	Timeout          time.Duration
}

// getProxyConfig gets the proxy configuration from environment or defaults
func (s *ClassificationAPIServer) getProxyConfig() ProxyConfig {
	backendURL := os.Getenv("VSR_LLM_BACKEND_URL")
	if backendURL == "" {
		backendURL = defaultLLMBackendURL
	}

	// Read blocking config from environment (default to true for security)
	blockOnPII := true
	if val := os.Getenv("VSR_BLOCK_ON_PII"); val == "false" {
		blockOnPII = false
	}

	blockOnJailbreak := true
	if val := os.Getenv("VSR_BLOCK_ON_JAILBREAK"); val == "false" {
		blockOnJailbreak = false
	}

	return ProxyConfig{
		LLMBackendURL:    backendURL,
		BlockOnPII:       blockOnPII,
		BlockOnJailbreak: blockOnJailbreak,
		Timeout:          proxyTimeout,
	}
}

// handleProxyChatCompletions handles /v1/chat/completions in proxy mode
// It uses the FULL vSR pipeline when router is available
func (s *ClassificationAPIServer) handleProxyChatCompletions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	config := s.getProxyConfig()

	// Read the request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		logging.Errorf("proxy: failed to read request body: %v", err)
		s.writeProxyError(w, http.StatusBadRequest, "failed to read request body")
		return
	}
	r.Body.Close()

	// Extract user content from messages
	_, userContent, err := cache.ExtractQueryFromOpenAIRequest(body)
	if err != nil {
		logging.Warnf("proxy: failed to parse request body: %v", err)
		// If we can't parse, forward anyway (fail open)
		s.forwardToBackend(w, r, body, config)
		return
	}

	if userContent == "" {
		// No content to classify, forward the request
		s.forwardToBackend(w, r, body, config)
		return
	}

	// Use full router pipeline if available, otherwise fall back to simplified classification
	if s.router != nil {
		s.handleWithFullPipeline(w, r, body, userContent, config, start)
	} else {
		s.handleWithSimplifiedClassification(w, r, body, userContent, config, start)
	}
}

// handleWithFullPipeline processes the request using the full vSR pipeline
// This includes Decision Engine, Semantic Cache, System Prompts, etc.
func (s *ClassificationAPIServer) handleWithFullPipeline(w http.ResponseWriter, r *http.Request, body []byte, userContent string, config ProxyConfig, start time.Time) {
	logging.Infof("proxy: using full vSR pipeline")

	// Process through the full vSR router
	result, err := s.router.ProcessHTTPRequest(body, userContent)
	if err != nil {
		logging.Errorf("proxy: router processing failed: %v", err)
		// Fall back to forwarding on error
		s.forwardToBackend(w, r, body, config)
		return
	}

	// Set all vSR headers (using existing header constants from headers.go)
	w.Header().Set(headers.VSRSelectedCategory, result.Category)
	w.Header().Set(headers.VSRSelectedDecision, result.DecisionName)
	if result.SelectedModel != "" {
		w.Header().Set(headers.VSRSelectedModel, result.SelectedModel)
	}
	if result.ReasoningEnabled {
		w.Header().Set(headers.VSRSelectedReasoning, "on")
	} else {
		w.Header().Set(headers.VSRSelectedReasoning, "off")
	}
	w.Header().Set(headers.VSRInjectedSystemPrompt, strconv.FormatBool(result.SystemPromptAdded))

	// Set security headers
	if result.IsJailbreak {
		w.Header().Set(headers.VSRJailbreakBlocked, "true")
		w.Header().Set(headers.VSRJailbreakType, result.JailbreakType)
		w.Header().Set(headers.VSRJailbreakConfidence, strconv.FormatFloat(float64(result.JailbreakConfidence), 'f', 3, 64))
	}
	if result.HasPII {
		w.Header().Set(headers.VSRPIIViolation, "true")
		if len(result.PIITypes) > 0 {
			w.Header().Set(headers.VSRPIITypes, strings.Join(result.PIITypes, ","))
		}
	}

	// Set cache header
	if result.CacheHit {
		w.Header().Set(headers.VSRCacheHit, "true")
	}

	// Set processing time
	w.Header().Set("X-Vsr-Processing-Time-Ms", strconv.FormatInt(time.Since(start).Milliseconds(), 10))

	// Check if request should be blocked
	if result.ShouldBlock {
		logging.Infof("proxy: blocking request, reason: %s", result.BlockReason)
		w.Header().Set("X-Vsr-Block-Reason", result.BlockReason)
		s.writeProxyError(w, http.StatusForbidden, "Request blocked: "+result.BlockReason)
		return
	}

	// Check if we have a cached response
	if result.CacheHit && len(result.CachedResponse) > 0 {
		logging.Infof("proxy: returning cached response")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(result.CachedResponse)
		return
	}

	// Forward the (possibly modified) request to LLM backend
	bodyToForward := body
	if len(result.ModifiedBody) > 0 {
		bodyToForward = result.ModifiedBody
	}
	s.forwardToBackend(w, r, bodyToForward, config)
}

// handleWithSimplifiedClassification is the fallback when router is not available
// This provides basic classification and blocking but not the full pipeline
func (s *ClassificationAPIServer) handleWithSimplifiedClassification(w http.ResponseWriter, r *http.Request, body []byte, userContent string, config ProxyConfig, start time.Time) {
	logging.Infof("proxy: using simplified classification (router not available)")

	// Perform classification using classification service
	classificationResult := s.classifyContent(userContent)

	// Set classification headers (using existing header constants)
	w.Header().Set(headers.VSRSelectedCategory, classificationResult.Category)
	w.Header().Set("X-Vsr-Confidence", strconv.FormatFloat(classificationResult.Confidence, 'f', 4, 64))
	w.Header().Set("X-Vsr-Processing-Time-Ms", strconv.FormatInt(time.Since(start).Milliseconds(), 10))

	// Set security headers
	if classificationResult.IsJailbreak {
		w.Header().Set(headers.VSRJailbreakBlocked, "true")
		if classificationResult.ThreatType != "" {
			w.Header().Set(headers.VSRJailbreakType, classificationResult.ThreatType)
		}
	}
	if classificationResult.HasPII {
		w.Header().Set(headers.VSRPIIViolation, "true")
	}

	// Check if request should be blocked
	if classificationResult.IsJailbreak && config.BlockOnJailbreak {
		logging.Infof("proxy: blocking jailbreak attempt")
		w.Header().Set("X-Vsr-Block-Reason", "jailbreak_detected")
		s.writeProxyError(w, http.StatusForbidden, "Request blocked: jailbreak attempt detected")
		return
	}

	if classificationResult.HasPII && config.BlockOnPII {
		logging.Infof("proxy: blocking PII content")
		w.Header().Set("X-Vsr-Block-Reason", "pii_detected")
		s.writeProxyError(w, http.StatusForbidden, "Request blocked: PII detected")
		return
	}

	// Request is allowed, forward to LLM backend
	s.forwardToBackend(w, r, body, config)
}

// ClassificationResult holds the results of content classification
type ClassificationResult struct {
	Category    string
	Confidence  float64
	HasPII      bool
	IsJailbreak bool
	ThreatType  string
}

// classifyContent performs classification on the content (simplified fallback)
func (s *ClassificationAPIServer) classifyContent(content string) ClassificationResult {
	result := ClassificationResult{
		Category:   "unknown",
		Confidence: 0.0,
	}

	if s.classificationSvc == nil {
		return result
	}

	// Try unified classifier first
	if s.classificationSvc.HasUnifiedClassifier() {
		results, err := s.classificationSvc.ClassifyBatchUnified([]string{content})
		if err != nil {
			logging.Warnf("proxy: unified classification failed: %v", err)
			return result
		}

		if len(results.IntentResults) > 0 {
			result.Category = results.IntentResults[0].Category
			result.Confidence = float64(results.IntentResults[0].Confidence)
		}

		if len(results.PIIResults) > 0 {
			result.HasPII = results.PIIResults[0].HasPII
		}

		if len(results.SecurityResults) > 0 {
			result.IsJailbreak = results.SecurityResults[0].IsJailbreak
			result.ThreatType = results.SecurityResults[0].ThreatType
		}

		return result
	}

	// Fallback to individual classifiers
	if s.classificationSvc.HasClassifier() {
		// Intent classification
		intentResp, err := s.classificationSvc.ClassifyIntent(services.IntentRequest{Text: content})
		if err == nil && intentResp != nil {
			result.Category = intentResp.Classification.Category
			result.Confidence = intentResp.Classification.Confidence
		}

		// PII detection
		piiResp, err := s.classificationSvc.DetectPII(services.PIIRequest{Text: content})
		if err == nil && piiResp != nil {
			result.HasPII = piiResp.HasPII
		}

		// Security detection
		secResp, err := s.classificationSvc.CheckSecurity(services.SecurityRequest{Text: content})
		if err == nil && secResp != nil {
			result.IsJailbreak = secResp.IsJailbreak
			if len(secResp.DetectionTypes) > 0 {
				result.ThreatType = secResp.DetectionTypes[0]
			}
		}
	}

	return result
}

// forwardToBackend forwards the request to the LLM backend
func (s *ClassificationAPIServer) forwardToBackend(w http.ResponseWriter, r *http.Request, body []byte, config ProxyConfig) {
	backendURL, err := url.Parse(config.LLMBackendURL)
	if err != nil {
		logging.Errorf("proxy: invalid backend URL %s: %v", config.LLMBackendURL, err)
		s.writeProxyError(w, http.StatusInternalServerError, "invalid backend configuration")
		return
	}

	// Create reverse proxy
	proxy := httputil.NewSingleHostReverseProxy(backendURL)

	// Configure timeout
	proxy.Transport = &http.Transport{
		ResponseHeaderTimeout: config.Timeout,
	}

	// Modify the director to set the correct host and path
	originalDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		originalDirector(req)
		req.Host = backendURL.Host
		req.URL.Host = backendURL.Host
		req.URL.Scheme = backendURL.Scheme
		// Keep the original path (e.g., /v1/chat/completions)
	}

	// Handle errors
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		logging.Errorf("proxy: backend request failed: %v", err)
		s.writeProxyError(w, http.StatusBadGateway, "backend request failed")
	}

	// Create a new request with the body
	ctx, cancel := context.WithTimeout(r.Context(), config.Timeout)
	defer cancel()

	proxyReq, err := http.NewRequestWithContext(ctx, r.Method, r.URL.String(), bytes.NewReader(body))
	if err != nil {
		logging.Errorf("proxy: failed to create proxy request: %v", err)
		s.writeProxyError(w, http.StatusInternalServerError, "failed to create proxy request")
		return
	}

	// Copy headers from original request
	for key, values := range r.Header {
		for _, value := range values {
			proxyReq.Header.Add(key, value)
		}
	}

	// Set content length
	proxyReq.ContentLength = int64(len(body))
	proxyReq.Header.Set("Content-Length", strconv.Itoa(len(body)))

	logging.Infof("proxy: forwarding request to %s%s", config.LLMBackendURL, r.URL.Path)

	// Forward the request
	proxy.ServeHTTP(w, proxyReq)
}

// handleProxyCompletions handles /v1/completions in proxy mode (legacy API)
func (s *ClassificationAPIServer) handleProxyCompletions(w http.ResponseWriter, r *http.Request) {
	// Use the same handler as chat completions
	s.handleProxyChatCompletions(w, r)
}

// handleProxyHealth handles health check for proxy mode
func (s *ClassificationAPIServer) handleProxyHealth(w http.ResponseWriter, _ *http.Request) {
	config := s.getProxyConfig()

	// Check if LLM backend is reachable
	client := &http.Client{Timeout: 5 * time.Second}
	healthURL := strings.TrimSuffix(config.LLMBackendURL, "/") + "/health"

	resp, err := client.Get(healthURL)
	if err != nil {
		logging.Warnf("proxy: LLM backend health check failed: %v", err)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		// Include router status
		routerStatus := "not_available"
		if s.router != nil {
			routerStatus = "available"
		}
		_, _ = w.Write([]byte(`{"status": "healthy", "backend": "unreachable", "router": "` + routerStatus + `", "note": "vSR is healthy but LLM backend may be down"}`))
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	// Include router status
	routerStatus := "not_available"
	pipelineMode := "simplified"
	if s.router != nil {
		routerStatus = "available"
		pipelineMode = "full"
	}
	_, _ = w.Write([]byte(`{"status": "healthy", "backend": "reachable", "router": "` + routerStatus + `", "pipeline": "` + pipelineMode + `"}`))
}

// writeProxyError writes an error response for proxy mode
func (s *ClassificationAPIServer) writeProxyError(w http.ResponseWriter, statusCode int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	_, _ = w.Write([]byte(`{"error": {"message": "` + message + `", "type": "proxy_error"}}`))
}
