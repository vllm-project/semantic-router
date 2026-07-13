package handlers

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/jsonunicode"
)

const (
	topologyMaxQueryBytes        = 16 << 10
	topologyMaxResponseBodyBytes = 4 << 20
	topologyRouterTimeout        = 10 * time.Second
)

// TestQueryMode represents the test query mode
type TestQueryMode string

const (
	TestQueryModeSimulate TestQueryMode = "simulate"
	TestQueryModeDryRun   TestQueryMode = "dry-run"
)

// TestQueryRequest represents a test query request
type TestQueryRequest struct {
	Query string        `json:"query"`
	Mode  TestQueryMode `json:"mode"`
}

// MatchedSignal represents a matched signal
type MatchedSignal struct {
	Type       string   `json:"type"`
	Name       string   `json:"name"`
	Confidence float64  `json:"confidence"`
	Value      *float64 `json:"value,omitempty"`
	Reason     string   `json:"reason,omitempty"`
}

// EvaluatedRule represents an evaluated decision rule
type EvaluatedRule struct {
	DecisionName  string   `json:"decisionName"`
	RuleOperator  string   `json:"ruleOperator"`
	Conditions    []string `json:"conditions"`
	MatchedCount  int      `json:"matchedCount"`
	TotalCount    int      `json:"totalCount"`
	IsMatch       bool     `json:"isMatch"`
	Priority      int      `json:"priority"`
	MatchedModels []string `json:"matchedModels,omitempty"`
}

// TestQueryResult represents the test query result
type TestQueryResult struct {
	Query              string          `json:"query"`
	Mode               TestQueryMode   `json:"mode"`
	MatchedSignals     []MatchedSignal `json:"matchedSignals"`
	MatchedDecision    string          `json:"matchedDecision"`
	MatchedModels      []string        `json:"matchedModels"`
	HighlightedPath    []string        `json:"highlightedPath"`
	IsAccurate         bool            `json:"isAccurate"`
	EvaluatedRules     []EvaluatedRule `json:"evaluatedRules,omitempty"`
	RoutingLatency     int64           `json:"routingLatency,omitempty"`
	Warning            string          `json:"warning,omitempty"`
	IsFallbackDecision bool            `json:"isFallbackDecision,omitempty"` // True if matched decision is a system fallback
	FallbackReason     string          `json:"fallbackReason,omitempty"`     // Reason for fallback (e.g., "low_confidence", "no_match")
}

// TopologyTestQueryHandler handles test query requests for topology visualization
// routerAPIURL: the Router API URL for dry-run mode (real classification)
// configPath: path to config.yaml for simulate mode (local simulation)
func TopologyTestQueryHandler(configPath, routerAPIURL string) http.HandlerFunc {
	client := newTopologyRouterHTTPClient()
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Parse request
		var req TestQueryRequest
		if status, err := decodeBoundedJSON(w, r, smallJSONRequestBodyLimit, &req); err != nil {
			http.Error(w, "Invalid request body", status)
			return
		}

		if req.Query == "" {
			http.Error(w, "Query cannot be empty", http.StatusBadRequest)
			return
		}
		if len([]byte(req.Query)) > topologyMaxQueryBytes {
			http.Error(w, "Query is too large", http.StatusBadRequest)
			return
		}

		// Default to dry-run mode
		if req.Mode == "" {
			req.Mode = TestQueryModeDryRun
		}

		start := time.Now()

		var result *TestQueryResult

		if req.Mode == TestQueryModeDryRun && routerAPIURL != "" {
			// Dry-run mode: call real Router API for actual classification
			result = callRouterAPI(r.Context(), client, req, routerAPIURL, configPath)
		} else {
			// Simulate mode is no longer supported
			result = &TestQueryResult{
				Query:           req.Query,
				Mode:            req.Mode,
				HighlightedPath: []string{"client"},
				Warning:         "Simulate mode is no longer supported. Please use dry-run mode.",
			}
		}

		result.RoutingLatency = time.Since(start).Milliseconds()

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(result); err != nil {
			log.Printf("Error encoding response: %v", err)
		}
	}
}

// RouterIntentRequest is the request body for Router's /api/v1/classify/intent
type RouterIntentRequest struct {
	Text    string               `json:"text"`
	Options *RouterIntentOptions `json:"options,omitempty"`
}

type RouterIntentOptions struct {
	ReturnProbabilities bool `json:"return_probabilities,omitempty"`
}

type RouterMatchedSignals struct {
	Keywords     []string `json:"keywords,omitempty"`
	Embeddings   []string `json:"embeddings,omitempty"`
	Domains      []string `json:"domains,omitempty"`
	FactCheck    []string `json:"fact_check,omitempty"`
	UserFeedback []string `json:"user_feedback,omitempty"`
	Preferences  []string `json:"preferences,omitempty"`
	Language     []string `json:"language,omitempty"`
	Context      []string `json:"context,omitempty"`
	Structure    []string `json:"structure,omitempty"`
	Complexity   []string `json:"complexity,omitempty"`
	Modality     []string `json:"modality,omitempty"`
	Authz        []string `json:"authz,omitempty"`
	Jailbreak    []string `json:"jailbreak,omitempty"`
	PII          []string `json:"pii,omitempty"`
	KB           []string `json:"kb,omitempty"`
	Conversation []string `json:"conversation,omitempty"`
	Event        []string `json:"event,omitempty"`
	Projection   []string `json:"projection,omitempty"`
}

type RouterEvalDecisionResult struct {
	DecisionName     string                `json:"decision_name"`
	UsedSignals      *RouterMatchedSignals `json:"used_signals,omitempty"`
	MatchedSignals   *RouterMatchedSignals `json:"matched_signals,omitempty"`
	UnmatchedSignals *RouterMatchedSignals `json:"unmatched_signals,omitempty"`
}

// RouterEvalResponse is the response from Router's /api/v1/eval endpoint.
type RouterEvalResponse struct {
	OriginalText      string                    `json:"original_text,omitempty"`
	DecisionResult    *RouterEvalDecisionResult `json:"decision_result,omitempty"`
	RecommendedModels []string                  `json:"recommended_models,omitempty"`
	RoutingDecision   string                    `json:"routing_decision,omitempty"`
	SignalConfidences map[string]float64        `json:"signal_confidences,omitempty"`
	SignalValues      map[string]float64        `json:"signal_values,omitempty"`
}

// callRouterAPI calls the real Router API for classification
func callRouterAPI(
	ctx context.Context,
	client *http.Client,
	req TestQueryRequest,
	routerAPIURL,
	configPath string,
) *TestQueryResult {
	// Prepare request to Router API
	intentReq := RouterIntentRequest{
		Text: req.Query,
		Options: &RouterIntentOptions{
			ReturnProbabilities: true,
		},
	}

	reqBody, err := json.Marshal(intentReq)
	if err != nil {
		return &TestQueryResult{
			Query:           req.Query,
			Mode:            req.Mode,
			HighlightedPath: []string{"client"},
			Warning:         fmt.Sprintf("Failed to marshal request: %v", err),
		}
	}

	// Call Router eval API so topology can inspect all matched signals and signal scores.
	apiURL, err := topologyRouterEvalURL(routerAPIURL)
	if err != nil || client == nil {
		return topologyRouterUnavailableResult(req)
	}
	callCtx, cancel := context.WithTimeout(ctx, topologyRouterTimeout)
	defer cancel()
	httpReq, err := http.NewRequestWithContext(callCtx, http.MethodPost, apiURL.String(), bytes.NewReader(reqBody))
	if err != nil {
		return topologyRouterUnavailableResult(req)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(httpReq)
	if err != nil {
		log.Printf("Router API call failed")
		return topologyRouterUnavailableResult(req)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("Router API returned %d for topology eval", resp.StatusCode)
		return &TestQueryResult{
			Query:           req.Query,
			Mode:            req.Mode,
			HighlightedPath: []string{"client"},
			Warning:         fmt.Sprintf("Router API error (status %d)", resp.StatusCode),
			IsAccurate:      false,
		}
	}

	// Parse response
	body, err := readBoundedOutboundBody(resp.Body, topologyMaxResponseBodyBytes)
	if err != nil {
		log.Printf("Router API topology response exceeded its read budget or failed")
		return topologyRouterInvalidResponseResult(req)
	}
	var routerResp RouterEvalResponse
	if !jsonunicode.Valid(body) || json.Unmarshal(body, &routerResp) != nil {
		log.Printf("Failed to decode Router API topology response")
		return topologyRouterInvalidResponseResult(req)
	}

	// Convert Router response to TestQueryResult
	return convertRouterResponse(req, &routerResp, configPath)
}

func topologyRouterEvalURL(raw string) (*url.URL, error) {
	base, err := parseRouterClassifierProxyBaseURL(raw)
	if err != nil {
		return nil, err
	}
	target := *base
	target.Path = strings.TrimRight(base.Path, "/") + "/api/v1/eval"
	target.RawPath = ""
	target.RawQuery = ""
	target.Fragment = ""
	return &target, nil
}

func newTopologyRouterHTTPClient() *http.Client {
	dialer := &net.Dialer{Timeout: topologyRouterTimeout, KeepAlive: 30 * time.Second}
	transport := &http.Transport{
		Proxy:                  nil,
		DialContext:            dialer.DialContext,
		ForceAttemptHTTP2:      true,
		MaxIdleConns:           20,
		MaxIdleConnsPerHost:    10,
		IdleConnTimeout:        90 * time.Second,
		TLSHandshakeTimeout:    topologyRouterTimeout,
		ResponseHeaderTimeout:  topologyRouterTimeout,
		ExpectContinueTimeout:  time.Second,
		MaxResponseHeaderBytes: outboundMaxResponseHeaderBytes,
		TLSClientConfig:        &tls.Config{MinVersion: tls.VersionTLS12},
	}
	return &http.Client{
		Transport: transport,
		Timeout:   topologyRouterTimeout,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

func topologyRouterUnavailableResult(req TestQueryRequest) *TestQueryResult {
	return &TestQueryResult{
		Query:           req.Query,
		Mode:            req.Mode,
		HighlightedPath: []string{"client"},
		Warning:         "Router API unavailable",
		IsAccurate:      false,
	}
}

func topologyRouterInvalidResponseResult(req TestQueryRequest) *TestQueryResult {
	return &TestQueryResult{
		Query:           req.Query,
		Mode:            req.Mode,
		HighlightedPath: []string{"client"},
		Warning:         "Failed to parse Router API response",
		IsAccurate:      false,
	}
}

// System fallback decisions - these are hardcoded in the router, not from config
var systemFallbackDecisions = map[string]string{
	"low_confidence_general":      "Classification confidence below threshold (default: 0.7)",
	"high_confidence_specialized": "Classification confidence above threshold (default: 0.7)",
}

// isSystemFallbackDecision checks if a decision name is a system fallback
func isSystemFallbackDecision(decisionName string) bool {
	_, exists := systemFallbackDecisions[decisionName]
	return exists
}

// getFallbackReason returns the reason for a system fallback decision
func getFallbackReason(decisionName string) string {
	if reason, exists := systemFallbackDecisions[decisionName]; exists {
		return reason
	}
	return "Unknown fallback reason"
}

// normalizeSignalName normalizes signal name for consistent matching
// Converts spaces to underscores and lowercases for matching "computer science" with "computer_science"
func normalizeSignalName(name string) string {
	return strings.ToLower(strings.ReplaceAll(name, " ", "_"))
}

// normalizeModelName normalizes model name for consistent ID matching
// Replaces non-alphanumeric characters with dashes, matching frontend behavior
func normalizeModelName(name string) string {
	var result strings.Builder
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			result.WriteRune(r)
		} else {
			result.WriteRune('-')
		}
	}
	return result.String()
}
