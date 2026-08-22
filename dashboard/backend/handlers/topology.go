package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

// TestQueryMode represents the test query mode
type TestQueryMode string

const (
	TestQueryModeSimulate TestQueryMode = "simulate"
	TestQueryModeDryRun   TestQueryMode = "dry-run"
)

// TestQueryRequest represents a test query request. Messages, metadata, and
// tools are optional and forwarded to Eval as-is; simple text remains the
// default request shape.
type TestQueryRequest struct {
	Query    string                `json:"query"`
	Mode     TestQueryMode         `json:"mode"`
	Model    string                `json:"model,omitempty"`
	Messages []RouterIntentMessage `json:"messages,omitempty"`
	Metadata map[string]string     `json:"metadata,omitempty"`
	Tools    []json.RawMessage     `json:"tools,omitempty"`
}

// MatchedSignal represents a matched signal
type MatchedSignal struct {
	Type       string   `json:"type"`
	Name       string   `json:"name"`
	Confidence float64  `json:"confidence"`
	Value      *float64 `json:"value,omitempty"`
	Reason     string   `json:"reason,omitempty"`
}

// EvaluatedRule is a derived, best-effort rule summary used only when the
// router did not return a trace (older router, or a request that failed
// before eval_trace could be produced). It is flat and cannot represent
// nested rules, so IsAccurate must never be true while it is the source.
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
	Query string        `json:"query"`
	Mode  TestQueryMode `json:"mode"`
	// RequestedModel and Recipe echo back what the router actually resolved,
	// so the caller can verify the response matches the selected scope.
	RequestedModel     string                   `json:"requestedModel,omitempty"`
	Recipe             string                   `json:"recipe,omitempty"`
	MatchedSignals     []MatchedSignal          `json:"matchedSignals"`
	MatchedDecision    string                   `json:"matchedDecision"`
	Algorithm          string                   `json:"algorithm,omitempty"`
	MatchedModels      []string                 `json:"matchedModels"`
	HighlightedPath    []string                 `json:"highlightedPath"`
	IsAccurate         bool                     `json:"isAccurate"`
	EvalTrace          []decision.DecisionTrace `json:"evalTrace,omitempty"`
	EvaluatedRules     []EvaluatedRule          `json:"evaluatedRules,omitempty"`
	RoutingLatency     int64                    `json:"routingLatency,omitempty"`
	Warning            string                   `json:"warning,omitempty"`
	IsFallbackDecision bool                     `json:"isFallbackDecision,omitempty"` // True if matched decision is a system fallback
	FallbackReason     string                   `json:"fallbackReason,omitempty"`     // Reason for fallback (e.g., "low_confidence", "no_match")
}

// TopologyTestQueryHandler handles test query requests for topology visualization
// routerAPIURL: the Router API URL for dry-run mode (real classification)
// configPath: path to config.yaml for simulate mode (local simulation)
func TopologyTestQueryHandler(configPath, routerAPIURL string, credentialProvider ...routerauth.CredentialProvider) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Parse request
		var req TestQueryRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		if req.Query == "" {
			http.Error(w, "Query cannot be empty", http.StatusBadRequest)
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
			result = callRouterAPI(req, routerAPIURL, configPath, credentialProvider...)
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

// RouterIntentMessage mirrors services.IntentMessage for request forwarding.
type RouterIntentMessage struct {
	Role       string            `json:"role"`
	Content    json.RawMessage   `json:"content"`
	ToolCalls  []json.RawMessage `json:"tool_calls,omitempty"`
	ToolCallID string            `json:"tool_call_id,omitempty"`
}

// RouterIntentRequest is the request body for Router's /api/v1/eval.
type RouterIntentRequest struct {
	Text     string                `json:"text"`
	Messages []RouterIntentMessage `json:"messages,omitempty"`
	Tools    []json.RawMessage     `json:"tools,omitempty"`
	Model    string                `json:"model,omitempty"`
	Metadata map[string]string     `json:"metadata,omitempty"`
	Options  *RouterIntentOptions  `json:"options,omitempty"`
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
	Algorithm        string                `json:"algorithm,omitempty"`
	Plugins          []string              `json:"plugins,omitempty"`
	UsedSignals      *RouterMatchedSignals `json:"used_signals,omitempty"`
	MatchedSignals   *RouterMatchedSignals `json:"matched_signals,omitempty"`
	UnmatchedSignals *RouterMatchedSignals `json:"unmatched_signals,omitempty"`
}

// RouterEvalResponse is the response from Router's /api/v1/eval endpoint.
// Field names and JSON tags mirror services.EvalResponse; EvalTrace reuses
// the router's own decision.DecisionTrace type rather than a second schema.
type RouterEvalResponse struct {
	OriginalText      string                    `json:"original_text,omitempty"`
	RequestedModel    string                    `json:"requested_model,omitempty"`
	Recipe            string                    `json:"recipe,omitempty"`
	DecisionResult    *RouterEvalDecisionResult `json:"decision_result,omitempty"`
	EvalTrace         []decision.DecisionTrace  `json:"eval_trace,omitempty"`
	RecommendedModels []string                  `json:"recommended_models,omitempty"`
	RoutingDecision   string                    `json:"routing_decision,omitempty"`
	SignalConfidences map[string]float64        `json:"signal_confidences,omitempty"`
	SignalValues      map[string]float64        `json:"signal_values,omitempty"`
	SignalErrors      map[string]string         `json:"signal_errors,omitempty"`
}

// callRouterAPI calls the real Router API for classification
func callRouterAPI(req TestQueryRequest, routerAPIURL, configPath string, credentialProvider ...routerauth.CredentialProvider) *TestQueryResult {
	// Prepare request to Router API
	intentReq := RouterIntentRequest{
		Text:     req.Query,
		Messages: req.Messages,
		Tools:    req.Tools,
		Model:    req.Model,
		Metadata: req.Metadata,
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

	// Call Router eval API with trace=true so topology can render the exact
	// recipe-scoped evaluation tree instead of reconstructing it from flat signals.
	apiURL := fmt.Sprintf("%s/api/v1/eval?trace=true", strings.TrimSuffix(routerAPIURL, "/"))
	httpReq, err := http.NewRequest("POST", apiURL, bytes.NewReader(reqBody))
	if err != nil {
		return &TestQueryResult{
			Query:           req.Query,
			Mode:            req.Mode,
			HighlightedPath: []string{"client"},
			Warning:         fmt.Sprintf("Failed to create request: %v", err),
		}
	}
	httpReq.Header.Set("Content-Type", "application/json")
	var provider routerauth.CredentialProvider
	if len(credentialProvider) > 0 {
		provider = credentialProvider[0]
	}
	if authErr := routerauth.RewriteAuthorization(httpReq, provider); authErr != nil {
		return &TestQueryResult{
			Query:           req.Query,
			Mode:            req.Mode,
			HighlightedPath: []string{"client"},
			Warning:         "Router management credential is unavailable",
			IsAccurate:      false,
		}
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		log.Printf("Router API call failed: %v", err)
		return &TestQueryResult{
			Query:           req.Query,
			Mode:            req.Mode,
			HighlightedPath: []string{"client"},
			Warning:         fmt.Sprintf("Router API unavailable: %v", err),
			IsAccurate:      false,
		}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
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
	var routerResp RouterEvalResponse
	if err := json.NewDecoder(resp.Body).Decode(&routerResp); err != nil {
		log.Printf("Failed to decode Router API response: %v", err)
		return &TestQueryResult{
			Query:           req.Query,
			Mode:            req.Mode,
			HighlightedPath: []string{"client"},
			Warning:         "Failed to parse Router API response",
			IsAccurate:      false,
		}
	}

	// Convert Router response to TestQueryResult
	return convertRouterResponse(req, &routerResp, configPath)
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
