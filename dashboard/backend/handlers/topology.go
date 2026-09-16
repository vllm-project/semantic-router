package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	Model string        `json:"model,omitempty"`
}

// MatchedSignal represents a matched signal
type MatchedSignal struct {
	Type                string   `json:"type"`
	Name                string   `json:"name"`
	Confidence          float64  `json:"confidence"`
	ConfidenceAvailable *bool    `json:"confidenceAvailable,omitempty"`
	Value               *float64 `json:"value,omitempty"`
	Reason              string   `json:"reason,omitempty"`
}

// EvaluatedRule represents an evaluated decision rule
type EvaluatedRule struct {
	DecisionName  string   `json:"decisionName"`
	Expression    string   `json:"expression,omitempty"`
	State         string   `json:"state,omitempty"`
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
	HTTPStatus                  int               `json:"-"`
	EvalTrace                   json.RawMessage   `json:"evalTrace,omitempty"`
	SignalErrors                map[string]string `json:"signalErrors,omitempty"`
	AppliedUnknownPolicies      map[string]string `json:"appliedUnknownPolicies,omitempty"`
	DecisionError               string            `json:"decisionError,omitempty"`
	SelectedModel               string            `json:"selectedModel,omitempty"`
	RecommendedModels           []string          `json:"recommendedModels,omitempty"`
	SelectionStatus             string            `json:"selectionStatus,omitempty"`
	SelectionMethod             string            `json:"selectionMethod,omitempty"`
	SelectionReason             string            `json:"selectionReason,omitempty"`
	DecisionConfidence          *float64          `json:"decisionConfidence"`
	DecisionConfidenceAvailable *bool             `json:"decisionConfidenceAvailable,omitempty"`
	SignalErrorMatches          map[string]bool   `json:"signalErrorMatches,omitempty"`
	Query                       string            `json:"query"`
	Mode                        TestQueryMode     `json:"mode"`
	MatchedSignals              []MatchedSignal   `json:"matchedSignals"`
	MatchedDecision             string            `json:"matchedDecision"`
	MatchedModels               []string          `json:"matchedModels"`
	HighlightedPath             []string          `json:"highlightedPath"`
	IsAccurate                  bool              `json:"isAccurate"`
	EvaluatedRules              []EvaluatedRule   `json:"evaluatedRules,omitempty"`
	RoutingLatency              int64             `json:"routingLatency,omitempty"`
	Warning                     string            `json:"warning,omitempty"`
	IsFallbackDecision          bool              `json:"isFallbackDecision,omitempty"` // True if matched decision is a system fallback
	FallbackReason              string            `json:"fallbackReason,omitempty"`     // Reason for fallback (e.g., "low_confidence", "no_match")
}

// TopologyTestQueryHandler handles test query requests for topology visualization
// routerAPIURL: the Router API URL for dry-run mode (real classification)
// configPath: path to config.yaml for decision display metadata
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

		if strings.TrimSpace(req.Query) == "" {
			http.Error(w, "Query cannot be empty", http.StatusBadRequest)
			return
		}

		// Default to dry-run mode
		if req.Mode == "" {
			req.Mode = TestQueryModeDryRun
		}

		start := time.Now()

		var result *TestQueryResult

		switch {
		case req.Mode != TestQueryModeDryRun:
			result = failedTestQueryResult(req, "Only dry-run mode is supported by the Router API.", http.StatusBadRequest)
		case routerAPIURL == "":
			result = failedTestQueryResult(req, "Router API is not configured.", http.StatusServiceUnavailable)
		default:
			result = callRouterAPI(r.Context(), req, routerAPIURL, configPath, credentialProvider...)
		}

		result.RoutingLatency = time.Since(start).Milliseconds()
		if r.Context().Err() != nil {
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if result.HTTPStatus != 0 {
			w.WriteHeader(result.HTTPStatus)
		}
		if err := json.NewEncoder(w).Encode(result); err != nil {
			log.Printf("Error encoding response: %v", err)
		}
	}
}

// RouterIntentRequest is the request body for Router's /api/v1/diagnostics/classify/intent
type RouterIntentRequest struct {
	Text    string               `json:"text"`
	Model   string               `json:"model,omitempty"`
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
	Confidence          *float64              `json:"confidence"`
	ConfidenceAvailable *bool                 `json:"confidence_available,omitempty"`
	DecisionName        string                `json:"decision_name"`
	UsedSignals         *RouterMatchedSignals `json:"used_signals,omitempty"`
	MatchedSignals      *RouterMatchedSignals `json:"matched_signals,omitempty"`
	UnmatchedSignals    *RouterMatchedSignals `json:"unmatched_signals,omitempty"`
}

// RouterEvalResponse is the response from Router's /api/v1/routing/preview endpoint.
type RouterEvalResponse struct {
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
	EvalTrace              json.RawMessage           `json:"eval_trace,omitempty"`
	SignalErrors           map[string]string         `json:"signal_errors,omitempty"`
	AppliedUnknownPolicies map[string]string         `json:"applied_unknown_policies,omitempty"`
	DecisionError          string                    `json:"decision_error,omitempty"`
	SelectedModel          string                    `json:"selected_model,omitempty"`
	SelectionStatus        string                    `json:"selection_status,omitempty"`
	SelectionMethod        string                    `json:"selection_method,omitempty"`
	SelectionReason        string                    `json:"selection_reason,omitempty"`
	SignalErrorMatches     map[string]bool           `json:"signal_error_matches,omitempty"`
	OriginalText           string                    `json:"original_text,omitempty"`
	DecisionResult         *RouterEvalDecisionResult `json:"decision_result,omitempty"`
	RecommendedModels      []string                  `json:"recommended_models,omitempty"`
	RoutingDecision        string                    `json:"routing_decision,omitempty"`
	SignalConfidences      map[string]float64        `json:"signal_confidences,omitempty"`
	SignalValues           map[string]float64        `json:"signal_values,omitempty"`
}

func failedTestQueryResult(req TestQueryRequest, warning string, status int) *TestQueryResult {
	result := newTestQueryResult(req)
	result.IsAccurate = false
	result.Warning = warning
	result.HTTPStatus = status
	return result
}

// callRouterAPI calls the real Router API and preserves its evaluation diagnostics.
func callRouterAPI(ctx context.Context, req TestQueryRequest, routerAPIURL, configPath string, credentialProvider ...routerauth.CredentialProvider) *TestQueryResult {
	ctx, cancel := context.WithTimeout(ctx, topologyPreviewTimeout(configPath))
	defer cancel()
	intentReq := RouterIntentRequest{
		Text:    req.Query,
		Model:   req.Model,
		Options: &RouterIntentOptions{ReturnProbabilities: true},
	}
	reqBody, err := json.Marshal(intentReq)
	if err != nil {
		return failedTestQueryResult(req, "Failed to encode preview request", http.StatusInternalServerError)
	}

	apiURL := fmt.Sprintf("%s/api/v1/routing/preview?trace=true", strings.TrimSuffix(routerAPIURL, "/"))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, apiURL, bytes.NewReader(reqBody))
	if err != nil {
		return failedTestQueryResult(req, "Failed to create Router API request", http.StatusBadGateway)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	var provider routerauth.CredentialProvider
	if len(credentialProvider) > 0 {
		provider = credentialProvider[0]
	}
	if authErr := routerauth.RewriteAuthorization(httpReq, provider); authErr != nil {
		return failedTestQueryResult(req, "Router management credential is unavailable", http.StatusServiceUnavailable)
	}

	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		log.Printf("Router API preview failed: %v", err)
		if errors.Is(err, context.DeadlineExceeded) {
			return failedTestQueryResult(req, "Router preview timed out. Retry after the classifiers finish warming up.", http.StatusGatewayTimeout)
		}
		return failedTestQueryResult(req, "Router API unavailable; live preview could not complete.", http.StatusBadGateway)
	}
	defer resp.Body.Close()

	// Even a 503 can contain the real trace and signal errors for an unresolved decision.
	var routerResp RouterEvalResponse
	decodeErr := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&routerResp)
	if resp.StatusCode != http.StatusOK {
		status := resp.StatusCode
		if status < http.StatusBadRequest {
			status = http.StatusBadGateway
		}
		warning := fmt.Sprintf("Router API error (status %d)", resp.StatusCode)
		if decodeErr == nil && routerResp.Error != nil && routerResp.Error.Message != "" {
			warning = fmt.Sprintf("%s: %s", routerResp.Error.Code, routerResp.Error.Message)
		}
		result := failedTestQueryResult(req, warning, status)
		if decodeErr == nil && (routerResp.DecisionError != "" || len(routerResp.EvalTrace) > 0) {
			result = convertRouterResponse(req, &routerResp, configPath)
			result.IsAccurate = false
			result.HTTPStatus = status
			if result.Warning == "" {
				result.Warning = warning
			}
		}
		return result
	}
	if decodeErr != nil {
		log.Printf("Failed to decode Router API preview: %v", decodeErr)
		return failedTestQueryResult(req, "Failed to parse Router API response", http.StatusBadGateway)
	}
	return convertRouterResponse(req, &routerResp, configPath)
}

func topologyPreviewTimeout(configPath string) time.Duration {
	settings := routerconfig.RoutingPreviewConfig{}
	if cfg, err := routerconfig.Parse(configPath); err == nil {
		settings = cfg.API.RoutingPreview
	}
	return settings.RequestTimeout() + routerconfig.RoutingPreviewResponseWriteAllowance
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
