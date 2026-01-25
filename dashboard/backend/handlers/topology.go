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
}

// MatchedSignal represents a matched signal
type MatchedSignal struct {
	Type       string  `json:"type"`
	Name       string  `json:"name"`
	Confidence float64 `json:"confidence"`
	Reason     string  `json:"reason,omitempty"`
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
			result = callRouterAPI(req, routerAPIURL, configPath)
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

// RouterIntentResponse is the response from Router's /api/v1/classify/intent
type RouterIntentResponse struct {
	Classification struct {
		Category         string  `json:"category"`
		Confidence       float64 `json:"confidence"`
		ProcessingTimeMs int64   `json:"processing_time_ms"`
	} `json:"classification"`
	Probabilities    map[string]float64 `json:"probabilities,omitempty"`
	RecommendedModel string             `json:"recommended_model,omitempty"`
	RoutingDecision  string             `json:"routing_decision,omitempty"`
	MatchedSignals   *struct {
		Keywords     []string `json:"keywords,omitempty"`
		Embeddings   []string `json:"embeddings,omitempty"`
		Domains      []string `json:"domains,omitempty"`
		FactCheck    []string `json:"fact_check,omitempty"`
		UserFeedback []string `json:"user_feedback,omitempty"`
		Preferences  []string `json:"preferences,omitempty"`
		Language     []string `json:"language,omitempty"`
		Latency      []string `json:"latency,omitempty"`
		Context      []string `json:"context,omitempty"`
	} `json:"matched_signals,omitempty"`
	DecisionResult *struct {
		DecisionName string   `json:"decision_name"`
		Confidence   float64  `json:"confidence"`
		MatchedRules []string `json:"matched_rules"`
	} `json:"decision_result,omitempty"`
}

// callRouterAPI calls the real Router API for classification
func callRouterAPI(req TestQueryRequest, routerAPIURL, configPath string) *TestQueryResult {
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

	// Call Router API
	apiURL := fmt.Sprintf("%s/api/v1/classify/intent", strings.TrimSuffix(routerAPIURL, "/"))
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
		body, _ := io.ReadAll(resp.Body)
		log.Printf("Router API returned %d: %s", resp.StatusCode, string(body))
		return &TestQueryResult{
			Query:           req.Query,
			Mode:            req.Mode,
			HighlightedPath: []string{"client"},
			Warning:         fmt.Sprintf("Router API error (status %d)", resp.StatusCode),
			IsAccurate:      false,
		}
	}

	// Parse response
	var routerResp RouterIntentResponse
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

// convertRouterResponse converts Router API response to TestQueryResult
func convertRouterResponse(req TestQueryRequest, routerResp *RouterIntentResponse, configPath string) *TestQueryResult {
	result := &TestQueryResult{
		Query:           req.Query,
		Mode:            req.Mode,
		MatchedSignals:  []MatchedSignal{},
		MatchedModels:   []string{},
		HighlightedPath: []string{"client"},
		IsAccurate:      true,
		EvaluatedRules:  []EvaluatedRule{},
	}

	// Convert matched signals
	if routerResp.MatchedSignals != nil {
		for _, kw := range routerResp.MatchedSignals.Keywords {
			result.MatchedSignals = append(result.MatchedSignals, MatchedSignal{
				Type:       "keyword",
				Name:       kw,
				Confidence: 1.0,
				Reason:     "Keyword rule matched",
			})
			result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("signal-keyword-%s", kw))
		}
		for _, emb := range routerResp.MatchedSignals.Embeddings {
			result.MatchedSignals = append(result.MatchedSignals, MatchedSignal{
				Type:       "embedding",
				Name:       emb,
				Confidence: 0.85,
				Reason:     "Embedding similarity matched",
			})
			result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("signal-embedding-%s", emb))
		}
		for _, domain := range routerResp.MatchedSignals.Domains {
			result.MatchedSignals = append(result.MatchedSignals, MatchedSignal{
				Type:       "domain",
				Name:       domain,
				Confidence: routerResp.Classification.Confidence,
				Reason:     "Domain classification matched",
			})
			result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("signal-domain-%s", domain))
		}
		for _, fc := range routerResp.MatchedSignals.FactCheck {
			result.MatchedSignals = append(result.MatchedSignals, MatchedSignal{
				Type:       "fact_check",
				Name:       fc,
				Confidence: 0.9,
				Reason:     "Fact check signal matched",
			})
		}
		for _, pref := range routerResp.MatchedSignals.Preferences {
			result.MatchedSignals = append(result.MatchedSignals, MatchedSignal{
				Type:       "preference",
				Name:       pref,
				Confidence: 1.0,
				Reason:     "User preference matched",
			})
			result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("signal-preference-%s", pref))
		}
		for _, uf := range routerResp.MatchedSignals.UserFeedback {
			result.MatchedSignals = append(result.MatchedSignals, MatchedSignal{
				Type:       "user_feedback",
				Name:       uf,
				Confidence: 1.0,
				Reason:     "User feedback matched",
			})
			result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("signal-user_feedback-%s", uf))
		}
		for _, lang := range routerResp.MatchedSignals.Language {
			result.MatchedSignals = append(result.MatchedSignals, MatchedSignal{
				Type:       "language",
				Name:       lang,
				Confidence: 0.95,
				Reason:     "Language detected",
			})
			result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("signal-language-%s", lang))
		}
		for _, lat := range routerResp.MatchedSignals.Latency {
			result.MatchedSignals = append(result.MatchedSignals, MatchedSignal{
				Type:       "latency",
				Name:       lat,
				Confidence: 1.0,
				Reason:     "Latency requirement matched",
			})
			result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("signal-latency-%s", lat))
		}
		for _, ctx := range routerResp.MatchedSignals.Context {
			result.MatchedSignals = append(result.MatchedSignals, MatchedSignal{
				Type:       "context",
				Name:       ctx,
				Confidence: 1.0,
				Reason:     "Context token count matched",
			})
			result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("signal-context-%s", ctx))
		}
	}

	// Add signal group nodes if signals matched
	if len(result.MatchedSignals) > 0 {
		signalTypes := make(map[string]bool)
		for _, s := range result.MatchedSignals {
			signalTypes[s.Type] = true
		}
		for st := range signalTypes {
			result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("signal-group-%s", st))
		}
	}

	// Set decision result
	if routerResp.DecisionResult != nil {
		result.MatchedDecision = routerResp.DecisionResult.DecisionName
		result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("decision-%s", routerResp.DecisionResult.DecisionName))

		// Add evaluated rule
		result.EvaluatedRules = append(result.EvaluatedRules, EvaluatedRule{
			DecisionName: routerResp.DecisionResult.DecisionName,
			Conditions:   routerResp.DecisionResult.MatchedRules,
			MatchedCount: len(routerResp.DecisionResult.MatchedRules),
			TotalCount:   len(routerResp.DecisionResult.MatchedRules),
			IsMatch:      true,
		})
	} else if routerResp.RoutingDecision != "" {
		result.MatchedDecision = routerResp.RoutingDecision
		result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("decision-%s", routerResp.RoutingDecision))

		// Check if this is a system fallback decision
		if isSystemFallbackDecision(routerResp.RoutingDecision) {
			result.IsFallbackDecision = true
			result.FallbackReason = getFallbackReason(routerResp.RoutingDecision)
			// Add fallback decision node to highlighted path
			result.HighlightedPath = append(result.HighlightedPath, "fallback-decision")
		}
	}

	// Set matched model
	if routerResp.RecommendedModel != "" {
		result.MatchedModels = append(result.MatchedModels, routerResp.RecommendedModel)
		// Normalize model ID to match frontend format (replace non-alphanumeric with -)
		normalizedModel := normalizeModelName(routerResp.RecommendedModel)
		result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("model-%s", normalizedModel))
	}

	// Build normalized matched signal names map for decision evaluation
	matchedSignalNames := make(map[string]bool)
	for _, signal := range result.MatchedSignals {
		// Store both original and normalized versions for matching
		key := fmt.Sprintf("%s:%s", signal.Type, signal.Name)
		normalizedKey := fmt.Sprintf("%s:%s", signal.Type, normalizeSignalName(signal.Name))
		matchedSignalNames[key] = true
		matchedSignalNames[normalizedKey] = true
	}

	// Load config to get evaluated rules (for showing all decisions)
	if parsedConfig, err := routerconfig.Parse(configPath); err == nil && parsedConfig != nil {
		for _, decision := range parsedConfig.IntelligentRouting.Decisions {
			if result.MatchedDecision != "" && decision.Name == result.MatchedDecision {
				continue // Already added
			}
			rule := EvaluatedRule{
				DecisionName: decision.Name,
				RuleOperator: strings.ToUpper(decision.Rules.Operator),
				IsMatch:      false,
				Priority:     decision.Priority,
			}
			if rule.RuleOperator == "" {
				rule.RuleOperator = "AND"
			}

			// Evaluate each condition against matched signals
			for _, cond := range decision.Rules.Conditions {
				condKey := fmt.Sprintf("%s:%s", cond.Type, cond.Name)
				normalizedCondKey := fmt.Sprintf("%s:%s", cond.Type, normalizeSignalName(cond.Name))
				rule.Conditions = append(rule.Conditions, condKey)
				rule.TotalCount++

				// Check if condition matches (using both original and normalized)
				if matchedSignalNames[condKey] || matchedSignalNames[normalizedCondKey] {
					rule.MatchedCount++
				}
			}

			// Determine if decision matches based on operator
			if rule.TotalCount == 0 {
				rule.IsMatch = true // No conditions = always match
			} else if rule.RuleOperator == "OR" {
				rule.IsMatch = rule.MatchedCount > 0
			} else { // AND
				rule.IsMatch = rule.MatchedCount == rule.TotalCount
			}

			result.EvaluatedRules = append(result.EvaluatedRules, rule)
		}
	}

	return result
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
