package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

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
			// Simulate mode: use local config-based simulation
			result = evaluateTestQueryLocally(req, configPath)
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
		result := evaluateTestQueryLocally(req, configPath)
		result.Warning = fmt.Sprintf("Failed to marshal request: %v", err)
		return result
	}

	// Call Router API
	apiURL := fmt.Sprintf("%s/api/v1/classify/intent", strings.TrimSuffix(routerAPIURL, "/"))
	httpReq, err := http.NewRequest("POST", apiURL, bytes.NewReader(reqBody))
	if err != nil {
		result := evaluateTestQueryLocally(req, configPath)
		result.Warning = fmt.Sprintf("Failed to create request: %v", err)
		return result
	}
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		log.Printf("Router API call failed: %v, falling back to local simulation", err)
		result := evaluateTestQueryLocally(req, configPath)
		result.Warning = fmt.Sprintf("Router API unavailable (%v), using local simulation", err)
		result.IsAccurate = false
		return result
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		log.Printf("Router API returned %d: %s", resp.StatusCode, string(body))
		result := evaluateTestQueryLocally(req, configPath)
		result.Warning = fmt.Sprintf("Router API error (status %d), using local simulation", resp.StatusCode)
		result.IsAccurate = false
		return result
	}

	// Parse response
	var routerResp RouterIntentResponse
	if err := json.NewDecoder(resp.Body).Decode(&routerResp); err != nil {
		log.Printf("Failed to decode Router API response: %v", err)
		result := evaluateTestQueryLocally(req, configPath)
		result.Warning = "Failed to parse Router API response, using local simulation"
		result.IsAccurate = false
		return result
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

// evaluateTestQueryLocally evaluates a test query using local config (simulate mode)
func evaluateTestQueryLocally(req TestQueryRequest, configPath string) *TestQueryResult {
	// Load current config
	data, err := os.ReadFile(configPath)
	if err != nil {
		return &TestQueryResult{
			Query:           req.Query,
			Mode:            req.Mode,
			HighlightedPath: []string{"client"},
			Warning:         fmt.Sprintf("Failed to read config: %v", err),
		}
	}

	var rawConfig map[string]interface{}
	if unmarshalErr := yaml.Unmarshal(data, &rawConfig); unmarshalErr != nil {
		return &TestQueryResult{
			Query:           req.Query,
			Mode:            req.Mode,
			HighlightedPath: []string{"client"},
			Warning:         fmt.Sprintf("Failed to parse config: %v", unmarshalErr),
		}
	}

	// Parse config using router's config parser
	parsedConfig, parseErr := routerconfig.Parse(configPath)
	if parseErr != nil {
		log.Printf("Warning: Config parse error (using fallback): %v", parseErr)
	}

	return evaluateTestQuery(req, parsedConfig, rawConfig)
}

// evaluateTestQuery evaluates a test query against the configuration
func evaluateTestQuery(req TestQueryRequest, parsedConfig *routerconfig.RouterConfig, rawConfig map[string]interface{}) *TestQueryResult {
	result := &TestQueryResult{
		Query:           req.Query,
		Mode:            req.Mode,
		MatchedSignals:  []MatchedSignal{},
		MatchedModels:   []string{},
		HighlightedPath: []string{},
		IsAccurate:      req.Mode == TestQueryModeDryRun,
		EvaluatedRules:  []EvaluatedRule{},
	}

	// Start with client node
	result.HighlightedPath = append(result.HighlightedPath, "client")

	queryLower := strings.ToLower(req.Query)

	// If we have parsed config with decisions, use proper evaluation
	if parsedConfig != nil && len(parsedConfig.IntelligentRouting.Decisions) > 0 {
		evaluateDecisionsWithConfig(result, req.Query, queryLower, parsedConfig)
	} else {
		// Fallback: use raw config for basic signal matching
		evaluateSignalsFromRawConfig(result, req.Query, queryLower, rawConfig)
	}

	return result
}

// evaluateDecisionsWithConfig evaluates query against parsed router config
func evaluateDecisionsWithConfig(result *TestQueryResult, query, queryLower string, cfg *routerconfig.RouterConfig) {
	// Evaluate global plugins
	evaluateGlobalPlugins(result, query, queryLower, cfg)

	// Collect all matched signals (store both original and normalized versions)
	matchedSignalNames := make(map[string]bool)

	// Evaluate signals from config
	signals := cfg.IntelligentRouting.Signals

	// 1. Keyword rules
	for _, kw := range signals.KeywordRules {
		if matchKeywordRule(query, queryLower, kw) {
			signal := MatchedSignal{
				Type:       "keyword",
				Name:       kw.Name,
				Confidence: 1.0,
				Reason:     "Keyword match",
			}
			result.MatchedSignals = append(result.MatchedSignals, signal)
			key := fmt.Sprintf("keyword:%s", kw.Name)
			matchedSignalNames[key] = true
			matchedSignalNames[fmt.Sprintf("keyword:%s", normalizeSignalName(kw.Name))] = true
		}
	}

	// 2. Domain signals (from Categories)
	for _, category := range signals.Categories {
		if matchDomainCategory(queryLower, category) {
			signal := MatchedSignal{
				Type:       "domain",
				Name:       category.Name,
				Confidence: 0.8, // Simulated confidence
				Reason:     fmt.Sprintf("Domain match: %s", category.Description),
			}
			result.MatchedSignals = append(result.MatchedSignals, signal)
			key := fmt.Sprintf("domain:%s", category.Name)
			matchedSignalNames[key] = true
			matchedSignalNames[fmt.Sprintf("domain:%s", normalizeSignalName(category.Name))] = true
		}
	}

	// 3. Embedding rules
	for _, emb := range signals.EmbeddingRules {
		if matchEmbeddingRule(queryLower, emb) {
			signal := MatchedSignal{
				Type:       "embedding",
				Name:       emb.Name,
				Confidence: 0.85,
				Reason:     "Embedding similarity match",
			}
			result.MatchedSignals = append(result.MatchedSignals, signal)
			key := fmt.Sprintf("embedding:%s", emb.Name)
			matchedSignalNames[key] = true
			matchedSignalNames[fmt.Sprintf("embedding:%s", normalizeSignalName(emb.Name))] = true
		}
	}

	// 4. FactCheck rules (these are ML-based, simplified here)
	for _, fc := range signals.FactCheckRules {
		// FactCheck rules are ML-based, we use simplified keyword matching for simulation
		if matchFactCheckRuleSimple(queryLower, fc.Name) {
			signal := MatchedSignal{
				Type:       "fact_check",
				Name:       fc.Name,
				Confidence: 0.9,
				Reason:     fmt.Sprintf("Fact check signal: %s", fc.Description),
			}
			result.MatchedSignals = append(result.MatchedSignals, signal)
			key := fmt.Sprintf("fact_check:%s", fc.Name)
			matchedSignalNames[key] = true
			matchedSignalNames[fmt.Sprintf("fact_check:%s", normalizeSignalName(fc.Name))] = true
		}
	}

	// Add signal group nodes to path if signals matched
	if len(result.MatchedSignals) > 0 {
		signalTypes := make(map[string]bool)
		for _, s := range result.MatchedSignals {
			signalTypes[s.Type] = true
		}
		for st := range signalTypes {
			result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("signal-group-%s", st))
		}
	}

	// Evaluate decisions
	var matchedDecision *routerconfig.Decision
	highestPriority := -1

	for _, decision := range cfg.IntelligentRouting.Decisions {
		evalRule := evaluateDecisionRules(decision, matchedSignalNames)
		result.EvaluatedRules = append(result.EvaluatedRules, evalRule)

		if evalRule.IsMatch {
			priority := decision.Priority
			if priority == 0 {
				priority = 100 // Default priority
			}
			if matchedDecision == nil || priority > highestPriority {
				matchedDecision = &decision
				highestPriority = priority
			}
		}
	}

	// Set matched decision
	if matchedDecision != nil {
		result.MatchedDecision = matchedDecision.Name
		result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("decision-%s", matchedDecision.Name))

		// Add algorithm node if configured (after decision, before plugin chain)
		if matchedDecision.Algorithm != nil && matchedDecision.Algorithm.Type != "" {
			result.HighlightedPath = append(result.HighlightedPath,
				fmt.Sprintf("algorithm-%s", matchedDecision.Name))
		}

		// Add plugin chain if configured (after decision/algorithm, before models)
		if len(matchedDecision.Plugins) > 0 {
			result.HighlightedPath = append(result.HighlightedPath,
				fmt.Sprintf("plugin-chain-%s", matchedDecision.Name))
		}

		// Get matched models from decision (last in the chain)
		for _, modelRef := range matchedDecision.ModelRefs {
			modelName := modelRef.Model
			if modelRef.LoRAName != "" {
				modelName = modelRef.LoRAName
			}
			result.MatchedModels = append(result.MatchedModels, modelName)

			// Add physical model ID (for node highlighting - models are aggregated by physical model)
			physicalKey := generatePhysicalModelKey(modelRef)
			normalizedPhysicalKey := normalizeModelName(physicalKey)
			physicalModelId := fmt.Sprintf("model-%s", normalizedPhysicalKey)

			// Add config-specific model ID (for edge highlighting - edges differ by reasoning mode)
			configKey := generateModelKey(modelRef)
			normalizedConfigKey := normalizeModelName(configKey)
			configModelId := fmt.Sprintf("model-%s", normalizedConfigKey)

			// Add both IDs to path (dedup if same)
			result.HighlightedPath = append(result.HighlightedPath, physicalModelId)
			if configModelId != physicalModelId {
				result.HighlightedPath = append(result.HighlightedPath, configModelId)
			}
		}
	} else {
		// No decision matched, use default model via default route
		// Add default-route to highlighted path for visual feedback
		result.HighlightedPath = append(result.HighlightedPath, "default-route")

		if cfg.BackendModels.DefaultModel != "" {
			result.MatchedModels = append(result.MatchedModels, cfg.BackendModels.DefaultModel)
			normalizedModel := normalizeModelName(cfg.BackendModels.DefaultModel)
			result.HighlightedPath = append(result.HighlightedPath,
				fmt.Sprintf("model-%s", normalizedModel))
		}
		result.Warning = "No decision matched, using default model"
	}
}

// evaluateGlobalPlugins evaluates global plugins (jailbreak, PII, cache)
func evaluateGlobalPlugins(result *TestQueryResult, query, queryLower string, cfg *routerconfig.RouterConfig) {
	// Check prompt guard (jailbreak detection)
	if cfg.PromptGuard.Enabled {
		result.HighlightedPath = append(result.HighlightedPath, "global-plugin-jailbreak")

		// Simple jailbreak pattern detection for simulation
		jailbreakPatterns := []string{
			"ignore previous", "ignore all", "disregard", "bypass",
			"pretend you", "act as", "jailbreak", "dan mode",
		}
		for _, pattern := range jailbreakPatterns {
			if strings.Contains(queryLower, pattern) {
				signal := MatchedSignal{
					Type:       "jailbreak",
					Name:       "prompt_injection",
					Confidence: 0.9,
					Reason:     fmt.Sprintf("Jailbreak pattern detected: %s", pattern),
				}
				result.MatchedSignals = append(result.MatchedSignals, signal)
				break
			}
		}
	}

	// Check PII detection
	if cfg.PIIModel.ModelID != "" {
		result.HighlightedPath = append(result.HighlightedPath, "global-plugin-pii")

		// Simple PII pattern detection for simulation
		if containsPIIPatterns(query) {
			signal := MatchedSignal{
				Type:       "pii",
				Name:       "pii_detected",
				Confidence: 0.85,
				Reason:     "PII pattern detected",
			}
			result.MatchedSignals = append(result.MatchedSignals, signal)
		}
	}

	// Check semantic cache
	if cfg.SemanticCache.Enabled {
		result.HighlightedPath = append(result.HighlightedPath, "global-plugin-cache")
	}
}

// evaluateDecisionRules evaluates a decision's rules against matched signals
func evaluateDecisionRules(decision routerconfig.Decision, matchedSignals map[string]bool) EvaluatedRule {
	evalRule := EvaluatedRule{
		DecisionName: decision.Name,
		RuleOperator: "AND", // Default
		Conditions:   []string{},
		MatchedCount: 0,
		TotalCount:   0,
		Priority:     decision.Priority,
	}

	if decision.Priority == 0 {
		evalRule.Priority = 100
	}

	// Get models for this decision
	for _, modelRef := range decision.ModelRefs {
		modelName := modelRef.Model
		if modelRef.LoRAName != "" {
			modelName = modelRef.LoRAName
		}
		evalRule.MatchedModels = append(evalRule.MatchedModels, modelName)
	}

	// Check if rules are configured (empty struct check)
	rules := decision.Rules
	if rules.Operator == "" && len(rules.Conditions) == 0 {
		// No rules = always match (default decision)
		evalRule.IsMatch = true
		return evalRule
	}

	evalRule.RuleOperator = strings.ToUpper(rules.Operator)
	if evalRule.RuleOperator == "" {
		evalRule.RuleOperator = "AND"
	}

	// Evaluate conditions
	for _, cond := range rules.Conditions {
		condKey := fmt.Sprintf("%s:%s", cond.Type, cond.Name)
		normalizedCondKey := fmt.Sprintf("%s:%s", cond.Type, normalizeSignalName(cond.Name))
		evalRule.Conditions = append(evalRule.Conditions, condKey)
		evalRule.TotalCount++

		// Check both original and normalized keys for matching
		if matchedSignals[condKey] || matchedSignals[normalizedCondKey] {
			evalRule.MatchedCount++
		}
	}

	// Determine if decision matches based on operator
	if evalRule.TotalCount == 0 {
		evalRule.IsMatch = true // No conditions = always match
	} else if evalRule.RuleOperator == "OR" {
		evalRule.IsMatch = evalRule.MatchedCount > 0
	} else { // AND
		evalRule.IsMatch = evalRule.MatchedCount == evalRule.TotalCount
	}

	return evalRule
}

// matchKeywordRule checks if query matches a keyword rule
func matchKeywordRule(query, queryLower string, kw routerconfig.KeywordRule) bool {
	matchedCount := 0
	for _, keyword := range kw.Keywords {
		var matched bool
		if kw.CaseSensitive {
			// Case-sensitive: use original query and keyword as-is
			matched = strings.Contains(query, keyword)
		} else {
			// Case-insensitive: use lowercase versions
			matched = strings.Contains(queryLower, strings.ToLower(keyword))
		}
		if matched {
			matchedCount++
		}
	}

	// Apply operator logic
	operator := strings.ToUpper(kw.Operator)
	switch operator {
	case "AND":
		return matchedCount == len(kw.Keywords)
	default: // "", "OR"
		return matchedCount > 0
	}
}

// matchDomainCategory checks if query likely matches a domain category
func matchDomainCategory(queryLower string, category routerconfig.Category) bool {
	// Simple keyword-based domain matching for simulation
	domainKeywords := map[string][]string{
		"math":      {"calculate", "math", "equation", "solve", "formula", "number", "计算", "数学"},
		"code":      {"code", "program", "function", "debug", "compile", "代码", "编程"},
		"coding":    {"code", "program", "function", "debug", "compile", "代码", "编程"},
		"writing":   {"write", "essay", "story", "article", "blog", "写作", "文章"},
		"reasoning": {"think", "reason", "analyze", "logic", "deduce", "推理", "分析"},
		"creative":  {"creative", "imagine", "design", "art", "创意", "设计"},
		"science":   {"science", "experiment", "theory", "research", "科学", "研究"},
		"general":   {"help", "what", "how", "why", "explain", "帮助", "什么", "如何"},
	}

	categoryLower := strings.ToLower(category.Name)
	if keywords, ok := domainKeywords[categoryLower]; ok {
		for _, kw := range keywords {
			if strings.Contains(queryLower, kw) {
				return true
			}
		}
	}

	// Also check description keywords
	if category.Description != "" {
		descWords := strings.Fields(strings.ToLower(category.Description))
		for _, word := range descWords {
			if len(word) > 3 && strings.Contains(queryLower, word) {
				return true
			}
		}
	}

	return false
}

// matchEmbeddingRule checks if query might match an embedding rule
func matchEmbeddingRule(queryLower string, emb routerconfig.EmbeddingRule) bool {
	// Simple simulation: check if query contains embedding name or candidate terms
	embLower := strings.ToLower(emb.Name)
	if strings.Contains(queryLower, embLower) {
		return true
	}

	// Check candidates
	for _, candidate := range emb.Candidates {
		candidateLower := strings.ToLower(candidate)
		// Check for significant word overlap
		candidateWords := strings.Fields(candidateLower)
		queryWords := strings.Fields(queryLower)
		matchCount := 0
		for _, cw := range candidateWords {
			for _, qw := range queryWords {
				if len(cw) > 3 && cw == qw {
					matchCount++
				}
			}
		}
		if matchCount >= 2 {
			return true
		}
	}

	return false
}

// matchFactCheckRuleSimple checks if query might need fact checking (simplified)
func matchFactCheckRuleSimple(queryLower string, ruleName string) bool {
	// Simplified fact-check detection based on rule name and query patterns
	factCheckPatterns := []string{
		"is it true", "fact check", "verify", "真的吗", "是真的",
		"actually", "claim", "statistics", "data shows",
	}

	if strings.Contains(strings.ToLower(ruleName), "needs_fact_check") {
		for _, pattern := range factCheckPatterns {
			if strings.Contains(queryLower, pattern) {
				return true
			}
		}
	}

	return false
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

// generateModelKey generates a unique key for a model configuration
// matching the frontend format: model|reasoning|effort-{level}|lora-{name}
// Used for edge highlighting (edges differ by reasoning mode)
func generateModelKey(modelRef routerconfig.ModelRef) string {
	parts := []string{modelRef.Model}
	if modelRef.UseReasoning != nil && *modelRef.UseReasoning {
		parts = append(parts, "reasoning")
	}
	if modelRef.ReasoningEffort != "" {
		parts = append(parts, fmt.Sprintf("effort-%s", modelRef.ReasoningEffort))
	}
	if modelRef.LoRAName != "" {
		parts = append(parts, fmt.Sprintf("lora-%s", modelRef.LoRAName))
	}
	return strings.Join(parts, "|")
}

// generatePhysicalModelKey generates a key for the physical model only
// This excludes reasoning configuration since same physical model can have different modes
// Used for node highlighting (models are aggregated by physical model)
func generatePhysicalModelKey(modelRef routerconfig.ModelRef) string {
	parts := []string{modelRef.Model}
	if modelRef.LoRAName != "" {
		parts = append(parts, fmt.Sprintf("lora-%s", modelRef.LoRAName))
	}
	return strings.Join(parts, "|")
}

// containsPIIPatterns checks for common PII patterns
func containsPIIPatterns(text string) bool {
	// Simple pattern checks for simulation
	patterns := []string{
		"@",                          // Email
		"xxx-xx-xxxx", "XXX-XX-XXXX", // SSN-like
		"credit card", "card number",
		"phone number", "电话",
		"身份证", "银行卡",
	}

	textLower := strings.ToLower(text)
	for _, p := range patterns {
		if strings.Contains(textLower, strings.ToLower(p)) {
			return true
		}
	}

	return false
}

// evaluateSignalsFromRawConfig handles evaluation when parsed config is not available
func evaluateSignalsFromRawConfig(result *TestQueryResult, query, queryLower string, rawConfig map[string]interface{}) {
	result.Warning = "Using simplified evaluation (config parsing unavailable)"

	// Try to extract signals from raw config
	if ir, ok := rawConfig["intelligent_routing"].(map[string]interface{}); ok {
		// Extract signals
		if signals, ok := ir["signals"].(map[string]interface{}); ok {
			// Keyword rules
			if keywordRules, ok := signals["keyword_rules"].([]interface{}); ok {
				for _, kw := range keywordRules {
					if kwMap, ok := kw.(map[string]interface{}); ok {
						name, _ := kwMap["name"].(string)
						keywords, _ := kwMap["keywords"].([]interface{})
						for _, k := range keywords {
							if keyword, ok := k.(string); ok {
								if strings.Contains(queryLower, strings.ToLower(keyword)) {
									result.MatchedSignals = append(result.MatchedSignals, MatchedSignal{
										Type:       "keyword",
										Name:       name,
										Confidence: 1.0,
									})
									break
								}
							}
						}
					}
				}
			}

			// Categories (domains)
			if categories, ok := signals["categories"].([]interface{}); ok {
				for _, c := range categories {
					if cMap, ok := c.(map[string]interface{}); ok {
						name, _ := cMap["name"].(string)
						desc, _ := cMap["description"].(string)
						// Simple match on name
						if strings.Contains(queryLower, strings.ToLower(name)) {
							result.MatchedSignals = append(result.MatchedSignals, MatchedSignal{
								Type:       "domain",
								Name:       name,
								Confidence: 0.7,
								Reason:     desc,
							})
						}
					}
				}
			}
		}

		// Extract decisions
		if decisions, ok := ir["decisions"].([]interface{}); ok {
			for _, d := range decisions {
				if dMap, ok := d.(map[string]interface{}); ok {
					name, _ := dMap["name"].(string)
					result.EvaluatedRules = append(result.EvaluatedRules, EvaluatedRule{
						DecisionName: name,
						IsMatch:      false,
					})
				}
			}
			// Use first decision as fallback
			if len(decisions) > 0 {
				if dMap, ok := decisions[0].(map[string]interface{}); ok {
					name, _ := dMap["name"].(string)
					result.MatchedDecision = name
					result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("decision-%s", name))
				}
			}
		}
	}

	// Fallback default model
	if bm, ok := rawConfig["backend_models"].(map[string]interface{}); ok {
		if defaultModel, ok := bm["default_model"].(string); ok {
			result.MatchedModels = append(result.MatchedModels, defaultModel)
			result.HighlightedPath = append(result.HighlightedPath, fmt.Sprintf("model-%s", defaultModel))
		}
	}
}
