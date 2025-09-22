package rules

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// RuleManagementAPI provides HTTP endpoints for rule management
type RuleManagementAPI struct {
	hybridRouter *HybridRouter
	config       *config.RouterConfig
}

// NewRuleManagementAPI creates a new rule management API instance
func NewRuleManagementAPI(hybridRouter *HybridRouter, routerConfig *config.RouterConfig) *RuleManagementAPI {
	return &RuleManagementAPI{
		hybridRouter: hybridRouter,
		config:       routerConfig,
	}
}

// Rule management request/response types

type CreateRuleRequest struct {
	Rule config.RoutingRule `json:"rule"`
}

type CreateRuleResponse struct {
	ID      string `json:"id"`
	Message string `json:"message"`
}

type ListRulesResponse struct {
	Rules []RuleInfo `json:"rules"`
	Count int        `json:"count"`
}

type RuleInfo struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Enabled     bool   `json:"enabled"`
	Priority    int    `json:"priority"`
	Conditions  int    `json:"condition_count"`
	Actions     int    `json:"action_count"`
}

type RuleEvaluationRequest struct {
	Content       string            `json:"content"`
	Headers       map[string]string `json:"headers,omitempty"`
	OriginalModel string            `json:"original_model,omitempty"`
}

type RuleEvaluationResponse struct {
	Decision *RoutingDecision `json:"decision"`
	Success  bool             `json:"success"`
	Error    string           `json:"error,omitempty"`
}

type DecisionExplanationResponse struct {
	DecisionID  string            `json:"decision_id"`
	Explanation DecisionExplanation `json:"explanation"`
	Timestamp   time.Time         `json:"timestamp"`
}

// RegisterRoutes registers rule management API routes
func (api *RuleManagementAPI) RegisterRoutes(mux *http.ServeMux) {
	// Rule management endpoints
	mux.HandleFunc("/api/v1/rules", api.handleRules)
	mux.HandleFunc("/api/v1/rules/", api.handleRuleOperations)
	
	// Rule evaluation and debugging endpoints
	mux.HandleFunc("/api/v1/rules/evaluate", api.handleRuleEvaluation)
	mux.HandleFunc("/api/v1/rules/explain/", api.handleDecisionExplanation)
	mux.HandleFunc("/api/v1/rules/test", api.handleRuleTest)
}

// handleRules handles GET /api/v1/rules (list) and POST /api/v1/rules (create)
func (api *RuleManagementAPI) handleRules(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		api.listRules(w, r)
	case http.MethodPost:
		api.createRule(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleRuleOperations handles rule-specific operations (GET, PUT, DELETE /api/v1/rules/{id})
func (api *RuleManagementAPI) handleRuleOperations(w http.ResponseWriter, r *http.Request) {
	// Extract rule ID from path
	ruleName := r.URL.Path[len("/api/v1/rules/"):]
	if ruleName == "" {
		http.Error(w, "Rule name required", http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodGet:
		api.getRule(w, r, ruleName)
	case http.MethodPut:
		api.updateRule(w, r, ruleName)
	case http.MethodDelete:
		api.deleteRule(w, r, ruleName)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// listRules returns all configured rules
func (api *RuleManagementAPI) listRules(w http.ResponseWriter, r *http.Request) {
	rules := make([]RuleInfo, 0, len(api.config.RoutingRules))
	
	for _, rule := range api.config.RoutingRules {
		ruleInfo := RuleInfo{
			Name:        rule.Name,
			Description: rule.Description,
			Enabled:     rule.Enabled,
			Priority:    rule.Priority,
			Conditions:  len(rule.Conditions),
			Actions:     len(rule.Actions),
		}
		rules = append(rules, ruleInfo)
	}

	response := ListRulesResponse{
		Rules: rules,
		Count: len(rules),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// createRule creates a new routing rule
func (api *RuleManagementAPI) createRule(w http.ResponseWriter, r *http.Request) {
	var req CreateRuleRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	// Validate rule
	if req.Rule.Name == "" {
		http.Error(w, "Rule name is required", http.StatusBadRequest)
		return
	}

	// Check if rule already exists
	for _, existingRule := range api.config.RoutingRules {
		if existingRule.Name == req.Rule.Name {
			http.Error(w, "Rule with this name already exists", http.StatusConflict)
			return
		}
	}

	// Add rule to configuration (in real implementation, this would persist to storage)
	api.config.RoutingRules = append(api.config.RoutingRules, req.Rule)

	observability.Infof("Created new routing rule: %s", req.Rule.Name)

	response := CreateRuleResponse{
		ID:      req.Rule.Name,
		Message: "Rule created successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// getRule returns a specific rule
func (api *RuleManagementAPI) getRule(w http.ResponseWriter, r *http.Request, ruleName string) {
	for _, rule := range api.config.RoutingRules {
		if rule.Name == ruleName {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(rule)
			return
		}
	}

	http.Error(w, "Rule not found", http.StatusNotFound)
}

// updateRule updates an existing rule
func (api *RuleManagementAPI) updateRule(w http.ResponseWriter, r *http.Request, ruleName string) {
	var updatedRule config.RoutingRule
	if err := json.NewDecoder(r.Body).Decode(&updatedRule); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	// Find and update rule
	for i, rule := range api.config.RoutingRules {
		if rule.Name == ruleName {
			// Ensure name consistency
			updatedRule.Name = ruleName
			api.config.RoutingRules[i] = updatedRule
			
			observability.Infof("Updated routing rule: %s", ruleName)
			
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(updatedRule)
			return
		}
	}

	http.Error(w, "Rule not found", http.StatusNotFound)
}

// deleteRule deletes a rule
func (api *RuleManagementAPI) deleteRule(w http.ResponseWriter, r *http.Request, ruleName string) {
	for i, rule := range api.config.RoutingRules {
		if rule.Name == ruleName {
			// Remove rule from slice
			api.config.RoutingRules = append(api.config.RoutingRules[:i], api.config.RoutingRules[i+1:]...)
			
			observability.Infof("Deleted routing rule: %s", ruleName)
			
			w.WriteHeader(http.StatusNoContent)
			return
		}
	}

	http.Error(w, "Rule not found", http.StatusNotFound)
}

// handleRuleEvaluation evaluates rules against provided content
func (api *RuleManagementAPI) handleRuleEvaluation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req RuleEvaluationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if req.Content == "" {
		http.Error(w, "Content is required", http.StatusBadRequest)
		return
	}

	// Create evaluation context
	headers := req.Headers
	if headers == nil {
		headers = make(map[string]string)
	}

	// Evaluate rules using hybrid router
	decision, err := api.hybridRouter.RouteRequest(r.Context(), req.Content, nil, headers, req.OriginalModel)
	
	response := RuleEvaluationResponse{
		Success: err == nil,
	}

	if err != nil {
		response.Error = err.Error()
		observability.Errorf("Rule evaluation failed: %v", err)
	} else {
		response.Decision = decision
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleDecisionExplanation provides detailed explanation for a decision
func (api *RuleManagementAPI) handleDecisionExplanation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract decision ID from path
	decisionID := r.URL.Path[len("/api/v1/rules/explain/"):]
	if decisionID == "" {
		http.Error(w, "Decision ID required", http.StatusBadRequest)
		return
	}

	// In a real implementation, this would lookup stored decision explanations
	// For now, return a placeholder response
	response := DecisionExplanationResponse{
		DecisionID:  decisionID,
		Timestamp:   time.Now(),
		Explanation: DecisionExplanation{
			DecisionType: "placeholder",
			Reasoning:    "Decision explanation feature requires persistent storage implementation",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleRuleTest tests a rule with sample data
func (api *RuleManagementAPI) handleRuleTest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Rule    config.RoutingRule    `json:"rule"`
		Content string                `json:"content"`
		Headers map[string]string     `json:"headers,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	// Create a temporary rule engine for testing
	testRules := []config.RoutingRule{req.Rule}
	testEngine := NewRuleEngine(testRules, api.hybridRouter.classifier, api.config)

	// Create evaluation context
	headers := req.Headers
	if headers == nil {
		headers = make(map[string]string)
	}

	evalCtx := &EvaluationContext{
		UserContent:    req.Content,
		NonUserContent: nil,
		AllContent:     req.Content,
		Headers:        headers,
		RequestID:      "test-" + strconv.FormatInt(time.Now().Unix(), 10),
		Timestamp:      time.Now(),
		OriginalModel:  "test",
		ExternalData:   make(map[string]interface{}),
	}

	// Evaluate the test rule
	decision, err := testEngine.EvaluateRules(r.Context(), evalCtx)
	
	response := RuleEvaluationResponse{
		Success: err == nil,
	}

	if err != nil {
		response.Error = err.Error()
	} else {
		response.Decision = decision
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}