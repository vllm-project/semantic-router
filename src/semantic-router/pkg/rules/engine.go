package rules

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/classification"
)

// RuleEngine evaluates routing rules and provides routing decisions
type RuleEngine struct {
	rules      []config.RoutingRule
	classifier *classification.Classifier
	config     *config.RouterConfig
}

// NewRuleEngine creates a new rule engine instance
func NewRuleEngine(rules []config.RoutingRule, classifier *classification.Classifier, config *config.RouterConfig) *RuleEngine {
	// Sort rules by priority (higher priority first)
	sortedRules := make([]config.RoutingRule, len(rules))
	copy(sortedRules, rules)
	sort.Slice(sortedRules, func(i, j int) bool {
		return sortedRules[i].Priority > sortedRules[j].Priority
	})

	return &RuleEngine{
		rules:      sortedRules,
		classifier: classifier,
		config:     config,
	}
}

// RoutingDecision represents the result of rule evaluation
type RoutingDecision struct {
	// Whether any rule matched
	RuleMatched bool

	// The matching rule (if any)
	MatchedRule *config.RoutingRule

	// Selected model
	SelectedModel string

	// Reasoning configuration
	UseReasoning    bool
	ReasoningEffort string

	// Decision explanation
	Explanation DecisionExplanation

	// Whether to block the request
	BlockRequest bool
	BlockMessage string

	// Additional headers to set
	Headers map[string]string

	// Processing time for rule evaluation
	EvaluationTimeMs int64
}

// DecisionExplanation provides detailed explanation of the routing decision
type DecisionExplanation struct {
	// Decision type: "rule_based", "model_based", or "fallback"
	DecisionType string

	// For rule-based decisions
	RuleName         string
	MatchedConditions []ConditionResult
	ExecutedActions   []ActionResult

	// For model-based decisions
	CategoryClassification *CategoryClassificationResult

	// Reasoning behind the decision
	Reasoning string

	// Confidence score for the decision
	Confidence float64
}

// ConditionResult represents the result of evaluating a rule condition
type ConditionResult struct {
	ConditionType string
	Matched       bool
	ActualValue   interface{}
	ExpectedValue interface{}
	Confidence    float64
	Details       string
}

// ActionResult represents the result of executing a rule action
type ActionResult struct {
	ActionType string
	Executed   bool
	Details    string
	Error      string
}

// CategoryClassificationResult represents ML classification results
type CategoryClassificationResult struct {
	Category    string
	Confidence  float64
	Probabilities map[string]float64
}

// EvaluationContext contains context for rule evaluation
type EvaluationContext struct {
	// Request content
	UserContent    string
	NonUserContent []string
	AllContent     string

	// Request metadata
	Headers   map[string]string
	RequestID string
	Timestamp time.Time

	// Model information
	OriginalModel string
	
	// External context (can be extended)
	ExternalData map[string]interface{}
}

// EvaluateRules evaluates all rules against the given context and returns routing decision
func (re *RuleEngine) EvaluateRules(ctx context.Context, evalCtx *EvaluationContext) (*RoutingDecision, error) {
	startTime := time.Now()
	
	decision := &RoutingDecision{
		RuleMatched:      false,
		SelectedModel:    re.config.DefaultModel,
		UseReasoning:     false,
		ReasoningEffort:  re.config.DefaultReasoningEffort,
		Headers:          make(map[string]string),
		Explanation: DecisionExplanation{
			DecisionType: "fallback",
			Reasoning:    "No rules matched, using default model",
		},
	}

	observability.Infof("Evaluating %d routing rules", len(re.rules))

	// Evaluate rules in priority order
	for _, rule := range re.rules {
		if !rule.Enabled {
			continue
		}

		observability.Infof("Evaluating rule: %s (priority: %d)", rule.Name, rule.Priority)

		// Check if rule matches
		matched, conditionResults, err := re.evaluateRuleConditions(ctx, &rule, evalCtx)
		if err != nil {
			observability.Errorf("Error evaluating rule %s: %v", rule.Name, err)
			continue
		}

		if matched {
			observability.Infof("Rule %s matched! Executing actions", rule.Name)
			
			// Execute rule actions
			actionResults, err := re.executeRuleActions(ctx, &rule, evalCtx, decision)
			if err != nil {
				observability.Errorf("Error executing actions for rule %s: %v", rule.Name, err)
				continue
			}

			// Update decision with rule match
			decision.RuleMatched = true
			decision.MatchedRule = &rule
			decision.Explanation = DecisionExplanation{
				DecisionType:      "rule_based",
				RuleName:          rule.Name,
				MatchedConditions: conditionResults,
				ExecutedActions:   actionResults,
				Reasoning:         fmt.Sprintf("Rule '%s' matched and actions executed", rule.Name),
				Confidence:        re.calculateRuleConfidence(conditionResults),
			}

			// Rule matched and executed, stop evaluation
			break
		}
	}

	// Calculate evaluation time
	decision.EvaluationTimeMs = time.Since(startTime).Milliseconds()

	observability.Infof("Rule evaluation completed in %dms, rule matched: %v", 
		decision.EvaluationTimeMs, decision.RuleMatched)

	return decision, nil
}

// evaluateRuleConditions evaluates all conditions for a rule
func (re *RuleEngine) evaluateRuleConditions(ctx context.Context, rule *config.RoutingRule, evalCtx *EvaluationContext) (bool, []ConditionResult, error) {
	results := make([]ConditionResult, 0, len(rule.Conditions))
	allMatched := true

	for _, condition := range rule.Conditions {
		result, err := re.evaluateCondition(ctx, &condition, evalCtx)
		if err != nil {
			return false, results, fmt.Errorf("failed to evaluate condition %s: %w", condition.Type, err)
		}

		results = append(results, result)
		if !result.Matched {
			allMatched = false
		}
	}

	return allMatched, results, nil
}

// evaluateCondition evaluates a single condition
func (re *RuleEngine) evaluateCondition(ctx context.Context, condition *config.RuleCondition, evalCtx *EvaluationContext) (ConditionResult, error) {
	result := ConditionResult{
		ConditionType: condition.Type,
		Matched:       false,
	}

	switch condition.Type {
	case "category_classification":
		return re.evaluateCategoryCondition(condition, evalCtx)
	case "content_complexity":
		return re.evaluateContentComplexityCondition(condition, evalCtx)
	case "request_header":
		return re.evaluateHeaderCondition(condition, evalCtx)
	case "time_based":
		return re.evaluateTimeCondition(condition, evalCtx)
	case "pattern_match":
		return re.evaluatePatternCondition(condition, evalCtx)
	default:
		return result, fmt.Errorf("unsupported condition type: %s", condition.Type)
	}
}

// evaluateCategoryCondition evaluates category classification conditions
func (re *RuleEngine) evaluateCategoryCondition(condition *config.RuleCondition, evalCtx *EvaluationContext) (ConditionResult, error) {
	result := ConditionResult{
		ConditionType: condition.Type,
		ExpectedValue: fmt.Sprintf("%s %s %.2f", condition.Category, condition.Operator, condition.Threshold),
	}

	if re.classifier == nil {
		return result, fmt.Errorf("classifier not available for category classification")
	}

	// Perform classification
	categoryName, confidence, err := re.classifier.ClassifyCategory(evalCtx.AllContent)
	if err != nil {
		return result, fmt.Errorf("classification failed: %w", err)
	}

	result.ActualValue = fmt.Sprintf("%s (confidence: %.2f)", categoryName, confidence)
	result.Confidence = float64(confidence)
	result.Details = fmt.Sprintf("Classified as '%s' with confidence %.2f", categoryName, confidence)

	// Check if category matches and confidence meets threshold
	categoryMatches := (condition.Category == "" || categoryName == condition.Category)
	confidenceMatches := re.compareFloat(float64(confidence), condition.Threshold, condition.Operator)

	result.Matched = categoryMatches && confidenceMatches

	return result, nil
}

// evaluateContentComplexityCondition evaluates content complexity conditions
func (re *RuleEngine) evaluateContentComplexityCondition(condition *config.RuleCondition, evalCtx *EvaluationContext) (ConditionResult, error) {
	result := ConditionResult{
		ConditionType: condition.Type,
		ExpectedValue: fmt.Sprintf("%s %s %.2f", condition.Metric, condition.Operator, condition.Threshold),
	}

	var actualValue float64
	switch condition.Metric {
	case "token_count":
		// Simple token count estimation (split by whitespace)
		actualValue = float64(len(strings.Fields(evalCtx.AllContent)))
	case "character_count":
		actualValue = float64(len(evalCtx.AllContent))
	case "line_count":
		actualValue = float64(len(strings.Split(evalCtx.AllContent, "\n")))
	default:
		return result, fmt.Errorf("unsupported complexity metric: %s", condition.Metric)
	}

	result.ActualValue = actualValue
	result.Details = fmt.Sprintf("%s: %.0f", condition.Metric, actualValue)
	result.Matched = re.compareFloat(actualValue, condition.Threshold, condition.Operator)

	return result, nil
}

// evaluateHeaderCondition evaluates request header conditions
func (re *RuleEngine) evaluateHeaderCondition(condition *config.RuleCondition, evalCtx *EvaluationContext) (ConditionResult, error) {
	result := ConditionResult{
		ConditionType: condition.Type,
		ExpectedValue: fmt.Sprintf("%s %s %s", condition.HeaderName, condition.Operator, condition.Value),
	}

	headerValue, exists := evalCtx.Headers[condition.HeaderName]
	if !exists {
		result.ActualValue = "<missing>"
		result.Details = fmt.Sprintf("Header '%s' not found", condition.HeaderName)
		result.Matched = false
		return result, nil
	}

	result.ActualValue = headerValue
	result.Details = fmt.Sprintf("Header '%s' = '%s'", condition.HeaderName, headerValue)
	result.Matched = re.compareString(headerValue, condition.Value, condition.Operator)

	return result, nil
}

// evaluateTimeCondition evaluates time-based conditions
func (re *RuleEngine) evaluateTimeCondition(condition *config.RuleCondition, evalCtx *EvaluationContext) (ConditionResult, error) {
	result := ConditionResult{
		ConditionType: condition.Type,
		ExpectedValue: condition.TimeRange,
	}

	// Simple time range check (could be extended)
	currentHour := evalCtx.Timestamp.Hour()
	result.ActualValue = fmt.Sprintf("Hour: %d", currentHour)
	result.Details = fmt.Sprintf("Current time: %s", evalCtx.Timestamp.Format("15:04:05"))
	
	// For now, always match (this could be extended with proper time range parsing)
	result.Matched = true

	return result, nil
}

// evaluatePatternCondition evaluates pattern matching conditions
func (re *RuleEngine) evaluatePatternCondition(condition *config.RuleCondition, evalCtx *EvaluationContext) (ConditionResult, error) {
	result := ConditionResult{
		ConditionType: condition.Type,
		ExpectedValue: condition.PatternMatch,
	}

	// Simple pattern matching (contains check)
	matched := strings.Contains(strings.ToLower(evalCtx.AllContent), strings.ToLower(condition.PatternMatch))
	
	result.ActualValue = evalCtx.AllContent
	result.Details = fmt.Sprintf("Pattern '%s' in content", condition.PatternMatch)
	result.Matched = matched

	return result, nil
}

// executeRuleActions executes all actions for a matched rule
func (re *RuleEngine) executeRuleActions(ctx context.Context, rule *config.RoutingRule, evalCtx *EvaluationContext, decision *RoutingDecision) ([]ActionResult, error) {
	results := make([]ActionResult, 0, len(rule.Actions))

	for _, action := range rule.Actions {
		result := re.executeAction(&action, evalCtx, decision)
		results = append(results, result)
	}

	return results, nil
}

// executeAction executes a single rule action
func (re *RuleEngine) executeAction(action *config.RuleAction, evalCtx *EvaluationContext, decision *RoutingDecision) ActionResult {
	result := ActionResult{
		ActionType: action.Type,
		Executed:   false,
	}

	switch action.Type {
	case "route_to_model":
		if action.Model != "" {
			decision.SelectedModel = action.Model
			result.Executed = true
			result.Details = fmt.Sprintf("Routed to model: %s", action.Model)
		}
	case "enable_reasoning":
		decision.UseReasoning = action.EnableReasoning
		if action.ReasoningEffort != "" {
			decision.ReasoningEffort = action.ReasoningEffort
		}
		result.Executed = true
		result.Details = fmt.Sprintf("Reasoning: %v, Effort: %s", action.EnableReasoning, action.ReasoningEffort)
	case "set_headers":
		for key, value := range action.Headers {
			decision.Headers[key] = value
		}
		result.Executed = true
		result.Details = fmt.Sprintf("Set %d headers", len(action.Headers))
	case "block_request":
		decision.BlockRequest = true
		decision.BlockMessage = action.BlockWithMessage
		result.Executed = true
		result.Details = fmt.Sprintf("Blocked: %s", action.BlockWithMessage)
	default:
		result.Error = fmt.Sprintf("Unsupported action type: %s", action.Type)
	}

	return result
}

// Helper functions for condition evaluation

func (re *RuleEngine) compareFloat(actual, expected float64, operator string) bool {
	switch operator {
	case "gte", ">=":
		return actual >= expected
	case "gt", ">":
		return actual > expected
	case "lte", "<=":
		return actual <= expected
	case "lt", "<":
		return actual < expected
	case "equals", "==":
		return actual == expected
	default:
		return false
	}
}

func (re *RuleEngine) compareString(actual, expected, operator string) bool {
	switch operator {
	case "equals":
		return actual == expected
	case "contains":
		return strings.Contains(strings.ToLower(actual), strings.ToLower(expected))
	default:
		return false
	}
}

func (re *RuleEngine) calculateRuleConfidence(conditionResults []ConditionResult) float64 {
	if len(conditionResults) == 0 {
		return 0.0
	}

	total := 0.0
	for _, result := range conditionResults {
		if result.Matched {
			total += result.Confidence
		}
	}

	return total / float64(len(conditionResults))
}