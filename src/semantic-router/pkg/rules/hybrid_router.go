package rules

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/classification"
)

// HybridRouter combines rule-based and model-based routing
type HybridRouter struct {
	ruleEngine *RuleEngine
	classifier *classification.Classifier
	config     *config.RouterConfig
}

// NewHybridRouter creates a new hybrid router instance
func NewHybridRouter(routerConfig *config.RouterConfig, classifier *classification.Classifier) *HybridRouter {
	var ruleEngine *RuleEngine
	
	// Initialize rule engine if rules are configured
	if len(routerConfig.RoutingRules) > 0 {
		ruleEngine = NewRuleEngine(routerConfig.RoutingRules, classifier, routerConfig)
	}

	return &HybridRouter{
		ruleEngine: ruleEngine,
		classifier: classifier,
		config:     routerConfig,
	}
}

// RouteRequest determines the best model and configuration for a request
func (hr *HybridRouter) RouteRequest(ctx context.Context, userContent string, nonUserContent []string, headers map[string]string, originalModel string) (*RoutingDecision, error) {
	startTime := time.Now()

	// Combine all content for analysis
	allContent := userContent
	if len(nonUserContent) > 0 {
		allContent = strings.Join(append([]string{userContent}, nonUserContent...), " ")
	}

	// Create evaluation context
	evalCtx := &EvaluationContext{
		UserContent:    userContent,
		NonUserContent: nonUserContent,
		AllContent:     allContent,
		Headers:        headers,
		RequestID:      headers["x-request-id"],
		Timestamp:      time.Now(),
		OriginalModel:  originalModel,
		ExternalData:   make(map[string]interface{}),
	}

	// Determine routing strategy
	strategy := hr.getRoutingStrategy()
	observability.Infof("Using routing strategy: %s", strategy)

	var decision *RoutingDecision
	var err error

	switch strategy {
	case "rules":
		decision, err = hr.routeWithRules(ctx, evalCtx)
	case "model":
		decision, err = hr.routeWithModel(ctx, evalCtx)
	case "hybrid":
		decision, err = hr.routeWithHybrid(ctx, evalCtx)
	default:
		// Fallback to model-based routing
		decision, err = hr.routeWithModel(ctx, evalCtx)
	}

	if err != nil {
		return nil, fmt.Errorf("routing failed: %w", err)
	}

	// Update evaluation time
	decision.EvaluationTimeMs = time.Since(startTime).Milliseconds()

	observability.Infof("Routing completed: model=%s, reasoning=%v, time=%dms",
		decision.SelectedModel, decision.UseReasoning, decision.EvaluationTimeMs)

	return decision, nil
}

// getRoutingStrategy determines which routing strategy to use
func (hr *HybridRouter) getRoutingStrategy() string {
	if hr.config.RoutingStrategy.Type != "" {
		return hr.config.RoutingStrategy.Type
	}

	// Auto-determine strategy based on configuration
	hasRules := len(hr.config.RoutingRules) > 0 && hr.ruleEngine != nil
	hasModel := hr.classifier != nil

	if hasRules && hasModel {
		return "hybrid"
	} else if hasRules {
		return "rules"
	} else if hasModel {
		return "model"
	}

	return "model" // Default fallback
}

// routeWithRules uses only rule-based routing
func (hr *HybridRouter) routeWithRules(ctx context.Context, evalCtx *EvaluationContext) (*RoutingDecision, error) {
	if hr.ruleEngine == nil {
		return hr.createFallbackDecision(evalCtx, "No rule engine available")
	}

	decision, err := hr.ruleEngine.EvaluateRules(ctx, evalCtx)
	if err != nil {
		return nil, fmt.Errorf("rule evaluation failed: %w", err)
	}

	// If no rule matched and fallback to model is enabled, try model routing
	if !decision.RuleMatched && hr.config.RoutingStrategy.RuleRouting.FallbackToModel {
		observability.Infof("No rules matched, falling back to model-based routing")
		modelDecision, modelErr := hr.routeWithModel(ctx, evalCtx)
		if modelErr == nil {
			// Merge model decision with rule decision
			decision.SelectedModel = modelDecision.SelectedModel
			decision.UseReasoning = modelDecision.UseReasoning
			decision.ReasoningEffort = modelDecision.ReasoningEffort
			decision.Explanation.DecisionType = "fallback_to_model"
			decision.Explanation.CategoryClassification = modelDecision.Explanation.CategoryClassification
			decision.Explanation.Reasoning = "Rules did not match, used model-based routing"
		}
	}

	return decision, nil
}

// routeWithModel uses only model-based routing
func (hr *HybridRouter) routeWithModel(ctx context.Context, evalCtx *EvaluationContext) (*RoutingDecision, error) {
	if hr.classifier == nil {
		return hr.createFallbackDecision(evalCtx, "No classifier available")
	}

	// Perform classification
	categoryName, confidence, err := hr.classifier.ClassifyCategory(evalCtx.AllContent)
	if err != nil {
		return nil, fmt.Errorf("classification failed: %w", err)
	}

	// Get model for category
	selectedModel := hr.classifier.ClassifyAndSelectBestModel(evalCtx.AllContent)
	if selectedModel == "" {
		selectedModel = hr.config.DefaultModel
	}

	// Get reasoning configuration for the category
	useReasoning, reasoningEffort := hr.getReasoningConfig(categoryName, selectedModel)

	decision := &RoutingDecision{
		RuleMatched:     false,
		SelectedModel:   selectedModel,
		UseReasoning:    useReasoning,
		ReasoningEffort: reasoningEffort,
		Headers:         make(map[string]string),
		Explanation: DecisionExplanation{
			DecisionType: "model_based",
			CategoryClassification: &CategoryClassificationResult{
				Category:   categoryName,
				Confidence: float64(confidence),
			},
			Reasoning:  fmt.Sprintf("Model-based classification selected '%s' with confidence %.2f", categoryName, confidence),
			Confidence: float64(confidence),
		},
	}

	return decision, nil
}

// routeWithHybrid uses hybrid routing (rules first, then model)
func (hr *HybridRouter) routeWithHybrid(ctx context.Context, evalCtx *EvaluationContext) (*RoutingDecision, error) {
	// First try rule-based routing
	decision, err := hr.routeWithRules(ctx, evalCtx)
	if err != nil {
		return nil, fmt.Errorf("rule evaluation failed in hybrid mode: %w", err)
	}

	// If rule matched, use rule decision
	if decision.RuleMatched {
		observability.Infof("Rule matched in hybrid mode: %s", decision.MatchedRule.Name)
		return decision, nil
	}

	// No rule matched, try model-based routing
	observability.Infof("No rules matched in hybrid mode, trying model-based routing")
	
	// Check model routing configuration
	if !hr.config.RoutingStrategy.ModelRouting.Enabled {
		observability.Infof("Model routing disabled, using rule decision with default model")
		return decision, nil
	}

	modelDecision, err := hr.routeWithModel(ctx, evalCtx)
	if err != nil {
		observability.Errorf("Model routing failed in hybrid mode: %v", err)
		return decision, nil // Return rule decision as fallback
	}

	// Check if model confidence meets threshold
	confidenceThreshold := hr.config.RoutingStrategy.ModelRouting.ConfidenceThreshold
	if confidenceThreshold > 0 && modelDecision.Explanation.Confidence < confidenceThreshold {
		observability.Infof("Model confidence %.2f below threshold %.2f, using default routing",
			modelDecision.Explanation.Confidence, confidenceThreshold)
		return decision, nil
	}

	// Use model decision
	modelDecision.Explanation.DecisionType = "hybrid_model"
	modelDecision.Explanation.Reasoning = fmt.Sprintf("No rules matched, used model-based routing with confidence %.2f", 
		modelDecision.Explanation.Confidence)
	
	return modelDecision, nil
}

// createFallbackDecision creates a fallback decision when routing fails
func (hr *HybridRouter) createFallbackDecision(evalCtx *EvaluationContext, reason string) (*RoutingDecision, error) {
	decision := &RoutingDecision{
		RuleMatched:     false,
		SelectedModel:   hr.config.DefaultModel,
		UseReasoning:    false,
		ReasoningEffort: hr.config.DefaultReasoningEffort,
		Headers:         make(map[string]string),
		Explanation: DecisionExplanation{
			DecisionType: "fallback",
			Reasoning:    reason,
			Confidence:   0.0,
		},
	}

	return decision, nil
}

// getReasoningConfig determines reasoning configuration for a category/model
func (hr *HybridRouter) getReasoningConfig(categoryName, modelName string) (bool, string) {
	// Find category configuration
	for _, category := range hr.config.Categories {
		if category.Name == categoryName {
			// Find model in category
			for _, modelScore := range category.ModelScores {
				if modelScore.Model == modelName && modelScore.UseReasoning != nil {
					reasoningEffort := category.ReasoningEffort
					if reasoningEffort == "" {
						reasoningEffort = hr.config.DefaultReasoningEffort
					}
					return *modelScore.UseReasoning, reasoningEffort
				}
			}
		}
	}

	// Default reasoning configuration
	return false, hr.config.DefaultReasoningEffort
}

// IsRulesEnabled returns true if rule-based routing is enabled
func (hr *HybridRouter) IsRulesEnabled() bool {
	return hr.ruleEngine != nil && hr.config.RoutingStrategy.RuleRouting.Enabled
}

// IsModelEnabled returns true if model-based routing is enabled
func (hr *HybridRouter) IsModelEnabled() bool {
	return hr.classifier != nil && hr.config.RoutingStrategy.ModelRouting.Enabled
}

// GetRuleCount returns the number of configured rules
func (hr *HybridRouter) GetRuleCount() int {
	if hr.ruleEngine == nil {
		return 0
	}
	return len(hr.ruleEngine.rules)
}