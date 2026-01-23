package extproc

import (
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// setFactCheckFromSignals sets fact-check context fields from signal evaluation results
// This is called after EvaluateAllSignals to populate ctx.FactCheckNeeded and ctx.FactCheckConfidence
// Also checks for tools in the request that can provide context for hallucination detection
// Signal names: "needs_fact_check" or "no_fact_check_needed"
func (r *OpenAIRouter) setFactCheckFromSignals(ctx *RequestContext, matchedFactCheckRules []string) {
	// Check if fact-check signal was evaluated
	if len(matchedFactCheckRules) == 0 {
		logging.Debugf("No fact-check signals matched")
		return
	}

	// Check if "needs_fact_check" signal was matched
	needsFactCheck := false
	for _, rule := range matchedFactCheckRules {
		if rule == "needs_fact_check" {
			needsFactCheck = true
			break
		}
	}

	ctx.FactCheckNeeded = needsFactCheck
	// Note: We don't have the raw confidence score from the signal evaluation
	// Set a default confidence of 1.0 since the signal already passed the threshold
	ctx.FactCheckConfidence = 1.0

	// Record metrics - signal match is already recorded in classifier.go
	// No need to record again here

	logging.Infof("Fact-check from signals: needs_fact_check=%v, matched_rules=%v",
		needsFactCheck, matchedFactCheckRules)

	// Check if request has tools that can provide context for fact-checking
	// This is done here because tool context is only needed when fact-check signal is evaluated
	r.checkRequestHasTools(ctx)
}

// checkRequestHasTools checks if the request body contains tools that could provide
// context for fact-checking (tool results from previous turns or tool definitions)
func (r *OpenAIRouter) checkRequestHasTools(ctx *RequestContext) {
	// Check for RAG-injected context first (NEW)
	if ctx.RAGRetrievedContext != "" {
		ctx.HasToolsForFactCheck = true
		ctx.ToolResultsContext = ctx.RAGRetrievedContext
		logging.Infof("Using RAG-retrieved context for hallucination detection (%d chars)",
			len(ctx.RAGRetrievedContext))
		return
	}

	if len(ctx.OriginalRequestBody) == 0 {
		return
	}

	// Parse request to check for tools
	var requestMap map[string]interface{}
	if err := json.Unmarshal(ctx.OriginalRequestBody, &requestMap); err != nil {
		logging.Debugf("Failed to parse request for tool check: %v", err)
		return
	}

	// Check for tool definitions
	if tools, ok := requestMap["tools"]; ok {
		if toolsArray, isArray := tools.([]interface{}); isArray && len(toolsArray) > 0 {
			ctx.HasToolsForFactCheck = true
			logging.Debugf("Request has %d tool definitions", len(toolsArray))
		}
	}

	// Check for tool results in messages (from previous tool calls)
	if messages, ok := requestMap["messages"]; ok {
		if messagesArray, isArray := messages.([]interface{}); isArray {
			toolResults := extractToolResultsFromMessages(messagesArray)
			if len(toolResults) > 0 {
				ctx.HasToolsForFactCheck = true
				ctx.ToolResultsContext = strings.Join(toolResults, "\n\n")
				logging.Infof("Extracted %d tool results for hallucination context (%d chars)",
					len(toolResults), len(ctx.ToolResultsContext))
			}
		}
	}
}

// extractToolResultsFromMessages extracts content from tool role messages
func extractToolResultsFromMessages(messages []interface{}) []string {
	var toolResults []string

	for _, msg := range messages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}

		// Check for tool role messages
		role, ok := msgMap["role"].(string)
		if !ok || role != "tool" {
			continue
		}

		// Extract content from tool message
		if content, ok := msgMap["content"].(string); ok && content != "" {
			toolResults = append(toolResults, content)
		}
	}

	return toolResults
}

// shouldPerformHallucinationDetection determines if hallucination detection should run
// on the response based on decision plugin configuration, fact-check classification and tool presence
func (r *OpenAIRouter) shouldPerformHallucinationDetection(ctx *RequestContext) bool {
	// Must have hallucination detection models available
	if r.Classifier == nil || !r.Classifier.IsHallucinationDetectionEnabled() {
		return false
	}

	// Check if hallucination plugin is enabled for the matched decision
	if !r.isHallucinationEnabledForDecision(ctx.VSRSelectedDecision) {
		logging.Infof("Skipping hallucination detection: not enabled for decision %s (VSRSelectedDecision=%v)",
			ctx.VSRSelectedDecisionName, ctx.VSRSelectedDecision != nil)
		return false
	}

	// Only run if fact-check was needed
	if !ctx.FactCheckNeeded {
		logging.Infof("Skipping hallucination detection: fact-check not needed (FactCheckNeeded=%v)", ctx.FactCheckNeeded)
		return false
	}

	// Only run if tools were used (to have context for grounding)
	if !ctx.HasToolsForFactCheck || ctx.ToolResultsContext == "" {
		logging.Infof("Skipping hallucination detection: no tool context available (HasToolsForFactCheck=%v, ToolResultsContextLen=%d)",
			ctx.HasToolsForFactCheck, len(ctx.ToolResultsContext))
		return false
	}

	logging.Infof("Hallucination detection will be performed: decision=%s, factCheckNeeded=%v, hasTools=%v, contextLen=%d",
		ctx.VSRSelectedDecisionName, ctx.FactCheckNeeded, ctx.HasToolsForFactCheck, len(ctx.ToolResultsContext))
	return true
}

// isHallucinationEnabledForDecision checks if hallucination detection is enabled for the given decision
func (r *OpenAIRouter) isHallucinationEnabledForDecision(decision *config.Decision) bool {
	if decision == nil {
		return false
	}

	hallucinationConfig := decision.GetHallucinationConfig()
	if hallucinationConfig == nil {
		return false
	}

	return hallucinationConfig.Enabled
}

// getHallucinationActionForDecision returns the action to take when hallucination is detected
// Returns "header", "body", "block", or "none", defaults to "header" if not configured
func (r *OpenAIRouter) getHallucinationActionForDecision(decision *config.Decision) string {
	if decision == nil {
		return "header"
	}

	hallucinationConfig := decision.GetHallucinationConfig()
	if hallucinationConfig == nil {
		return "header"
	}

	action := hallucinationConfig.HallucinationAction
	if action == "" {
		return "header"
	}

	return action
}

// getUnverifiedFactualActionForDecision returns the action to take when fact-check is needed but no tool context
// Returns "header", "body", or "none", defaults to "header" if not configured
func (r *OpenAIRouter) getUnverifiedFactualActionForDecision(decision *config.Decision) string {
	if decision == nil {
		return "header"
	}

	hallucinationConfig := decision.GetHallucinationConfig()
	if hallucinationConfig == nil {
		return "header"
	}

	action := hallucinationConfig.UnverifiedFactualAction
	if action == "" {
		return "header"
	}

	return action
}
