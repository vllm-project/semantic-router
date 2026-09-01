package classification

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ConversationFacts holds request-shape facts extracted from the incoming
// request for use by the conversation signal evaluator.
type ConversationFacts struct {
	HasDeveloperMessage       bool
	UserMessageCount          int
	AssistantMessageCount     int
	SystemMessageCount        int
	ToolMessageCount          int
	ToolDefinitionCount       int
	ToolChoiceRequired        bool
	ToolChoiceNone            bool
	AssistantToolCallCount    int
	ToolResultCount           int
	ImageContentCount         int
	LastMessageRole           string
	LastMessageToolResult     bool
	LastMessageFlowToolResult bool
	LastAssistantToolCall     bool
	LastUserAfterToolResult   bool
}

func (c *Classifier) evaluateConversationSignal(
	results *SignalResults,
	mu *sync.Mutex,
	facts ConversationFacts,
	usedSignals map[string]bool,
) {
	rules := c.Config.ConversationRules
	if len(rules) == 0 {
		return
	}

	start := time.Now()
	matchedAny := false

	for _, rule := range rules {
		if !signalRuleUsed(
			usedSignals,
			config.SignalTypeConversation,
			rule.Name,
		) {
			continue
		}
		value := resolveConversationValue(rule.Feature, facts)
		matched := conversationPredicateMatches(rule, value)
		elapsed := time.Since(start)
		mu.Lock()
		key := signalConfidenceKey(config.SignalTypeConversation, rule.Name)
		results.SignalValues[key] = value
		if matched {
			matchedAny = true
			results.SignalConfidences[key] = 1.0
			results.MatchedConversationRules = append(results.MatchedConversationRules, rule.Name)
		} else {
			results.SignalConfidences[key] = 0
		}
		mu.Unlock()

		c.recordSignalExtraction(config.SignalTypeConversation, rule.Name, elapsed.Seconds())
		if matched {
			c.recordSignalMatch(config.SignalTypeConversation, rule.Name)
		}
	}

	elapsed := time.Since(start)
	results.Metrics.Conversation.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if matchedAny {
		results.Metrics.Conversation.Confidence = 1.0
	} else {
		results.Metrics.Conversation.Confidence = 0
	}
	logging.Debugf("[Signal Computation] Conversation signal evaluation completed in %v", elapsed)
}

func resolveConversationValue(feature config.ConversationFeature, facts ConversationFacts) float64 {
	raw := resolveConversationRawCount(feature, facts)

	switch feature.Type {
	case "exists":
		if raw > 0 {
			return 1.0
		}
		return 0.0
	default:
		return float64(raw)
	}
}

func resolveConversationRawCount(feature config.ConversationFeature, facts ConversationFacts) int {
	switch feature.Source.Type {
	case "message":
		return countMessagesByRole(feature.Source.Role, facts)
	case "tool_definition":
		return facts.ToolDefinitionCount
	case "tool_choice_required":
		return conversationBoolCount(facts.ToolChoiceRequired)
	case "tool_choice_none":
		return conversationBoolCount(facts.ToolChoiceNone)
	case "assistant_tool_call":
		return facts.AssistantToolCallCount
	case "assistant_tool_cycle":
		return facts.ToolResultCount
	case "active_tool_loop":
		return activeToolLoopCount(facts)
	case "flow_tool_state":
		return conversationBoolCount(facts.LastMessageFlowToolResult)
	case "image_content":
		return facts.ImageContentCount
	default:
		return 0
	}
}

func conversationBoolCount(value bool) int {
	if value {
		return 1
	}
	return 0
}

func activeToolLoopCount(facts ConversationFacts) int {
	if facts.LastMessageToolResult ||
		facts.LastMessageRole == "tool" ||
		facts.LastUserAfterToolResult ||
		facts.LastAssistantToolCall {
		return 1
	}
	return 0
}

func countMessagesByRole(role string, facts ConversationFacts) int {
	switch role {
	case "user":
		return facts.UserMessageCount
	case "assistant":
		return facts.AssistantMessageCount
	case "system":
		return facts.SystemMessageCount
	case "developer":
		if facts.HasDeveloperMessage {
			return 1
		}
		return 0
	case "tool":
		return facts.ToolMessageCount
	case "non_user":
		total := facts.AssistantMessageCount + facts.SystemMessageCount + facts.ToolMessageCount
		if facts.HasDeveloperMessage {
			total++
		}
		return total
	case "":
		total := facts.UserMessageCount + facts.AssistantMessageCount + facts.SystemMessageCount + facts.ToolMessageCount
		if facts.HasDeveloperMessage {
			total++
		}
		return total
	default:
		return 0
	}
}

func conversationPredicateMatches(rule config.ConversationRule, value float64) bool {
	if rule.Feature.Type == "exists" {
		return value > 0
	}
	if rule.Predicate == nil {
		return true
	}
	p := rule.Predicate
	if p.GT != nil && value <= *p.GT {
		return false
	}
	if p.GTE != nil && value < *p.GTE {
		return false
	}
	if p.LT != nil && value >= *p.LT {
		return false
	}
	if p.LTE != nil && value > *p.LTE {
		return false
	}
	return true
}
