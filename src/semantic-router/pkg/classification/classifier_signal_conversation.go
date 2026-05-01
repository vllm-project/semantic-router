package classification

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// ConversationFacts holds request-shape facts extracted from the incoming
// request for use by the conversation signal evaluator.
type ConversationFacts struct {
	HasDeveloperMessage    bool
	UserMessageCount       int
	AssistantMessageCount  int
	SystemMessageCount     int
	ToolMessageCount       int
	ToolDefinitionCount    int
	AssistantToolCallCount int
	ToolResultCount        int
}

func (c *Classifier) evaluateConversationSignal(results *SignalResults, mu *sync.Mutex, facts ConversationFacts) {
	rules := c.Config.ConversationRules
	if len(rules) == 0 {
		return
	}

	start := time.Now()

	for _, rule := range rules {
		value := resolveConversationValue(rule.Feature, facts)
		if !conversationPredicateMatches(rule, value) {
			continue
		}

		elapsed := time.Since(start)
		mu.Lock()
		key := signalConfidenceKey(config.SignalTypeConversation, rule.Name)
		results.MatchedConversationRules = append(results.MatchedConversationRules, rule.Name)
		results.SignalConfidences[key] = 1.0
		results.SignalValues[key] = value
		mu.Unlock()

		metrics.RecordSignalExtraction(config.SignalTypeConversation, rule.Name, elapsed.Seconds())
		metrics.RecordSignalMatch(config.SignalTypeConversation, rule.Name)
	}

	elapsed := time.Since(start)
	results.Metrics.Conversation.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	results.Metrics.Conversation.Confidence = 1.0
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
	case "assistant_tool_call":
		return facts.AssistantToolCallCount
	case "assistant_tool_cycle":
		return facts.ToolResultCount
	default:
		return 0
	}
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
