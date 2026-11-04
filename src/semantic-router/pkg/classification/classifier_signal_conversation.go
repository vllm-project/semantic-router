package classification

import (
	"encoding/json"
	"regexp"
	"slices"
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
	LastAssistantToolNames    []string
	LastAssistantToolArgs     []string // raw arguments, aligned with LastAssistantToolNames
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
		ruleStart := time.Now()
		value := resolveConversationValue(rule.Feature, facts)
		matched := conversationPredicateMatches(rule, value)
		elapsed := time.Since(ruleStart)
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
	case "last_tool_call":
		return countLastToolCalls(feature.Source, facts.LastAssistantToolNames, facts.LastAssistantToolArgs)
	default:
		return 0
	}
}

// countLastToolCalls counts the tool calls of the latest assistant turn that
// match the source (by name in tools, or by any entry of calls), or with match
// no_match the calls that match none. No tools and no calls match every call.
func countLastToolCalls(source config.ConversationSource, names []string, arguments []string) int {
	matchers := source.Calls
	if len(matchers) == 0 {
		matchers = []config.ToolCallMatch{{Tools: source.Tools}}
	}
	want := source.Match != "no_match"
	count := 0
	for i, name := range names {
		args := ""
		if i < len(arguments) {
			args = arguments[i]
		}
		matched := slices.ContainsFunc(matchers, func(m config.ToolCallMatch) bool {
			return toolCallMatches(m, name, args)
		})
		if matched == want {
			count++
		}
	}
	return count
}

func toolCallMatches(match config.ToolCallMatch, name string, arguments string) bool {
	if len(match.Tools) > 0 && !slices.Contains(match.Tools, name) {
		return false
	}
	if match.Pattern == "" {
		return true
	}
	pattern, err := toolArgumentPattern(match.Pattern)
	if err != nil {
		return false
	}
	value, ok := toolArgument(arguments, match.Argument)
	return ok && pattern.MatchString(value)
}

// maxToolArgumentPatterns bounds the compiled-pattern cache, which outlives
// configuration reloads; patterns past the bound are compiled on each use.
const maxToolArgumentPatterns = 256

var toolArgumentPatterns = struct {
	sync.Mutex
	compiled map[string]*regexp.Regexp
}{compiled: map[string]*regexp.Regexp{}}

func toolArgumentPattern(source string) (*regexp.Regexp, error) {
	toolArgumentPatterns.Lock()
	defer toolArgumentPatterns.Unlock()
	if pattern, ok := toolArgumentPatterns.compiled[source]; ok {
		return pattern, nil
	}
	pattern, err := regexp.Compile(source)
	if err == nil && len(toolArgumentPatterns.compiled) < maxToolArgumentPatterns {
		toolArgumentPatterns.compiled[source] = pattern
	}
	return pattern, err
}

// toolArgument returns one string field of a call's JSON arguments, or the raw
// arguments when no field is named. It reports false when the arguments are not
// a JSON object or the field is missing, null or not a string, so no pattern,
// not even one matching an empty string, can match a call it cannot read.
func toolArgument(arguments string, field string) (string, bool) {
	if field == "" {
		return arguments, true
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal([]byte(arguments), &fields) != nil {
		return "", false
	}
	raw, ok := fields[field]
	if !ok || string(raw) == "null" {
		return "", false
	}
	var value string
	if json.Unmarshal(raw, &value) != nil {
		return "", false
	}
	return value, true
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
