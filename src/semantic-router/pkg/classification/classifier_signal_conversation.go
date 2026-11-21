package classification

import (
	"encoding/json"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
	LastMessageRole           string
	LastMessageToolResult     bool
	LastMessageFlowToolResult bool
	LastAssistantToolCall     bool
	LastUserAfterToolResult   bool
	// TurnIndex is the zero-based user turn index within the session: tool
	// continuations are grouped with the user message that started them,
	// matching the router-replay TurnIndex semantics. It stays 0 when no user
	// history is visible to the router.
	TurnIndex int
	// HistoryTokens estimates the token size of the conversation history the
	// request carries, excluding the current user message.
	HistoryTokens int
	// FailedToolResults is ToolFailureCount.Failed for the request.
	FailedToolResults int
}

// toolFailureRecoveryRun is how many consecutive successful tool results
// clear the failed_tool_result count, the default number of recoveries the
// progress gate needs before a session switches back.
const toolFailureRecoveryRun = 2

// ToolFailureCount is the failed_tool_result fact: the failed tool results
// of the current user turn since the last two successes with no failure
// between them. The client resends the history every turn, so the count
// needs no session state and reads the same on every replica.
type ToolFailureCount struct {
	Failed     int
	successRun int
}

// ToolResultStatus is what a tool result says about its call.
type ToolResultStatus int

const (
	// ToolResultUnknown is a result that says neither, such as a command
	// still running or a tool whose results carry no status.
	ToolResultUnknown ToolResultStatus = iota
	ToolResultFailed
	ToolResultSucceeded
)

// Observe counts one tool result, in conversation order. An unknown result
// changes nothing.
func (c *ToolFailureCount) Observe(status ToolResultStatus) {
	switch status {
	case ToolResultFailed:
		c.Failed++
		c.successRun = 0
	case ToolResultSucceeded:
		c.successRun++
		if c.successRun >= toolFailureRecoveryRun {
			c.Failed = 0
		}
	}
}

// codexShellTools are the Codex tools that run a command, above whose
// output Codex writes the command's exit code.
var codexShellTools = map[string]bool{"shell": true, "shell_command": true, "exec_command": true, "write_stdin": true}

// codexHeaderKeys start the lines Codex writes above a command's output.
var codexHeaderKeys = []string{
	"Chunk ID: ", "Wall time: ", "Exit code: ", "Process exited with code ",
	"Process running with session ID ", "Original token count: ", "Total output lines: ",
}

// ToolResultStatusOf reads the status of one tool result of a request in
// format. call is the call it answers, nil when the request does not carry
// it. A result whose flag, Anthropic is_error, is set failed. Claude Code
// sets the flag on every failure, but a clear flag says the work succeeded
// only for its Bash tool run in the foreground: a read, a plan update or a
// command started in the background prove nothing. OpenAI Chat and Responses
// outputs have no flag, so there only the exit code Codex writes above a
// shell command's output counts.
func ToolResultStatusOf(format llmprotocol.WireFormat, call *llmprotocol.ToolCall, isError *bool, text string) ToolResultStatus {
	switch {
	case isError != nil && *isError:
		return ToolResultFailed
	case call == nil || call.Namespace != "":
		return ToolResultUnknown
	case format == llmprotocol.AnthropicMessagesV1:
		if call.Name == "Bash" && !runsInBackground(call.Arguments) {
			return ToolResultSucceeded
		}
		return ToolResultUnknown
	case codexShellTools[call.Name]:
		return codexExitStatus(text)
	default:
		return ToolResultUnknown
	}
}

func runsInBackground(arguments string) bool {
	var input struct {
		RunInBackground bool `json:"run_in_background"`
	}
	return json.Unmarshal([]byte(arguments), &input) == nil && input.RunInBackground
}

// codexExitStatus reads the header Codex ends with an "Output:" line above a
// command's output. Every line of it must be a Codex header line, so neither
// the command's output nor text from elsewhere passes for a status. A zero
// exit code succeeded, any other failed, and a command still running has
// none yet.
func codexExitStatus(text string) ToolResultStatus {
	header, _, found := strings.Cut(text, "\nOutput:\n")
	if !found {
		return ToolResultUnknown
	}
	status := ToolResultUnknown
	for _, line := range strings.Split(header, "\n") {
		key := codexHeaderKey(line)
		switch key {
		case "":
			return ToolResultUnknown
		case "Exit code: ", "Process exited with code ":
			code, err := strconv.Atoi(line[len(key):])
			if err != nil {
				return ToolResultUnknown
			}
			status = ToolResultFailed
			if code == 0 {
				status = ToolResultSucceeded
			}
		}
	}
	return status
}

func codexHeaderKey(line string) string {
	for _, key := range codexHeaderKeys {
		if strings.HasPrefix(line, key) {
			return key
		}
	}
	return ""
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
	case "turn_index":
		return facts.TurnIndex
	case "session_history_tokens":
		return facts.HistoryTokens
	case "failed_tool_result":
		return facts.FailedToolResults
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
