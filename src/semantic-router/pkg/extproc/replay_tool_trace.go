package extproc

import (
	"encoding/json"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const (
	replayToolStepUserInput              = "user_input"
	replayToolStepAssistantToolCall      = "assistant_tool_call"
	replayToolStepClientToolResult       = "client_tool_result"
	replayToolStepAssistantFinalResponse = "assistant_final_response"

	replayToolSourceRequest  = "request"
	replayToolSourceResponse = "response"
	replayToolSourceStream   = "stream"
)

type replayToolTraceChatRequest struct {
	Messages []replayToolTraceChatMessage `json:"messages"`
}

type replayToolTraceChatResponse struct {
	Choices []replayToolTraceChatChoice `json:"choices"`
}

type replayToolTraceChatChoice struct {
	Message replayToolTraceChatMessage `json:"message"`
}

type replayToolTraceChatMessage struct {
	Role       string                    `json:"role"`
	Content    json.RawMessage           `json:"content"`
	ToolCalls  []replayToolTraceToolCall `json:"tool_calls,omitempty"`
	ToolCallID string                    `json:"tool_call_id,omitempty"`
}

type replayToolTraceToolCall struct {
	ID       string                      `json:"id"`
	Type     string                      `json:"type"`
	Function replayToolTraceFunctionCall `json:"function"`
}

type replayToolTraceFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type replayToolTraceResponseAPIResponse struct {
	Output []replayToolTraceResponseAPIItem `json:"output"`
}

type replayToolTraceResponseAPIItem struct {
	Type      string          `json:"type"`
	Role      string          `json:"role,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	Name      string          `json:"name,omitempty"`
	CallID    string          `json:"call_id,omitempty"`
	Arguments string          `json:"arguments,omitempty"`
	Output    json.RawMessage `json:"output,omitempty"`
}

type replayToolTraceCollector struct {
	steps             []routerreplay.ToolTraceStep
	toolNamesByCallID map[string]string
}

type replayStreamingIndexedToolCall struct {
	rawIndex int
	toolCall map[string]interface{}
}

func newReplayToolTraceCollector(capacity int) *replayToolTraceCollector {
	return &replayToolTraceCollector{
		steps:             make([]routerreplay.ToolTraceStep, 0, capacity),
		toolNamesByCallID: make(map[string]string),
	}
}

func (collector *replayToolTraceCollector) addUserInput(source string, role string, raw json.RawMessage) {
	collector.addTextStep(replayToolStepUserInput, source, role, extractReplayJSONText(raw))
}

func (collector *replayToolTraceCollector) addUserText(source string, role string, text string) {
	collector.addTextStep(replayToolStepUserInput, source, role, text)
}

func (collector *replayToolTraceCollector) addAssistantToolCall(
	source string,
	role string,
	toolName string,
	toolCallID string,
	arguments string,
) {
	collector.steps = append(collector.steps, routerreplay.ToolTraceStep{
		Type:       replayToolStepAssistantToolCall,
		Source:     source,
		Role:       role,
		ToolName:   toolName,
		ToolCallID: toolCallID,
		Arguments:  arguments,
	})
	if toolCallID != "" && toolName != "" {
		collector.toolNamesByCallID[toolCallID] = toolName
	}
}

func (collector *replayToolTraceCollector) addToolResult(source string, role string, raw json.RawMessage, toolCallID string) {
	collector.addToolResultText(source, role, extractReplayJSONText(raw), collector.toolNamesByCallID[toolCallID], toolCallID)
}

func (collector *replayToolTraceCollector) addToolResultText(
	source string,
	role string,
	text string,
	toolName string,
	toolCallID string,
) {
	if text == "" && toolName == "" && toolCallID == "" {
		return
	}
	collector.steps = append(collector.steps, routerreplay.ToolTraceStep{
		Type:       replayToolStepClientToolResult,
		Source:     source,
		Role:       role,
		Text:       text,
		ToolName:   toolName,
		ToolCallID: toolCallID,
	})
}

func (collector *replayToolTraceCollector) addAssistantFinalResponse(source string, role string, raw json.RawMessage) {
	collector.addTextStep(replayToolStepAssistantFinalResponse, source, role, extractReplayJSONText(raw))
}

func (collector *replayToolTraceCollector) addTextStep(stepType string, source string, role string, text string) {
	if text == "" {
		return
	}
	collector.steps = append(collector.steps, routerreplay.ToolTraceStep{
		Type:   stepType,
		Source: source,
		Role:   role,
		Text:   text,
	})
}

func (collector *replayToolTraceCollector) trace() *routerreplay.ToolTrace {
	return newReplayToolTrace(collector.steps)
}

func buildReplayRequestToolTrace(ctx *RequestContext) *routerreplay.ToolTrace {
	if ctx == nil {
		return nil
	}
	if isResponseAPIRequest(ctx) && ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.OriginalRequest != nil {
		return parseResponseAPIRequestToolTrace(ctx.ResponseAPICtx.OriginalRequest.Input)
	}
	return parseChatCompletionRequestToolTrace(ctx.OriginalRequestBody)
}

func buildReplayResponseToolTrace(
	ctx *RequestContext,
	responseBody []byte,
) *routerreplay.ToolTrace {
	if ctx == nil {
		return nil
	}
	if len(ctx.StreamingChunks) > 0 || len(ctx.StreamingToolCalls) > 0 {
		return buildReplayStreamingToolTrace(ctx)
	}
	if isResponseAPIRequest(ctx) {
		return parseResponseAPIResponseToolTrace(responseBody)
	}
	return parseChatCompletionResponseToolTrace(responseBody)
}

func mergeReplayToolTraces(
	left *routerreplay.ToolTrace,
	right *routerreplay.ToolTrace,
) *routerreplay.ToolTrace {
	switch {
	case left == nil:
		return cloneReplayToolTraceForRecord(right)
	case right == nil:
		return cloneReplayToolTraceForRecord(left)
	}

	steps := append([]routerreplay.ToolTraceStep(nil), left.Steps...)
	for _, step := range right.Steps {
		if len(steps) > 0 && replayToolTraceStepsEqual(steps[len(steps)-1], step) {
			continue
		}
		steps = append(steps, step)
	}

	return newReplayToolTrace(steps)
}

func buildReplayStreamingToolTrace(ctx *RequestContext) *routerreplay.ToolTrace {
	if ctx == nil {
		return nil
	}

	steps := make([]routerreplay.ToolTraceStep, 0, len(ctx.StreamingToolCalls)+1)
	if len(ctx.StreamingToolCalls) > 0 {
		indexes := make([]int, 0, len(ctx.StreamingToolCalls))
		for index := range ctx.StreamingToolCalls {
			indexes = append(indexes, index)
		}
		sort.Ints(indexes)

		for _, index := range indexes {
			call := ctx.StreamingToolCalls[index]
			if call == nil {
				continue
			}
			if call.ID == "" && call.Name == "" && call.Arguments == "" {
				continue
			}
			steps = append(steps, routerreplay.ToolTraceStep{
				Type:       replayToolStepAssistantToolCall,
				Source:     replayToolSourceStream,
				Role:       "assistant",
				ToolName:   call.Name,
				ToolCallID: call.ID,
				Arguments:  call.Arguments,
			})
		}
	}

	if content := strings.TrimSpace(ctx.StreamingContent); content != "" {
		steps = append(steps, routerreplay.ToolTraceStep{
			Type:   replayToolStepAssistantFinalResponse,
			Source: replayToolSourceStream,
			Role:   "assistant",
			Text:   content,
		})
	}

	return newReplayToolTrace(steps)
}

func parseChatCompletionRequestToolTrace(body []byte) *routerreplay.ToolTrace {
	messages, ok := decodeReplayChatRequestMessages(body)
	if !ok {
		return nil
	}

	collector := newReplayToolTraceCollector(len(messages))
	for _, message := range replayChatMessagesSinceLastUser(messages) {
		appendReplayChatRequestMessage(collector, message)
	}
	return collector.trace()
}

func parseChatCompletionResponseToolTrace(body []byte) *routerreplay.ToolTrace {
	if len(body) == 0 {
		return nil
	}

	var response replayToolTraceChatResponse
	if err := json.Unmarshal(body, &response); err != nil || len(response.Choices) == 0 {
		return nil
	}

	return buildReplayTraceFromChatMessage(response.Choices[0].Message, replayToolSourceResponse)
}

func buildReplayTraceFromChatMessage(
	message replayToolTraceChatMessage,
	source string,
) *routerreplay.ToolTrace {
	collector := newReplayToolTraceCollector(len(message.ToolCalls) + 1)
	for _, toolCall := range message.ToolCalls {
		collector.addAssistantToolCall(source, message.Role, toolCall.Function.Name, toolCall.ID, toolCall.Function.Arguments)
	}
	collector.addAssistantFinalResponse(source, message.Role, message.Content)
	return collector.trace()
}

func parseResponseAPIRequestToolTrace(input json.RawMessage) *routerreplay.ToolTrace {
	if len(input) == 0 {
		return nil
	}

	if trace := parseReplayResponseAPITextInput(input); trace != nil {
		return trace
	}

	items, ok := decodeReplayResponseAPIItems(input)
	if !ok {
		return nil
	}

	collector := newReplayToolTraceCollector(len(items))
	for _, item := range replayResponseAPIItemsSinceLastUser(items) {
		appendReplayResponseAPIRequestItem(collector, item)
	}
	return collector.trace()
}

func parseResponseAPIResponseToolTrace(body []byte) *routerreplay.ToolTrace {
	if len(body) == 0 {
		return nil
	}

	var response replayToolTraceResponseAPIResponse
	if err := json.Unmarshal(body, &response); err != nil || len(response.Output) == 0 {
		return nil
	}

	collector := newReplayToolTraceCollector(len(response.Output))
	for _, item := range response.Output {
		appendReplayResponseAPIResponseItem(collector, item)
	}
	return collector.trace()
}

func newReplayToolTrace(steps []routerreplay.ToolTraceStep) *routerreplay.ToolTrace {
	if len(steps) == 0 {
		return nil
	}

	clonedSteps := append([]routerreplay.ToolTraceStep(nil), steps...)
	flowParts := make([]string, 0, len(clonedSteps))
	toolNames := make([]string, 0, len(clonedSteps))
	seenToolNames := make(map[string]struct{})
	lastFlowLabel := ""

	for _, step := range clonedSteps {
		label := replayToolTraceStepLabel(step.Type)
		if label != "" && label != lastFlowLabel {
			flowParts = append(flowParts, label)
			lastFlowLabel = label
		}
		if step.ToolName == "" {
			continue
		}
		if _, ok := seenToolNames[step.ToolName]; ok {
			continue
		}
		seenToolNames[step.ToolName] = struct{}{}
		toolNames = append(toolNames, step.ToolName)
	}

	return &routerreplay.ToolTrace{
		Flow:      strings.Join(flowParts, " -> "),
		Stage:     replayToolTraceStepLabel(clonedSteps[len(clonedSteps)-1].Type),
		ToolNames: toolNames,
		Steps:     clonedSteps,
	}
}

func cloneReplayToolTraceForRecord(trace *routerreplay.ToolTrace) *routerreplay.ToolTrace {
	if trace == nil {
		return nil
	}
	cloned := *trace
	cloned.ToolNames = append([]string(nil), trace.ToolNames...)
	cloned.Steps = append([]routerreplay.ToolTraceStep(nil), trace.Steps...)
	return &cloned
}

func replayToolTraceStepsEqual(
	left routerreplay.ToolTraceStep,
	right routerreplay.ToolTraceStep,
) bool {
	return left.Type == right.Type &&
		left.Source == right.Source &&
		left.Role == right.Role &&
		left.Text == right.Text &&
		left.ToolName == right.ToolName &&
		left.ToolCallID == right.ToolCallID &&
		left.Arguments == right.Arguments
}

func replayToolTraceStepLabel(stepType string) string {
	switch stepType {
	case replayToolStepUserInput:
		return "User Query"
	case replayToolStepAssistantToolCall:
		return "LLM Tool Call"
	case replayToolStepClientToolResult:
		return "Client Tool Result"
	case replayToolStepAssistantFinalResponse:
		return "LLM Final Response"
	default:
		return ""
	}
}

func lastReplayUserMessageIndex(messages []replayToolTraceChatMessage) int {
	for index := len(messages) - 1; index >= 0; index-- {
		if messages[index].Role == "user" {
			return index
		}
	}
	return -1
}

func lastReplayResponseAPIUserIndex(items []replayToolTraceResponseAPIItem) int {
	for index := len(items) - 1; index >= 0; index-- {
		if items[index].Type == "message" && (items[index].Role == "" || items[index].Role == "user") {
			return index
		}
	}
	return -1
}

func extractReplayJSONText(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}

	var value interface{}
	if err := json.Unmarshal(raw, &value); err != nil {
		return strings.TrimSpace(string(raw))
	}

	return extractReplayValueText(value)
}

func extractReplayValueText(value interface{}) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return typed
	case []interface{}:
		parts := make([]string, 0, len(typed))
		for _, item := range typed {
			if text := extractReplayValueText(item); text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n")
	case map[string]interface{}:
		for _, key := range []string{"text", "content", "output_text", "output"} {
			if text := extractReplayValueText(typed[key]); text != "" {
				return text
			}
		}

		serialized, err := json.Marshal(typed)
		if err != nil {
			return ""
		}
		return string(serialized)
	default:
		serialized, err := json.Marshal(typed)
		if err != nil {
			return ""
		}
		return string(serialized)
	}
}

func extractStreamingToolCalls(ctx *RequestContext, chunkData map[string]interface{}) {
	if ctx == nil {
		return
	}

	choices := replayStreamingChoices(chunkData)
	if len(choices) == 0 {
		return
	}
	if ctx.StreamingToolCalls == nil {
		ctx.StreamingToolCalls = make(map[int]*StreamingToolCallState)
	}
	for _, choice := range choices {
		for _, indexedToolCall := range replayStreamingToolCalls(choice) {
			mergeReplayStreamingToolCall(ctx, indexedToolCall.rawIndex, indexedToolCall.toolCall)
		}
	}
}

func decodeReplayChatRequestMessages(body []byte) ([]replayToolTraceChatMessage, bool) {
	if len(body) == 0 {
		return nil, false
	}

	var request replayToolTraceChatRequest
	if err := json.Unmarshal(body, &request); err != nil || len(request.Messages) == 0 {
		return nil, false
	}
	return request.Messages, true
}

func replayChatMessagesSinceLastUser(messages []replayToolTraceChatMessage) []replayToolTraceChatMessage {
	start := lastReplayUserMessageIndex(messages)
	if start < 0 {
		start = 0
	}
	return messages[start:]
}

func appendReplayChatRequestMessage(collector *replayToolTraceCollector, message replayToolTraceChatMessage) {
	switch message.Role {
	case "user":
		collector.addUserInput(replayToolSourceRequest, message.Role, message.Content)
	case "assistant":
		for _, toolCall := range message.ToolCalls {
			collector.addAssistantToolCall(
				replayToolSourceRequest,
				message.Role,
				toolCall.Function.Name,
				toolCall.ID,
				toolCall.Function.Arguments,
			)
		}
	case "tool":
		collector.addToolResult(replayToolSourceRequest, message.Role, message.Content, message.ToolCallID)
	}
}

func parseReplayResponseAPITextInput(input json.RawMessage) *routerreplay.ToolTrace {
	var inputText string
	if err := json.Unmarshal(input, &inputText); err != nil {
		return nil
	}

	collector := newReplayToolTraceCollector(1)
	collector.addUserText(replayToolSourceRequest, "user", inputText)
	return collector.trace()
}

func decodeReplayResponseAPIItems(input json.RawMessage) ([]replayToolTraceResponseAPIItem, bool) {
	var items []replayToolTraceResponseAPIItem
	if err := json.Unmarshal(input, &items); err != nil || len(items) == 0 {
		return nil, false
	}
	return items, true
}

func replayResponseAPIItemsSinceLastUser(items []replayToolTraceResponseAPIItem) []replayToolTraceResponseAPIItem {
	start := lastReplayResponseAPIUserIndex(items)
	if start < 0 {
		start = 0
	}
	return items[start:]
}

func appendReplayResponseAPIRequestItem(collector *replayToolTraceCollector, item replayToolTraceResponseAPIItem) {
	switch item.Type {
	case "message":
		appendReplayResponseAPIUserMessage(collector, item)
	case "function_call":
		collector.addAssistantToolCall(replayToolSourceRequest, "assistant", item.Name, item.CallID, item.Arguments)
	case "function_call_output":
		collector.addToolResult(replayToolSourceRequest, "tool", item.Output, item.CallID)
	}
}

func appendReplayResponseAPIResponseItem(collector *replayToolTraceCollector, item replayToolTraceResponseAPIItem) {
	switch item.Type {
	case "function_call":
		collector.addAssistantToolCall(replayToolSourceResponse, "assistant", item.Name, item.CallID, item.Arguments)
	case "function_call_output":
		collector.addToolResult(replayToolSourceResponse, "tool", item.Output, item.CallID)
	case "message":
		if item.Role != "" && item.Role != "assistant" {
			return
		}
		collector.addAssistantFinalResponse(replayToolSourceResponse, "assistant", item.Content)
	}
}

func appendReplayResponseAPIUserMessage(collector *replayToolTraceCollector, item replayToolTraceResponseAPIItem) {
	role := item.Role
	if role == "" {
		role = "user"
	}
	if role != "user" {
		return
	}
	collector.addUserInput(replayToolSourceRequest, role, item.Content)
}

func replayStreamingChoices(chunkData map[string]interface{}) []map[string]interface{} {
	rawChoices, ok := chunkData["choices"].([]interface{})
	if !ok || len(rawChoices) == 0 {
		return nil
	}

	choices := make([]map[string]interface{}, 0, len(rawChoices))
	for _, rawChoice := range rawChoices {
		choice, ok := rawChoice.(map[string]interface{})
		if ok {
			choices = append(choices, choice)
		}
	}
	return choices
}

func replayStreamingToolCalls(choice map[string]interface{}) []replayStreamingIndexedToolCall {
	delta, ok := choice["delta"].(map[string]interface{})
	if !ok {
		return nil
	}
	rawToolCalls, ok := delta["tool_calls"].([]interface{})
	if !ok {
		return nil
	}

	toolCalls := make([]replayStreamingIndexedToolCall, 0, len(rawToolCalls))
	for rawIndex, rawToolCall := range rawToolCalls {
		toolCall, ok := rawToolCall.(map[string]interface{})
		if ok {
			toolCalls = append(toolCalls, replayStreamingIndexedToolCall{
				rawIndex: rawIndex,
				toolCall: toolCall,
			})
		}
	}
	return toolCalls
}

func mergeReplayStreamingToolCall(ctx *RequestContext, rawIndex int, toolCall map[string]interface{}) {
	index := replayStreamingToolCallIndex(rawIndex, toolCall)
	state := replayStreamingToolCallState(ctx, index)

	if id, ok := toolCall["id"].(string); ok && id != "" {
		state.ID = mergeReplayStreamingFragment(state.ID, id)
	}
	if fn, ok := toolCall["function"].(map[string]interface{}); ok {
		mergeReplayStreamingFunctionFragment(state, fn)
	}
}

func replayStreamingToolCallIndex(rawIndex int, toolCall map[string]interface{}) int {
	if value, ok := toolCall["index"].(float64); ok {
		return int(value)
	}
	return rawIndex
}

func replayStreamingToolCallState(ctx *RequestContext, index int) *StreamingToolCallState {
	state := ctx.StreamingToolCalls[index]
	if state != nil {
		return state
	}

	state = &StreamingToolCallState{}
	ctx.StreamingToolCalls[index] = state
	return state
}

func mergeReplayStreamingFunctionFragment(state *StreamingToolCallState, fn map[string]interface{}) {
	if name, ok := fn["name"].(string); ok && name != "" {
		state.Name = mergeReplayStreamingFragment(state.Name, name)
	}
	if arguments, ok := fn["arguments"].(string); ok && arguments != "" {
		state.Arguments = mergeReplayStreamingFragment(state.Arguments, arguments)
	}
}

func mergeReplayStreamingFragment(current string, fragment string) string {
	if fragment == "" {
		return current
	}
	if current == "" {
		return fragment
	}
	if strings.HasPrefix(fragment, current) {
		return fragment
	}
	if strings.HasPrefix(current, fragment) {
		return current
	}
	return current + fragment
}
