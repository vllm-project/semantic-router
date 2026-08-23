package extproc

import (
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const (
	replayToolStepUserInput              = "user_input"
	replayToolStepAssistantToolCall      = "assistant_tool_call"
	replayToolStepClientToolResult       = "client_tool_result"
	replayToolStepAssistantFinalResponse = "assistant_final_response"
	replayToolStepAssistantReasoningDone = "assistant_reasoning_complete"

	replayToolSourceRequest  = "request"
	replayToolSourceResponse = "response"
	replayToolSourceStream   = "stream"
)

// extractSemanticPromptAndTools captures replay metadata from the semantic
// request before recorder truncation. Wire bodies are never reparsed here.
func extractSemanticPromptAndTools(request *llmprotocol.Request) (prompt string, toolDefs string) {
	if request == nil {
		return "", ""
	}
	for index := len(request.Messages) - 1; index >= 0; index-- {
		if request.Messages[index].Role == llmprotocol.RoleUser {
			prompt = semanticText(request.Messages[index].Content)
			break
		}
	}
	if len(request.Tools) > 0 {
		if encoded, err := json.Marshal(request.Tools); err == nil {
			toolDefs = string(encoded)
		}
	}
	return prompt, toolDefs
}

func buildReplayRequestToolTrace(ctx *RequestContext) *routerreplay.ToolTrace {
	if ctx == nil || ctx.SemanticRequest == nil {
		return nil
	}
	steps := make([]routerreplay.ToolTraceStep, 0, len(ctx.SemanticRequest.Messages))
	apiType := string(ctx.SourceFormat)
	for _, message := range ctx.SemanticRequest.Messages {
		role := string(message.Role)
		if message.Role == llmprotocol.RoleUser {
			if text := strings.TrimSpace(semanticText(message.Content)); text != "" {
				steps = append(steps, routerreplay.ToolTraceStep{
					Type: replayToolStepUserInput, Source: replayToolSourceRequest,
					Role: role, Text: text, APIType: apiType,
				})
			}
		}
		for _, content := range message.Content {
			switch content.Kind {
			case llmprotocol.ContentToolCall:
				if content.ToolCall == nil {
					continue
				}
				steps = append(steps, replayToolCallStep(
					replayToolSourceRequest, role, apiType, *content.ToolCall,
				))
			case llmprotocol.ContentToolResult:
				if content.ToolResult == nil {
					continue
				}
				steps = append(steps, replayToolResultStep(
					replayToolSourceRequest, role, apiType, *content.ToolResult,
				))
			}
		}
	}
	return newReplayToolTrace(steps)
}

func buildReplayResponseToolTrace(
	ctx *RequestContext,
	_ []byte,
) *routerreplay.ToolTrace {
	if ctx == nil || ctx.SemanticResponse == nil {
		return nil
	}
	source := replayToolSourceResponse
	if ctx.ExpectStreamingResponse || ctx.IsStreamingResponse {
		source = replayToolSourceStream
	}
	return buildReplaySemanticResponseToolTrace(ctx.SemanticResponse, source, string(ctx.SourceFormat))
}

func buildReplaySemanticResponseToolTrace(
	response *llmprotocol.Response,
	source string,
	apiType string,
) *routerreplay.ToolTrace {
	if response == nil {
		return nil
	}
	steps := make([]routerreplay.ToolTraceStep, 0, len(response.Output)+1)
	hasVisibleText := false
	hasReasoning := false
	for _, output := range response.Output {
		role := string(output.Role)
		if role == "" {
			role = string(llmprotocol.RoleAssistant)
		}
		for _, content := range output.Content {
			switch content.Kind {
			case llmprotocol.ContentToolCall:
				if content.ToolCall != nil {
					steps = append(steps, replayToolCallStep(source, role, apiType, *content.ToolCall))
				}
			case llmprotocol.ContentText, llmprotocol.ContentRefusal:
				if text := strings.TrimSpace(content.Text); text != "" {
					hasVisibleText = true
					steps = append(steps, routerreplay.ToolTraceStep{
						Type: replayToolStepAssistantFinalResponse, Source: source,
						Role: role, Text: text, APIType: apiType,
					})
				}
			case llmprotocol.ContentReasoning:
				hasReasoning = hasReasoning || content.Text != "" || content.Signature != ""
			}
		}
	}
	if hasReasoning && !hasVisibleText {
		steps = append(steps, routerreplay.ToolTraceStep{
			Type: replayToolStepAssistantReasoningDone, Source: source,
			Role: string(llmprotocol.RoleAssistant), APIType: apiType,
		})
	}
	return newReplayToolTrace(steps)
}

func buildReplayStreamingToolTrace(ctx *RequestContext) *routerreplay.ToolTrace {
	if ctx == nil {
		return nil
	}
	return buildReplaySemanticResponseToolTrace(
		ctx.SemanticResponse,
		replayToolSourceStream,
		string(ctx.SourceFormat),
	)
}

func replayToolCallStep(source, role, apiType string, call llmprotocol.ToolCall) routerreplay.ToolTraceStep {
	return routerreplay.ToolTraceStep{
		Type: replayToolStepAssistantToolCall, Source: source, Role: role,
		ToolName: call.Name, ToolCallID: call.ID,
		Arguments: call.Arguments, RawArguments: call.Arguments, APIType: apiType,
	}
}

func replayToolResultStep(source, role, apiType string, result llmprotocol.ToolResult) routerreplay.ToolTraceStep {
	encoded, _ := json.Marshal(result.Content)
	text := semanticText(result.Content)
	return routerreplay.ToolTraceStep{
		Type: replayToolStepClientToolResult, Source: source, Role: role,
		Text: text, ToolCallID: result.CallID,
		RawOutput: string(encoded), Output: text, APIType: apiType,
	}
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

func replayToolTraceStepsEqual(left, right routerreplay.ToolTraceStep) bool {
	return left.Type == right.Type && left.Source == right.Source &&
		left.Role == right.Role && left.Text == right.Text &&
		left.ToolName == right.ToolName && left.ToolCallID == right.ToolCallID &&
		left.Arguments == right.Arguments && left.RawArguments == right.RawArguments &&
		left.RawOutput == right.RawOutput && left.Output == right.Output &&
		left.APIType == right.APIType && left.Truncated == right.Truncated
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
	case replayToolStepAssistantReasoningDone:
		return "LLM Reasoning Complete"
	default:
		return ""
	}
}
