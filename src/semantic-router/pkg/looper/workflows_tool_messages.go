package looper

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go"
)

func findWorkflowToolStateID(req *openai.ChatCompletionNewParams) (string, bool) {
	reqMap, ok := requestAsMap(req)
	if !ok {
		return "", false
	}
	messages, ok := reqMap["messages"].([]interface{})
	if !ok {
		return "", false
	}
	for _, message := range trailingWorkflowToolMessages(messages) {
		if id, ok := message["tool_call_id"].(string); ok {
			if stateID, parsed := parseWorkflowToolStateID(id); parsed {
				return stateID, true
			}
		}
	}
	return "", false
}

func parseWorkflowToolStateID(toolCallID string) (string, bool) {
	if !strings.HasPrefix(toolCallID, workflowToolCallIDPrefix) {
		return "", false
	}
	rest := strings.TrimPrefix(toolCallID, workflowToolCallIDPrefix)
	idx := strings.Index(rest, workflowToolCallIDSeparator)
	if idx <= 0 {
		return "", false
	}
	return rest[:idx], true
}

func requestHasTools(req *openai.ChatCompletionNewParams) bool {
	reqMap, ok := requestAsMap(req)
	if !ok {
		return false
	}
	tools, ok := reqMap["tools"].([]interface{})
	return ok && len(tools) > 0
}

func requestAsMap(req *openai.ChatCompletionNewParams) (map[string]interface{}, bool) {
	if req == nil {
		return nil, false
	}
	data, err := json.Marshal(req)
	if err != nil {
		return nil, false
	}
	var reqMap map[string]interface{}
	if err := json.Unmarshal(data, &reqMap); err != nil {
		return nil, false
	}
	return reqMap, true
}

func workflowToolMessagesForState(req *openai.ChatCompletionNewParams, state *workflowPendingToolState) ([]map[string]interface{}, error) {
	stateID, err := workflowToolStateID(state)
	if err != nil {
		return nil, err
	}
	trailingTools, err := workflowTrailingToolMessagesForRequest(req, stateID)
	if err != nil {
		return nil, err
	}
	pending := workflowPendingToolCallIDSet(state.ToolCallIDs)
	return workflowValidatedToolMessages(trailingTools, stateID, pending)
}

func workflowToolStateID(state *workflowPendingToolState) (string, error) {
	if state == nil {
		return "", fmt.Errorf("workflow tool state missing")
	}
	if state.ID == "" {
		return "", fmt.Errorf("workflow tool state ID missing")
	}
	return state.ID, nil
}

func workflowAgentID(phase string, step workflowPlanStep, model string, modelIndex int) string {
	if phase == workflowToolPhaseFinal {
		return "final:" + strings.TrimSpace(model)
	}
	stepID := strings.TrimSpace(step.ID)
	if stepID == "" {
		stepID = "step"
	}
	modelName := strings.TrimSpace(model)
	if modelName == "" {
		modelName = "model"
	}
	return fmt.Sprintf("%s:%d:%s", stepID, modelIndex, modelName)
}

func workflowTrailingToolMessagesForRequest(req *openai.ChatCompletionNewParams, stateID string) ([]map[string]interface{}, error) {
	reqMap, ok := requestAsMap(req)
	if !ok {
		return nil, fmt.Errorf("could not parse request messages")
	}
	messages, ok := reqMap["messages"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("request messages missing")
	}
	trailingTools := trailingWorkflowToolMessages(messages)
	if len(trailingTools) == 0 {
		return nil, fmt.Errorf("request does not end with workflow tool messages for state %q", stateID)
	}
	return trailingTools, nil
}

func workflowValidatedToolMessages(
	trailingTools []map[string]interface{},
	stateID string,
	pending map[string]bool,
) ([]map[string]interface{}, error) {
	seen := map[string]bool{}
	var toolMessages []map[string]interface{}
	for _, message := range trailingTools {
		id, ok := workflowToolMessageID(message)
		if !ok {
			continue
		}
		if err := validateWorkflowToolMessageID(id, stateID, pending); err != nil {
			return nil, err
		}
		if !seen[id] {
			toolMessages = append(toolMessages, cloneWorkflowMap(message))
			seen[id] = true
		}
	}
	if len(toolMessages) == 0 {
		return nil, fmt.Errorf("no tool messages found for workflow state %q", stateID)
	}
	if err := workflowRequirePendingToolMessages(pending, seen); err != nil {
		return nil, err
	}
	return toolMessages, nil
}

func workflowToolMessageID(message map[string]interface{}) (string, bool) {
	id, ok := message["tool_call_id"].(string)
	return id, ok && id != ""
}

func validateWorkflowToolMessageID(id string, stateID string, pending map[string]bool) error {
	parsed, isFlow := parseWorkflowToolStateID(id)
	if !isFlow {
		return fmt.Errorf("tool result %q is not a Router Flow tool_call_id", id)
	}
	if parsed != stateID {
		return fmt.Errorf("tool result %q belongs to workflow state %q, not %q", id, parsed, stateID)
	}
	if len(pending) > 0 && !pending[id] {
		return fmt.Errorf("tool result %q was not requested by workflow state %q", id, stateID)
	}
	return nil
}

func workflowRequirePendingToolMessages(pending map[string]bool, seen map[string]bool) error {
	for id := range pending {
		if !seen[id] {
			return fmt.Errorf("missing tool result for workflow tool_call_id %q", id)
		}
	}
	return nil
}

func trailingWorkflowToolMessages(messages []interface{}) []map[string]interface{} {
	var reversed []map[string]interface{}
	for i := len(messages) - 1; i >= 0; i-- {
		message, ok := messages[i].(map[string]interface{})
		if !ok || message["role"] != "tool" {
			break
		}
		reversed = append(reversed, message)
	}
	if len(reversed) == 0 {
		return nil
	}
	ordered := make([]map[string]interface{}, len(reversed))
	for i := range reversed {
		ordered[len(reversed)-1-i] = reversed[i]
	}
	return ordered
}

func workflowPendingToolCallIDSet(ids []string) map[string]bool {
	if len(ids) == 0 {
		return nil
	}
	pending := make(map[string]bool, len(ids))
	for _, id := range ids {
		if strings.TrimSpace(id) != "" {
			pending[id] = true
		}
	}
	return pending
}

func patchWorkflowToolCallResponse(raw []byte, state *workflowPendingToolState) ([]byte, []string, error) {
	if len(raw) == 0 {
		return nil, nil, fmt.Errorf("empty tool-call response")
	}
	ensureWorkflowToolStateID(state)
	completion, message, err := workflowToolCallCompletionMessage(raw)
	if err != nil {
		return nil, nil, err
	}
	toolCalls, err := workflowToolCallsFromMessage(message)
	if err != nil {
		return nil, nil, err
	}
	ids := patchWorkflowToolCallIDs(toolCalls, state)
	normalizeWorkflowToolCallFinishReason(completion)
	patched, err := json.Marshal(completion)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal workflow tool-call response: %w", err)
	}
	return patched, ids, nil
}

func normalizeWorkflowToolCallFinishReason(completion map[string]interface{}) {
	choices, ok := completion["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return
	}
	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return
	}
	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return
	}
	if toolCalls, ok := message["tool_calls"].([]interface{}); ok && len(toolCalls) > 0 {
		choice["finish_reason"] = "tool_calls"
	}
}

func ensureWorkflowToolStateID(state *workflowPendingToolState) string {
	if state.ID != "" {
		return state.ID
	}
	state.ID = newWorkflowToolStateID()
	return state.ID
}

func workflowToolCallCompletionMessage(raw []byte) (map[string]interface{}, map[string]interface{}, error) {
	var completion map[string]interface{}
	if err := json.Unmarshal(raw, &completion); err != nil {
		return nil, nil, fmt.Errorf("parse workflow tool-call response: %w", err)
	}
	choices, ok := completion["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return nil, nil, fmt.Errorf("workflow tool-call response missing choices")
	}
	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return nil, nil, fmt.Errorf("workflow tool-call response choice is invalid")
	}
	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return nil, nil, fmt.Errorf("workflow tool-call response missing message")
	}
	return completion, message, nil
}

func workflowToolCallsFromMessage(message map[string]interface{}) ([]interface{}, error) {
	toolCalls, ok := message["tool_calls"].([]interface{})
	if !ok || len(toolCalls) == 0 {
		return nil, fmt.Errorf("workflow tool-call response missing tool_calls")
	}
	return toolCalls, nil
}

func patchWorkflowToolCallIDs(toolCalls []interface{}, state *workflowPendingToolState) []string {
	stateID := ensureWorkflowToolStateID(state)
	ids := make([]string, 0, len(toolCalls))
	for _, rawCall := range toolCalls {
		call, ok := rawCall.(map[string]interface{})
		if !ok {
			continue
		}
		originalID, _ := call["id"].(string)
		call["id"] = workflowPatchedToolCallID(originalID, stateID, nextWorkflowToolCallSeq(state))
		if id, ok := call["id"].(string); ok {
			ids = append(ids, id)
		}
	}
	return ids
}

func nextWorkflowToolCallSeq(state *workflowPendingToolState) int {
	if state == nil {
		return 0
	}
	seq := state.ToolCallSeq
	state.ToolCallSeq++
	return seq
}

func workflowPatchedToolCallID(originalID string, stateID string, sequence int) string {
	if originalID == "" {
		originalID = newWorkflowToolStateID()
	}
	if _, alreadyFlow := parseWorkflowToolStateID(originalID); alreadyFlow {
		return originalID
	}
	return fmt.Sprintf("%s%s%s%d%s%s", workflowToolCallIDPrefix, stateID, workflowToolCallIDSeparator, sequence, workflowToolCallIDSeparator, originalID)
}

func workflowAssistantMessageFromRaw(raw []byte) (map[string]interface{}, error) {
	var completion struct {
		Choices []struct {
			Message map[string]interface{} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &completion); err != nil {
		return nil, fmt.Errorf("parse workflow assistant tool-call message: %w", err)
	}
	if len(completion.Choices) == 0 || completion.Choices[0].Message == nil {
		return nil, fmt.Errorf("workflow assistant tool-call message missing")
	}
	message := cloneWorkflowMap(completion.Choices[0].Message)
	message["role"] = "assistant"
	return message, nil
}

func appendWorkflowRawMessages(req *openai.ChatCompletionNewParams, rawMessages ...map[string]interface{}) (*openai.ChatCompletionNewParams, error) {
	reqMap, ok := requestAsMap(req)
	if !ok {
		return nil, fmt.Errorf("could not parse request")
	}
	messages, ok := reqMap["messages"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("request messages missing")
	}
	for _, message := range rawMessages {
		messages = append(messages, cloneWorkflowMap(message))
	}
	reqMap["messages"] = messages
	data, err := json.Marshal(reqMap)
	if err != nil {
		return nil, fmt.Errorf("marshal request with workflow tool messages: %w", err)
	}
	var appended openai.ChatCompletionNewParams
	if err := json.Unmarshal(data, &appended); err != nil {
		return nil, fmt.Errorf("parse request with workflow tool messages: %w", err)
	}
	return &appended, nil
}

func cloneWorkflowMap(src map[string]interface{}) map[string]interface{} {
	cloned := make(map[string]interface{}, len(src))
	for key, value := range src {
		cloned[key] = value
	}
	return cloned
}
