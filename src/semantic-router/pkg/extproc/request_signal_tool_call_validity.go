package extproc

import (
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// countInvalidRecentToolCalls returns how many tool calls the request's last
// assistant turn carries that violate the schema of the tool they name:
// arguments that are not a JSON object, or that miss a required property. A
// small model can keep emitting such calls late in a session until the client
// harness terminates the conversation on its validation loop, so the router
// exposes the count as the invalid_tool_call conversation fact and a decision
// can escalate to a stronger lane before that happens. The client resends the
// history every turn, so the count needs no session state and reads the same
// on every replica.
func countInvalidRecentToolCalls(request *llmprotocol.Request) int {
	if request == nil || len(request.Tools) == 0 {
		return 0
	}
	required := make(map[string][]string, len(request.Tools))
	for _, tool := range request.Tools {
		if tool.CustomFormat != nil {
			continue
		}
		properties, ok := requiredProperties(tool.InputSchema)
		if !ok {
			continue
		}
		required[tool.Name] = properties
	}
	if len(required) == 0 {
		return 0
	}
	last := -1
	for index := len(request.Messages) - 1; index >= 0; index-- {
		if request.Messages[index].Role == llmprotocol.RoleAssistant {
			last = index
			break
		}
	}
	if last < 0 {
		return 0
	}
	// Adjacent assistant messages are one turn: a Responses input carries each
	// function_call item as its own message, and the codec joins them when it
	// projects the turn. Counting only the last message would miss the calls
	// of the items before it.
	invalid := 0
	for index := last; index >= 0 && request.Messages[index].Role == llmprotocol.RoleAssistant; index-- {
		for _, content := range request.Messages[index].Content {
			if content.Kind != llmprotocol.ContentToolCall || content.ToolCall == nil {
				continue
			}
			if content.ToolCall.Kind != "" {
				continue // a custom call carries free-form input, not arguments
			}
			if missingRequiredProperty(required[content.ToolCall.Name], content.ToolCall.Arguments) {
				invalid++
			}
		}
	}
	return invalid
}

// requiredProperties reads a tool schema's required property names. A schema
// without required properties validates any arguments, so it reports false
// and the fact stays conservative.
func requiredProperties(schema json.RawMessage) ([]string, bool) {
	if len(schema) == 0 {
		return nil, false
	}
	var object struct {
		Required []string `json:"required"`
	}
	if err := json.Unmarshal(schema, &object); err != nil || len(object.Required) == 0 {
		return nil, false
	}
	return object.Required, true
}

// missingRequiredProperty reports whether the arguments string is not a JSON
// object or lacks one of the required properties.
func missingRequiredProperty(required []string, arguments string) bool {
	if len(required) == 0 {
		return false
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal([]byte(arguments), &object); err != nil {
		return true
	}
	for _, property := range required {
		if _, present := object[property]; !present {
			return true
		}
	}
	return false
}
