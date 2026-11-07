package protocolcodec

import (
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type responsesNamespaceToolWire struct {
	Type        string            `json:"type"`
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
	Tools       []json.RawMessage `json:"tools"`
}

// decodeResponsesNamespaceTool flattens a namespace into the neutral tool list.
// Every backend receives flat names, including a Responses backend, so one
// mapping serves all three formats and no backend needs namespace support.
func decodeResponsesNamespaceTool(body json.RawMessage, request *llmprotocol.Request, policy llmprotocol.Policy) error {
	var wire responsesNamespaceToolWire
	if err := decodeWireValue(body, &wire, policy); err != nil {
		return err
	}
	if strings.TrimSpace(wire.Name) == "" || len(wire.Tools) == 0 {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool", "Responses namespace requires a name and tools", nil)
	}
	for _, entry := range wire.Tools {
		if err := decodeResponsesNamespaceEntry(entry, wire, request, policy); err != nil {
			return err
		}
	}
	request.Generation++
	return nil
}

func decodeResponsesNamespaceEntry(
	body json.RawMessage,
	namespace responsesNamespaceToolWire,
	request *llmprotocol.Request,
	policy llmprotocol.Policy,
) error {
	toolType, err := responsesToolDiscriminator(body)
	if err != nil {
		return err
	}
	if err = validateResponsesToolVariant(body, toolType); err != nil {
		return err
	}
	switch toolType {
	case "function":
		err = decodeResponsesFunctionTool(body, request, policy)
	case "custom":
		err = decodeResponsesCustomTool(body, request, policy)
	default:
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_tool", "a Responses namespace can hold only function and custom tools", nil)
	}
	if err != nil {
		return err
	}
	tool := &request.Tools[len(request.Tools)-1]
	if strings.TrimSpace(tool.Name) == "" {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_tool", "Responses namespace tool requires a name", nil)
	}
	tool.Namespace, tool.LocalName = namespace.Name, tool.Name
	tool.Name = llmprotocol.FlattenToolName(namespace.Name, tool.Name)
	tool.Description = joinNamespaceDescription(namespace.Description, tool.Description)
	return nil
}

// A flat tool has no namespace left to describe it, so its description carries
// what the namespace told the model.
func joinNamespaceDescription(namespace, tool string) string {
	if namespace == "" {
		return tool
	}
	if tool == "" {
		return namespace
	}
	return namespace + "\n\n" + tool
}

// rejectToolNamespaceCollisions fails closed: a shared flat name could not be
// mapped back to one tool. Duplicates among plain tools keep the validator's error.
func rejectToolNamespaceCollisions(tools []llmprotocol.Tool) error {
	namespaced := make(map[string]bool, len(tools))
	for _, tool := range tools {
		wasNamespaced, seen := namespaced[tool.Name]
		if seen && (wasNamespaced || tool.Namespace != "") {
			return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "tool_namespace_collision",
				"namespaced tools flatten to the same name as another tool: "+tool.Name, nil)
		}
		namespaced[tool.Name] = tool.Namespace != ""
	}
	return nil
}

// projectToolCallNamespaces keeps the loss report for a Responses provider call
// that names a namespace on its way to a client format that has none.
func projectToolCallNamespaces(source, target llmprotocol.WireFormat, response *llmprotocol.Response) llmprotocol.Diagnostics {
	if target == llmprotocol.OpenAIResponsesV1 {
		return nil
	}
	var diagnostics llmprotocol.Diagnostics
	for _, item := range response.Output {
		for index := range item.Content {
			if item.Content[index].ToolCall == nil || item.Content[index].ToolCall.Namespace == "" {
				continue
			}
			call := *item.Content[index].ToolCall
			call.Namespace = ""
			item.Content[index].ToolCall = &call
			diagnostics = append(diagnostics, llmprotocol.Diagnostic{
				Source: source, Target: target, Field: "output.namespace",
				Action: llmprotocol.DiagnosticDropped, Reason: "the client format has no tool namespaces",
			})
		}
	}
	return diagnostics
}
