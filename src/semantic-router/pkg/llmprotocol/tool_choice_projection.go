package llmprotocol

// ProjectToolChoiceNone sends a Chat Completions backend a request whose
// tool_choice is none without its tools, tool_choice and parallel_tool_calls.
// Model servers such as vLLM render the declared tools into the prompt
// whatever the choice, so a model can still start a call the server then
// drops, leaving only the sentence before it. Messages backends keep the
// tools, which they require while the history holds tool_use blocks.
// Responses backends keep them too: the Responses API enforces tool_choice
// itself, and its tool list also holds hosted tools and is part of the prompt
// prefix that providers cache. The caller must report the diagnostic.
func ProjectToolChoiceNone(request Request, target WireFormat) (Request, Diagnostics) {
	if target != OpenAIChatV1 || request.ToolChoice.Mode != ToolChoiceNone || len(request.Tools) == 0 {
		return request, nil
	}
	projected := request
	projected.Tools = nil
	projected.ToolChoice = ToolChoice{}
	projected.ParallelToolCalls = nil
	projected.Generation++ // The source envelope still carries the tools.
	return projected, Diagnostics{{
		Source: request.Trusted.SourceFormat,
		Target: target,
		Field:  "tools",
		Action: DiagnosticDropped,
		Reason: "a Chat Completions backend receives a tool_choice none request without its tools",
	}}
}
