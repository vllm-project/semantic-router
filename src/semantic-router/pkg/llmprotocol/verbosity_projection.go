package llmprotocol

// ProjectTextVerbosity removes OpenAI's output-detail hint only for Messages,
// which has no corresponding request control. The caller must report the
// diagnostic to the client instead of silently discarding the hint.
func ProjectTextVerbosity(request Request, target WireFormat) (Request, Diagnostics) {
	if target != AnthropicMessagesV1 || request.TextVerbosity == "" {
		return request, nil
	}

	projected := request
	projected.TextVerbosity = ""
	projected.Generation++ // The original wire body still carries verbosity.
	field := "text.verbosity"
	if request.Trusted.SourceFormat == OpenAIChatV1 {
		field = "verbosity"
	}
	return projected, Diagnostics{{
		Source: request.Trusted.SourceFormat,
		Target: target,
		Field:  field,
		Action: DiagnosticDropped,
		Reason: "Messages has no output verbosity control",
	}}
}
