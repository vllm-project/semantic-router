package llmprotocol

// DeveloperRoleSentAsSystem is the reason of the diagnostic a model that lacks
// the developer role reports.
const DeveloperRoleSentAsSystem = "the selected model does not support the developer role; developer messages were sent as system messages"

// ProjectUnsupportedDeveloperRole sends every developer instruction and message
// as a system one to a model that declares the developer role unsupported, and
// reports the change. OpenAI reads the two roles alike, but DeepSeek V4's
// encoding treats a developer message as a turn of its own and, when it thinks
// without tools, drops any developer message before the last user message. The
// caller projects per candidate, so another model still receives the role.
func ProjectUnsupportedDeveloperRole(request Request, unsupported CapabilitySet, target WireFormat) (Request, Diagnostics) {
	if !unsupported.Supports(CapabilityDeveloperRole) {
		return request, nil
	}
	sent := 0
	instructions := append([]InstructionBlock(nil), request.Instructions...)
	for index := range instructions {
		if instructions[index].Role == RoleDeveloper {
			instructions[index].Role = RoleSystem
			sent++
		}
	}
	messages := append([]Message(nil), request.Messages...)
	for index := range messages {
		if messages[index].Role == RoleDeveloper {
			messages[index].Role = RoleSystem
			sent++
		}
	}
	if sent == 0 {
		return request, nil
	}
	projected := request
	projected.Instructions = instructions
	projected.Messages = messages
	projected.Generation++ // The source envelope still names the developer role.
	return projected, Diagnostics{{
		Source: request.Trusted.SourceFormat, Target: target, Field: LossyFieldSystemMessageRole,
		Action: DiagnosticApproximated, Reason: DeveloperRoleSentAsSystem,
	}}
}
