package llmprotocol

// SetSystemInstruction applies a deterministic system instruction policy to the
// neutral request. insert prepends to the first system block; other modes
// replace every system and developer block the caller sent with one system
// block, matching the router's existing system_prompt contract.
func SetSystemInstruction(request *Request, text, mode string) bool {
	if request == nil || text == "" {
		return false
	}
	content := Content{Kind: ContentText, Text: text}
	if mode != "insert" {
		request.Instructions = []InstructionBlock{{Role: RoleSystem, Content: []Content{content}}}
		return true
	}
	for i := range request.Instructions {
		instruction := &request.Instructions[i]
		if instruction.Role != RoleSystem {
			continue
		}
		instruction.Content = append([]Content{content}, instruction.Content...)
		return true
	}
	request.Instructions = append([]InstructionBlock{{Role: RoleSystem, Content: []Content{content}}}, request.Instructions...)
	return true
}
