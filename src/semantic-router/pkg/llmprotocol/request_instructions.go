package llmprotocol

// SetSystemInstruction applies a deterministic system instruction policy to the
// neutral request. insert prepends to the first system block; other modes replace
// that block, matching the router's existing system_prompt contract.
func SetSystemInstruction(request *Request, text, mode string) bool {
	if request == nil || text == "" {
		return false
	}
	content := Content{Kind: ContentText, Text: text}
	for i := range request.Instructions {
		instruction := &request.Instructions[i]
		if instruction.Role != RoleSystem {
			continue
		}
		if mode == "insert" {
			instruction.Content = append([]Content{content}, instruction.Content...)
		} else {
			instruction.Content = []Content{content}
		}
		return true
	}
	request.Instructions = append([]InstructionBlock{{Role: RoleSystem, Content: []Content{content}}}, request.Instructions...)
	return true
}
