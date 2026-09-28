package llmprotocol

// StripTools removes tool controls and optionally linked calls/results.
//
//nolint:cyclop // Tool history removal walks every closed neutral content variant.
func StripTools(request *Request, stripHistory bool) (bool, int) {
	if request == nil {
		return false, 0
	}
	changed := len(request.Tools) > 0 || request.ToolChoice.Mode != "" ||
		request.ToolChoice.Name != "" || request.ParallelToolCalls != nil
	request.Tools = nil
	request.ToolChoice = ToolChoice{}
	request.ParallelToolCalls = nil
	if !stripHistory {
		return changed, 0
	}
	filtered := make([]Message, 0, len(request.Messages))
	removed := 0
	for _, message := range request.Messages {
		if message.Role == RoleTool {
			removed++
			changed = true
			continue
		}
		content := message.Content[:0]
		for _, block := range message.Content {
			if block.Kind == ContentToolCall || block.Kind == ContentToolResult {
				removed++
				changed = true
				continue
			}
			content = append(content, block)
		}
		if len(content) == 0 {
			if len(message.Content) > 0 {
				removed++
				changed = true
			}
			continue
		}
		message.Content = content
		filtered = append(filtered, message)
	}
	request.Messages = filtered
	return changed, removed
}
