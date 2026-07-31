package extproc

func boundedSelectionReasoning(value string) string {
	const maxReasoningBytes = 1024
	if len(value) <= maxReasoningBytes {
		return value
	}
	end := 0
	for index := range value {
		if index > maxReasoningBytes {
			break
		}
		end = index
	}
	return value[:end]
}
