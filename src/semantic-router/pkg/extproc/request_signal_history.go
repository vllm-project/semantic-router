package extproc

import "github.com/openai/openai-go"

type signalConversationHistory struct {
	currentUserMessage string
	priorUserMessages  []string
	nonUserMessages    []string
}

func signalConversationHistoryFromFastExtract(result *FastExtractResult) signalConversationHistory {
	if result == nil {
		return signalConversationHistory{}
	}
	return signalConversationHistory{
		currentUserMessage: result.UserContent,
		priorUserMessages:  append([]string(nil), result.PriorUserMessages...),
		nonUserMessages:    append([]string(nil), result.NonUserMessages...),
	}
}

func extractSignalConversationHistory(req *openai.ChatCompletionNewParams) signalConversationHistory {
	var history signalConversationHistory

	for _, msg := range req.Messages {
		role, textContent := extractMessageRoleAndContent(msg)
		if textContent == "" {
			continue
		}
		switch role {
		case "user":
			if history.currentUserMessage != "" {
				history.priorUserMessages = append(history.priorUserMessages, history.currentUserMessage)
			}
			history.currentUserMessage = textContent
		case "system", "assistant":
			history.nonUserMessages = append(history.nonUserMessages, textContent)
		}
	}

	return history
}
