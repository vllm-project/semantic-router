package services

import (
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
)

type IntentMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type intentSignalInput struct {
	evaluationText    string
	contextText       string
	currentUserText   string
	priorUserMessages []string
	nonUserMessages   []string
	hasAssistantReply bool
	imageURL          string
}

type intentConversationHistory struct {
	currentUserMessage  string
	currentUserImageURL string
	priorUserMessages   []string
	nonUserMessages     []string
	hasAssistantReply   bool
}

type intentMessageImageURL struct {
	URL string `json:"url"`
}

type intentMessageContentPart struct {
	Type     string                 `json:"type"`
	Text     string                 `json:"text"`
	ImageURL *intentMessageImageURL `json:"image_url,omitempty"`
}

func (req IntentRequest) resolveSignalInput() (intentSignalInput, error) {
	if input, ok := resolveIntentSignalInputFromMessages(req.Messages); ok {
		return input, nil
	}

	text := strings.TrimSpace(req.Text)
	if text == "" {
		return intentSignalInput{}, ErrEmptyText
	}

	return intentSignalInput{
		evaluationText:  text,
		contextText:     text,
		currentUserText: text,
	}, nil
}

func resolveIntentSignalInputFromMessages(messages []IntentMessage) (intentSignalInput, bool) {
	if len(messages) == 0 {
		return intentSignalInput{}, false
	}

	history := extractIntentConversationHistory(messages)
	input := intentSignalInput{
		evaluationText:    history.currentUserMessage,
		contextText:       strings.Join(history.nonUserMessages, " "),
		currentUserText:   history.currentUserMessage,
		priorUserMessages: append([]string(nil), history.priorUserMessages...),
		nonUserMessages:   append([]string(nil), history.nonUserMessages...),
		hasAssistantReply: history.hasAssistantReply,
		imageURL:          history.currentUserImageURL,
	}

	if input.evaluationText == "" && len(history.nonUserMessages) > 0 {
		input.evaluationText = strings.Join(history.nonUserMessages, " ")
		input.contextText = input.evaluationText
	}

	if history.currentUserMessage != "" && len(history.nonUserMessages) > 0 {
		allMessages := make([]string, 0, len(history.nonUserMessages)+1)
		allMessages = append(allMessages, history.nonUserMessages...)
		allMessages = append(allMessages, history.currentUserMessage)
		input.contextText = strings.Join(allMessages, " ")
	} else if history.currentUserMessage != "" {
		input.contextText = history.currentUserMessage
	}

	// An image-only user turn (no accompanying text) is still a valid input for
	// image-modality signals, so accept the message path when an image is present
	// even if there is no evaluation text to score.
	return input, strings.TrimSpace(input.evaluationText) != "" || input.imageURL != ""
}

func extractIntentConversationHistory(messages []IntentMessage) intentConversationHistory {
	var history intentConversationHistory

	for _, msg := range messages {
		text := extractIntentMessageText(msg.Content)
		role := strings.ToLower(strings.TrimSpace(msg.Role))

		// A user turn may carry only an image (no text). Capture the image even
		// when the text is empty so image-modality signals still receive it.
		if role == "user" {
			imageURL := extractIntentMessageImageURL(msg.Content)
			if text == "" && imageURL == "" {
				continue
			}
			if history.currentUserMessage != "" {
				history.priorUserMessages = append(history.priorUserMessages, history.currentUserMessage)
			}
			history.currentUserMessage = text
			history.currentUserImageURL = imageURL
			continue
		}

		if text == "" {
			continue
		}

		switch role {
		case "system", "assistant":
			history.nonUserMessages = append(history.nonUserMessages, text)
			if role == "assistant" {
				history.hasAssistantReply = true
			}
		}
	}

	return history
}

// extractIntentMessageImageURL returns the first safe inline base64 image data
// URI from a message's content parts, mirroring the ExtProc request path. Only
// data URIs are accepted (HTTP(S) URLs are rejected to prevent SSRF).
func extractIntentMessageImageURL(raw json.RawMessage) string {
	raw = bytesTrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}

	var parts []intentMessageContentPart
	if err := json.Unmarshal(raw, &parts); err == nil {
		return firstSafeImageURL(parts)
	}

	var part intentMessageContentPart
	if err := json.Unmarshal(raw, &part); err == nil {
		return firstSafeImageURL([]intentMessageContentPart{part})
	}

	return ""
}

func firstSafeImageURL(parts []intentMessageContentPart) string {
	for _, part := range parts {
		if part.ImageURL == nil {
			continue
		}
		if url := strings.TrimSpace(part.ImageURL.URL); imageurl.IsSafeImageDataURL(url) {
			return url
		}
	}
	return ""
}

func extractIntentMessageText(raw json.RawMessage) string {
	raw = bytesTrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return strings.TrimSpace(text)
	}

	var parts []intentMessageContentPart
	if err := json.Unmarshal(raw, &parts); err == nil {
		return joinIntentMessageContentParts(parts)
	}

	var part intentMessageContentPart
	if err := json.Unmarshal(raw, &part); err == nil {
		return joinIntentMessageContentParts([]intentMessageContentPart{part})
	}

	return ""
}

func joinIntentMessageContentParts(parts []intentMessageContentPart) string {
	textParts := make([]string, 0, len(parts))
	for _, part := range parts {
		partType := strings.ToLower(strings.TrimSpace(part.Type))
		if partType != "" && partType != "text" && partType != "input_text" {
			continue
		}
		if text := strings.TrimSpace(part.Text); text != "" {
			textParts = append(textParts, text)
		}
	}
	return strings.Join(textParts, " ")
}

func bytesTrimSpace(raw []byte) []byte {
	start := 0
	for start < len(raw) && (raw[start] == ' ' || raw[start] == '\n' || raw[start] == '\t' || raw[start] == '\r') {
		start++
	}
	end := len(raw)
	for end > start && (raw[end-1] == ' ' || raw[end-1] == '\n' || raw[end-1] == '\t' || raw[end-1] == '\r') {
		end--
	}
	return raw[start:end]
}
