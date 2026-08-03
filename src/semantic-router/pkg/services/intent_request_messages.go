package services

import (
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
)

type IntentMessage struct {
	Role       string            `json:"role"`
	Content    json.RawMessage   `json:"content"`
	ToolCalls  []json.RawMessage `json:"tool_calls,omitempty"`
	ToolCallID string            `json:"tool_call_id,omitempty"`
}

type intentSignalInput struct {
	evaluationText    string
	contextText       string
	currentUserText   string
	priorUserMessages []string
	nonUserMessages   []string
	hasAssistantReply bool
	imageURL          string
	conversationFacts classification.ConversationFacts
}

type intentConversationHistory struct {
	currentUserMessage  string
	currentUserImageURL string
	priorUserMessages   []string
	nonUserMessages     []string
	hasAssistantReply   bool
	conversationFacts   classification.ConversationFacts
}

type intentMessageImageURL struct {
	URL string `json:"url"`
}

// UnmarshalJSON accepts both the Chat Completions object form {"url": "..."} and
// the Responses API bare-string form, so a string-valued image_url part cannot
// fail the whole content-parts unmarshal (which would drop text extraction).
func (u *intentMessageImageURL) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		u.URL = s
		return nil
	}
	// Best-effort: an unknown shape (number/array/bool) yields no image rather
	// than failing the whole content-parts unmarshal and dropping sibling text.
	type plain intentMessageImageURL
	var p plain
	if err := json.Unmarshal(data, &p); err == nil {
		*u = intentMessageImageURL(p)
	}
	return nil
}

type intentMessageContentPart struct {
	Type     string                 `json:"type"`
	Text     string                 `json:"text"`
	ImageURL *intentMessageImageURL `json:"image_url,omitempty"`
}

func (req IntentRequest) resolveSignalInput() (intentSignalInput, error) {
	text := strings.TrimSpace(req.Text)

	if input, ok := resolveIntentSignalInputFromMessages(req.Messages, len(req.Tools)); ok {
		return applyTopLevelTextFallback(input, text), nil
	}

	if text == "" {
		return intentSignalInput{}, ErrEmptyText
	}

	return intentSignalInput{
		evaluationText:  text,
		contextText:     text,
		currentUserText: text,
		conversationFacts: classification.ConversationFacts{
			UserMessageCount:    1,
			ToolDefinitionCount: len(req.Tools),
			LastMessageRole:     "user",
		},
	}, nil
}

// applyTopLevelTextFallback fills empty text slots from req.Text when the
// messages path was accepted solely because it carries an image, so image safety
// cannot toggle whether the caller-supplied text is scored.
func applyTopLevelTextFallback(input intentSignalInput, text string) intentSignalInput {
	if text == "" || strings.TrimSpace(input.evaluationText) != "" {
		return input
	}
	input.evaluationText = text
	if strings.TrimSpace(input.contextText) == "" {
		input.contextText = text
	}
	if strings.TrimSpace(input.currentUserText) == "" {
		input.currentUserText = text
	}
	return input
}

func resolveIntentSignalInputFromMessages(messages []IntentMessage, toolDefinitionCount int) (intentSignalInput, bool) {
	if len(messages) == 0 {
		return intentSignalInput{}, false
	}

	history := extractIntentConversationHistory(messages, toolDefinitionCount)
	input := intentSignalInput{
		evaluationText:    history.currentUserMessage,
		contextText:       strings.Join(history.nonUserMessages, " "),
		currentUserText:   history.currentUserMessage,
		priorUserMessages: append([]string(nil), history.priorUserMessages...),
		nonUserMessages:   append([]string(nil), history.nonUserMessages...),
		hasAssistantReply: history.hasAssistantReply,
		imageURL:          history.currentUserImageURL,
		conversationFacts: history.conversationFacts,
	}

	// Promote system/assistant text only with no user text AND no image; the
	// image guard stops an image-only turn from promoting assistant text, leaving
	// the slot empty for the caller's req.Text fallback.
	if input.evaluationText == "" && input.imageURL == "" && len(history.nonUserMessages) > 0 {
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

func extractIntentConversationHistory(messages []IntentMessage, toolDefinitionCount int) intentConversationHistory {
	history := intentConversationHistory{
		conversationFacts: classification.ConversationFacts{
			ToolDefinitionCount: toolDefinitionCount,
		},
	}
	sawToolResult := false

	for _, msg := range messages {
		text := extractIntentMessageText(msg.Content)
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		sawToolResult = observeIntentConversationMessage(
			&history.conversationFacts,
			role,
			len(msg.ToolCalls),
			sawToolResult,
		)
		if recordIntentUserMessage(&history, role, text, msg.Content) {
			continue
		}
		recordIntentNonUserMessage(&history, role, text)
	}

	return history
}

func observeIntentConversationMessage(
	facts *classification.ConversationFacts,
	role string,
	toolCallCount int,
	sawToolResult bool,
) bool {
	if role == "" {
		return sawToolResult
	}
	facts.LastMessageRole = role
	facts.LastMessageToolResult = role == "tool"
	switch role {
	case "user":
		facts.UserMessageCount++
		facts.LastUserAfterToolResult = sawToolResult
	case "assistant":
		facts.AssistantMessageCount++
		facts.AssistantToolCallCount += toolCallCount
	case "system":
		facts.SystemMessageCount++
	case "developer":
		facts.HasDeveloperMessage = true
	case "tool":
		facts.ToolMessageCount++
		facts.ToolResultCount++
		return true
	}
	return sawToolResult
}

func recordIntentUserMessage(
	history *intentConversationHistory,
	role string,
	text string,
	content json.RawMessage,
) bool {
	if role != "user" {
		return false
	}
	imageURL := extractIntentMessageImageURL(content)
	if text == "" && imageURL == "" {
		return true
	}
	// An image-only turn attaches its image without clobbering the most recent
	// user text, which stays the best text to score.
	if text == "" {
		history.currentUserImageURL = imageURL
		return true
	}
	if history.currentUserMessage != "" {
		history.priorUserMessages = append(history.priorUserMessages, history.currentUserMessage)
	}
	history.currentUserMessage = text
	history.currentUserImageURL = imageURL
	return true
}

func recordIntentNonUserMessage(history *intentConversationHistory, role, text string) {
	if text == "" || (role != "system" && role != "developer" && role != "assistant") {
		return
	}
	history.nonUserMessages = append(history.nonUserMessages, text)
	history.hasAssistantReply = history.hasAssistantReply || role == "assistant"
}

// extractIntentMessageImageURL returns the first safe inline base64 image data
// URI (canonicalized) from a message's content parts. Only data URIs are
// accepted; HTTP(S) URLs are rejected to prevent SSRF. This shares the same
// imageurl gate as the ExtProc request path but is not a full behavioral mirror
// of it (e.g. the ExtProc path fills the first safe image once across all user
// turns, whereas this HTTP path resolves per turn).
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
		// Return the canonical form (lowercased scheme/MIME/";base64," marker,
		// payload preserved) rather than the raw URI. The classifier backend that
		// ultimately consumes this scans for ";base64," case-sensitively, so an
		// accepted uppercase-scheme data URI would otherwise yield no image signal
		// on the classify/eval path even though the gate admitted it.
		if canonical, ok := imageurl.CanonicalDataURL(strings.TrimSpace(part.ImageURL.URL)); ok {
			return canonical
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
