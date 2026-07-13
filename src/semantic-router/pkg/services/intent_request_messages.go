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

	input, ok, err := resolveIntentSignalInputFromMessages(req.Messages)
	if err != nil {
		return intentSignalInput{}, err
	}
	if ok {
		return applyTopLevelTextFallback(input, text), nil
	}

	if text == "" {
		return intentSignalInput{}, ErrEmptyText
	}

	return intentSignalInput{
		evaluationText:  text,
		contextText:     text,
		currentUserText: text,
	}, nil
}

// HasInlineImageInput reports whether resolving this request will perform the
// bounded full-image validation used by the classify/eval path. It deliberately
// applies only the cheap canonical data-URI shape check so request entrypoints
// can acquire process admission before any base64 decode or pixel allocation.
func (req IntentRequest) HasInlineImageInput() bool {
	for _, message := range req.Messages {
		if !strings.EqualFold(strings.TrimSpace(message.Role), "user") {
			continue
		}
		for _, part := range parseIntentMessageContentParts(message.Content) {
			if part.ImageURL == nil {
				continue
			}
			if _, ok := imageurl.CanonicalDataURL(strings.TrimSpace(part.ImageURL.URL)); ok {
				return true
			}
		}
	}
	return false
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

func resolveIntentSignalInputFromMessages(messages []IntentMessage) (intentSignalInput, bool, error) {
	if len(messages) == 0 {
		return intentSignalInput{}, false, nil
	}

	history, err := extractIntentConversationHistory(messages)
	if err != nil {
		return intentSignalInput{}, false, err
	}
	input := intentSignalInput{
		evaluationText:    history.currentUserMessage,
		contextText:       strings.Join(history.nonUserMessages, " "),
		currentUserText:   history.currentUserMessage,
		priorUserMessages: append([]string(nil), history.priorUserMessages...),
		nonUserMessages:   append([]string(nil), history.nonUserMessages...),
		hasAssistantReply: history.hasAssistantReply,
		imageURL:          history.currentUserImageURL,
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
	return input, strings.TrimSpace(input.evaluationText) != "" || input.imageURL != "", nil
}

func extractIntentConversationHistory(messages []IntentMessage) (intentConversationHistory, error) {
	var history intentConversationHistory
	var imageBudget imageurl.RequestImageBudget

	for _, msg := range messages {
		text := extractIntentMessageText(msg.Content)
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		switch role {
		case "user":
			if err := history.applyUserMessage(msg.Content, text, &imageBudget); err != nil {
				return intentConversationHistory{}, err
			}
		case "system", "assistant":
			history.applyNonUserMessage(role, text)
		}
	}

	return history, nil
}

func (history *intentConversationHistory) applyUserMessage(
	raw json.RawMessage,
	text string,
	imageBudget *imageurl.RequestImageBudget,
) error {
	imageURL, err := extractIntentMessageImageURL(raw, imageBudget)
	if err != nil {
		return err
	}
	if text == "" {
		// An image-only turn attaches its image without clobbering the most
		// recent user text, which stays the best text to score.
		if imageURL != "" {
			history.currentUserImageURL = imageURL
		}
		return nil
	}
	if history.currentUserMessage != "" {
		history.priorUserMessages = append(history.priorUserMessages, history.currentUserMessage)
	}
	history.currentUserMessage = text
	history.currentUserImageURL = imageURL
	return nil
}

func (history *intentConversationHistory) applyNonUserMessage(role, text string) {
	if text == "" {
		return
	}
	history.nonUserMessages = append(history.nonUserMessages, text)
	if role == "assistant" {
		history.hasAssistantReply = true
	}
}

// extractIntentMessageImageURL returns the first strictly decodable inline JPEG
// or PNG data URI (canonicalized) from a message's content parts, after
// validating every generically allowlisted image part. Invalid base64, corrupt
// images, MIME/format mismatches, and formats unsupported by Candle fail the
// request; HTTP(S), file, unsupported-MIME, and unknown shapes retain their
// SSRF-safe drop semantics. The generic ExtProc allowlist stays broader.
func extractIntentMessageImageURL(raw json.RawMessage, imageBudget *imageurl.RequestImageBudget) (string, error) {
	return firstSafeImageURL(parseIntentMessageContentParts(raw), imageBudget)
}

func parseIntentMessageContentParts(raw json.RawMessage) []intentMessageContentPart {
	raw = bytesTrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}

	var parts []intentMessageContentPart
	if err := json.Unmarshal(raw, &parts); err == nil {
		return parts
	}

	var part intentMessageContentPart
	if err := json.Unmarshal(raw, &part); err == nil {
		return []intentMessageContentPart{part}
	}
	return nil
}

func firstSafeImageURL(parts []intentMessageContentPart, imageBudget *imageurl.RequestImageBudget) (string, error) {
	var firstImageURL string
	for _, part := range parts {
		if part.ImageURL == nil {
			continue
		}
		imageURL := strings.TrimSpace(part.ImageURL.URL)
		_, ok := imageurl.CanonicalDataURL(imageURL)
		if !ok {
			// CanonicalDataURL intentionally rejects an empty payload. Probe the
			// same shared parser with a harmless payload so an otherwise allowlisted
			// empty data URI is distinguished from HTTP(S), file, unsupported MIME,
			// and unknown shapes, which retain their SSRF-safe drop semantics.
			if imageurl.IsSafeImageDataURL(imageURL + "AA==") {
				return "", ErrInvalidImageInput
			}
			continue
		}
		validated, ok := imageurl.ValidateJPEGOrPNGDataURL(imageURL, imageBudget)
		if !ok {
			return "", ErrInvalidImageInput
		}
		// Return the canonical form (lowercased scheme/MIME/";base64," marker,
		// payload preserved) rather than the raw URI. The classifier backend that
		// ultimately consumes this scans for ";base64," case-sensitively, so an
		// accepted uppercase-scheme data URI would otherwise yield no image signal
		// on the classify/eval path even though the gate admitted it.
		// Continue scanning after selecting it: every later allowlisted image part
		// must satisfy the same strict decoder contract, even though only the first
		// image is scored.
		if firstImageURL == "" {
			firstImageURL = validated.DataURL
		}
	}
	return firstImageURL, nil
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
