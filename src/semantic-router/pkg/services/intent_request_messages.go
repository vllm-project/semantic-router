package services

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
)

const (
	maxIntentMetadataEntries    = 32
	maxIntentMetadataKeyBytes   = 128
	maxIntentMetadataValueBytes = 1024
)

type IntentMessage struct {
	Role             string            `json:"role"`
	Name             string            `json:"name,omitempty"`
	Content          json.RawMessage   `json:"content"`
	ToolCalls        []json.RawMessage `json:"tool_calls,omitempty"`
	ToolCallID       string            `json:"tool_call_id,omitempty"`
	FunctionCall     json.RawMessage   `json:"function_call,omitempty"`
	Refusal          json.RawMessage   `json:"refusal,omitempty"`
	ReasoningContent json.RawMessage   `json:"reasoning_content,omitempty"`
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
	requestFacts      classification.RequestFacts
}

type intentConversationHistory struct {
	currentUserMessage  string
	currentUserRawText  string
	currentUserImageURL string
	priorUserMessages   []string
	nonUserMessages     []string
	hasAssistantReply   bool
	conversationFacts   classification.ConversationFacts
	inputModality       classification.InputModalityFacts
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
	if err := validateIntentMetadata(req.Metadata); err != nil {
		return intentSignalInput{}, err
	}
	rawText := req.Text
	text := strings.TrimSpace(rawText)

	toolDefinitionCount := len(req.Tools) + len(req.Functions)
	if input, ok := resolveIntentSignalInputFromMessages(req.Messages, toolDefinitionCount); ok {
		useTopLevelTextFallback := rawText != "" && strings.TrimSpace(input.evaluationText) == ""
		input = applyTopLevelTextFallback(input, rawText)
		contextEstimate, err := estimateIntentRequestContext(
			req,
			fallbackText(rawText, useTopLevelTextFallback),
		)
		if err != nil {
			return intentSignalInput{}, err
		}
		inputModality := input.requestFacts.InputModality
		input.requestFacts = requestFactsForIntent(
			req.Metadata,
			contextEstimate,
		)
		input.requestFacts.InputModality = inputModality
		if useTopLevelTextFallback && text != "" && input.requestFacts.InputModality.TextContentCount == 0 {
			// req.Text supplied user text the message walk could not see.
			// Promoted system/assistant text must not count: it is not user
			// input, and counting it would diverge from the data-plane path.
			input.requestFacts.InputModality.TextContentCount = 1
		}
		return input, nil
	}

	if rawText == "" && len(req.Metadata) == 0 {
		return intentSignalInput{}, ErrEmptyText
	}

	contextEstimate, err := estimateIntentRequestContext(
		req,
		rawText,
	)
	if err != nil {
		return intentSignalInput{}, err
	}

	fallbackInput := intentSignalInput{
		evaluationText:  text,
		contextText:     text,
		currentUserText: rawText,
		conversationFacts: classification.ConversationFacts{
			UserMessageCount:    1,
			ToolDefinitionCount: toolDefinitionCount,
			LastMessageRole:     "user",
		},
		requestFacts: requestFactsForIntent(req.Metadata, contextEstimate),
	}
	if text != "" {
		fallbackInput.requestFacts.InputModality.TextContentCount = 1
	}
	return fallbackInput, nil
}

func fallbackText(text string, enabled bool) string {
	if !enabled {
		return ""
	}
	return text
}

func requestFactsForIntent(
	metadata map[string]string,
	estimate classification.RequestContextEstimate,
) classification.RequestFacts {
	return classification.RequestFacts{
		Metadata:               cloneIntentMetadata(metadata),
		ContextTokenFloor:      estimate.TokenFloor,
		ContextTextBytes:       estimate.TextBytes,
		ContextEquivalentBytes: estimate.EquivalentBytes,
		ContextHasNonText:      estimate.HasNonText,
	}
}

// estimateIntentRequestContext serializes only the prompt-bearing subset of
// the eval/classify request and delegates to the exact same OpenAI envelope
// estimator used by ExtProc. json.RawMessage preserves structured numbers
// lexically; no request content is logged or retained after this call.
func estimateIntentRequestContext(
	req IntentRequest,
	additionalUserText string,
) (classification.RequestContextEstimate, error) {
	estimateMessages := append([]IntentMessage(nil), req.Messages...)
	if additionalUserText != "" {
		content, err := json.Marshal(additionalUserText)
		if err != nil {
			return classification.RequestContextEstimate{}, err
		}
		estimateMessages = append(estimateMessages, IntentMessage{
			Role:    "user",
			Content: content,
		})
	}

	envelope, err := json.Marshal(struct {
		Messages            []IntentMessage   `json:"messages"`
		Tools               []json.RawMessage `json:"tools,omitempty"`
		Functions           []json.RawMessage `json:"functions,omitempty"`
		ToolChoice          json.RawMessage   `json:"tool_choice,omitempty"`
		FunctionCall        json.RawMessage   `json:"function_call,omitempty"`
		ResponseFormat      json.RawMessage   `json:"response_format,omitempty"`
		MaxTokens           json.RawMessage   `json:"max_tokens,omitempty"`
		MaxCompletionTokens json.RawMessage   `json:"max_completion_tokens,omitempty"`
	}{
		Messages:            estimateMessages,
		Tools:               req.Tools,
		Functions:           req.Functions,
		ToolChoice:          req.ToolChoice,
		FunctionCall:        req.FunctionCall,
		ResponseFormat:      req.ResponseFormat,
		MaxTokens:           req.MaxTokens,
		MaxCompletionTokens: req.MaxCompletionTokens,
	})
	if err != nil {
		return classification.RequestContextEstimate{}, fmt.Errorf(
			"estimate intent request context: %w",
			err,
		)
	}
	return classification.EstimateOpenAIRequestContext(envelope), nil
}

// applyTopLevelTextFallback fills empty text slots from req.Text when the
// messages path was accepted solely because it carries an image, so image safety
// cannot toggle whether the caller-supplied text is scored.
func applyTopLevelTextFallback(input intentSignalInput, rawText string) intentSignalInput {
	text := strings.TrimSpace(rawText)
	if rawText == "" || strings.TrimSpace(input.evaluationText) != "" {
		return input
	}
	input.evaluationText = text
	if strings.TrimSpace(input.contextText) == "" {
		input.contextText = text
	}
	if input.currentUserText == "" {
		input.currentUserText = rawText
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
		currentUserText:   history.currentUserRawText,
		priorUserMessages: append([]string(nil), history.priorUserMessages...),
		nonUserMessages:   append([]string(nil), history.nonUserMessages...),
		hasAssistantReply: history.hasAssistantReply,
		imageURL:          history.currentUserImageURL,
		conversationFacts: history.conversationFacts,
		requestFacts:      classification.RequestFacts{InputModality: history.inputModality},
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
	return input,
		strings.TrimSpace(input.evaluationText) != "" ||
			input.currentUserText != "" ||
			input.imageURL != "" ||
			hasIntentConversationFacts(input.conversationFacts)
}

func hasIntentConversationFacts(facts classification.ConversationFacts) bool {
	return facts.UserMessageCount > 0 ||
		facts.AssistantMessageCount > 0 ||
		facts.SystemMessageCount > 0 ||
		facts.ToolMessageCount > 0 ||
		facts.ImageContentCount > 0 ||
		facts.ToolDefinitionCount > 0 ||
		facts.AssistantToolCallCount > 0 ||
		facts.ToolResultCount > 0 ||
		facts.HasDeveloperMessage
}

func extractIntentConversationHistory(messages []IntentMessage, toolDefinitionCount int) intentConversationHistory {
	history := intentConversationHistory{
		conversationFacts: classification.ConversationFacts{
			ToolDefinitionCount: toolDefinitionCount,
		},
	}
	previousWasToolResult := false

	for _, msg := range messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		rawText := extractIntentMessageRawText(msg.Content)
		text := strings.TrimSpace(rawText)
		if role == "user" {
			if rawText != "" {
				history.currentUserRawText = rawText
			}
			counts := countIntentMessageModalities(msg.Content)
			history.conversationFacts.ImageContentCount += counts.ImageContentCount
			history.inputModality.ImageContentCount += counts.ImageContentCount
			history.inputModality.AudioContentCount += counts.AudioContentCount
			history.inputModality.VideoContentCount += counts.VideoContentCount
			if text != "" {
				history.inputModality.TextContentCount++
			}
		}
		previousWasToolResult = observeIntentConversationMessage(
			&history.conversationFacts,
			role,
			len(msg.ToolCalls),
			previousWasToolResult,
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
	previousWasToolResult bool,
) bool {
	facts.LastMessageRole = role
	facts.LastMessageToolResult = false
	facts.LastUserAfterToolResult = false
	switch role {
	case "user":
		facts.UserMessageCount++
		facts.LastUserAfterToolResult = previousWasToolResult
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
		facts.LastMessageToolResult = true
	}
	return facts.LastMessageToolResult
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

// countIntentMessageModalities counts image, audio, and video content parts in
// a message using the shared part-type table in
// config.InputModalityForContentPartType. Text presence is tracked by the
// caller from extracted text so plain string content counts too;
// TextContentCount is never set here.
func countIntentMessageModalities(raw json.RawMessage) classification.InputModalityFacts {
	var counts classification.InputModalityFacts
	raw = bytesTrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return counts
	}
	var parts []intentMessageContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		var part intentMessageContentPart
		if err := json.Unmarshal(raw, &part); err != nil {
			return counts
		}
		parts = []intentMessageContentPart{part}
	}
	for _, part := range parts {
		modality, _ := config.InputModalityForContentPartType(part.Type)
		switch modality {
		case config.InputModalityImage:
			counts.ImageContentCount++
		case config.InputModalityAudio:
			counts.AudioContentCount++
		case config.InputModalityVideo:
			counts.VideoContentCount++
		}
	}
	return counts
}

func extractIntentMessageRawText(raw json.RawMessage) string {
	raw = bytesTrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text
	}

	var parts []intentMessageContentPart
	if err := json.Unmarshal(raw, &parts); err == nil {
		return joinIntentMessageRawTextParts(parts)
	}

	var part intentMessageContentPart
	if err := json.Unmarshal(raw, &part); err == nil {
		return joinIntentMessageRawTextParts([]intentMessageContentPart{part})
	}

	return ""
}

func joinIntentMessageRawTextParts(parts []intentMessageContentPart) string {
	textParts := make([]string, 0, len(parts))
	for _, part := range parts {
		partType := strings.ToLower(strings.TrimSpace(part.Type))
		if partType != "" && partType != "text" && partType != "input_text" {
			continue
		}
		if part.Text != "" {
			textParts = append(textParts, part.Text)
		}
	}
	return strings.Join(textParts, " ")
}

func validateIntentMetadata(metadata map[string]string) error {
	if len(metadata) > maxIntentMetadataEntries {
		return fmt.Errorf(
			"%w: metadata cannot contain more than %d entries",
			ErrInvalidRequestFacts,
			maxIntentMetadataEntries,
		)
	}
	for key, value := range metadata {
		if len(key) == 0 || len(key) > maxIntentMetadataKeyBytes {
			return fmt.Errorf(
				"%w: metadata key length must be between 1 and %d bytes",
				ErrInvalidRequestFacts,
				maxIntentMetadataKeyBytes,
			)
		}
		if len(value) > maxIntentMetadataValueBytes {
			return fmt.Errorf(
				"%w: metadata value for key %q exceeds %d bytes",
				ErrInvalidRequestFacts,
				key,
				maxIntentMetadataValueBytes,
			)
		}
	}
	return nil
}

func cloneIntentMetadata(metadata map[string]string) map[string]string {
	if len(metadata) == 0 {
		return nil
	}
	cloned := make(map[string]string, len(metadata))
	for key, value := range metadata {
		cloned[key] = value
	}
	return cloned
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
