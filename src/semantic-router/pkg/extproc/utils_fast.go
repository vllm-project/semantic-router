package extproc

import (
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// FastExtractResult holds fields extracted from the request body via gjson
// without allocating the full OpenAI SDK struct.
type FastExtractResult struct {
	Model             string
	Stream            bool
	UserContent       string
	PriorUserMessages []string
	NonUserMessages   []string
	HasAssistantReply bool
	FirstImageURL     string

	// Conversation-shape fields for the conversation signal family.
	HasDeveloperMessage    bool
	UserMessageCount       int
	AssistantMessageCount  int
	SystemMessageCount     int
	ToolMessageCount       int
	ToolDefinitionCount    int
	AssistantToolCallCount int
	CompletedToolCycles    int
}

// extractContentFast extracts model, stream, message content, and the first
// image URL from raw JSON bytes using gjson. This avoids the two expensive
// json.Unmarshal calls (extractStreamParam + parseOpenAIRequest) that dominate
// E2E latency for large request bodies (~300ms for 64KB → ~1ms with gjson).
func extractContentFast(body []byte) (*FastExtractResult, error) {
	r := &FastExtractResult{}
	if err := extractModelAndStreamFast(body, r); err != nil {
		return nil, err
	}

	messages := gjson.GetBytes(body, "messages")
	if !messages.Exists() || !messages.IsArray() {
		return r, nil
	}

	populateFastExtractMessages(messages, r)
	countFastToolDefinitions(body, r)

	return r, nil
}

func countFastToolDefinitions(body []byte, result *FastExtractResult) {
	tools := gjson.GetBytes(body, "tools")
	if !tools.Exists() || !tools.IsArray() {
		return
	}
	tools.ForEach(func(_, _ gjson.Result) bool {
		result.ToolDefinitionCount++
		return true
	})
}

func extractModelAndStreamFast(body []byte, result *FastExtractResult) error {
	modelResult := gjson.GetBytes(body, "model")
	if !modelResult.Exists() {
		return errMissingModel
	}
	if modelResult.Type != gjson.String {
		return &jsonTypeError{field: "model", want: "string", got: modelResult.Type.String()}
	}
	result.Model = modelResult.String()

	streamResult := gjson.GetBytes(body, "stream")
	if !streamResult.Exists() {
		return nil
	}
	if streamResult.Type != gjson.True && streamResult.Type != gjson.False {
		return &jsonTypeError{field: "stream", want: "boolean", got: streamResult.Type.String()}
	}
	result.Stream = streamResult.Type == gjson.True
	return nil
}

func populateFastExtractMessages(messages gjson.Result, result *FastExtractResult) {
	messages.ForEach(func(_, msg gjson.Result) bool {
		consumeFastExtractMessage(msg, result)
		return true
	})
}

func consumeFastExtractMessage(msg gjson.Result, result *FastExtractResult) {
	role := msg.Get("role").String()
	content := msg.Get("content")
	text := extractTextFromContent(content)

	switch role {
	case "user":
		result.UserMessageCount++
		recordFastExtractUserMessage(result, text, content)
	case "system":
		result.SystemMessageCount++
		recordFastExtractNonUserMessage(result, role, text)
	case "assistant":
		result.AssistantMessageCount++
		recordFastExtractNonUserMessage(result, role, text)
		countAssistantToolCalls(msg, result)
	case "developer":
		result.HasDeveloperMessage = true
		recordFastExtractNonUserMessage(result, role, text)
	case "tool":
		result.ToolMessageCount++
		result.CompletedToolCycles++
	}
}

func countAssistantToolCalls(msg gjson.Result, result *FastExtractResult) {
	toolCalls := msg.Get("tool_calls")
	if !toolCalls.Exists() || !toolCalls.IsArray() {
		return
	}
	toolCalls.ForEach(func(_, _ gjson.Result) bool {
		result.AssistantToolCallCount++
		return true
	})
}

func recordFastExtractUserMessage(result *FastExtractResult, text string, content gjson.Result) {
	if result.FirstImageURL == "" {
		result.FirstImageURL = extractImageURLFromContent(content)
	}
	if text == "" {
		return
	}
	if result.UserContent != "" {
		result.PriorUserMessages = append(result.PriorUserMessages, result.UserContent)
	}
	result.UserContent = text
}

func recordFastExtractNonUserMessage(result *FastExtractResult, role string, text string) {
	if text == "" {
		return
	}
	result.NonUserMessages = append(result.NonUserMessages, text)
	if role == "assistant" {
		result.HasAssistantReply = true
	}
}

// extractTextFromContent extracts text from a message content field that can
// be either a plain string or an array of content parts.
func extractTextFromContent(content gjson.Result) string {
	if content.Type == gjson.String {
		return content.String()
	}
	if !content.IsArray() {
		return ""
	}
	var parts []string
	content.ForEach(func(_, part gjson.Result) bool {
		partType := part.Get("type").String()
		if partType == "text" || partType == "" {
			if t := part.Get("text").String(); t != "" {
				parts = append(parts, t)
			}
		}
		return true
	})
	if len(parts) == 1 {
		return parts[0]
	}
	return strings.Join(parts, " ")
}

// extractImageURLFromContent returns the first safe base64 image data URI
// from content parts. Only inline data URIs are accepted (no HTTP URLs).
func extractImageURLFromContent(content gjson.Result) string {
	if !content.IsArray() {
		return ""
	}
	var found string
	content.ForEach(func(_, part gjson.Result) bool {
		if part.Get("type").String() != "image_url" {
			return true
		}
		url := part.Get("image_url.url").String()
		if isSafeImageDataURL(url) {
			found = url
			return false
		}
		return true
	})
	return found
}

// extractStreamParamFast extracts the "stream" boolean from raw JSON without
// allocating a map[string]interface{}. Falls back to false for missing/invalid.
func extractStreamParamFast(body []byte) bool {
	return gjson.GetBytes(body, "stream").Bool()
}

// extractModelFast extracts just the "model" string from raw JSON.
func extractModelFast(body []byte) string {
	return gjson.GetBytes(body, "model").String()
}

// rewriteModelInBodyFast replaces the "model" field in raw JSON using sjson,
// avoiding the unmarshal → set → marshal cycle of rewriteModelInBody.
func rewriteModelInBodyFast(body []byte, newModel string) ([]byte, error) {
	return sjson.SetBytes(body, "model", newModel)
}

// addStreamFieldsFast adds stream=true and stream_options.include_usage=true
// to raw JSON bytes using sjson.
func addStreamFieldsFast(body []byte) []byte {
	out, err := sjson.SetBytes(body, "stream", true)
	if err != nil {
		return body
	}
	out, err = sjson.SetBytes(out, "stream_options.include_usage", true)
	if err != nil {
		return body
	}
	logging.Infof("Added stream_options.include_usage=true for streaming request")
	return out
}

// errMissingModel is returned when the model field is absent from the JSON body.
var errMissingModel = &jsonFieldError{field: "model"}

type jsonFieldError struct{ field string }

func (e *jsonFieldError) Error() string { return "missing required field: " + e.field }

type jsonTypeError struct{ field, want, got string }

func (e *jsonTypeError) Error() string {
	return "field " + e.field + " must be " + e.want + ", got " + e.got
}
