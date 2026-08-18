package extproc

import (
	"fmt"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	maxRoutingMetadataEntries  = 32
	maxRoutingMetadataKeyLen   = 128
	maxRoutingMetadataValueLen = 1024
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
	ImageContentCount int
	Metadata          map[string]string
	// Request-aware context estimate. These are content-free scalar summaries;
	// raw tool results, schemas, image URLs, and image bytes are never retained.
	ContextTokenFloor      int
	ContextTextBytes       int
	ContextEquivalentBytes int
	ContextHasNonText      bool

	// Conversation-shape fields for the conversation signal family.
	HasDeveloperMessage     bool
	UserMessageCount        int
	AssistantMessageCount   int
	SystemMessageCount      int
	ToolMessageCount        int
	ToolDefinitionCount     int
	AssistantToolCallCount  int
	ToolResultCount         int
	AssistantToolNames      []string
	LastMessageRole         string
	LastMessageToolResult   bool
	LastUserAfterToolResult bool
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
	if err := extractMetadataFast(body, r); err != nil {
		return nil, err
	}
	contextEstimate := classification.EstimateOpenAIRequestContext(body)
	r.ContextTokenFloor = contextEstimate.TokenFloor
	r.ContextTextBytes = contextEstimate.TextBytes
	r.ContextEquivalentBytes = contextEstimate.EquivalentBytes
	r.ContextHasNonText = contextEstimate.HasNonText

	countFastToolDefinitions(body, r)
	messages := gjson.GetBytes(body, "messages")
	if !messages.Exists() || !messages.IsArray() {
		return r, nil
	}

	populateFastExtractMessages(messages, r)

	return r, nil
}

func countFastToolDefinitions(body []byte, result *FastExtractResult) {
	for _, field := range []string{"tools", "functions"} {
		definitions := gjson.GetBytes(body, field)
		if !definitions.Exists() || !definitions.IsArray() {
			continue
		}
		definitions.ForEach(func(_, _ gjson.Result) bool {
			result.ToolDefinitionCount++
			return true
		})
	}
}

func extractMetadataFast(body []byte, result *FastExtractResult) error {
	metadata := gjson.GetBytes(body, "metadata")
	if !metadata.Exists() {
		return nil
	}
	if !metadata.IsObject() {
		return &jsonTypeError{field: "metadata", want: "object", got: metadata.Type.String()}
	}
	values := make(map[string]string)
	var invalidKey string
	invalidValue := false
	var validationErr error
	metadata.ForEach(func(key, value gjson.Result) bool {
		if len(values) >= maxRoutingMetadataEntries {
			validationErr = fmt.Errorf("metadata cannot contain more than %d entries", maxRoutingMetadataEntries)
			return false
		}
		if value.Type != gjson.String {
			invalidKey = key.String()
			invalidValue = true
			return false
		}
		keyValue := key.String()
		stringValue := value.String()
		if len(keyValue) == 0 || len(keyValue) > maxRoutingMetadataKeyLen {
			validationErr = fmt.Errorf("metadata key length must be between 1 and %d bytes", maxRoutingMetadataKeyLen)
			return false
		}
		if len(stringValue) > maxRoutingMetadataValueLen {
			validationErr = fmt.Errorf(
				"metadata value for key %q exceeds %d bytes",
				keyValue,
				maxRoutingMetadataValueLen,
			)
			return false
		}
		values[keyValue] = stringValue
		return true
	})
	if validationErr != nil {
		return validationErr
	}
	if invalidValue {
		field := "metadata"
		if invalidKey != "" {
			field += "." + invalidKey
		}
		return &jsonTypeError{
			field: field,
			want:  "string",
			got:   "non-string",
		}
	}
	result.Metadata = values
	return nil
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
	previousWasToolResult := result.LastMessageToolResult || result.LastMessageRole == "tool"
	role := msg.Get("role").String()
	content := msg.Get("content")
	text := extractTextFromContent(content)
	result.LastMessageRole = role
	result.LastMessageToolResult = false
	result.LastUserAfterToolResult = false

	switch role {
	case "user":
		result.UserMessageCount++
		result.LastUserAfterToolResult = previousWasToolResult
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
		result.ToolResultCount++
		result.LastMessageToolResult = true
	}
}

func countAssistantToolCalls(msg gjson.Result, result *FastExtractResult) {
	toolCalls := msg.Get("tool_calls")
	if !toolCalls.Exists() || !toolCalls.IsArray() {
		return
	}
	toolCalls.ForEach(func(_, toolCall gjson.Result) bool {
		result.AssistantToolCallCount++
		if name := toolCall.Get("function.name").String(); name != "" {
			result.AssistantToolNames = append(result.AssistantToolNames, name)
		}
		return true
	})
}

func recordFastExtractUserMessage(result *FastExtractResult, text string, content gjson.Result) {
	result.ImageContentCount += countImageContentParts(content)
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

func countImageContentParts(content gjson.Result) int {
	if !content.IsArray() {
		return 0
	}
	count := 0
	content.ForEach(func(_, part gjson.Result) bool {
		switch part.Get("type").String() {
		case "image_url", "input_image":
			count++
		}
		return true
	})
	return count
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
