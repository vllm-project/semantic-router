package http

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// BuildStreamingModalityBody converts a complete ChatCompletion JSON body into
// SSE chunks compatible with OpenAI streaming protocol. For multimodal
// responses (content array with non-text parts), the entire message is emitted
// as a single delta chunk. For text-only responses, content is split into
// word-level chunks for smooth streaming.
func BuildStreamingModalityBody(body []byte) []byte {
	// First pass: detect whether content is multimodal (JSON array) by
	// inspecting the raw JSON, since openai.ChatCompletionMessage.Content
	// is typed as string and would lose the array structure.
	if rawContent := extractRawContent(body); rawContent != nil && isMultimodalContent(rawContent) {
		return buildMultimodalSSE(body)
	}

	// Text-only path: use the typed struct
	var completion openai.ChatCompletion
	if err := json.Unmarshal(body, &completion); err != nil {
		logging.Errorf("Modality streaming: failed to parse response: %v", err)
		return []byte("data: {\"error\": \"Failed to process modality response\"}\n\ndata: [DONE]\n\n")
	}

	return buildTextSSE(completion)
}

// extractRawContent extracts the raw content value from a ChatCompletion JSON
// body without type coercion. Returns nil if the content cannot be extracted.
func extractRawContent(body []byte) interface{} {
	var raw struct {
		Choices []struct {
			Message struct {
				Content json.RawMessage `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil
	}
	if len(raw.Choices) == 0 || len(raw.Choices[0].Message.Content) == 0 {
		return nil
	}
	// Determine if it's an array
	var arr []interface{}
	if err := json.Unmarshal(raw.Choices[0].Message.Content, &arr); err == nil {
		return arr
	}
	// It's a string
	var s string
	if err := json.Unmarshal(raw.Choices[0].Message.Content, &s); err == nil {
		return s
	}
	return nil
}

// buildMultimodalSSE constructs SSE chunks for a multimodal response.
// The entire content array is emitted as a single delta chunk.
func buildMultimodalSSE(body []byte) []byte {
	unixTimeStep := time.Now().Unix()
	newID := fmt.Sprintf("chatcmpl-modality-%d", unixTimeStep)

	// Extract model, role, and content from raw JSON
	var raw struct {
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		return []byte("data: {\"error\": \"Failed to process multimodal response\"}\n\ndata: [DONE]\n\n")
	}
	if len(raw.Choices) == 0 {
		return []byte("data: {\"error\": \"No choices in multimodal response\"}\n\ndata: [DONE]\n\n")
	}

	model := raw.Model
	if model == "" {
		model = "router"
	}
	ch := raw.Choices[0]

	// Reconstruct the content as an interface value
	var content interface{}
	_ = json.Unmarshal(ch.Message.Content, &content)

	var sseChunks []string

	delta := map[string]interface{}{
		"role":    ch.Message.Role,
		"content": content,
	}
	chunk := buildSSEModalityChunk(newID, model, unixTimeStep, delta, nil)
	sseChunks = append(sseChunks, chunk)

	// Final chunk
	finalReason := ch.FinishReason
	if finalReason == "" {
		finalReason = "stop"
	}
	finalChunk := buildSSEModalityChunk(newID, model, unixTimeStep, map[string]interface{}{}, finalReason)
	sseChunks = append(sseChunks, finalChunk)
	sseChunks = append(sseChunks, "data: [DONE]\n\n")

	return []byte(strings.Join(sseChunks, ""))
}

// buildTextSSE constructs SSE chunks for a text-only ChatCompletion response,
// splitting the content into word-level chunks for smooth streaming.
func buildTextSSE(completion openai.ChatCompletion) []byte {
	unixTimeStep := time.Now().Unix()
	newID := fmt.Sprintf("chatcmpl-modality-%d", unixTimeStep)
	model := completion.Model
	if model == "" {
		model = "router"
	}

	var sseChunks []string

	if len(completion.Choices) == 0 {
		sseChunks = append(sseChunks, "data: [DONE]\n\n")
		return []byte(strings.Join(sseChunks, ""))
	}

	message := completion.Choices[0].Message
	content := message.Content
	finishReason := completion.Choices[0].FinishReason

	sseChunks = appendContentChunks(sseChunks, content, string(message.Role), newID, model, unixTimeStep)

	finalReason := finishReason
	if finalReason == "" {
		finalReason = "stop"
	}
	finalChunk := buildSSEModalityChunk(newID, model, unixTimeStep, map[string]interface{}{}, finalReason)
	sseChunks = append(sseChunks, finalChunk)

	sseChunks = append(sseChunks, "data: [DONE]\n\n")
	return []byte(strings.Join(sseChunks, ""))
}

// appendContentChunks splits text content into word-level SSE delta chunks and
// appends them to sseChunks. The first chunk includes the assistant role.
func appendContentChunks(sseChunks []string, content string, role string, id, model string, created int64) []string {
	if content == "" {
		return sseChunks
	}
	chunks := splitContentIntoChunks(content)
	for i, c := range chunks {
		delta := map[string]interface{}{"content": c}
		if i == 0 {
			delta["role"] = role
		}
		chunk := buildSSEModalityChunk(id, model, created, delta, nil)
		sseChunks = append(sseChunks, chunk)
	}
	return sseChunks
}

// isMultimodalContent returns true if the content is an array (multimodal
// content parts) rather than a plain string.
func isMultimodalContent(content interface{}) bool {
	_, isArray := content.([]interface{})
	return isArray
}

// extractStringContent returns the content as a string, or an empty string if
// the content is not a string.
func extractStringContent(content interface{}) string {
	s, _ := content.(string)
	return s
}

// buildSSEModalityChunk constructs a single SSE data frame for a streaming
// ChatCompletion chunk. When finishReason is empty, the chunk is a content
// delta; otherwise it is the terminal chunk.
func buildSSEModalityChunk(id, model string, created int64, delta map[string]interface{}, finishReason interface{}) string {
	chunk := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"delta":         delta,
				"finish_reason": finishReason,
			},
		},
	}
	chunkJSON, _ := json.Marshal(chunk)
	return fmt.Sprintf("data: %s\n\n", chunkJSON)
}

// isErrorResponse checks if a JSON response is an error response
func isErrorResponse(responseBytes []byte) bool {
	var responseMap map[string]interface{}
	if err := json.Unmarshal(responseBytes, &responseMap); err != nil {
		return false
	}
	// Check for common error response structures
	_, hasError := responseMap["error"]
	_, hasDetail := responseMap["detail"]
	// If it has "error" or "detail" but no "choices", it's likely an error response
	_, hasChoices := responseMap["choices"]
	return (hasError || hasDetail) && !hasChoices
}

// extractErrorMessage extracts error message from error response
func extractErrorMessage(responseBytes []byte) string {
	var responseMap map[string]interface{}
	if err := json.Unmarshal(responseBytes, &responseMap); err != nil {
		return "Failed to parse error response"
	}

	// Try to extract error message from various formats
	if errorObj, ok := responseMap["error"].(map[string]interface{}); ok {
		if msg, ok := errorObj["message"].(string); ok {
			return msg
		}
	}
	if detail, ok := responseMap["detail"].(string); ok {
		return detail
	}
	return "Error response from cache"
}

// splitContentIntoChunks splits content into word-by-word chunks for streaming
func splitContentIntoChunks(content string) []string {
	if content == "" {
		return []string{}
	}

	// Split by words (preserving spaces)
	words := strings.Fields(content)
	if len(words) == 0 {
		return []string{content}
	}

	chunks := make([]string, 0, len(words))
	for i, word := range words {
		if i < len(words)-1 {
			// Add space after word (except last word)
			chunks = append(chunks, word+" ")
		} else {
			// Last word without trailing space
			chunks = append(chunks, word)
		}
	}
	return chunks
}

// buildStreamingCacheErrorBody returns an SSE-formatted error body from a cached error response.
func buildStreamingCacheErrorBody(cachedResponse []byte) []byte {
	errorMsg := extractErrorMessage(cachedResponse)
	logging.Errorf("Cached response is an error response, cannot convert to streaming: %s", errorMsg)

	now := time.Now().Unix()
	errorChunk := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-cache-error-%d", now),
		"object":  "chat.completion.chunk",
		"created": now,
		"model":   "cache",
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"delta": map[string]interface{}{
					"role":    "assistant",
					"content": fmt.Sprintf("Error: %s", errorMsg),
				},
				"finish_reason": "error",
			},
		},
	}
	chunkJSON, err := json.Marshal(errorChunk)
	if err != nil {
		return []byte("data: {\"error\": \"Failed to convert cached error response\"}\n\ndata: [DONE]\n\n")
	}
	return []byte(fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", chunkJSON))
}

// buildStreamingCacheBody converts a cached ChatCompletion into SSE chunks.
func buildStreamingCacheBody(cachedResponse []byte) []byte {
	var cachedCompletion openai.ChatCompletion
	if err := json.Unmarshal(cachedResponse, &cachedCompletion); err != nil {
		logging.Errorf("Error parsing cached response for streaming conversion: %v", err)
		return []byte("data: {\"error\": \"Failed to convert cached response\"}\n\ndata: [DONE]\n\n")
	}

	if len(cachedCompletion.Choices) == 0 || cachedCompletion.Choices[0].Message.Content == "" {
		logging.Errorf("Cached response has no valid choices or content")
		return []byte("data: {\"error\": \"Cached response has no content\"}\n\ndata: [DONE]\n\n")
	}

	unixTimeStep := time.Now().Unix()
	newID := fmt.Sprintf("chatcmpl-cache-%d", unixTimeStep)
	content := cachedCompletion.Choices[0].Message.Content
	chunks := splitContentIntoChunks(content)
	if len(chunks) == 0 {
		chunks = []string{content}
	}

	var sseChunks []string
	for i, chunkContent := range chunks {
		streamChunk := map[string]interface{}{
			"id":      newID,
			"object":  "chat.completion.chunk",
			"created": unixTimeStep,
			"model":   cachedCompletion.Model,
			"choices": []map[string]interface{}{
				{
					"index": cachedCompletion.Choices[0].Index,
					"delta": map[string]interface{}{
						"content": chunkContent,
					},
					"finish_reason": nil,
				},
			},
		}
		chunkJSON, err := json.Marshal(streamChunk)
		if err != nil {
			logging.Errorf("Error marshaling streaming chunk %d: %v", i, err)
			sseChunks = append(sseChunks, fmt.Sprintf("data: {\"error\": \"Failed to marshal chunk %d\"}\n\n", i))
			continue
		}
		sseChunks = append(sseChunks, fmt.Sprintf("data: %s\n\n", chunkJSON))
	}

	finalChunk := map[string]interface{}{
		"id":      newID,
		"object":  "chat.completion.chunk",
		"created": unixTimeStep,
		"model":   cachedCompletion.Model,
		"choices": []map[string]interface{}{
			{
				"index":         cachedCompletion.Choices[0].Index,
				"delta":         map[string]interface{}{},
				"finish_reason": cachedCompletion.Choices[0].FinishReason,
			},
		},
	}
	finalChunkJSON, err := json.Marshal(finalChunk)
	if err != nil {
		logging.Errorf("Error marshaling final streaming chunk: %v", err)
		sseChunks = append(sseChunks, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
	} else {
		sseChunks = append(sseChunks, fmt.Sprintf("data: %s\n\n", finalChunkJSON))
	}
	sseChunks = append(sseChunks, "data: [DONE]\n\n")

	var buf bytes.Buffer
	for _, chunk := range sseChunks {
		buf.WriteString(chunk)
	}
	return buf.Bytes()
}

// buildNonStreamingCacheBody regenerates ID/timestamp on a cached response.
func buildNonStreamingCacheBody(cachedResponse []byte) []byte {
	if isErrorResponse(cachedResponse) {
		return cachedResponse
	}

	var cachedCompletion openai.ChatCompletion
	if err := json.Unmarshal(cachedResponse, &cachedCompletion); err != nil {
		logging.Errorf("Error parsing cached response for ID regeneration: %v", err)
		return cachedResponse
	}

	unixTimeStep := time.Now().Unix()
	cachedCompletion.ID = fmt.Sprintf("chatcmpl-cache-%d", unixTimeStep)
	cachedCompletion.Created = unixTimeStep

	marshaledBody, err := json.Marshal(cachedCompletion)
	if err != nil {
		logging.Errorf("Error marshaling regenerated cache response: %v", err)
		return cachedResponse
	}
	return marshaledBody
}

// KeystoneHeaderOptions returns the v0.4 keystone header options
// (x-vsr-schema-version + x-vsr-response-path) for an immediate response on the
// given response path. See issue #2203.
func KeystoneHeaderOptions(path string) []*core.HeaderValueOption {
	return []*core.HeaderValueOption{
		{Header: &core.HeaderValue{Key: headers.VSRSchemaVersion, RawValue: []byte(headers.SchemaVersionValue)}},
		{Header: &core.HeaderValue{Key: headers.VSRResponsePath, RawValue: []byte(path)}},
	}
}

// CreateCacheHitResponse creates an immediate response from cache.
//
// content-type, the cache-hit marker and the final routing fact (selected
// decision) ride on the default surface. The intermediate category,
// cache-similarity and matched-keyword headers are demoted off the default
// surface (#2205): each is emitted only when non-empty, so the caller demotes
// them by passing empty values unless the request opted into x-vsr-debug.
func CreateCacheHitResponse(cachedResponse []byte, isStreaming bool, category string, decisionName string, matchedKeywords []string, similarity ...float32) *ext_proc.ProcessingResponse {
	var responseBody []byte
	var contentType string

	if isStreaming {
		contentType = "text/event-stream"
		if isErrorResponse(cachedResponse) {
			responseBody = buildStreamingCacheErrorBody(cachedResponse)
		} else {
			responseBody = buildStreamingCacheBody(cachedResponse)
		}
	} else {
		contentType = "application/json"
		responseBody = buildNonStreamingCacheBody(cachedResponse)
	}

	setHeaders := []*core.HeaderValueOption{
		{
			Header: &core.HeaderValue{
				Key:      "content-type",
				RawValue: []byte(contentType),
			},
		},
		{
			Header: &core.HeaderValue{
				Key:      headers.VSRCacheHit,
				RawValue: []byte("true"),
			},
		},
		{
			Header: &core.HeaderValue{
				Key:      headers.VSRSelectedDecision,
				RawValue: []byte(decisionName),
			},
		},
	}

	// Demoted intermediate detail (#2205): emitted only when non-empty.
	if category != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRSelectedCategory,
				RawValue: []byte(category),
			},
		})
	}
	if len(similarity) > 0 && similarity[0] > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      "x-vsr-cache-similarity",
				RawValue: []byte(fmt.Sprintf("%.4f", similarity[0])),
			},
		})
	}
	if len(matchedKeywords) > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRMatchedKeywords,
				RawValue: []byte(strings.Join(matchedKeywords, ",")),
			},
		})
	}

	// v0.4 keystone headers: this is the semantic-cache path (#2203).
	setHeaders = append(setHeaders, KeystoneHeaderOptions(headers.ResponsePathCache)...)

	immediateResponse := &ext_proc.ImmediateResponse{
		Status: &typev3.HttpStatus{
			Code: typev3.StatusCode_OK,
		},
		Headers: &ext_proc.HeaderMutation{
			SetHeaders: setHeaders,
		},
		Body: responseBody,
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: immediateResponse,
		},
	}
}

// buildStreamingFastBody converts a message into SSE chunks for the fast_response plugin.
func buildStreamingFastBody(message string, unixTimeStep int64) []byte {
	chunks := splitContentIntoChunks(message)
	if len(chunks) == 0 {
		chunks = []string{message}
	}

	var sseChunks []string
	newID := fmt.Sprintf("chatcmpl-fast-%d", unixTimeStep)

	for i, chunkContent := range chunks {
		streamChunk := map[string]interface{}{
			"id":      newID,
			"object":  "chat.completion.chunk",
			"created": unixTimeStep,
			"model":   "router",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"content": chunkContent,
					},
					"finish_reason": nil,
				},
			},
		}
		if i == 0 {
			streamChunk["choices"].([]map[string]interface{})[0]["delta"].(map[string]interface{})["role"] = "assistant"
		}

		chunkJSON, err := json.Marshal(streamChunk)
		if err != nil {
			continue
		}
		sseChunks = append(sseChunks, fmt.Sprintf("data: %s\n\n", chunkJSON))
	}

	finalChunk := map[string]interface{}{
		"id":      newID,
		"object":  "chat.completion.chunk",
		"created": unixTimeStep,
		"model":   "router",
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"delta":         map[string]interface{}{},
				"finish_reason": "stop",
			},
		},
	}
	finalJSON, _ := json.Marshal(finalChunk)
	sseChunks = append(sseChunks, fmt.Sprintf("data: %s\n\n", finalJSON))
	sseChunks = append(sseChunks, "data: [DONE]\n\n")

	return []byte(strings.Join(sseChunks, ""))
}

// buildNonStreamingFastBody creates a JSON ChatCompletion body for the fast_response plugin.
func buildNonStreamingFastBody(message string, unixTimeStep int64) []byte {
	openAIResponse := openai.ChatCompletion{
		ID:      fmt.Sprintf("chatcmpl-fast-%d", unixTimeStep),
		Object:  "chat.completion",
		Created: unixTimeStep,
		Model:   "router",
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: message,
				},
				FinishReason: "stop",
			},
		},
		Usage: openai.CompletionUsage{
			PromptTokens:     0,
			CompletionTokens: 0,
			TotalTokens:      0,
		},
	}

	responseBody, err := json.Marshal(openAIResponse)
	if err != nil {
		logging.Errorf("Error marshaling fast_response: %v", err)
		return []byte(`{"error": "Failed to generate response"}`)
	}
	return responseBody
}

// CreateFastResponse creates an OpenAI-compatible immediate response for the
// fast_response plugin. It supports both streaming (SSE) and non-streaming (JSON)
// formats based on the original request's "stream" flag.
// The response is returned as 200 OK with finish_reason "stop".
func CreateFastResponse(message string, isStreaming bool, decisionName string) *ext_proc.ProcessingResponse {
	unixTimeStep := time.Now().Unix()
	var responseBody []byte
	var contentType string

	if isStreaming {
		contentType = "text/event-stream"
		responseBody = buildStreamingFastBody(message, unixTimeStep)
	} else {
		contentType = "application/json"
		responseBody = buildNonStreamingFastBody(message, unixTimeStep)
	}

	setHeaders := []*core.HeaderValueOption{
		{
			Header: &core.HeaderValue{
				Key:      "content-type",
				RawValue: []byte(contentType),
			},
		},
		{
			Header: &core.HeaderValue{
				Key:      headers.VSRSelectedDecision,
				RawValue: []byte(decisionName),
			},
		},
		{
			Header: &core.HeaderValue{
				Key:      headers.VSRFastResponse,
				RawValue: []byte("true"),
			},
		},
	}

	// v0.4 keystone headers: this is the fast_response plugin path (#2203).
	setHeaders = append(setHeaders, KeystoneHeaderOptions(headers.ResponsePathFastResponse)...)

	immediateResponse := &ext_proc.ImmediateResponse{
		Status: &typev3.HttpStatus{
			Code: typev3.StatusCode_OK,
		},
		Headers: &ext_proc.HeaderMutation{
			SetHeaders: setHeaders,
		},
		Body: responseBody,
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: immediateResponse,
		},
	}
}
