package extproc

import (
	"encoding/json"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// sendResponse sends a response with proper error handling and logging
func sendResponse(stream ext_proc.ExternalProcessor_ProcessServer, response *ext_proc.ProcessingResponse, msgType string) error {
	logging.Debugf("Processing at stage [%s]: %+v", msgType, response)

	// Debug: dump response structure if needed
	if err := stream.Send(response); err != nil {
		logging.Errorf("Error sending %s response: %v", msgType, err)
		return err
	}
	return nil
}

// parseOpenAIRequest parses the raw JSON using the OpenAI SDK types
func parseOpenAIRequest(data []byte) (*openai.ChatCompletionNewParams, error) {
	var req openai.ChatCompletionNewParams
	if err := json.Unmarshal(data, &req); err != nil {
		return nil, err
	}
	return &req, nil
}

// extractStreamParam extracts the stream parameter from the original request body
func extractStreamParam(originalBody []byte) bool {
	var requestMap map[string]interface{}
	if err := json.Unmarshal(originalBody, &requestMap); err != nil {
		return false
	}

	if streamValue, exists := requestMap["stream"]; exists {
		if stream, ok := streamValue.(bool); ok {
			return stream
		}
	}
	return false
}

// serializeOpenAIRequestWithStream converts request back to JSON, preserving the stream parameter from original request
func serializeOpenAIRequestWithStream(req *openai.ChatCompletionNewParams, hasStreamParam bool) ([]byte, error) {
	// First serialize the SDK object
	sdkBytes, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	// If original request had stream parameter, add it back along with stream_options
	if hasStreamParam {
		var sdkMap map[string]interface{}
		if err := json.Unmarshal(sdkBytes, &sdkMap); err == nil {
			sdkMap["stream"] = true

			// Automatically add stream_options to enable usage tracking in streaming responses
			// This ensures vLLM returns token usage information in the final chunk
			sdkMap["stream_options"] = map[string]interface{}{
				"include_usage": true,
			}

			logging.Infof("Added stream_options.include_usage=true for streaming request")

			if modifiedBytes, err := json.Marshal(sdkMap); err == nil {
				return modifiedBytes, nil
			}
		}
	}

	return sdkBytes, nil
}

// extractUserAndNonUserContent extracts content from request messages
func extractUserAndNonUserContent(req *openai.ChatCompletionNewParams) (string, []string) {
	var userContent string
	var nonUser []string

	for _, msg := range req.Messages {
		// Extract content based on message type
		var textContent string
		var role string

		if msg.OfUser != nil {
			role = "user"
			// Handle user message content
			if msg.OfUser.Content.OfString.Value != "" {
				textContent = msg.OfUser.Content.OfString.Value
			} else if len(msg.OfUser.Content.OfArrayOfContentParts) > 0 {
				// Extract text from content parts
				var parts []string
				for _, part := range msg.OfUser.Content.OfArrayOfContentParts {
					if part.OfText != nil {
						parts = append(parts, part.OfText.Text)
					}
				}
				textContent = strings.Join(parts, " ")
			}
		} else if msg.OfSystem != nil {
			role = "system"
			if msg.OfSystem.Content.OfString.Value != "" {
				textContent = msg.OfSystem.Content.OfString.Value
			} else if len(msg.OfSystem.Content.OfArrayOfContentParts) > 0 {
				// Extract text from content parts
				var parts []string
				for _, part := range msg.OfSystem.Content.OfArrayOfContentParts {
					if part.Text != "" {
						parts = append(parts, part.Text)
					}
				}
				textContent = strings.Join(parts, " ")
			}
		} else if msg.OfAssistant != nil {
			role = "assistant"
			if msg.OfAssistant.Content.OfString.Value != "" {
				textContent = msg.OfAssistant.Content.OfString.Value
			} else if len(msg.OfAssistant.Content.OfArrayOfContentParts) > 0 {
				// Extract text from content parts
				var parts []string
				for _, part := range msg.OfAssistant.Content.OfArrayOfContentParts {
					if part.OfText != nil {
						parts = append(parts, part.OfText.Text)
					}
				}
				textContent = strings.Join(parts, " ")
			}
		}

		// Categorize by role
		if role == "user" {
			userContent = textContent
		} else if role != "" {
			nonUser = append(nonUser, textContent)
		}
	}

	return userContent, nonUser
}

// statusCodeToEnum converts HTTP status code to typev3.StatusCode enum
func statusCodeToEnum(statusCode int) typev3.StatusCode {
	switch statusCode {
	case 200:
		return typev3.StatusCode_OK
	case 400:
		return typev3.StatusCode_BadRequest
	case 404:
		return typev3.StatusCode_NotFound
	case 500:
		return typev3.StatusCode_InternalServerError
	default:
		return typev3.StatusCode_OK
	}
}

// rewriteRequestModel rewrites the model field in the request body JSON
// Used by looper internal requests to route to specific models
func rewriteRequestModel(originalBody []byte, newModel string) ([]byte, error) {
	var requestMap map[string]interface{}
	if err := json.Unmarshal(originalBody, &requestMap); err != nil {
		return nil, err
	}

	requestMap["model"] = newModel

	return json.Marshal(requestMap)
}
