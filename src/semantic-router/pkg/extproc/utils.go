package extproc

import (
	"encoding/json"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// sendResponse sends a response with proper error handling and logging.
// If response is nil, a CONTINUE BodyResponse is sent as a safe fallback
// to prevent nil pointer dereferences in Envoy or test assertions.
func sendResponse(stream ext_proc.ExternalProcessor_ProcessServer, response *ext_proc.ProcessingResponse, msgType string) error {
	if response == nil {
		logging.Warnf("Nil response for %s stage — sending CONTINUE fallback to avoid nil dereference", msgType)
		response = &ext_proc.ProcessingResponse{
			Response: &ext_proc.ProcessingResponse_RequestBody{
				RequestBody: &ext_proc.BodyResponse{
					Response: &ext_proc.CommonResponse{
						Status: ext_proc.CommonResponse_CONTINUE,
					},
				},
			},
		}
	}

	logging.Debugf("Processing at stage [%s]: %+v", msgType, response)

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

// extractStreamParam extracts the stream parameter from the original request body.
// Uses gjson for O(scan) extraction without allocating a map.
func extractStreamParam(originalBody []byte) bool {
	return extractStreamParamFast(originalBody)
}

// serializeOpenAIRequestWithStream converts request back to JSON, preserving
// the stream parameter from the original request. Uses sjson for in-place
// field insertion instead of unmarshal → modify → marshal.
func serializeOpenAIRequestWithStream(req *openai.ChatCompletionNewParams, hasStreamParam bool) ([]byte, error) {
	sdkBytes, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	if hasStreamParam {
		sdkBytes = addStreamFieldsFast(sdkBytes)
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

// isSafeImageDataURL returns true only for inline base64-encoded image data URIs
// with an allowlisted MIME type (e.g. "data:image/png;base64,...").
// HTTP(S) URLs, non-image data URIs, and file paths are rejected to prevent
// SSRF, local file access, and decode errors on non-image payloads.
func isSafeImageDataURL(url string) bool {
	if url == "" {
		return false
	}
	lower := strings.ToLower(url)
	if !strings.HasPrefix(lower, "data:image/") {
		return false
	}
	const base64Sep = ";base64,"
	sepIdx := strings.Index(lower, base64Sep)
	if sepIdx == -1 {
		return false
	}
	mime := lower[len("data:"):sepIdx]
	switch mime {
	case "image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp":
	default:
		return false
	}
	payload := strings.TrimSpace(url[sepIdx+len(base64Sep):])
	return payload != ""
}

// rewriteRequestModel rewrites the model field in the request body JSON.
// Uses sjson for in-place field replacement.
func rewriteRequestModel(originalBody []byte, newModel string) ([]byte, error) {
	return rewriteModelInBodyFast(originalBody, newModel)
}
