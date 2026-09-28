package extproc

import (
	"encoding/json"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
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

	// Redact credentials (Authorization / x-api-key / ...) before dumping the
	// mutation: the credential resolver injects the upstream provider key as a
	// set-header, so a verbatim %+v would leak it to the log (CWE-532). Gate on
	// the debug level so the clone+redact cost is paid only when it is logged.
	if logging.DebugEnabled() {
		logging.Debugf("Processing at stage [%s]: %+v", msgType, redactResponseForLog(response))
	}

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
	// The SDK union unmarshal keeps only response_format.type; rebuild the
	// full union from the raw bytes so re-serialized bodies keep json_schema
	// payloads (issue #3024).
	if err := restoreResponseFormat(data, &req); err != nil {
		return nil, err
	}
	return &req, nil
}

// extractUserAndNonUserContent extracts content from request messages
func extractUserAndNonUserContent(req *openai.ChatCompletionNewParams) (string, []string) {
	if req == nil {
		return "", nil
	}
	var currentUser string
	nonUser := make([]string, 0, len(req.Messages))
	for _, message := range req.Messages {
		role, content := extractMessageRoleAndContent(message)
		if content == "" {
			continue
		}
		if role == "user" {
			currentUser = content
			continue
		}
		if role == "tool" {
			continue
		}
		nonUser = append(nonUser, content)
	}
	return currentUser, nonUser
}

func extractMessageRoleAndContent(msg openai.ChatCompletionMessageParamUnion) (string, string) {
	switch {
	case msg.OfUser != nil:
		return "user", extractUserMessageContent(msg.OfUser.Content)
	case msg.OfSystem != nil:
		return "system", extractSystemMessageContent(msg.OfSystem.Content)
	case msg.OfAssistant != nil:
		return "assistant", extractAssistantMessageContent(msg.OfAssistant.Content)
	case msg.OfDeveloper != nil:
		return "developer", extractDeveloperMessageContent(msg.OfDeveloper.Content)
	case msg.OfTool != nil:
		return "tool", extractToolMessageContent(msg.OfTool.Content)
	default:
		return "", ""
	}
}

func extractUserMessageContent(content openai.ChatCompletionUserMessageParamContentUnion) string {
	if content.OfString.Value != "" {
		return content.OfString.Value
	}
	return joinUserContentParts(content.OfArrayOfContentParts)
}

func extractSystemMessageContent(content openai.ChatCompletionSystemMessageParamContentUnion) string {
	if content.OfString.Value != "" {
		return content.OfString.Value
	}
	return joinSystemContentParts(content.OfArrayOfContentParts)
}

func extractAssistantMessageContent(
	content openai.ChatCompletionAssistantMessageParamContentUnion,
) string {
	if content.OfString.Value != "" {
		return content.OfString.Value
	}
	return joinAssistantContentParts(content.OfArrayOfContentParts)
}

func extractDeveloperMessageContent(content openai.ChatCompletionDeveloperMessageParamContentUnion) string {
	if content.OfString.Value != "" {
		return content.OfString.Value
	}
	return joinSystemContentParts(content.OfArrayOfContentParts)
}

func extractToolMessageContent(content openai.ChatCompletionToolMessageParamContentUnion) string {
	if content.OfString.Value != "" {
		return content.OfString.Value
	}
	return joinSystemContentParts(content.OfArrayOfContentParts)
}

func joinUserContentParts(parts []openai.ChatCompletionContentPartUnionParam) string {
	textParts := make([]string, 0, len(parts))
	for _, part := range parts {
		if part.OfText != nil {
			textParts = append(textParts, part.OfText.Text)
		}
	}
	return strings.Join(textParts, " ")
}

func joinSystemContentParts(parts []openai.ChatCompletionContentPartTextParam) string {
	textParts := make([]string, 0, len(parts))
	for _, part := range parts {
		if part.Text != "" {
			textParts = append(textParts, part.Text)
		}
	}
	return strings.Join(textParts, " ")
}

func joinAssistantContentParts(
	parts []openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion,
) string {
	textParts := make([]string, 0, len(parts))
	for _, part := range parts {
		if part.OfText != nil {
			textParts = append(textParts, part.OfText.Text)
		}
	}
	return strings.Join(textParts, " ")
}

var httpStatusToEnvoyCode = map[int]typev3.StatusCode{
	200: typev3.StatusCode_OK,
	400: typev3.StatusCode_BadRequest,
	401: typev3.StatusCode_Unauthorized,
	403: typev3.StatusCode_Forbidden,
	404: typev3.StatusCode_NotFound,
	405: typev3.StatusCode_MethodNotAllowed,
	413: typev3.StatusCode_PayloadTooLarge,
	422: typev3.StatusCode_UnprocessableEntity,
	429: typev3.StatusCode_TooManyRequests,
	500: typev3.StatusCode_InternalServerError,
	502: typev3.StatusCode_BadGateway,
	503: typev3.StatusCode_ServiceUnavailable,
}

func statusCodeToEnum(statusCode int) typev3.StatusCode {
	if code, ok := httpStatusToEnvoyCode[statusCode]; ok {
		return code
	}
	return typev3.StatusCode_OK
}

// isSafeImageDataURL returns true only for inline base64-encoded image data URIs
// with an allowlisted MIME type (e.g. "data:image/png;base64,...").
// HTTP(S) URLs, non-image data URIs, and file paths are rejected to prevent
// SSRF, local file access, and decode errors on non-image payloads.
//
// The implementation lives in the shared imageurl package so the ExtProc path
// and the HTTP classification API enforce an identical gate.
func isSafeImageDataURL(url string) bool {
	return imageurl.IsSafeImageDataURL(url)
}
