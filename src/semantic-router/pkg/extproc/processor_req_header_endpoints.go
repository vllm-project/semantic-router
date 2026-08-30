package extproc

import (
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (r *OpenAIRouter) handleModelsRequestHeaders(
	method string,
	path string,
) (*ext_proc.ProcessingResponse, error) {
	if method != "GET" || !strings.HasPrefix(path, "/v1/models") {
		return nil, nil
	}

	logging.ComponentDebugEvent("extproc", "models_request_intercepted", map[string]interface{}{
		"method": method,
		"path":   path,
	})
	response, err := r.handleModelsRequest(path)
	if err != nil {
		return nil, err
	}
	return response, nil
}

func (r *OpenAIRouter) handleResponseAPIRequestHeaders(
	method string,
	path string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	if !strings.HasPrefix(path, "/v1/responses") {
		return nil, nil
	}

	switch method {
	case "GET":
		return r.handleResponseAPIGet(path, ctx)
	case "DELETE":
		return r.handleResponseAPIDelete(path, ctx)
	case "POST":
		logging.ComponentDebugEvent("extproc", "response_api_request_detected", map[string]interface{}{
			"request_id": ctx.RequestID,
			"method":     method,
			"path":       path,
			"operation":  "create_response",
		})
	}

	return nil, nil
}

func (r *OpenAIRouter) handleResponseAPIGet(
	path string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	if responseID := extractResponseIDFromInputItemsPath(path); responseID != "" {
		logResponseAPIOperation(ctx, "GET", path, "get_input_items", responseID)
		if r.ResponseAPIFilter == nil {
			return responseObjectNotFound(responseID), nil
		}
		return r.ResponseAPIFilter.HandleGetInputItems(ctx.TraceContext, responseID)
	}
	responseID := extractResponseIDFromPath(path)
	if responseID == "" {
		return nil, nil
	}
	logResponseAPIOperation(ctx, "GET", path, "get_response", responseID)
	if r.ResponseAPIFilter == nil {
		return responseObjectNotFound(responseID), nil
	}
	return r.ResponseAPIFilter.HandleGetResponse(ctx.TraceContext, responseID)
}

func (r *OpenAIRouter) handleResponseAPIDelete(
	path string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	responseID := extractResponseIDFromPath(path)
	if responseID == "" {
		return nil, nil
	}
	logResponseAPIOperation(ctx, "DELETE", path, "delete_response", responseID)
	if r.ResponseAPIFilter == nil {
		return responseObjectNotFound(responseID), nil
	}
	return r.ResponseAPIFilter.HandleDeleteResponse(ctx.TraceContext, responseID)
}

func logResponseAPIOperation(
	ctx *RequestContext,
	method, path, operation, responseID string,
) {
	logging.ComponentDebugEvent("extproc", "response_api_request_intercepted", map[string]interface{}{
		"request_id":  ctx.RequestID,
		"method":      method,
		"path":        path,
		"operation":   operation,
		"response_id": responseID,
	})
}

// extractResponseIDFromPath extracts the response ID from a path like /v1/responses/{id}.
func extractResponseIDFromPath(path string) string {
	if idx := strings.Index(path, "?"); idx != -1 {
		path = path[:idx]
	}

	const prefix = "/v1/responses/"
	if !strings.HasPrefix(path, prefix) {
		return ""
	}

	responseID := strings.TrimSuffix(strings.TrimPrefix(path, prefix), "/")
	if strings.Contains(responseID, "/") {
		return ""
	}
	if responseID != "" && strings.HasPrefix(responseID, "resp_") {
		return responseID
	}
	return ""
}

// extractResponseIDFromInputItemsPath extracts the response ID from /v1/responses/{id}/input_items.
func extractResponseIDFromInputItemsPath(path string) string {
	if idx := strings.Index(path, "?"); idx != -1 {
		path = path[:idx]
	}

	const (
		prefix = "/v1/responses/"
		suffix = "/input_items"
	)
	if !strings.HasPrefix(path, prefix) || !strings.HasSuffix(path, suffix) {
		return ""
	}

	responseID := strings.TrimSuffix(strings.TrimPrefix(path, prefix), suffix)
	if responseID != "" && strings.HasPrefix(responseID, "resp_") {
		return responseID
	}
	return ""
}

// detectSourceFormat classifies the public wire format from the request path.
func detectSourceFormat(path string, ctx *RequestContext) {
	switch {
	case strings.HasPrefix(path, "/v1/messages"):
		ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
		logging.Debugf("Detected Anthropic client protocol from path: %s", path)
	case strings.HasPrefix(path, "/v1/responses"):
		ctx.SourceFormat = llmprotocol.OpenAIResponsesV1
	default:
		ctx.SourceFormat = llmprotocol.OpenAIChatV1
	}
}
