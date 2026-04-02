package extproc

import (
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

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
	if r.ResponseAPIFilter == nil || !r.ResponseAPIFilter.IsEnabled() || !strings.HasPrefix(path, "/v1/responses") {
		return nil, nil
	}

	if method == "GET" && strings.HasSuffix(path, "/input_items") {
		responseID := extractResponseIDFromInputItemsPath(path)
		if responseID != "" {
			logging.ComponentDebugEvent("extproc", "response_api_request_intercepted", map[string]interface{}{
				"request_id":  ctx.RequestID,
				"method":      method,
				"path":        path,
				"operation":   "get_input_items",
				"response_id": responseID,
			})
			return r.ResponseAPIFilter.HandleGetInputItems(ctx.TraceContext, responseID)
		}
	}

	if method == "GET" {
		responseID := extractResponseIDFromPath(path)
		if responseID != "" {
			logging.ComponentDebugEvent("extproc", "response_api_request_intercepted", map[string]interface{}{
				"request_id":  ctx.RequestID,
				"method":      method,
				"path":        path,
				"operation":   "get_response",
				"response_id": responseID,
			})
			return r.ResponseAPIFilter.HandleGetResponse(ctx.TraceContext, responseID)
		}
	}

	if method == "DELETE" {
		responseID := extractResponseIDFromPath(path)
		if responseID != "" {
			logging.ComponentDebugEvent("extproc", "response_api_request_intercepted", map[string]interface{}{
				"request_id":  ctx.RequestID,
				"method":      method,
				"path":        path,
				"operation":   "delete_response",
				"response_id": responseID,
			})
			return r.ResponseAPIFilter.HandleDeleteResponse(ctx.TraceContext, responseID)
		}
	}

	if method == "POST" {
		ctx.ResponseAPICtx = &ResponseAPIContext{IsResponseAPIRequest: true}
		logging.ComponentDebugEvent("extproc", "response_api_request_detected", map[string]interface{}{
			"request_id": ctx.RequestID,
			"method":     method,
			"path":       path,
			"operation":  "create_response",
		})
	}

	return nil, nil
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
