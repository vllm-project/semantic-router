package extproc

import (
	"context"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (r *OpenAIRouter) handleModelsRequestHeaders(
	method string,
	path string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	if method != "GET" || normalizeRequestPath(path) != "/v1/models" {
		return nil, nil
	}

	logging.ComponentDebugEvent("extproc", "models_request_intercepted", map[string]interface{}{
		"method": method,
		"path":   path,
	})
	traceContext := context.Background()
	if ctx != nil && ctx.TraceContext != nil {
		traceContext = ctx.TraceContext
	}
	response, err := r.handleAuthorizedModelsRequest(traceContext, ctx)
	if err != nil {
		return nil, err
	}
	if ctx != nil && response != nil {
		ctx.ImmediateResponseEncoded = true
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
	owner, _ := r.responseObjectOwner(ctx)

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
			response := responseObjectNotFound(responseID)
			var err error
			if r.ResponseAPIFilter != nil {
				response, err = r.ResponseAPIFilter.HandleGetInputItems(ctx.TraceContext, owner, responseID)
			}
			ctx.ImmediateResponseEncoded = response != nil
			return response, err
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
			response := responseObjectNotFound(responseID)
			var err error
			if r.ResponseAPIFilter != nil {
				response, err = r.ResponseAPIFilter.HandleGetResponse(ctx.TraceContext, owner, responseID)
			}
			ctx.ImmediateResponseEncoded = response != nil
			return response, err
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
			response := responseObjectNotFound(responseID)
			var err error
			if r.ResponseAPIFilter != nil {
				response, err = r.ResponseAPIFilter.HandleDeleteResponse(ctx.TraceContext, owner, responseID)
			}
			ctx.ImmediateResponseEncoded = response != nil
			return response, err
		}
	}

	if method == "POST" {
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

// detectSourceFormat classifies the inbound wire format from the request path.
// Paths under /v1/messages (the Anthropic Messages API surface, including
// /v1/messages/count_tokens) are tagged as Anthropic; everything else falls
// through to the OpenAI-compatible default represented by the zero value.
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
