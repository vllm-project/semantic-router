package extproc

import (
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

func (r *OpenAIRouter) validateRequestHeaders(method string, path string) *ext_proc.ProcessingResponse {
	response, _ := r.classifyRequestHeaders(method, path)
	return response
}

// classifyRequestHeaders validates the public route and reports whether an
// accepted request is an inference call that must carry a JSON body.
func (r *OpenAIRouter) classifyRequestHeaders(method string, path string) (*ext_proc.ProcessingResponse, bool) {
	normalizedPath := normalizeRequestPath(path)

	switch normalizedPath {
	case "/v1/chat/completions", azureV1ChatPath:
		return validateInferenceMethod(r, method)
	case "/v1/messages":
		return validateInferenceMethod(r, method)
	case "/v1/models":
		return validateAllowedMethod(r, method, "GET"), false
	case "/v1/responses", azureResponsesPath, azureV1ResponsesPath:
		return r.validateResponseAPICollectionMethod(method)
	}

	if extractResponseIDFromInputItemsPath(normalizedPath) != "" {
		return validateAllowedMethod(r, method, "GET"), false
	}

	if extractResponseIDFromPath(normalizedPath) != "" {
		return r.validateResponseAPIItemMethod(method), false
	}

	if _, ok := azureChatDeployment(normalizedPath); ok {
		return validateInferenceMethod(r, method)
	}

	if isAzureOpenAIPath(normalizedPath) {
		return r.createErrorResponse(404, "endpoint not found"), false
	}

	if normalizedPath == routerReplayAPIBasePath || strings.HasPrefix(normalizedPath, routerReplayAPIBasePath+"/") {
		return r.createErrorResponse(404, "endpoint not found"), false
	}

	if normalizedPath == "/v1" || strings.HasPrefix(normalizedPath, "/v1/") {
		return r.createErrorResponse(404, "endpoint not found"), false
	}

	return nil, false
}

// validateInferenceMethod accepts only POST, which always carries the JSON request.
func validateInferenceMethod(r *OpenAIRouter, method string) (*ext_proc.ProcessingResponse, bool) {
	if response := validateAllowedMethod(r, method, "POST"); response != nil {
		return response, false
	}
	return nil, true
}

// rejectBodylessInferenceRequest runs the ingress codec on the body Envoy will
// never send, so end_of_stream headers get the body stage's own 400.
func (r *OpenAIRouter) rejectBodylessInferenceRequest(ctx *RequestContext) *ext_proc.ProcessingResponse {
	engine, err := r.protocolEngine()
	if err != nil {
		return r.createErrorResponse(503, "protocol runtime unavailable")
	}
	_, _, _, err = engine.DecodeRequestForMutation(ctx.SourceFormat, nil)
	return r.ingressDecodeErrorResponse(ctx, err)
}

func (r *OpenAIRouter) validateResponseAPICollectionMethod(method string) (*ext_proc.ProcessingResponse, bool) {
	if r.ResponseAPIFilter == nil || !r.ResponseAPIFilter.IsEnabled() {
		return r.createErrorResponse(404, "endpoint not found"), false
	}

	return validateInferenceMethod(r, method)
}

func (r *OpenAIRouter) validateResponseAPIItemMethod(method string) *ext_proc.ProcessingResponse {
	if method == "GET" || method == "DELETE" {
		return nil
	}

	return r.createErrorResponse(405, "method not allowed")
}

func validateAllowedMethod(r *OpenAIRouter, method string, allowed string) *ext_proc.ProcessingResponse {
	if method == allowed {
		return nil
	}
	return r.createErrorResponse(405, "method not allowed")
}

func normalizeRequestPath(path string) string {
	if idx := strings.Index(path, "?"); idx != -1 {
		path = path[:idx]
	}
	if len(path) > 1 {
		path = strings.TrimSuffix(path, "/")
	}
	return path
}
