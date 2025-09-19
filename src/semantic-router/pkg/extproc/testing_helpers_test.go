package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// Test helper methods to expose private functionality for testing

// HandleRequestHeaders exposes handleRequestHeaders for testing
func (r *OpenAIRouter) HandleRequestHeaders(v *ext_proc.ProcessingRequest_RequestHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	return r.handleRequestHeaders(v, ctx)
}

// HandleRequestBody exposes handleRequestBody for testing
func (r *OpenAIRouter) HandleRequestBody(v *ext_proc.ProcessingRequest_RequestBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	return r.handleRequestBody(v, ctx)
}

// HandleResponseHeaders exposes handleResponseHeaders for testing
func (r *OpenAIRouter) HandleResponseHeaders(v *ext_proc.ProcessingRequest_ResponseHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	return r.handleResponseHeaders(v, ctx)
}

// HandleResponseBody exposes handleResponseBody for testing
func (r *OpenAIRouter) HandleResponseBody(v *ext_proc.ProcessingRequest_ResponseBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	return r.handleResponseBody(v, ctx)
}
