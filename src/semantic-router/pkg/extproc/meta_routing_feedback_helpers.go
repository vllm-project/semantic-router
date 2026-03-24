package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
)

func immediateResponseStatusCode(resp *ext_proc.ProcessingResponse) int {
	if resp == nil || resp.GetImmediateResponse() == nil || resp.GetImmediateResponse().GetStatus() == nil {
		return 0
	}
	return envoyStatusCodeToHTTP(resp.GetImmediateResponse().GetStatus().GetCode())
}

func responseStatusOrOK(ctx *RequestContext) int {
	if ctx != nil && ctx.UpstreamResponseStatus != 0 {
		return ctx.UpstreamResponseStatus
	}
	return 200
}

func envoyStatusCodeToHTTP(code typev3.StatusCode) int {
	switch code {
	case typev3.StatusCode_OK:
		return 200
	case typev3.StatusCode_BadRequest:
		return 400
	case typev3.StatusCode_Unauthorized:
		return 401
	case typev3.StatusCode_Forbidden:
		return 403
	case typev3.StatusCode_NotFound:
		return 404
	case typev3.StatusCode_MethodNotAllowed:
		return 405
	case typev3.StatusCode_PayloadTooLarge:
		return 413
	case typev3.StatusCode_UnprocessableEntity:
		return 422
	case typev3.StatusCode_TooManyRequests:
		return 429
	case typev3.StatusCode_BadGateway:
		return 502
	case typev3.StatusCode_ServiceUnavailable:
		return 503
	default:
		return 500
	}
}
