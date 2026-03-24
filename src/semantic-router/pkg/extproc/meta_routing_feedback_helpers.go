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
	if statusCode, ok := envoyToHTTPStatusCodes()[code]; ok {
		return statusCode
	}
	return 500
}

func envoyToHTTPStatusCodes() map[typev3.StatusCode]int {
	return map[typev3.StatusCode]int{
		typev3.StatusCode_OK:                  200,
		typev3.StatusCode_BadRequest:          400,
		typev3.StatusCode_Unauthorized:        401,
		typev3.StatusCode_Forbidden:           403,
		typev3.StatusCode_NotFound:            404,
		typev3.StatusCode_MethodNotAllowed:    405,
		typev3.StatusCode_PayloadTooLarge:     413,
		typev3.StatusCode_UnprocessableEntity: 422,
		typev3.StatusCode_TooManyRequests:     429,
		typev3.StatusCode_BadGateway:          502,
		typev3.StatusCode_ServiceUnavailable:  503,
	}
}
