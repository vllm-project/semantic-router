package extproc

import typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

func statusCodeToImmediateResponseCode(statusCode int) typev3.StatusCode {
	switch statusCode {
	case 200, 400, 404, 500:
		return statusCodeToEnum(statusCode)
	}

	switch statusCode {
	case 401:
		return typev3.StatusCode_Unauthorized
	case 403:
		return typev3.StatusCode_Forbidden
	case 405:
		return typev3.StatusCode_MethodNotAllowed
	case 429:
		return typev3.StatusCode_TooManyRequests
	case 502:
		return typev3.StatusCode_BadGateway
	case 503:
		return typev3.StatusCode_ServiceUnavailable
	default:
		return typev3.StatusCode_OK
	}
}
