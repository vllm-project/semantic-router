package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

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
