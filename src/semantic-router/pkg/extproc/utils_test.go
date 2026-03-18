package extproc

import (
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
)

func TestStatusCodeToEnumIncludesClientAndUpstreamErrors(t *testing.T) {
	tests := []struct {
		statusCode int
		want       typev3.StatusCode
	}{
		{statusCode: 400, want: typev3.StatusCode_BadRequest},
		{statusCode: 401, want: typev3.StatusCode_Unauthorized},
		{statusCode: 403, want: typev3.StatusCode_Forbidden},
		{statusCode: 404, want: typev3.StatusCode_NotFound},
		{statusCode: 405, want: typev3.StatusCode_MethodNotAllowed},
		{statusCode: 429, want: typev3.StatusCode_TooManyRequests},
		{statusCode: 500, want: typev3.StatusCode_InternalServerError},
		{statusCode: 502, want: typev3.StatusCode_BadGateway},
		{statusCode: 503, want: typev3.StatusCode_ServiceUnavailable},
	}

	for _, tt := range tests {
		t.Run(tt.want.String(), func(t *testing.T) {
			if got := statusCodeToImmediateResponseCode(tt.statusCode); got != tt.want {
				t.Fatalf("statusCodeToImmediateResponseCode(%d) = %v, want %v", tt.statusCode, got, tt.want)
			}
		})
	}
}
