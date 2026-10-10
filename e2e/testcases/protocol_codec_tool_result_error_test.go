package testcases

import (
	"net/http"
	"testing"
)

func TestToolResultErrorRejectionAssertion(t *testing.T) {
	valid := `{"type":"error","error":{"type":"invalid_request_error","message":"translation would lose tool_result.is_error"}}`
	for _, test := range []struct {
		name       string
		status     int
		body       string
		dispatched bool
		wantError  bool
	}{
		{"native Messages error", http.StatusBadRequest, valid, false, false},
		{"dispatched despite error", http.StatusBadRequest, valid, true, true},
		{"success status", http.StatusOK, valid, false, true},
		{"unrelated rejection", http.StatusBadRequest, `{"type":"error","error":{"type":"invalid_request_error","message":"invalid model"}}`, false, true},
		{"wrong protocol shape", http.StatusBadRequest, `{"error":{"code":"lossy_translation"}}`, false, true},
		{"invalid JSON", http.StatusBadRequest, "not JSON", false, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := assertToolResultErrorRejection(protocolMatrixHTTPResult{StatusCode: test.status, Body: []byte(test.body)}, test.dispatched)
			if (err != nil) != test.wantError {
				t.Fatalf("error=%v, wantError=%t", err, test.wantError)
			}
		})
	}
}
