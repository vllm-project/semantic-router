package testcases

import (
	"net/http"
	"strings"
	"testing"
)

func TestValidateMalformedImageRejection(t *testing.T) {
	valid := &httpResponse{
		StatusCode: http.StatusBadRequest,
		Body:       []byte(`{"error":{"code":"INVALID_IMAGE","message":"invalid image"}}`),
	}
	if err := validateMalformedImageRejection("/api/v1/eval", valid, malformedImageE2EPayload); err != nil {
		t.Fatalf("expected valid rejection contract: %v", err)
	}

	tests := []struct {
		name string
		resp *httpResponse
		want string
	}{
		{
			name: "wrong status",
			resp: &httpResponse{StatusCode: http.StatusOK, Body: []byte(`{"error":{"code":"INVALID_IMAGE"}}`)},
			want: "status 400",
		},
		{
			name: "wrong code",
			resp: &httpResponse{StatusCode: http.StatusBadRequest, Body: []byte(`{"error":{"code":"INVALID_INPUT"}}`)},
			want: "INVALID_IMAGE",
		},
		{
			name: "payload reflected",
			resp: &httpResponse{StatusCode: http.StatusBadRequest, Body: []byte(`{"error":{"code":"INVALID_IMAGE","message":"` + malformedImageE2EPayload + `"}}`)},
			want: "reflected",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateMalformedImageRejection("/api/v1/eval", tt.resp, malformedImageE2EPayload)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("expected error containing %q, got %v", tt.want, err)
			}
		})
	}
}
