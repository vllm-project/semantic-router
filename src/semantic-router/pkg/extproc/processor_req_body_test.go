package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
)

func TestHandleRequestBodyReturnsBadRequestForMalformedJSON(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{Headers: make(map[string]string)}

	response, err := router.HandleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(`{not json`)},
	}, ctx)
	if err != nil {
		t.Fatalf("HandleRequestBody returned error: %v", err)
	}

	assertBodyImmediateErrorResponse(t, response, typev3.StatusCode_BadRequest, "invalid inference request")
}

func TestHandleRequestBodyReturnsBadRequestForMissingMessages(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{Headers: make(map[string]string)}

	response, err := router.HandleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(`{"model":"model-a"}`)},
	}, ctx)
	if err != nil {
		t.Fatalf("HandleRequestBody returned error: %v", err)
	}

	assertBodyImmediateErrorResponse(t, response, typev3.StatusCode_BadRequest, "invalid inference request")
}

func assertBodyImmediateErrorResponse(
	t *testing.T,
	response *ext_proc.ProcessingResponse,
	wantStatus typev3.StatusCode,
	wantMessage string,
) {
	t.Helper()

	immediate := response.GetImmediateResponse()
	if immediate == nil {
		t.Fatal("expected immediate response")
	}
	if immediate.Status == nil || immediate.Status.Code != wantStatus {
		t.Fatalf("expected status %v, got %v", wantStatus, immediate.Status)
	}

	var decoded struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(immediate.Body, &decoded); err != nil {
		t.Fatalf("failed to decode error response: %v", err)
	}
	if !strings.Contains(decoded.Error.Message, wantMessage) {
		t.Fatalf("expected error message to contain %q, got %q", wantMessage, decoded.Error.Message)
	}
}
