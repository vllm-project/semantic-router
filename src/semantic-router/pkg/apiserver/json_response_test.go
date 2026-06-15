//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWriteJSONResponseReturnsErrorPayloadOnEncodeFailure(t *testing.T) {
	apiServer := &ClassificationAPIServer{}
	rr := httptest.NewRecorder()

	apiServer.writeJSONResponse(rr, http.StatusOK, map[string]interface{}{
		"bad": func() {},
	})

	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("expected status %d, got %d: %s", http.StatusInternalServerError, rr.Code, rr.Body.String())
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode error response: %v", err)
	}

	errorPayload, ok := resp["error"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected error payload, got %#v", resp)
	}
	if errorPayload["code"] != "JSON_ENCODE_ERROR" {
		t.Fatalf("expected JSON_ENCODE_ERROR, got %#v", errorPayload["code"])
	}
}

func TestReadJSONRequestBodyRejectsOversizedPayload(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/api/v1/classify/intent", strings.NewReader(`{"text":"too large"}`))

	_, err := readJSONRequestBody(req, 8)
	if err == nil {
		t.Fatal("expected oversized request body error")
	}
	if !errors.Is(err, errRequestBodyTooLarge) {
		t.Fatalf("expected errRequestBodyTooLarge, got %v", err)
	}
	if !strings.Contains(err.Error(), "exceeds 8 bytes") {
		t.Fatalf("expected request body limit error, got %v", err)
	}
}

func TestWriteJSONRequestErrorMapsOversizedPayloadTo413(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/api/v1/classify/intent", strings.NewReader(`{"text":"too large"}`))
	_, err := readJSONRequestBody(req, 8)
	if err == nil {
		t.Fatal("expected oversized request body error")
	}

	rr := httptest.NewRecorder()
	server := &ClassificationAPIServer{}
	server.writeJSONRequestError(rr, err)

	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected status %d, got %d", http.StatusRequestEntityTooLarge, rr.Code)
	}

	var errorPayload map[string]interface{}
	if decodeErr := json.Unmarshal(rr.Body.Bytes(), &errorPayload); decodeErr != nil {
		t.Fatalf("failed to decode error response: %v", decodeErr)
	}
	errorBody, ok := errorPayload["error"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected nested error payload, got %#v", errorPayload)
	}
	if errorBody["code"] != "REQUEST_BODY_TOO_LARGE" {
		t.Fatalf("expected REQUEST_BODY_TOO_LARGE, got %#v", errorBody["code"])
	}
}

func TestDecodeJSONBodyWrapsSyntaxError(t *testing.T) {
	var payload map[string]interface{}

	err := decodeJSONBody([]byte(`{"text":`), &payload)
	if err == nil {
		t.Fatal("expected JSON syntax error")
	}
	if !strings.Contains(err.Error(), "failed to parse JSON") {
		t.Fatalf("expected wrapped JSON parse error, got %v", err)
	}
}
