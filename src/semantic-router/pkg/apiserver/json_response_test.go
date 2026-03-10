//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
