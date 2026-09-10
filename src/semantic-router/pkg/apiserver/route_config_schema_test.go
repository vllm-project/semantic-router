//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configschema"
)

func TestConfigSchemaRoutePublishesGeneratedContract(t *testing.T) {
	server := &ClassificationAPIServer{}
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, configschema.SchemaEndpoint, nil)
	server.handleConfigSchema(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if got := response.Header().Get("Content-Type"); got != "application/schema+json" {
		t.Fatalf("content type=%q", got)
	}
	if response.Header().Get("ETag") != configschema.ETag() {
		t.Fatal("schema response did not expose its generated content identity")
	}
	var document map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &document); err != nil {
		t.Fatalf("invalid schema JSON: %v", err)
	}
	if document["$id"] != configschema.SchemaID {
		t.Fatalf("schema id=%v", document["$id"])
	}

	notModified := httptest.NewRecorder()
	cachedRequest := httptest.NewRequest(http.MethodGet, configschema.SchemaEndpoint, nil)
	cachedRequest.Header.Set("If-None-Match", configschema.ETag())
	server.handleConfigSchema(notModified, cachedRequest)
	if notModified.Code != http.StatusNotModified || notModified.Body.Len() != 0 {
		t.Fatalf("conditional response status=%d body=%q", notModified.Code, notModified.Body.String())
	}
}
