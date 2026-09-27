//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestOpenAPIResponsesRequireExplicitSuccessContracts(t *testing.T) {
	for _, route := range apiRoutes() {
		t.Run(route.pattern(), func(t *testing.T) {
			success := false
			for status, response := range route.Responses {
				code, err := strconv.Atoi(status)
				if err != nil {
					t.Fatalf("invalid response status %q", status)
				}
				if code < 200 || code >= 300 {
					continue
				}
				success = true
				if response.Description == "" {
					t.Fatal("response description missing")
				}
				for mediaType, media := range response.Content {
					schema := media.Schema
					if schema == nil {
						t.Fatalf("%s response schema missing", mediaType)
					}
					if schema.Type == "object" && len(schema.Properties) == 0 && schema.AdditionalProperties == nil && schema.Ref == "" {
						t.Fatalf("%s is an unspecified object instead of a response contract", mediaType)
					}
				}
			}
			if !success {
				t.Fatal("handler has no declared successful response")
			}
		})
	}
}

func TestOpenAPIConfigMutationContractIncludesLifecycleAndConcurrency(t *testing.T) {
	spec := (&ClassificationAPIServer{}).generateOpenAPISpec()
	operations := []*OpenAPIOperation{
		spec.Paths[apiConfigPath].Patch, spec.Paths[apiConfigPath].Put,
		spec.Paths[apiConfigRollbackPath].Post, spec.Paths[apiRecipesPath+"/{name}"].Put,
		spec.Paths[apiRecipesPath+"/{name}"].Delete,
	}
	for _, operation := range operations {
		for _, status := range []string{"200", "202", "401", "403", "409", "412", "428", "503"} {
			if _, ok := operation.Responses[status]; !ok {
				t.Errorf("%s missing status %s", operation.OperationID, status)
			}
			if operation.Responses[status].Headers[managementRequestIDHeader].Schema.Type != "string" {
				t.Errorf("%s %s lacks request correlation header", operation.OperationID, status)
			}
		}
		for _, status := range []string{"200", "202"} {
			schema := operation.Responses[status].Content["application/json"].Schema
			if schema.Properties["activation_status"].Type != "string" || schema.Properties["generated_runtime_hash"].Type != "string" {
				t.Errorf("%s %s lacks runtime publication identity", operation.OperationID, status)
			}
			if operation.Responses[status].Headers["ETag"].Schema.Type != "string" {
				t.Errorf("%s lacks ETag", operation.OperationID)
			}
		}
		if operation.Responses["503"].Headers["ETag"].Schema.Type != "string" {
			t.Errorf("%s lacks persisted ETag after activation failure", operation.OperationID)
		}
	}
}

func TestOpenAPIResponseMediaMatchesServedContent(t *testing.T) {
	server, files := newFileUploadServer(t)
	record, err := files.Save("contract.txt", []byte("downloaded bytes"), "assistants")
	if err != nil {
		t.Fatal(err)
	}
	mux := server.setupRoutes()
	spec := server.generateOpenAPISpec()
	tests := []struct{ path, catalogPath, media, schemaType string }{
		{apiConfigVersionsPath, apiConfigVersionsPath, "application/json", "array"},
		{"/docs", "/docs", "text/html", "string"},
		{apiStorageFilesPath + "/" + record.ID + "/content", apiStorageFilesPath + "/{id}/content", "application/octet-stream", "string"},
		{apiConfigSchemaPath + "?view=full", apiConfigSchemaPath, "application/schema+json", "object"},
	}
	for _, test := range tests {
		t.Run(test.path, func(t *testing.T) {
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, test.path, nil))
			if response.Code != http.StatusOK {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
			if !strings.HasPrefix(response.Header().Get("Content-Type"), test.media) {
				t.Fatalf("content-type=%q", response.Header().Get("Content-Type"))
			}
			contract := spec.Paths[test.catalogPath].Get.Responses["200"].Content[test.media].Schema
			if contract == nil || contract.Type != test.schemaType {
				t.Fatalf("schema=%+v", contract)
			}
			if test.schemaType == "array" {
				var versions []RouterConfigVersionEntry
				if err := json.Unmarshal(response.Body.Bytes(), &versions); err != nil {
					t.Fatalf("versions is not an array: %v", err)
				}
			}
		})
	}
	first := httptest.NewRecorder()
	mux.ServeHTTP(first, httptest.NewRequest(http.MethodGet, apiConfigSchemaPath, nil))
	cached := httptest.NewRequest(http.MethodGet, apiConfigSchemaPath, nil)
	cached.Header.Set("If-None-Match", first.Header().Get("ETag"))
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, cached)
	if response.Code != http.StatusNotModified || response.Body.Len() != 0 {
		t.Fatalf("conditional schema status=%d body=%q", response.Code, response.Body.String())
	}
	contract, ok := spec.Paths[apiConfigSchemaPath].Get.Responses["304"]
	if !ok || len(contract.Content) != 0 || contract.Headers["ETag"].Schema.Type != "string" {
		t.Fatalf("304 contract=%+v", contract)
	}
}

func TestManagementErrorsMatchCommonResponseContract(t *testing.T) {
	runtime := &replayRuntimeStub{handled: true, response: routerruntime.ReplayResponse{
		StatusCode: http.StatusBadRequest,
		Body:       []byte(`{"error":{"message":"invalid replay filter","type":"invalid_request_error","code":400}}`),
	}}
	registry := routerruntime.NewRegistry(nil)
	registry.SetReplayRuntime(runtime)
	server := &ClassificationAPIServer{runtimeRegistry: registry}
	mux := server.setupRoutes()
	spec := server.generateOpenAPISpec()
	for _, test := range []struct{ path, catalogPath, wantCode string }{
		{apiConfigSchemaPath + "?view=unknown", apiConfigSchemaPath, "INVALID_SCHEMA_VIEW"},
		{apiObservabilityReplaysPath + "?limit=bad", apiObservabilityReplaysPath, "REPLAY_HTTP_400"},
	} {
		t.Run(test.path, func(t *testing.T) {
			response := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodGet, test.path, nil)
			request.Header.Set(managementRequestIDHeader, "contract-request")
			mux.ServeHTTP(response, request)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
			var payload managementErrorResponse
			if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
				t.Fatalf("non-conforming JSON error: %v", err)
			}
			if payload.Error.Code != test.wantCode || payload.Error.RequestID != "contract-request" {
				t.Fatalf("error=%+v", payload.Error)
			}
			schema := spec.Paths[test.catalogPath].Get.Responses["400"].Content["application/json"].Schema
			if schema.Properties["error"].Properties["code"].Type != "string" {
				t.Fatalf("error contract=%+v", schema)
			}
		})
	}
}

func TestOpenAPIReplayResponsesUseRuntimeDTOs(t *testing.T) {
	spec := (&ClassificationAPIServer{}).generateOpenAPISpec()
	list := spec.Paths[apiObservabilityReplaysPath].Get.Responses["200"].Content["application/json"].Schema
	if list.Properties["data"].Type != "array" || list.Properties["next_offset"].Type != "integer" || list.Properties["has_more"].Type != "boolean" {
		t.Fatalf("replay pagination contract=%+v", list)
	}
	aggregate := spec.Paths[apiObservabilityReplaysPath+"/aggregate"].Get.Responses["200"].Content["application/json"].Schema
	if aggregate.Properties["lifecycle"].Properties["failed"].Type != "integer" {
		t.Fatalf("aggregate lifecycle contract=%+v", aggregate)
	}
	trajectory := spec.Paths[apiObservabilityReplaysPath+"/trajectory"].Get.Responses["200"].Content["application/json"].Schema
	if trajectory.Properties["routes"].Items.Properties["record_id"].Type != "string" {
		t.Fatalf("trajectory record contract=%+v", trajectory)
	}
}
