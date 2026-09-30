//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

type retrievalRouteStub struct {
	rag   pluginruntime.RAGPreviewRequest
	tools pluginruntime.ToolsPreviewRequest
	calls int
}

func TestPluginRetrievalRequiresContentReadPermissions(t *testing.T) {
	t.Setenv("VSR_RAG_PERMISSION_TEST_TOKEN", "rag-permission-test")
	server := testManagementAPIServer(t, config.ManagementAPIConfig{Auth: config.ManagementAPIAuthConfig{
		Mode:   config.ManagementAuthModeBearer,
		Tokens: []config.ManagementAPITokenRef{{Env: "VSR_RAG_PERMISSION_TEST_TOKEN", Role: "classify-only"}},
		Roles:  map[string][]string{"classify-only": {string(PermClassifyInvoke)}},
	}})
	for _, plugin := range []string{"rag", "tools", "tool_selection"} {
		for _, mode := range []string{"preview", "probe"} {
			response := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, apiPluginsPath+"/"+plugin+"/preview", strings.NewReader(`{"mode":"`+mode+`","binding":{"recipe":"private","decision":"retrieve"},"request_body":{}}`))
			request.Header.Set("Authorization", "Bearer rag-permission-test")
			server.setupRoutes().ServeHTTP(response, request)
			if response.Code != http.StatusForbidden {
				t.Fatalf("classify-only role reached %s %s: %d %s", plugin, mode, response.Code, response.Body.String())
			}
		}
		operation := server.generateOpenAPISpec().Paths[apiPluginsPath+"/"+plugin+"/preview"].Post
		permission := PermConfigRead
		if plugin == "rag" {
			permission = PermDataRead
		}
		if operation.Permission != permission || operation.Sensitivity != SensitivityConfig {
			t.Fatalf("%s content permission contract: %+v", plugin, operation)
		}
	}
}

func (stub *retrievalRouteStub) PreviewRAG(_ context.Context, input pluginruntime.RAGPreviewRequest) (pluginruntime.RAGPreviewResponse, error) {
	stub.calls++
	stub.rag = input
	return pluginruntime.RAGPreviewResponse{Binding: input.Binding, Guarantees: pluginruntime.Guarantees{Mode: input.Mode}, RequestBody: input.RequestBody}, nil
}

func (stub *retrievalRouteStub) PreviewTools(_ context.Context, input pluginruntime.ToolsPreviewRequest) (pluginruntime.ToolsPreviewResponse, error) {
	stub.calls++
	stub.tools = input
	return pluginruntime.ToolsPreviewResponse{Binding: input.Binding, Plugin: input.Plugin, Guarantees: pluginruntime.Guarantees{Mode: input.Mode}, RequestBody: input.RequestBody}, nil
}

func TestPluginRetrievalRoutesUsePublishedRuntimeAndTypedContracts(t *testing.T) {
	stub := &retrievalRouteStub{}
	registry := routerruntime.NewRegistry(nil)
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Plugins: pluginruntime.Capabilities{Retrieval: stub}})
	server := &ClassificationAPIServer{runtimeRegistry: registry}
	mux := server.setupRoutes()
	spec := server.generateOpenAPISpec()
	for _, plugin := range []string{"rag", "tools", "tool_selection"} {
		path := apiPluginsPath + "/" + plugin + "/preview"
		t.Run(plugin, func(t *testing.T) {
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"binding":{"recipe":"isolated","decision":"selected"},"mode":"probe","request_body":{"model":"sample","messages":[{"role":"user","content":"hello"}]}}`)))
			if response.Code != http.StatusOK {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
			var body map[string]any
			if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if body["mode"] != "probe" || body["persisted"] != false {
				t.Fatalf("effects=%v", body)
			}
			if plugin == "rag" {
				if stub.rag.Binding.Recipe != "isolated" {
					t.Fatalf("binding=%+v", stub.rag.Binding)
				}
			} else if stub.tools.Plugin != plugin || stub.tools.Binding.Recipe != "isolated" {
				t.Fatalf("tool target=%+v", stub.tools)
			}
			operation := spec.Paths[path].Post
			if len(operation.PluginOperations) != 2 || operation.Responses["200"].Content["application/json"].Schema.Properties["request_body"].Type != "" {
				t.Fatalf("operation contract=%+v", operation)
			}
		})
	}
	// The route fixes plugin ownership and endpoint configuration; input cannot
	// redirect execution to another plugin or an arbitrary retrieval endpoint.
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodPost, apiPluginsPath+"/tools/preview", strings.NewReader(`{"binding":{"recipe":"isolated","decision":"selected"},"request_body":{},"plugin":"rag","endpoint":"http://example.invalid"}`)))
	if response.Code != http.StatusBadRequest || stub.calls != 3 {
		t.Fatalf("unknown input reached runtime: status=%d calls=%d", response.Code, stub.calls)
	}
}
