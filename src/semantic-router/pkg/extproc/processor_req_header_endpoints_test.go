package extproc

import (
	"encoding/json"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestModelsInterceptorServesOnlyTheListPath(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	tests := []struct {
		path   string
		status typev3.StatusCode
	}{
		{path: "/v1/models", status: typev3.StatusCode_OK},
		{path: "/v1/models?trace=1", status: typev3.StatusCode_OK},
		{path: "/v1/models/", status: typev3.StatusCode_OK},
		{path: "/v1/models/gpt-4o-mini", status: typev3.StatusCode_NotFound},
		{path: "/v1/models-extra", status: typev3.StatusCode_NotFound},
		{path: "/v1/modelsfoo", status: typev3.StatusCode_NotFound},
	}
	for _, test := range tests {
		t.Run(test.path, func(t *testing.T) {
			response, err := router.handleRequestHeaders(
				newRequestHeaders("GET", test.path),
				&RequestContext{Headers: map[string]string{}},
			)
			if err != nil {
				t.Fatalf("handleRequestHeaders: %v", err)
			}
			immediate := response.GetImmediateResponse()
			if immediate == nil {
				t.Fatalf("expected an immediate response, got %+v", response)
			}
			if got := immediate.GetStatus().GetCode(); got != test.status {
				t.Fatalf("status = %s, want %s; body: %s", got, test.status, immediate.GetBody())
			}
			if test.status != typev3.StatusCode_OK {
				return
			}
			var list OpenAIModelList
			if json.Unmarshal(immediate.GetBody(), &list) != nil || list.Object != "list" {
				t.Fatalf("body = %s, want a model list", immediate.GetBody())
			}
		})
	}
}
