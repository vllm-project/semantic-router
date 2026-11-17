package extproc

import (
	"strings"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The default response_api is enabled with a Redis store and no address, so
// the store fails to start and the router comes up without the Responses API.
// The client must learn that from the answer, not from a 404.
func TestResponsesEndpointNamesAFailedStore(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ResponseAPI.Enabled = true
	router := &OpenAIRouter{Config: cfg}
	for _, method := range []string{"POST", "GET"} {
		ctx := &RequestContext{Headers: map[string]string{}, SourceFormat: llmprotocol.OpenAIResponsesV1}
		path := "/v1/responses"
		if method == "GET" {
			path = "/v1/responses/resp_1"
		}
		response, err := router.handleResponseAPIRequestHeaders(method, path, ctx)
		if err != nil {
			t.Fatal(err)
		}
		response = router.encodeImmediateResponseForClient(response, ctx)
		immediate := response.GetImmediateResponse()
		if immediate.GetStatus().GetCode() != typev3.StatusCode_ServiceUnavailable {
			t.Fatalf("%s status = %v, want 503", method, immediate.GetStatus())
		}
		if !strings.Contains(string(immediate.GetBody()), "store failed to start") {
			t.Fatalf("%s body = %s, want the store named", method, immediate.GetBody())
		}
	}
}

func TestResponsesEndpointStaysNotFoundWhenDisabled(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	response := router.validateRequestHeaders("POST", "/v1/responses")
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != typev3.StatusCode_NotFound {
		t.Fatalf("status = %v, want 404", got)
	}
}
