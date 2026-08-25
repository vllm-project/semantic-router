package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestModelsCatalogInterceptorMatchesOnlyTheModelsResource(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{Headers: make(map[string]string)}

	response, err := router.handleRequestHeaders(newRequestHeaders("GET", "/v1/models-private"), ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders() error = %v", err)
	}
	if status := inferenceAccessDisposition(response); status != 404 {
		t.Fatalf("lookalike models path status = %d, want 404", status)
	}
	if response.GetImmediateResponse() == nil || ctx.ImmediateResponseEncoded {
		t.Fatalf("lookalike models path was handled as catalog: response=%+v context=%+v", response, ctx)
	}
}
