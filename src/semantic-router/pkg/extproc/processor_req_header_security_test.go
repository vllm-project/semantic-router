package extproc

import (
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestHandleRequestHeadersSanitizesClientSelectedModel(t *testing.T) {
	t.Run("normal processing", func(t *testing.T) {
		assertClientSelectedModelSanitized(t, &OpenAIRouter{}, false)
	})
	t.Run("skip processing", func(t *testing.T) {
		assertClientSelectedModelSanitized(t, newRouterWithSkipProcessingGate(true), true)
	})
}

func assertClientSelectedModelSanitized(t *testing.T, router *OpenAIRouter, skipProcessing bool) {
	t.Helper()
	values := []*core.HeaderValue{
		{Key: ":method", Value: "POST"},
		{Key: ":path", Value: "/v1/chat/completions"},
		{Key: headers.SelectedModel, Value: "forged-lower"},
		{Key: "X-SELECTED-MODEL", Value: "forged-upper"},
		{Key: "X-Selected-Model", Value: "forged-mixed"},
		{Key: "x-client-metadata", Value: "retained"},
	}
	if skipProcessing {
		values = append(values, &core.HeaderValue{
			Key: headers.VSRSkipProcessing, Value: "true",
		})
	}

	ctx := &RequestContext{Headers: make(map[string]string)}
	response, err := router.handleRequestHeaders(
		&ext_proc.ProcessingRequest_RequestHeaders{
			RequestHeaders: &ext_proc.HttpHeaders{
				Headers: &core.HeaderMap{Headers: values},
			},
		},
		ctx,
	)
	if err != nil {
		t.Fatalf("handleRequestHeaders failed: %v", err)
	}
	if ctx.SkipProcessing != skipProcessing {
		t.Fatalf("SkipProcessing = %v, want %v", ctx.SkipProcessing, skipProcessing)
	}
	for name := range ctx.Headers {
		if strings.EqualFold(name, headers.SelectedModel) {
			t.Fatalf("untrusted routing header %q entered request context", name)
		}
	}
	if got := ctx.Headers["x-client-metadata"]; got != "retained" {
		t.Fatalf("ordinary client metadata = %q, want retained", got)
	}

	common := response.GetRequestHeaders().GetResponse()
	if common == nil {
		t.Fatal("expected request-header continue response")
	}
	if !common.GetClearRouteCache() {
		t.Fatal("removing a client routing header must clear Envoy's cached route")
	}
	removed := common.GetHeaderMutation().GetRemoveHeaders()
	if !containsHeaderName(removed, headers.SelectedModel) {
		t.Fatalf("%s was not removed from the live Envoy request: %v", headers.SelectedModel, removed)
	}
}

func TestHandleRequestHeadersDoesNotClearRouteWithoutClientSelectedModel(t *testing.T) {
	ctx := &RequestContext{Headers: make(map[string]string)}
	response, err := (&OpenAIRouter{}).handleRequestHeaders(
		newRequestHeaders("POST", "/v1/chat/completions"),
		ctx,
	)
	if err != nil {
		t.Fatalf("handleRequestHeaders failed: %v", err)
	}
	if response.GetRequestHeaders().GetResponse().GetClearRouteCache() {
		t.Fatal("header phase should not recompute the route when no client routing header was removed")
	}
}
