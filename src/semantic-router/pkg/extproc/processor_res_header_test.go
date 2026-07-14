package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func TestGetStatusFromHeadersUsesRawValue(t *testing.T) {
	headerMap := &core.HeaderMap{
		Headers: []*core.HeaderValue{
			{Key: ":status", RawValue: []byte("429")},
		},
	}

	if got := getStatusFromHeaders(headerMap); got != 429 {
		t.Fatalf("getStatusFromHeaders() = %d, want 429", got)
	}
}

func TestHandleResponseHeadersSetsStreamingModeOverride(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{}
	responseHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "200"},
					{Key: "content-type", Value: "text/event-stream"},
				},
			},
		},
	}

	response, err := router.handleResponseHeaders(responseHeaders, ctx)
	if err != nil {
		t.Fatalf("handleResponseHeaders failed: %v", err)
	}
	if !ctx.IsStreamingResponse {
		t.Fatal("expected streaming response to be detected")
	}
	if response.ModeOverride == nil {
		t.Fatal("expected streaming response to set mode override")
	}
	if response.ModeOverride.ResponseBodyMode != http_ext.ProcessingMode_STREAMED {
		t.Fatalf("unexpected response body mode: %v", response.ModeOverride.ResponseBodyMode)
	}
}

func TestHandleResponseHeadersUsesResponseAPIStreamRequestFallback(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest: &responseapi.ResponseAPIRequest{
				Stream: true,
			},
		},
	}
	responseHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "200"},
					{Key: "content-type", Value: ""},
				},
			},
		},
	}

	response, err := router.handleResponseHeaders(responseHeaders, ctx)
	if err != nil {
		t.Fatalf("handleResponseHeaders failed: %v", err)
	}
	if !ctx.IsStreamingResponse {
		t.Fatal("expected stream:true Responses API request to force streaming response mode")
	}
	if response.ModeOverride == nil || response.ModeOverride.ResponseBodyMode != http_ext.ProcessingMode_STREAMED {
		t.Fatalf("expected streamed mode override, got %#v", response.ModeOverride)
	}

	mutation := response.GetResponseHeaders().Response.GetHeaderMutation()
	if mutation == nil {
		t.Fatal("expected response header mutation")
	}
	if got := headerValueForTest(mutation, "content-type"); got != "text/event-stream; charset=utf-8" {
		t.Fatalf("content-type mutation = %q, want text/event-stream; charset=utf-8", got)
	}
	if got := headerAppendActionForTest(mutation, "content-type"); got != core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD {
		t.Fatalf("content-type append action = %v, want OVERWRITE_IF_EXISTS_OR_ADD", got)
	}
	if !containsStringForTest(mutation.GetRemoveHeaders(), "content-length") {
		t.Fatalf("expected content-length removal, got %#v", mutation.GetRemoveHeaders())
	}
}

func TestHandleResponseHeadersDoesNotForceResponseAPIStreamForError(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest: &responseapi.ResponseAPIRequest{
				Stream: true,
			},
		},
	}
	responseHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "400"},
					{Key: "content-type", Value: "application/json"},
				},
			},
		},
	}

	response, err := router.handleResponseHeaders(responseHeaders, ctx)
	if err != nil {
		t.Fatalf("handleResponseHeaders failed: %v", err)
	}
	if ctx.IsStreamingResponse {
		t.Fatal("did not expect error JSON response to be forced into streaming mode")
	}
	if response.ModeOverride != nil {
		t.Fatalf("did not expect mode override, got %#v", response.ModeOverride)
	}
}

// The response-header phase must persist the upstream status on the context so
// the later response-body cache-write path can refuse to cache non-2xx bodies.
func TestHandleResponseHeadersCapturesUpstreamStatus(t *testing.T) {
	for _, status := range []string{"200", "400", "429", "503"} {
		router := &OpenAIRouter{}
		ctx := &RequestContext{}
		responseHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{
				Headers: &core.HeaderMap{
					Headers: []*core.HeaderValue{{Key: ":status", Value: status}},
				},
			},
		}
		if _, err := router.handleResponseHeaders(responseHeaders, ctx); err != nil {
			t.Fatalf("handleResponseHeaders(%s) failed: %v", status, err)
		}
		want := map[string]int{"200": 200, "400": 400, "429": 429, "503": 503}[status]
		if ctx.UpstreamStatusCode != want {
			t.Fatalf("status %s: ctx.UpstreamStatusCode = %d, want %d", status, ctx.UpstreamStatusCode, want)
		}
	}
}

func TestHandleResponseHeadersLooperUsesContinueResponse(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{LooperRequest: true}
	responseHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "204"},
				},
			},
		},
	}

	response, err := router.handleResponseHeaders(responseHeaders, ctx)
	if err != nil {
		t.Fatalf("handleResponseHeaders failed: %v", err)
	}
	if response.GetResponseHeaders() == nil {
		t.Fatal("expected response headers continue response")
	}
	if response.GetResponseHeaders().Response.Status != ext_proc.CommonResponse_CONTINUE {
		t.Fatalf("expected CONTINUE status, got %v", response.GetResponseHeaders().Response.Status)
	}
	if response.ModeOverride != nil {
		t.Fatal("did not expect mode override for looper response headers")
	}
}

func headerValueForTest(mutation *ext_proc.HeaderMutation, key string) string {
	for _, header := range mutation.GetSetHeaders() {
		if header.GetHeader().GetKey() == key {
			return extractHeaderValue(header.GetHeader())
		}
	}
	return ""
}

func headerAppendActionForTest(
	mutation *ext_proc.HeaderMutation,
	key string,
) core.HeaderValueOption_HeaderAppendAction {
	for _, header := range mutation.GetSetHeaders() {
		if header.GetHeader().GetKey() == key {
			return header.GetAppendAction()
		}
	}
	return core.HeaderValueOption_APPEND_IF_EXISTS_OR_ADD
}

func containsStringForTest(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
