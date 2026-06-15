package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
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
