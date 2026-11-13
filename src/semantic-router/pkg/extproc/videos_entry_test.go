package extproc

import (
	"bytes"
	"errors"
	"testing"

	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestSplitAndStampVideoID(t *testing.T) {
	route, providerID, ok := splitVideoID("route:provider:part")
	if !ok || route != "route" || providerID != "provider:part" {
		t.Fatalf("split = %q, %q, %t", route, providerID, ok)
	}
	if _, _, ok := splitVideoID("missing-colon"); ok {
		t.Fatal("an id without a route must fail")
	}
	if got := stampVideoID("route", "provider:part"); got != "route:provider:part" {
		t.Fatalf("stamped = %q", got)
	}
	if got := stampVideoID("route", "route:provider:part"); got != "route:provider:part" {
		t.Fatalf("re-stamp changed %q", got)
	}
}

func videosHeaderTestRouter() *OpenAIRouter {
	return &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{
			"video-route": {APIFormat: config.APIFormatVideos, Capabilities: []string{"video_generation"}},
			"chat-only":   {APIFormat: config.APIFormatChatCompletions, Capabilities: []string{"text"}},
			"bad:route":   {APIFormat: config.APIFormatVideos, Capabilities: []string{"video_generation"}},
		},
	}}}
}

func TestVideosObjectHeadersSetRouteModelAndProviderPath(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{}}
	response, err := videosHeaderTestRouter().handleRequestHeaders(newRequestHeaders("GET", "/v1/videos/video-route:provider:part"), ctx)
	if err != nil {
		t.Fatal(err)
	}
	mutation := response.GetRequestHeaders().GetResponse().GetHeaderMutation()
	if got, _ := accountingHeaderValue(mutation, headers.VSRRouteModel); got != "video-route" {
		t.Fatalf("route model header = %q", got)
	}
	if got, _ := accountingHeaderValue(mutation, ":path"); got != "/v1/videos/provider:part" {
		t.Fatalf("provider path = %q", got)
	}
	if !response.GetRequestHeaders().GetResponse().GetClearRouteCache() {
		t.Fatal("route header mutation must clear the route cache")
	}
}

func TestVideosObjectHeadersFailClosed(t *testing.T) {
	for _, path := range []string{
		"/v1/videos/missing:provider",
		"/v1/videos/chat-only:provider",
		"/v1/videos/bad:route:provider",
	} {
		t.Run(path, func(t *testing.T) {
			ctx := &RequestContext{Headers: map[string]string{}}
			response, err := videosHeaderTestRouter().handleRequestHeaders(newRequestHeaders("GET", path), ctx)
			if err != nil {
				t.Fatal(err)
			}
			if response.GetImmediateResponse().GetStatus().GetCode() != typev3.StatusCode_NotFound ||
				ctx.ImmediateProtocolError == nil || ctx.ImmediateProtocolError.Code != "video_not_found" {
				t.Fatalf("response = %+v, error = %+v", response, ctx.ImmediateProtocolError)
			}
		})
	}
}

func TestVideosContentHeadersStreamAndBodyPassesThrough(t *testing.T) {
	ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIVideosV1, videosContent: true}
	response := answerHeaders(t, &OpenAIRouter{}, ctx, "application/octet-stream")
	if response.ModeOverride == nil || response.ModeOverride.ResponseBodyMode != http_ext.ProcessingMode_STREAMED {
		t.Fatalf("content response mode = %+v", response.ModeOverride)
	}
	ctx.UpstreamStatusCode = 200
	chunk := []byte{0x00, 0x01, 0xfe, 0xff}
	bodyResponse, err := (&OpenAIRouter{}).handleResponseBody(responseBodyRequest(chunk, true), ctx)
	if err != nil {
		t.Fatal(err)
	}
	if mutation := bodyResponse.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
		t.Fatalf("content body was mutated: %+v", mutation)
	}
}

func TestVideosCreateResponseIsReencodedWithStampedID(t *testing.T) {
	router := &OpenAIRouter{}
	request := &llmprotocol.Request{
		Generation: 1, Model: "video-route", VideoGeneration: &llmprotocol.VideoGenerationOptions{Prompt: "a lighthouse"},
	}
	ctx := routingTestContext(llmprotocol.OpenAIVideosV1, request)
	ctx.TargetFormat, ctx.RequestModel, ctx.videosCreate = llmprotocol.OpenAIVideosV1, "video-route", true
	body := []byte(`{"id":"provider_1","object":"video","status":"queued","created_at":1,"provider_extra":"drop"}`)
	response := router.handleNonStreamingResponseBody(body, ctx, 0, false)
	rewritten := response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
	if !bytes.Contains(rewritten, []byte(`"id":"video-route:provider_1"`)) || bytes.Contains(rewritten, []byte("provider_extra")) {
		t.Fatalf("rewritten response = %s", rewritten)
	}

	canonical := []byte(`{"id":"video-route:provider_1","object":"video","status":"queued","created_at":1}`)
	ctx = routingTestContext(llmprotocol.OpenAIVideosV1, request)
	ctx.TargetFormat, ctx.RequestModel, ctx.videosCreate = llmprotocol.OpenAIVideosV1, "video-route", true
	response = router.handleNonStreamingResponseBody(canonical, ctx, 0, false)
	if got := response.GetResponseBody().GetResponse().GetBodyMutation().GetBody(); !bytes.Equal(got, canonical) {
		t.Fatalf("idempotent re-stamp = %s, want %s", got, canonical)
	}
}

func TestVideosCreateRequiresAVideoJobDecision(t *testing.T) {
	router := videosHeaderTestRouter()
	request := &llmprotocol.Request{
		Generation: 1, Model: "video-route", VideoGeneration: &llmprotocol.VideoGenerationOptions{Prompt: "a lighthouse"},
	}
	ctx := routingTestContext(llmprotocol.OpenAIVideosV1, request)
	ctx.videosCreate = true
	ctx.VSRSelectedDecision = &config.Decision{Name: "videos", Output: &config.DecisionOutput{Kind: config.OutputKinds{config.OutputKindVideo}}}
	if err := router.applyDecisionOutput(ctx); err != nil {
		t.Fatalf("video decision refused: %v", err)
	}
	if request.VideoGeneration == nil || !llmprotocol.RequiredCapabilities(*request).Supports(llmprotocol.CapabilityVideoGeneration) {
		t.Fatalf("video operation was not forced: %+v", request)
	}

	ctx.VSRSelectedDecision = &config.Decision{Name: "chat", Output: &config.DecisionOutput{Kind: config.OutputKinds{config.OutputKindText}}}
	err := router.applyDecisionOutput(ctx)
	var protocolErr *llmprotocol.ProtocolError
	if !errors.As(err, &protocolErr) || protocolErr.Code != "unsupported_capability" {
		t.Fatalf("chat decision error = %v", err)
	}
}

func TestVideosObjectResponseUsesTheClientVisibleID(t *testing.T) {
	for name, body := range map[string]string{
		"status": `{"id":"provider_1","object":"video","status":"completed","created_at":1}`,
		"delete": `{"id":"provider_1","object":"video.deleted","deleted":true}`,
	} {
		t.Run(name, func(t *testing.T) {
			ctx := routingTestContext(llmprotocol.OpenAIVideosV1, nil)
			ctx.TargetFormat, ctx.videosRoute = llmprotocol.OpenAIVideosV1, "video-route"
			response := (&OpenAIRouter{}).handleNonStreamingResponseBody([]byte(body), ctx, 0, false)
			rewritten := response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
			if !bytes.Contains(rewritten, []byte(`"id":"video-route:provider_1"`)) {
				t.Fatalf("rewritten object = %s", rewritten)
			}
		})
	}
}

func TestVideosRoutesUseBoundedTraceTemplates(t *testing.T) {
	for path, want := range map[string]string{
		"/v1/videos":                        "/v1/videos",
		"/v1/videos/video-route:id":         "/v1/videos/{video_id}",
		"/v1/videos/video-route:id/content": "/v1/videos/{video_id}/content",
	} {
		route, kind := requestTraceRoute(path)
		if route != want {
			t.Fatalf("trace route for %s = %q, want %q", path, route, want)
		}
		if path == "/v1/videos" && kind != "inference" || path != "/v1/videos" && kind != "video_object" {
			t.Fatalf("trace kind for %s = %q", path, kind)
		}
	}
}

func TestVideosObjectRequestBodyIsNotDecoded(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{}}
	if _, err := videosHeaderTestRouter().handleRequestHeaders(newRequestHeaders("GET", "/v1/videos/video-route:provider_1"), ctx); err != nil {
		t.Fatal(err)
	}
	if ctx.videosContent {
		t.Fatal("a status call was parsed as a content call")
	}
	response, err := (&OpenAIRouter{}).handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("a bodyless status call failed to decode: %v", err)
	}
	if response.GetImmediateResponse() != nil || ctx.SemanticRequest == nil || ctx.SemanticRequest.Model != "video-route" ||
		ctx.SourceFormat != llmprotocol.OpenAIVideosV1 || ctx.TargetFormat != llmprotocol.OpenAIVideosV1 {
		t.Fatalf("response = %+v, request = %+v", response.GetImmediateResponse(), ctx.SemanticRequest)
	}

	contentCtx := &RequestContext{Headers: map[string]string{}}
	if _, err := videosHeaderTestRouter().handleRequestHeaders(newRequestHeaders("GET", "/v1/videos/video-route:provider_1/content"), contentCtx); err != nil {
		t.Fatal(err)
	}
	if !contentCtx.videosContent || contentCtx.SourceFormat != llmprotocol.OpenAIVideosV1 {
		t.Fatalf("content call = %+v", contentCtx.SourceFormat)
	}
}

func TestVideosDecisionOnAChatClientIsRefused(t *testing.T) {
	router := videosHeaderTestRouter()
	request := &llmprotocol.Request{Generation: 1, Model: "video-route", Messages: []llmprotocol.Message{{Role: "user", Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hi"}}}}}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.VSRSelectedDecision = &config.Decision{Name: "videos", Output: &config.DecisionOutput{Kind: config.OutputKinds{config.OutputKindVideo}}}
	err := router.applyDecisionOutput(ctx)
	var protocolErr *llmprotocol.ProtocolError
	if !errors.As(err, &protocolErr) || protocolErr.Code != "unsupported_capability" {
		t.Fatalf("video decision on a chat client = %v", err)
	}
}

func responseBodyRequest(body []byte, end bool) *ext_proc.ProcessingRequest_ResponseBody {
	return &ext_proc.ProcessingRequest_ResponseBody{ResponseBody: &ext_proc.HttpBody{Body: body, EndOfStream: end}}
}
