package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestUpstreamContextErrorSurvivesResponsePipelineAndReplay(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, format := range extProcMatrixFormats {
			t.Run(string(format)+map[bool]string{false: "/buffered", true: "/stream"}[stream], func(t *testing.T) {
				recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
				r := &OpenAIRouter{ReplayRecorder: recorder}
				replayConfig := config.DefaultRouterReplayPluginConfig()
				replayConfig.Enabled = true
				replayConfig.CaptureResponseBody = true
				ctx := &RequestContext{RequestID: "context-request", RequestModel: "public", SourceFormat: format, TargetFormat: llmprotocol.OpenAIChatV1, SemanticRequest: testNeutralRequest("public", "long context"), RouterReplayPluginConfig: &replayConfig, ExpectStreamingResponse: stream}
				r.startRouterReplay(ctx, "public", "backend", "decision")
				header, err := r.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: "400"}, {Key: "content-type", Value: "application/json"}}}}}, ctx)
				if err != nil {
					t.Fatal(err)
				}
				if header.ModeOverride != nil || ctx.IsStreamingResponse || ctx.UpstreamStatusCode != 400 {
					t.Fatal("HTTP400 became an SSE success")
				}
				upstream := []byte(`{"error":{"message":"Maximum context length is 32768 tokens; reduce input or requested output.","type":"BadRequestError","param":"input_tokens","code":400}}`)
				response, err := r.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{ResponseBody: &ext_proc.HttpBody{Body: upstream, EndOfStream: true}}, ctx)
				if err != nil {
					t.Fatal(err)
				}
				common := response.GetResponseBody().GetResponse()
				body := common.GetBodyMutation().GetBody()
				translated, err := protocolcodec.NewBuiltinEngine().TranslateTransportError(format, format, body, nil)
				if err != nil || translated.TransportError.Error.Category != llmprotocol.ErrorInvalidRequest || translated.TransportError.Error.Message != "Maximum context length is 32768 tokens; reduce input or requested output." {
					t.Fatalf("context error lost: %s (%v)", body, err)
				}
				if headerValueForTest(common.GetHeaderMutation(), "content-type") != "application/json" {
					t.Fatal("not JSON error")
				}
				record, ok := recorder.GetRecord(ctx.RouterReplayID)
				if !ok || record.LifecycleState != routerreplay.LifecycleFailed || record.ResponseStatus != 400 || record.ResponseBody != string(body) {
					t.Fatalf("Replay does not match HTTP400: %+v", record)
				}
			})
		}
	}
}
