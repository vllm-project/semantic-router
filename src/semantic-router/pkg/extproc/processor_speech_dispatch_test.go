package extproc

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func speechTestRequest(t *testing.T, body string) *llmprotocol.Request {
	t.Helper()
	request, _, _, err := protocolcodec.SpeechCodec{}.DecodeRequest([]byte(body), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("decode speech request %s: %v", body, err)
	}
	return &request
}

func speechBackendConfig() config.ModelParams {
	return config.ModelParams{
		PreferredEndpoints: []string{"backend"},
		APIFormat:          config.APIFormatSpeech,
		ExternalModelIDs:   map[string]string{"vllm": "provider-speech"},
	}
}

func assertUnsupportedCapability(t *testing.T, err error) {
	t.Helper()
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) {
		t.Fatalf("error = %v (%T), want an unsupported_capability protocol error", err, err)
	}
	if protocolError.Code != "unsupported_capability" {
		t.Fatalf("protocol error code = %q, want unsupported_capability", protocolError.Code)
	}
}

func TestWireFormatForSpeechAPIFormat(t *testing.T) {
	for _, apiFormat := range []string{config.APIFormatSpeech, "openai.speech", string(llmprotocol.OpenAISpeechV1)} {
		format, err := wireFormatForModel(apiFormat)
		if err != nil {
			t.Fatalf("wireFormatForModel(%q): %v", apiFormat, err)
		}
		if format != llmprotocol.OpenAISpeechV1 {
			t.Fatalf("wireFormatForModel(%q) = %s, want %s", apiFormat, format, llmprotocol.OpenAISpeechV1)
		}
	}
}

func TestRequestWirePathForSpeechFormat(t *testing.T) {
	if path := requestWirePath(llmprotocol.OpenAISpeechV1); path != "/v1/audio/speech" {
		t.Fatalf("wire path = %q, want /v1/audio/speech", path)
	}
}

// The OpenAI provider profile resolves a chat create path; a speech dispatch
// must still reach the Speech API path, with every client field unchanged
// except the model.
func TestSpecifiedModelForwardsSpeechRequestWithOnlyTheModelSwapped(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAISpeechV1)
	body := `{"model":"` + model + `","input":"Namaste duniya","voice":{"id":"anushka"},` +
		`"ref_audio":["a.wav"],"ref_text":"reference","extra_params":{"prompt_id":"p1"},"response_format":"wav"}`
	request := speechTestRequest(t, body)
	ctx := routingTestContext(llmprotocol.OpenAISpeechV1, request)

	response, err := router.handleSpecifiedModelRouting(request, model, "", ctx)
	if err != nil {
		t.Fatalf("handleSpecifiedModelRouting: %v", err)
	}
	common := response.GetRequestBody().GetResponse()
	if got := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())[":path"]; got != "/v1/audio/speech" {
		t.Fatalf("backend path = %q, want /v1/audio/speech", got)
	}
	var got, want map[string]any
	if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &got); err != nil {
		t.Fatalf("decode backend body: %v", err)
	}
	if err := json.Unmarshal([]byte(body), &want); err != nil {
		t.Fatalf("decode client body: %v", err)
	}
	want["model"] = "provider-model"
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("backend body = %s, want the client body with only the model swapped", common.GetBodyMutation().GetBody())
	}
	if ctx.TargetFormat != llmprotocol.OpenAISpeechV1 {
		t.Fatalf("ctx.TargetFormat = %s, want %s", ctx.TargetFormat, llmprotocol.OpenAISpeechV1)
	}
}

// A decision that lists a chat model first and a speech model second sends a
// speech request to the speech model, because only its wire can express
// speech_generation.
func TestCapabilitySelectionChoosesSpeechWireSibling(t *testing.T) {
	router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	speechBackend := "speech-backend"
	router.Config.ModelConfig[speechBackend] = speechBackendConfig()
	decision := &config.Decision{
		Name:      "Voice",
		ModelRefs: []config.ModelRef{{Model: primary}, {Model: speechBackend}},
	}
	request := speechTestRequest(t, `{"model":"auto","input":"hello there","voice":"alloy"}`)
	ctx := routingTestContext(llmprotocol.OpenAISpeechV1, request)
	ctx.VSRSelectedDecision = decision

	dispatch, err := selectCapabilityTestDispatch(router, request, decision, ctx)
	if err != nil {
		t.Fatalf("expected selection of the speech sibling, got error: %v", err)
	}
	if dispatch.logicalModel != speechBackend || dispatch.targetFormat != llmprotocol.OpenAISpeechV1 {
		t.Fatalf("dispatch = %s on %s, want %s on %s", dispatch.logicalModel, dispatch.targetFormat, speechBackend, llmprotocol.OpenAISpeechV1)
	}
	if request.Model != "provider-speech" {
		t.Fatalf("request.Model = %s, want provider-speech", request.Model)
	}
}

func TestPrepareProviderDispatchRejectsSpeechOnChatOnlyDecision(t *testing.T) {
	router, primary := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	decision := &config.Decision{Name: "Chat", ModelRefs: []config.ModelRef{{Model: primary}}}
	request := speechTestRequest(t, `{"model":"auto","input":"hello there"}`)
	ctx := routingTestContext(llmprotocol.OpenAISpeechV1, request)
	ctx.VSRSelectedDecision = decision

	_, err := router.prepareProviderDispatch(request, primary, decision.Name, false, ctx)
	assertUnsupportedCapability(t, err)
}

// A chat request carries no speech options, so a speech-only backend must
// refuse it instead of speaking the chat prompt.
func TestSpecifiedModelRejectsChatRequestForSpeechBackend(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAISpeechV1)
	request := testNeutralRequest(model, "hello there")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)

	response, err := router.handleSpecifiedModelRouting(request, model, "", ctx)
	if err == nil && response.GetImmediateResponse() == nil {
		t.Fatalf("a chat request was dispatched to a speech backend: %+v", response)
	}
}

func speechResponseTestContext(t *testing.T, recorder *routerreplay.Recorder, router *OpenAIRouter, model string) *RequestContext {
	t.Helper()
	replayConfig := config.DefaultRouterReplayPluginConfig()
	replayConfig.Enabled = true
	replayConfig.CaptureResponseBody = true
	ctx := &RequestContext{
		RequestID: "speech-request", RequestModel: model,
		SourceFormat: llmprotocol.OpenAISpeechV1, TargetFormat: llmprotocol.OpenAISpeechV1,
		SemanticRequest:          speechTestRequest(t, `{"input":"hello there","voice":"alloy"}`),
		RouterReplayPluginConfig: &replayConfig,
	}
	router.startRouterReplay(ctx, model, "backend", "decision")
	return ctx
}

func speechResponseHeaders(status, contentType string) *ext_proc.ProcessingRequest_ResponseHeaders {
	return &ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{
		Headers: []*core.HeaderValue{{Key: ":status", Value: status}, {Key: "content-type", Value: contentType}},
	}}}
}

// Audio bypasses the neutral response: the router forwards the backend's bytes
// and content type unchanged, and the request still completes with zero token
// usage, releasing its in-flight slot and closing its replay record without
// storing the audio.
func TestSpeechAudioResponseIsForwardedUnchangedAndCompletesTheRequest(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	router := &OpenAIRouter{Config: &config.RouterConfig{}, ReplayRecorder: recorder}
	model := "speech-audio-response-test"
	ctx := speechResponseTestContext(t, recorder, router, model)
	ctx.InflightToken = inflight.Begin(model)

	header, err := router.handleResponseHeaders(speechResponseHeaders("200", "audio/wav"), ctx)
	if err != nil {
		t.Fatalf("handleResponseHeaders: %v", err)
	}
	if header.ModeOverride != nil || ctx.IsStreamingResponse {
		t.Fatalf("a WAV response must stay a buffered response")
	}
	for _, removed := range header.GetResponseHeaders().GetResponse().GetHeaderMutation().GetRemoveHeaders() {
		if removed == "content-length" {
			t.Fatalf("the audio is forwarded unchanged, so its content-length must be kept")
		}
	}

	audio := []byte("RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\xbb\x00\x00")
	response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: audio, EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("handleResponseBody: %v", err)
	}
	common := response.GetResponseBody().GetResponse()
	if common == nil || common.GetStatus() != ext_proc.CommonResponse_CONTINUE {
		t.Fatalf("audio response must continue to the client, got %+v", response)
	}
	if common.GetBodyMutation() != nil {
		t.Fatalf("audio bytes must not be rewritten: %+v", common.GetBodyMutation())
	}
	if got := headerValueForTest(common.GetHeaderMutation(), "content-type"); got != "" {
		t.Fatalf("content-type was overwritten with %q", got)
	}
	if got := inflight.Get(model); got != 0 {
		t.Fatalf("inflight(%s) = %d, want 0 once the audio response completes", model, got)
	}
	record, ok := recorder.GetRecord(ctx.RouterReplayID)
	if !ok || record.LifecycleState != routerreplay.LifecycleCompleted {
		t.Fatalf("replay record must complete, got %+v", record)
	}
	if record.ResponseBody != "" {
		t.Fatalf("replay must not store audio bytes, got %d bytes", len(record.ResponseBody))
	}
}

// vLLM-Omni reports errors with a numeric code; the client still receives the
// backend's message in the OpenAI error envelope.
func TestSpeechBackendErrorKeepsTheBackendMessage(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	router := &OpenAIRouter{Config: &config.RouterConfig{}, ReplayRecorder: recorder}
	ctx := speechResponseTestContext(t, recorder, router, "speech-error-test")

	if _, err := router.handleResponseHeaders(speechResponseHeaders("400", "application/json"), ctx); err != nil {
		t.Fatalf("handleResponseHeaders: %v", err)
	}
	upstream := []byte(`{"error":{"message":"voice 'bob' is not supported","type":"BadRequestError","param":null,"code":400}}`)
	response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: upstream, EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("handleResponseBody: %v", err)
	}
	body := string(response.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
	if !strings.Contains(body, `"message":"voice 'bob' is not supported"`) {
		t.Fatalf("client error body = %s, want the backend message", body)
	}
	record, ok := recorder.GetRecord(ctx.RouterReplayID)
	if !ok || record.LifecycleState != routerreplay.LifecycleFailed {
		t.Fatalf("replay record must record the failed request, got %+v", record)
	}
}

func TestSemanticCacheIsDisabledForSpeechRequests(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		SemanticCache: config.SemanticCache{Enabled: true},
	}}
	chat := &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1}
	if !router.semanticCacheEnabledForRequest(chat) {
		t.Fatal("test setup: the cache must be enabled for chat requests")
	}
	speech := &RequestContext{SourceFormat: llmprotocol.OpenAISpeechV1}
	if router.semanticCacheEnabledForRequest(speech) {
		t.Fatal("speech requests must neither read nor write the response cache")
	}
}
