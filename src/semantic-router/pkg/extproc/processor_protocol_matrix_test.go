package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

var extProcMatrixFormats = []llmprotocol.WireFormat{
	llmprotocol.OpenAIChatV1,
	llmprotocol.OpenAIResponsesV1,
	llmprotocol.AnthropicMessagesV1,
}

// This test owns the ExtProc orientation contract: an upstream response is
// decoded with the backend format and encoded with the original client format.
// The codec package has its own exhaustive semantic tests; this matrix makes
// sure the data-plane seam never reverses those two axes.
func TestExtProcBufferedResponseProtocolMatrix(t *testing.T) {
	router := &OpenAIRouter{}
	engine := protocolcodec.NewBuiltinEngine()
	for _, clientFormat := range extProcMatrixFormats {
		for _, backendFormat := range extProcMatrixFormats {
			t.Run(string(clientFormat)+"_client_"+string(backendFormat)+"_backend", func(t *testing.T) {
				ctx := &RequestContext{SourceFormat: clientFormat, TargetFormat: backendFormat}
				semantic, err := router.decodeClientResponse(extProcResponseFixture(backendFormat), ctx)
				if err != nil {
					t.Fatalf("decodeClientResponse(): %v", err)
				}
				body, err := router.encodeClientResponse(*semantic, ctx)
				if err != nil {
					t.Fatalf("encodeClientResponse(): %v", err)
				}
				translated, err := engine.TranslateResponse(clientFormat, clientFormat, body, nil)
				if err != nil {
					t.Fatalf("client response is not valid %s: %v\n%s", clientFormat, err, body)
				}
				assertExtProcMatrixResponse(t, translated.Response, "source-model")
			})
		}
	}
}

// HTTP failures have a separate wire contract from failed model-generation
// resources. This matrix locks the response-header/body seam so every backend
// error envelope is rendered for every client without becoming a 2xx response.
func TestExtProcBufferedTransportErrorProtocolMatrix(t *testing.T) {
	router := &OpenAIRouter{}
	engine := protocolcodec.NewBuiltinEngine()
	forEachExtProcMatrixPair(t, func(t *testing.T, clientFormat, backendFormat llmprotocol.WireFormat) {
		ctx := &RequestContext{
			SourceFormat: clientFormat, TargetFormat: backendFormat,
			RequestID: "request_1", RequestModel: "public-model",
		}
		headerResponse, err := router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
				{Key: ":status", Value: "429"},
				{Key: "content-type", Value: "application/json"},
			}}},
		}, ctx)
		if err != nil {
			t.Fatalf("handleResponseHeaders(): %v", err)
		}
		if headerResponse.ModeOverride != nil || ctx.IsStreamingResponse {
			t.Fatal("HTTP error was incorrectly promoted to a semantic stream")
		}
		bodyResponse, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: extProcTransportErrorFixture(backendFormat), EndOfStream: true},
		}, ctx)
		if err != nil {
			t.Fatalf("handleResponseBody(): %v", err)
		}
		common := bodyResponse.GetResponseBody().GetResponse()
		if common.GetBodyMutation() == nil {
			t.Fatal("translated transport error did not replace the upstream body")
		}
		clientBody := common.GetBodyMutation().GetBody()
		translated, err := engine.TranslateTransportError(clientFormat, clientFormat, clientBody, nil)
		if err != nil {
			t.Fatalf("client transport error is not valid %s: %v\n%s", clientFormat, err, clientBody)
		}
		if translated.TransportError.Error == nil ||
			translated.TransportError.Error.Category != llmprotocol.ErrorRateLimited ||
			translated.TransportError.Error.Message != "slow down" {
			t.Fatalf("transport error semantics changed: %+v", translated.TransportError)
		}
		if got := headerValueForTest(common.GetHeaderMutation(), "content-type"); got != "application/json" {
			t.Fatalf("content-type mutation = %q, want application/json", got)
		}
		if !containsStringForTest(common.GetHeaderMutation().GetRemoveHeaders(), "content-length") {
			t.Fatalf("content-length was not removed: %#v", common.GetHeaderMutation())
		}
	})
}

func TestExtProcResponsesClientOwnsBufferedResponseID(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		ResponseObjectState: &ResponseObjectState{
			GeneratedResponseID: "resp_router_owned",
			PreviousResponseID:  "resp_previous",
		},
	}

	response, err := router.decodeClientResponse(extProcResponseFixture(llmprotocol.OpenAIChatV1), ctx)
	if err != nil {
		t.Fatal(err)
	}
	if response.ID != "resp_router_owned" {
		t.Fatalf("response ID = %q, want Router-owned ID", response.ID)
	}
	body, err := router.encodeClientResponse(*response, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(body, []byte(`"previous_response_id":"resp_previous"`)) {
		t.Fatalf("client response did not preserve request lineage: %s", body)
	}
}

// Structured-output requests are especially sensitive to request-body
// rewrites: dropping one schema keyword still leaves valid JSON but changes
// the contract observed by the backend. Exercise both request modes at the
// exact neutral decode/mutate/dispatch seam used by ExtProc.
func TestExtProcStructuredOutputRequestProtocolMatrix(t *testing.T) {
	router := &OpenAIRouter{}
	for _, streaming := range []bool{false, true} {
		forEachExtProcMatrixPair(t, func(t *testing.T, clientFormat, backendFormat llmprotocol.WireFormat) {
			assertExtProcStructuredOutputPair(t, router, clientFormat, backendFormat, streaming)
		})
	}
}

func assertExtProcStructuredOutputPair(
	t *testing.T,
	router *OpenAIRouter,
	clientFormat, backendFormat llmprotocol.WireFormat,
	streaming bool,
) {
	t.Helper()
	ctx := &RequestContext{
		SourceFormat: clientFormat,
		TargetFormat: backendFormat,
		RequestID:    "request_structured_output",
		TraceContext: t.Context(),
	}
	request, immediate := router.prepareProtocolRequest(extProcStructuredOutputRequestFixture(clientFormat, streaming), ctx)
	if immediate != nil || request == nil {
		t.Fatalf("structured request was rejected: request=%+v immediate=%+v", request, immediate)
	}
	request.Model = "routed-model"
	request.Generation++
	dispatch, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	decoded, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(backendFormat, dispatch)
	if err != nil {
		t.Fatalf("backend request does not satisfy %s: %v\n%s", backendFormat, err, dispatch)
	}
	assertExtProcStructuredOutputRequest(t, decoded, streaming)
}

// A same-format streaming Chat request that nothing mutated is byte-replay
// eligible: the neutral request and its source envelope share Generation 1, so
// EncodeRequest forwards the original client bytes verbatim. The Router forces
// stream_options.include_usage on dispatch so accounting can observe
// authoritative tokens, and that forcing must survive replay. This pins that
// the dispatch body carries the usage request even when the client omitted it.
func TestExtProcDispatchForcesIncludeUsageOnReplayEligibleStream(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		RequestID:    "request_replay_include_usage",
		TraceContext: t.Context(),
	}
	// No stream_options in the client body, and no mutation before dispatch, so
	// the request keeps decode-time Generation 1 and stays replay eligible.
	body := []byte(`{"model":"public-model","messages":[{"role":"user","content":"hello"}],"stream":true}`)
	request, immediate := router.prepareProtocolRequest(body, ctx)
	if immediate != nil || request == nil {
		t.Fatalf("streaming request was rejected: request=%+v immediate=%+v", request, immediate)
	}
	if request.StreamOptions.IncludeUsage != nil {
		t.Fatalf("client omitted stream_options but decode set include_usage: %+v", request.StreamOptions)
	}

	dispatch, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}

	decoded, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1, dispatch)
	if err != nil {
		t.Fatalf("dispatch request does not satisfy Chat Completions: %v\n%s", err, dispatch)
	}
	if decoded.StreamOptions.IncludeUsage == nil || !*decoded.StreamOptions.IncludeUsage {
		t.Fatalf("dispatch dropped the forced usage request on replay: %s", dispatch)
	}
	// The client preference is untouched, so client-facing rendering still omits
	// the usage chunk. Only the backend dispatch carries the forced flag.
	if ctx.SemanticRequest.StreamOptions.IncludeUsage != nil {
		t.Fatalf("forcing usage on dispatch mutated the retained client preference: %+v", ctx.SemanticRequest.StreamOptions)
	}
}

// When the client already asked for usage, the forcing is a no-op and byte
// replay must be preserved: re-encoding a large accepted body just to restate a
// flag it already carries would defeat the optimization. This guards that the
// dispatch body is byte-identical to the original client body in that case.
func TestExtProcDispatchPreservesReplayWhenClientRequestsUsage(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		RequestID:    "request_replay_client_usage",
		TraceContext: t.Context(),
	}
	body := []byte(`{"model":"public-model","messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":true}}`)
	request, immediate := router.prepareProtocolRequest(body, ctx)
	if immediate != nil || request == nil {
		t.Fatalf("streaming request was rejected: request=%+v immediate=%+v", request, immediate)
	}
	if request.StreamOptions.IncludeUsage == nil || !*request.StreamOptions.IncludeUsage {
		t.Fatalf("client requested usage but decode did not record it: %+v", request.StreamOptions)
	}

	dispatch, err := router.encodeDispatchRequest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(dispatch, body) {
		t.Fatalf("byte replay was not preserved for a client that already requested usage:\n got=%s\nwant=%s", dispatch, body)
	}
}

// Streaming translation is stateful, so it needs an independent 3x3 matrix at
// the ExtProc seam rather than relying on buffered coverage or codec-only tests.
func TestExtProcStreamingResponseProtocolMatrix(t *testing.T) {
	router := &OpenAIRouter{}
	engine := protocolcodec.NewBuiltinEngine()
	forEachExtProcMatrixPair(t, func(t *testing.T, clientFormat, backendFormat llmprotocol.WireFormat) {
		assertExtProcStreamingPair(t, router, engine, clientFormat, backendFormat)
	})
}

func TestExtProcResponsesClientOwnsStreamingResponseID(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		RequestModel: "public-model",
		TraceContext: t.Context(),
		ResponseObjectState: &ResponseObjectState{
			GeneratedResponseID: "resp_router_owned",
			PreviousResponseID:  "resp_previous",
		},
	}
	if err := router.ensureSemanticResponseStream(ctx); err != nil {
		t.Fatal(err)
	}
	clientWire := pushExtProcStreamFixture(t, ctx, extProcStreamFixture(llmprotocol.OpenAIChatV1))
	semantic, err := ctx.SemanticStreamState.response()
	if err != nil {
		t.Fatal(err)
	}
	if semantic.ID != "resp_router_owned" {
		t.Fatalf("semantic response ID = %q, want Router-owned ID", semantic.ID)
	}
	responseID := decodeExtProcClientStreamResponseID(t, llmprotocol.OpenAIResponsesV1, clientWire.Bytes())
	if responseID != "resp_router_owned" {
		t.Fatalf("client response ID = %q, want Router-owned ID", responseID)
	}
	if !bytes.Contains(clientWire.Bytes(), []byte(`"previous_response_id":"resp_previous"`)) {
		t.Fatalf("client stream did not preserve request lineage: %s", clientWire.Bytes())
	}
}

func TestSameFormatChatAccountsBackendUsageWithoutPublishingIt(t *testing.T) {
	includeUsage := false
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		RequestModel: "public-model",
		TraceContext: t.Context(),
		SemanticRequest: &llmprotocol.Request{
			Generation:    1,
			Model:         "public-model",
			Stream:        true,
			StreamOptions: llmprotocol.StreamOptions{IncludeUsage: &includeUsage},
		},
	}
	response := router.handleSemanticStreamingResponseBody(
		extProcStreamFixture(llmprotocol.OpenAIChatV1), true, ctx,
	)
	mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		t.Fatal("same-format Chat usage filtering did not produce a public body mutation")
	}
	publicBody := mutation.GetBody()
	if bytes.Contains(publicBody, []byte(`"usage"`)) {
		t.Fatalf("backend accounting leaked into a client that did not request usage: %s", publicBody)
	}
	if !bytes.Contains(publicBody, []byte(`"content":"hello"`)) || !bytes.Contains(publicBody, []byte("data: [DONE]")) {
		t.Fatalf("usage filtering changed public content or terminal framing: %s", publicBody)
	}
	if ctx.SemanticResponse == nil || ctx.SemanticResponse.Usage.State != llmprotocol.UsageAvailable ||
		ctx.SemanticResponse.Usage.Total.Value == nil || *ctx.SemanticResponse.Usage.Total.Value != 3 {
		t.Fatalf("backend usage was not retained for Router accounting: %+v", ctx.SemanticResponse)
	}
}

func TestSameFormatChatPublishesCodecFailureWhenTransparentUsageFilteringFails(t *testing.T) {
	includeUsage := false
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		RequestModel: "public-model",
		TraceContext: t.Context(),
		SemanticRequest: &llmprotocol.Request{
			Generation: 1, Model: "public-model", Stream: true,
			StreamOptions: llmprotocol.StreamOptions{IncludeUsage: &includeUsage},
		},
	}
	payload := append(
		append([]byte(nil), extProcStreamFixture(llmprotocol.OpenAIChatV1)...),
		[]byte("data: {\n\n")...,
	)
	response := router.handleSemanticStreamingResponseBody(payload, true, ctx)
	mutation := response.GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		t.Fatal("malformed same-format stream did not produce a public failure body")
	}
	publicBody := mutation.GetBody()
	if bytes.Contains(publicBody, []byte("data: [DONE]")) {
		t.Fatalf("malformed same-format stream published a success terminal: %s", publicBody)
	}
	if !bytes.Contains(publicBody, []byte(`"error"`)) || !ctx.StreamingAborted {
		t.Fatalf("malformed same-format stream did not fail closed: aborted=%t body=%s", ctx.StreamingAborted, publicBody)
	}
}

func TestSameFormatChatSuppressesDeferredSuccessWhenTrailingFragmentFailsAtEOF(t *testing.T) {
	includeUsage := false
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1,
		TargetFormat: llmprotocol.OpenAIChatV1,
		RequestModel: "public-model",
		TraceContext: t.Context(),
		SemanticRequest: &llmprotocol.Request{
			Generation: 1, Model: "public-model", Stream: true,
			StreamOptions: llmprotocol.StreamOptions{IncludeUsage: &includeUsage},
		},
	}

	prefix := router.handleSemanticStreamingResponseBody(
		extProcStreamFixture(llmprotocol.OpenAIChatV1), false, ctx,
	)
	prefixMutation := prefix.GetResponseBody().GetResponse().GetBodyMutation()
	if prefixMutation == nil {
		t.Fatal("same-format stream prefix did not produce a body mutation")
	}
	if bytes.Contains(prefixMutation.GetBody(), []byte("data: [DONE]")) {
		t.Fatalf("success terminal escaped before HTTP EOS: %s", prefixMutation.GetBody())
	}

	terminal := router.handleSemanticStreamingResponseBody([]byte("data: {"), true, ctx)
	terminalMutation := terminal.GetResponseBody().GetResponse().GetBodyMutation()
	if terminalMutation == nil {
		t.Fatal("trailing fragment did not produce a public failure body")
	}
	publicBody := terminalMutation.GetBody()
	if bytes.Contains(publicBody, []byte("data: [DONE]")) {
		t.Fatalf("trailing fragment published the deferred success terminal: %s", publicBody)
	}
	if !bytes.Contains(publicBody, []byte(`"error"`)) || !ctx.StreamingAborted {
		t.Fatalf("trailing fragment did not fail closed: aborted=%t body=%s", ctx.StreamingAborted, publicBody)
	}
	if ctx.SemanticResponse != nil {
		t.Fatalf("failed stream was reconstructed as a successful semantic response: %+v", ctx.SemanticResponse)
	}
}

func forEachExtProcMatrixPair(
	t *testing.T,
	assertPair func(*testing.T, llmprotocol.WireFormat, llmprotocol.WireFormat),
) {
	t.Helper()
	for _, clientFormat := range extProcMatrixFormats {
		for _, backendFormat := range extProcMatrixFormats {
			t.Run(string(clientFormat)+"_client_"+string(backendFormat)+"_backend", func(t *testing.T) {
				assertPair(t, clientFormat, backendFormat)
			})
		}
	}
}

func assertExtProcStreamingPair(
	t *testing.T,
	router *OpenAIRouter,
	engine *protocolcodec.Engine,
	clientFormat,
	backendFormat llmprotocol.WireFormat,
) {
	t.Helper()
	ctx := &RequestContext{
		SourceFormat: clientFormat, TargetFormat: backendFormat,
		RequestModel: "public-model", TraceContext: context.Background(),
	}
	if err := router.ensureSemanticResponseStream(ctx); err != nil {
		t.Fatalf("ensureSemanticResponseStream(): %v", err)
	}
	clientWire := pushExtProcStreamFixture(t, ctx, extProcStreamFixture(backendFormat))
	semantic, err := ctx.SemanticStreamState.response()
	if err != nil {
		t.Fatalf("reconstruct response: %v", err)
	}
	assertExtProcMatrixResponse(t, *semantic, "public-model")
	assertClientStreamDecodes(t, engine, clientFormat, clientWire.Bytes())
}

func pushExtProcStreamFixture(t *testing.T, ctx *RequestContext, payload []byte) bytes.Buffer {
	t.Helper()
	var clientWire bytes.Buffer
	for offset := 0; offset < len(payload); {
		end := offset + 1 + offset%19
		if end > len(payload) {
			end = len(payload)
		}
		frames, events, diagnostics, err := ctx.ProtocolResponseStream.Push(payload[offset:end])
		if err != nil {
			t.Fatalf("Push(%d:%d): %v", offset, end, err)
		}
		observeExtProcStream(ctx, frames, events, diagnostics, &clientWire)
		offset = end
	}
	frames, events, diagnostics, err := ctx.ProtocolResponseStream.Finalize(nil)
	if err != nil {
		t.Fatalf("Finalize(): %v", err)
	}
	observeExtProcStream(ctx, frames, events, diagnostics, &clientWire)
	return clientWire
}

func observeExtProcStream(
	ctx *RequestContext,
	frames [][]byte,
	events []llmprotocol.Event,
	diagnostics llmprotocol.Diagnostics,
	clientWire *bytes.Buffer,
) {
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, diagnostics...)
	ctx.SemanticStreamState.observe(events)
	for _, frame := range frames {
		clientWire.Write(frame)
	}
}

func assertClientStreamDecodes(
	t *testing.T,
	engine *protocolcodec.Engine,
	format llmprotocol.WireFormat,
	body []byte,
) {
	t.Helper()
	stream, err := engine.NewStream(format, format, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	_, events, _, err := stream.Push(body)
	if err != nil {
		t.Fatalf("translated stream is not valid %s: %v\n%s", format, err, body)
	}
	_, terminal, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("translated stream finalize: %v", err)
	}
	events = append(events, terminal...)
	if len(events) == 0 || events[len(events)-1].Type != llmprotocol.EventResponseCompleted {
		t.Fatalf("translated stream has no terminal event: %+v", events)
	}
}

func decodeExtProcClientStreamResponseID(
	t *testing.T,
	format llmprotocol.WireFormat,
	body []byte,
) string {
	t.Helper()
	engine := protocolcodec.NewBuiltinEngine()
	stream, err := engine.NewStream(format, format, llmprotocol.StreamContext{
		Context: t.Context(), PublicModel: "public-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	_, events, _, err := stream.Push(body)
	if err != nil {
		t.Fatal(err)
	}
	_, terminal, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	events = append(events, terminal...)
	for _, event := range events {
		if event.ResponseID != "" {
			return event.ResponseID
		}
	}
	return ""
}

func assertExtProcMatrixResponse(t *testing.T, response llmprotocol.Response, expectedModel string) {
	t.Helper()
	if response.ID != "response_1" || response.Model != expectedModel ||
		len(response.Output) != 1 || len(response.Output[0].Content) != 1 ||
		response.Output[0].Content[0].Kind != llmprotocol.ContentText ||
		response.Output[0].Content[0].Text != "hello" {
		t.Fatalf("response semantics changed: %+v", response)
	}
	if response.Usage.State != llmprotocol.UsageAvailable ||
		response.Usage.InputTotal.Value == nil || *response.Usage.InputTotal.Value != 2 ||
		response.Usage.OutputTotal.Value == nil || *response.Usage.OutputTotal.Value != 1 {
		t.Fatalf("authoritative usage changed: %+v", response.Usage)
	}
}

func extProcResponseFixture(format llmprotocol.WireFormat) []byte {
	switch format {
	case llmprotocol.OpenAIChatV1:
		return []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"id":"output_1","role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	case llmprotocol.OpenAIResponsesV1:
		return []byte(`{"id":"response_1","model":"source-model","status":"completed","output":[{"type":"message","id":"output_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`{"id":"response_1","type":"message","role":"assistant","model":"source-model","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}`)
	default:
		panic(fmt.Sprintf("unsupported response fixture format %q", format))
	}
}

func extProcStructuredOutputRequestFixture(format llmprotocol.WireFormat, streaming bool) []byte {
	stream := "false"
	if streaming {
		stream = "true"
	}
	schema := `{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`
	switch format {
	case llmprotocol.OpenAIChatV1:
		return []byte(`{"model":"client-model","messages":[{"role":"user","content":"answer"}],"reasoning_effort":"high","stream":` + stream + `,"response_format":{"type":"json_schema","json_schema":{"name":"structured_output","strict":true,"schema":` + schema + `}}}`)
	case llmprotocol.OpenAIResponsesV1:
		return []byte(`{"model":"client-model","input":"answer","reasoning":{"effort":"high"},"stream":` + stream + `,"text":{"format":{"type":"json_schema","name":"structured_output","strict":true,"schema":` + schema + `}}}`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`{"model":"client-model","max_tokens":64,"messages":[{"role":"user","content":"answer"}],"stream":` + stream + `,"output_config":{"effort":"high","format":{"type":"json_schema","schema":` + schema + `}}}`)
	default:
		panic(fmt.Sprintf("unsupported request fixture format %q", format))
	}
}

func assertExtProcStructuredOutputRequest(t *testing.T, request llmprotocol.Request, streaming bool) {
	t.Helper()
	if !extProcStructuredRequestMetadataMatches(request, streaming) {
		t.Fatalf("structured request semantics changed: %+v", request)
	}
	assertExtProcStructuredSchema(t, request.OutputFormat.Schema)
}

func extProcStructuredRequestMetadataMatches(request llmprotocol.Request, streaming bool) bool {
	return request.Model == "routed-model" && request.Stream == streaming && request.ReasoningEffort == "high" &&
		request.OutputFormat.Kind == llmprotocol.OutputJSONSchema && request.OutputFormat.Name == "structured_output" &&
		request.OutputFormat.Strict != nil && *request.OutputFormat.Strict
}

func assertExtProcStructuredSchema(t *testing.T, schema json.RawMessage) {
	t.Helper()
	want := []byte(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`)
	var gotValue, wantValue any
	if err := json.Unmarshal(schema, &gotValue); err != nil {
		t.Fatalf("decoded schema is invalid: %v", err)
	}
	if err := json.Unmarshal(want, &wantValue); err != nil {
		t.Fatal(err)
	}
	gotJSON, err := json.Marshal(gotValue)
	if err != nil {
		t.Fatal(err)
	}
	wantJSON, err := json.Marshal(wantValue)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotJSON, wantJSON) {
		t.Fatalf("structured schema changed: got=%s want=%s", gotJSON, wantJSON)
	}
}

func extProcTransportErrorFixture(format llmprotocol.WireFormat) []byte {
	switch format {
	case llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1:
		return []byte(`{"error":{"message":"slow down","type":"rate_limit_error","param":null,"code":"rate_limit_exceeded"}}`)
	case llmprotocol.AnthropicMessagesV1:
		return []byte(`{"type":"error","request_id":"request_provider","error":{"type":"rate_limit_error","message":"slow down"}}`)
	default:
		panic(fmt.Sprintf("unsupported error fixture format %q", format))
	}
}

func extProcStreamFixture(format llmprotocol.WireFormat) []byte {
	join := func(events ...string) []byte { return []byte(strings.Join(events, "")) }
	switch format {
	case llmprotocol.OpenAIChatV1:
		return join(
			"data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n",
			"data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n",
			"data: [DONE]\n\n",
		)
	case llmprotocol.OpenAIResponsesV1:
		return join(
			"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"response_1\",\"object\":\"response\",\"created_at\":100,\"model\":\"source-model\",\"status\":\"in_progress\",\"output\":[]}}\n\n",
			"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"output_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
			"event: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"sequence_number\":2,\"output_index\":0,\"item_id\":\"output_1\",\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[],\"logprobs\":[]}}\n\n",
			"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":3,\"output_index\":0,\"content_index\":0,\"item_id\":\"output_1\",\"delta\":\"hello\",\"logprobs\":[]}\n\n",
			"event: response.output_text.done\ndata: {\"type\":\"response.output_text.done\",\"sequence_number\":4,\"output_index\":0,\"content_index\":0,\"item_id\":\"output_1\",\"text\":\"hello\",\"logprobs\":[]}\n\n",
			"event: response.content_part.done\ndata: {\"type\":\"response.content_part.done\",\"sequence_number\":5,\"output_index\":0,\"item_id\":\"output_1\",\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"hello\",\"annotations\":[],\"logprobs\":[]}}\n\n",
			"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":6,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"output_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\",\"annotations\":[],\"logprobs\":[]}]}}\n\n",
			"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":7,\"response\":{\"id\":\"response_1\",\"object\":\"response\",\"created_at\":100,\"model\":\"source-model\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"id\":\"output_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\",\"annotations\":[],\"logprobs\":[]}]}],\"usage\":{\"input_tokens\":2,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens\":1,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":3}}}\n\n",
		)
	case llmprotocol.AnthropicMessagesV1:
		return join(
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"response_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"source-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\n",
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"type\":\"message_delta\",\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n",
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
		)
	default:
		panic(fmt.Sprintf("unsupported stream fixture format %q", format))
	}
}
