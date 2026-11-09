package extproc

import (
	"bytes"
	"compress/gzip"
	"context"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A Responses client that asked for a stream can be routed to a provider that
// answers with one JSON object. The client asked for SSE, so it keeps SSE: the
// buffered answer arrives as the terminal event it already describes.
func TestBufferedResponsesAnswerReachesAStreamingClient(t *testing.T) {
	body := `{"id":"resp_1","object":"response","created_at":1,"model":"public-model",` +
		`"status":"completed","output":[{"type":"message","id":"msg_1","status":"completed",` +
		`"role":"assistant","content":[{"type":"output_text","text":"hello","annotations":[]}]}],` +
		`"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}`
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIResponsesV1,
		RequestModel: "public-model", TraceContext: context.Background(),
		SemanticRequest: &llmprotocol.Request{Generation: 1, Model: "public-model", Stream: true},
	}
	headers, err := router.handleResponseHeaders(looperTransportHeaders(200, "application/json"), ctx)
	if err != nil {
		t.Fatal(err)
	}
	if ctx.IsStreamingResponse != true {
		t.Fatal("stream:true Responses request must keep streaming response mode")
	}
	if mode := headers.GetModeOverride().GetResponseBodyMode(); mode == http_ext.ProcessingMode_STREAMED {
		t.Fatalf("a JSON provider body must be read whole, got %v", mode)
	}
	if got := ctx.UpstreamBodyStreamed; got {
		t.Fatal("an application/json provider body is not a stream")
	}
	response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatal(err)
	}
	client := string(response.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
	if !strings.Contains(client, "event: response.completed") {
		t.Fatalf("client body is not the terminal SSE event: %q", client)
	}
	if !strings.Contains(client, `"text":"hello"`) {
		t.Fatalf("the buffered answer was dropped: %q", client)
	}
	if strings.Contains(client, "stream_incomplete") {
		t.Fatalf("a complete upstream answer must not be reported as incomplete: %q", client)
	}
}

// A buffered provider reply is one body message, and its end_of_stream is false
// when trailers follow it. The object being complete ends the read, not the flag:
// without that, the answer reaches the SSE decoder and the client gets
// stream_incomplete instead of the terminal event.
func TestBufferedResponsesAnswerArrivesBeforeTrailers(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIResponsesV1,
		RequestModel: "public-model", TraceContext: context.Background(),
		SemanticRequest: &llmprotocol.Request{Generation: 1, Model: "public-model", Stream: true},
	}
	if _, err := router.handleResponseHeaders(looperTransportHeaders(200, "application/json"), ctx); err != nil {
		t.Fatal(err)
	}
	response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte(bufferedResponsesBody()), EndOfStream: false},
	}, ctx)
	if err != nil {
		t.Fatal(err)
	}
	client := string(response.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
	if !strings.Contains(client, "event: response.completed") || !strings.Contains(client, `"text":"hello"`) {
		t.Fatalf("the buffered answer was not served before the trailers: %q", client)
	}
	if strings.Contains(client, "stream_incomplete") {
		t.Fatalf("a complete upstream answer must not be reported as incomplete: %q", client)
	}
}

// A gzip provider body reaches the bridge compressed, because a streaming
// response leaves content coding to the client. It is decoded to recognize the
// object and encoded again under the header the client already has.
func TestBufferedResponsesAnswerDecodesGzipBeforeBridging(t *testing.T) {
	var compressed bytes.Buffer
	writer := gzip.NewWriter(&compressed)
	if _, err := writer.Write([]byte(bufferedResponsesBody())); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIResponsesV1,
		RequestModel: "public-model", TraceContext: context.Background(),
		SemanticRequest: &llmprotocol.Request{Generation: 1, Model: "public-model", Stream: true},
	}
	headers := looperTransportHeaders(200, "application/json")
	headers.ResponseHeaders.Headers.Headers = append(headers.ResponseHeaders.Headers.Headers,
		&core.HeaderValue{Key: "content-encoding", Value: "gzip"})
	if _, err := router.handleResponseHeaders(headers, ctx); err != nil {
		t.Fatal(err)
	}
	response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: compressed.Bytes(), EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatal(err)
	}
	sent := response.GetResponseBody().GetResponse().GetBodyMutation().GetBody()
	reader, err := gzip.NewReader(bytes.NewReader(sent))
	if err != nil {
		t.Fatalf("the client body no longer matches its content-encoding: %v", err)
	}
	var decoded bytes.Buffer
	if _, err := decoded.ReadFrom(reader); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(decoded.String(), "event: response.completed") || !strings.Contains(decoded.String(), `"text":"hello"`) {
		t.Fatalf("the gzip answer was not served as a stream: %q", decoded.String())
	}
}

func bufferedResponsesBody() string {
	return `{"id":"resp_1","object":"response","created_at":1,"model":"public-model",` +
		`"status":"completed","output":[{"type":"message","id":"msg_1","status":"completed",` +
		`"role":"assistant","content":[{"type":"output_text","text":"hello","annotations":[]}]}],` +
		`"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}`
}

// A provider that does stream SSE can still have its bytes split anywhere,
// including right at the nested response object of a terminal event. That read
// is valid JSON with object "response", so only the request's own knowledge
// that the body is SSE keeps the bridge away from it.
func TestBridgeDoesNotRewriteNestedObjectOfAStreamingBody(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIResponsesV1,
		RequestModel: "public-model", TraceContext: context.Background(),
		SemanticRequest:      &llmprotocol.Request{Generation: 1, Model: "public-model", Stream: true},
		IsStreamingResponse:  true,
		UpstreamBodyStreamed: true,
	}
	nested := `{"id":"resp_1","object":"response","status":"completed","output":[]}`
	if bridged, ok := router.bridgeBufferedResponsesBody([]byte(nested), ctx); ok {
		t.Fatalf("bridged a nested object out of a real SSE body: %q", bridged)
	}
}

func TestBridgeBufferedResponsesBodyLeavesOtherBodiesAlone(t *testing.T) {
	router := &OpenAIRouter{}
	streaming := &RequestContext{
		SourceFormat:    llmprotocol.OpenAIResponsesV1,
		SemanticRequest: &llmprotocol.Request{Stream: true},
	}
	chat := &RequestContext{
		SourceFormat:    llmprotocol.OpenAIChatV1,
		SemanticRequest: &llmprotocol.Request{Stream: true},
	}
	sse := "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\"}}\n\n"
	cases := []struct {
		name string
		ctx  *RequestContext
		body string
		want string
	}{
		{"sse passes through", streaming, sse, sse},
		{"chat object is not a responses object", chat, `{"object":"response"}`, `{"object":"response"}`},
		{"another json object", streaming, `{"object":"chat.completion"}`, `{"object":"chat.completion"}`},
		{
			"a failed status stays on the error path", streaming,
			`{"object":"response","status":"failed"}`, `{"object":"response","status":"failed"}`,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			bridged, ok := router.bridgeBufferedResponsesBody([]byte(c.body), c.ctx)
			if ok {
				t.Fatalf("body was bridged: %q", bridged)
			}
		})
	}
}
