package extproc

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// listenerStream is an ext_proc stream from an Envoy listener that names
// itself as listener in x-vsr-listener, or names no listener when it is
// empty. The first request is the request headers.
func listenerStream(listener string, requests ...*ext_proc.ProcessingRequest) *MockStream {
	if listener != "" {
		headerMap := requests[0].GetRequestHeaders().GetHeaders()
		headerMap.Headers = append(headerMap.Headers, &core.HeaderValue{Key: headers.VSRListener, Value: listener})
	}
	return NewMockStream(requests)
}

func listenerHeaders(method, target string, pairs ...string) *ext_proc.ProcessingRequest {
	values := []*core.HeaderValue{
		{Key: ":method", Value: method},
		{Key: ":path", Value: target},
		{Key: "content-type", Value: "application/json"},
	}
	for i := 0; i+1 < len(pairs); i += 2 {
		values = append(values, &core.HeaderValue{Key: pairs[i], Value: pairs[i+1]})
	}
	return &ext_proc.ProcessingRequest{Request: &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: values}},
	}}
}

func listenerBody(body string) *ext_proc.ProcessingRequest {
	return &ext_proc.ProcessingRequest{Request: &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true},
	}}
}

// messagesOnlyRouter routes to a backend of format behind a listener named
// claude that serves only messages, and one named open that names no apis.
func messagesOnlyRouter(format llmprotocol.WireFormat, apis config.ListenerAPIs) (*OpenAIRouter, string) {
	router, model := routingTestRouterForFormat(format)
	router.Config.Listeners = []config.Listener{
		{Name: "claude", Port: 18899, APIs: apis},
		{Name: "open", Port: 18898},
	}
	return router, model
}

func notServedDetail(t *testing.T, response *ext_proc.ProcessingResponse) string {
	t.Helper()
	immediate := response.GetImmediateResponse()
	if immediate == nil || immediate.GetStatus().GetCode() != 404 {
		t.Fatalf("want an immediate 404, got %v", response)
	}
	if got := immediateHeaderValue(response, "content-type"); got != "application/json" {
		t.Fatalf("content-type = %q", got)
	}
	var body struct{ Detail string }
	if err := json.Unmarshal(immediate.GetBody(), &body); err != nil {
		t.Fatalf("body %q: %v", immediate.GetBody(), err)
	}
	return body.Detail
}

// The grant reads the path the client sent, so the provider path the body
// phase sets cannot refuse a request the listener serves.
func TestEngineListenerServesItsAPIWhateverTheProviderPath(t *testing.T) {
	router, model := messagesOnlyRouter(llmprotocol.OpenAIChatV1, config.ListenerAPIs{"messages": {}})
	stream := listenerStream("claude",
		listenerHeaders("POST", "/v1/messages"),
		listenerBody(`{"model":"`+model+`","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`),
	)
	if err := router.Process(stream); err != nil {
		t.Fatal(err)
	}
	if len(stream.Responses) != 2 || stream.Responses[0].GetRequestHeaders() == nil {
		t.Fatalf("the header phase did not continue: %v", stream.Responses)
	}
	removed := stream.Responses[0].GetRequestHeaders().GetResponse().GetHeaderMutation().GetRemoveHeaders()
	if !slices.Contains(removed, headers.VSRListener) {
		t.Fatalf("x-vsr-listener reaches the provider; removed %v", removed)
	}
	var providerPath string
	for _, header := range stream.Responses[1].GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders() {
		if header.GetHeader().GetKey() == ":path" {
			providerPath = string(header.GetHeader().GetRawValue())
		}
	}
	if providerPath != "/v1/chat/completions" {
		t.Fatalf("provider path = %q, want the Chat Completions path; body reply %v", providerPath, stream.Responses[1])
	}
}

// A path the listener does not serve is refused at the header phase, before
// the body phase could rewrite it to a path the listener serves.
func TestEngineListenerRefusesAnAPIItDoesNotServe(t *testing.T) {
	router, model := messagesOnlyRouter(llmprotocol.AnthropicMessagesV1, config.ListenerAPIs{"messages": {}})
	stream := listenerStream("claude",
		listenerHeaders("POST", "/v1/chat/completions"),
		listenerBody(`{"model":"`+model+`","messages":[{"role":"user","content":"hi"}]}`),
	)
	if err := router.Process(stream); err != nil {
		t.Fatal(err)
	}
	if len(stream.Responses) == 0 {
		t.Fatal("no reply")
	}
	detail := notServedDetail(t, stream.Responses[0])
	if detail != `listener "claude" does not serve /v1/chat/completions; it serves messages` {
		t.Fatalf("detail = %q", detail)
	}
}

func TestEngineListenerGrantCases(t *testing.T) {
	for _, tt := range []struct {
		name     string
		listener string
		apis     config.ListenerAPIs
		method   string
		target   string
		pairs    []string
		refused  bool
	}{
		{
			name: "skip-processing does not pass the grant", listener: "claude", method: "POST", target: "/v1/chat/completions",
			pairs: []string{headers.VSRSkipProcessing, "true"}, refused: true,
		},
		{name: "a path of no API", listener: "claude", method: "POST", target: "/v2/anything", refused: true},
		{name: "the query is not the path", listener: "claude", method: "POST", target: "/v1/messages?beta=true"},
		{name: "every listener lists models", listener: "claude", method: "GET", target: "/v1/models"},
		{name: "a listener with no apis serves the default APIs", listener: "open", method: "POST", target: "/v1/chat/completions"},
		{name: "a gateway that names no listener gets no grant", method: "POST", target: "/v1/chat/completions"},
		{
			name: "an API under the listener's prefix", listener: "claude", method: "POST", target: "/anthropic/v1/messages",
			apis: config.ListenerAPIs{"messages": {Prefix: "/anthropic"}},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			apis := tt.apis
			if apis == nil {
				apis = config.ListenerAPIs{"messages": {}}
			}
			router, _ := messagesOnlyRouter(llmprotocol.OpenAIChatV1, apis)
			router.Config.SkipProcessing.Enabled = true
			stream := listenerStream(tt.listener, listenerHeaders(tt.method, tt.target, tt.pairs...))
			if err := router.Process(stream); err != nil {
				t.Fatal(err)
			}
			if len(stream.Responses) == 0 {
				t.Fatal("no reply")
			}
			response := stream.Responses[0]
			if tt.refused {
				if detail := notServedDetail(t, response); !strings.HasPrefix(detail, `listener "claude" does not serve`) {
					t.Fatalf("detail = %q", detail)
				}
				return
			}
			if status := response.GetImmediateResponse().GetStatus().GetCode(); status == 404 {
				t.Fatalf("refused: %s", response.GetImmediateResponse().GetBody())
			}
		})
	}
}

// Envoy names a listener the Router's configuration no longer has after a
// reload removed it: the request is refused rather than served ungated.
func TestEngineListenerUnknownToTheRouterIsRefused(t *testing.T) {
	router, _ := messagesOnlyRouter(llmprotocol.OpenAIChatV1, config.ListenerAPIs{"messages": {}})
	stream := listenerStream("renamed", listenerHeaders("POST", "/v1/messages"))
	if err := router.Process(stream); err != nil {
		t.Fatal(err)
	}
	if len(stream.Responses) == 0 || stream.Responses[0].GetImmediateResponse().GetStatus().GetCode() != 503 {
		t.Fatalf("want 503, got %v", stream.Responses)
	}
}

// The standalone gateway applies its own listener before an in-process
// session runs, so a client's x-vsr-listener there names nothing.
func TestOnlyAnEnvoyStreamReadsTheListenerHeader(t *testing.T) {
	router, _ := messagesOnlyRouter(llmprotocol.OpenAIChatV1, config.ListenerAPIs{"messages": {}})
	request := listenerHeaders("POST", "/v1/chat/completions", headers.VSRListener, "claude").GetRequestHeaders()
	inProcess := &RequestContext{Headers: map[string]string{}}
	response, err := router.handleRequestHeaders(&ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: request}, inProcess)
	if err != nil {
		t.Fatal(err)
	}
	if inProcess.GatewayListener != "" || response.GetImmediateResponse() != nil {
		t.Fatalf("an in-process request read the client's listener %q: %v", inProcess.GatewayListener, response)
	}
	if _, kept := inProcess.Headers[headers.VSRListener]; kept {
		t.Fatal("the Router kept the client's x-vsr-listener in its view")
	}
}
