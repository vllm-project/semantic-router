package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

const (
	umbrellaSchema = `{"type":"object","properties":{"umbrella":{"type":"boolean"},"reason":{"type":"string"}},"required":["umbrella","reason"],"additionalProperties":false}`
	// umbrellaAdvice is the answer held to the schema; madeUpAdvice is the one
	// a model server gives when the schema leaves the model no tool call.
	umbrellaAdvice = `{"umbrella":false,"reason":"It is sunny in Paris."}`
	madeUpAdvice   = `{"umbrella":true,"reason":"It is probably raining in Paris."}`
	weatherProse   = "It is sunny in Paris, so you need no umbrella."
)

// umbrellaBackend answers as vLLM does: an output format holds the whole
// completion, so a request with a format and callable tools gets made-up
// JSON instead of a tool call. Without a format it calls get_weather until
// the conversation holds the tool's result, then answers in prose. Answers
// without a format report 20 input and 5 output tokens, answers with one 21
// and 7. calls are the requests the Router sent it itself.
type umbrellaBackend struct {
	t      *testing.T
	format llmprotocol.WireFormat
	// status, when set, answers the Router's own calls with an error.
	status int
	mu     sync.Mutex
	calls  []llmprotocol.Request
}

func newUmbrellaBackend(t *testing.T, format llmprotocol.WireFormat) (*umbrellaBackend, *httptest.Server) {
	t.Helper()
	backend := &umbrellaBackend{t: t, format: format}
	server := httptest.NewServer(backend)
	t.Cleanup(server.Close)
	return backend, server
}

func (b *umbrellaBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		b.t.Errorf("read the Router's call: %v", err)
	}
	request := b.decode(body)
	b.mu.Lock()
	b.calls = append(b.calls, request)
	b.mu.Unlock()
	if b.status != 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(b.status)
		_, _ = w.Write(extProcTransportErrorFixture(b.format))
		return
	}
	reply, contentType := b.reply(request)
	w.Header().Set("Content-Type", contentType)
	_, _ = w.Write(reply)
}

func (b *umbrellaBackend) ownCalls() []llmprotocol.Request {
	b.mu.Lock()
	defer b.mu.Unlock()
	calls := b.calls
	b.calls = nil
	return calls
}

func (b *umbrellaBackend) decode(body []byte) llmprotocol.Request {
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(b.format, body)
	if err != nil {
		b.t.Errorf("the backend cannot read %s: %v", body, err)
	}
	return request
}

// reply is the backend's answer to request in its own format, streamed when
// the request asks for a stream.
func (b *umbrellaBackend) reply(request llmprotocol.Request) ([]byte, string) {
	answer := umbrellaAnswer(request)
	engine := protocolcodec.NewBuiltinEngine()
	if request.Stream {
		frames, _, err := engine.EncodeResponseStream(b.format, answer, llmprotocol.StreamContext{
			Context: context.Background(), Source: b.format, Target: b.format, Options: request.StreamOptions,
			PublicModel: answer.Model, ResponseID: answer.ID,
		})
		if err != nil {
			b.t.Errorf("encode the backend's stream: %v", err)
		}
		return frames, "text/event-stream"
	}
	encoded, err := engine.EncodeResponse(b.format, answer, llmprotocol.Envelope{})
	if err != nil {
		b.t.Errorf("encode the backend's answer: %v", err)
	}
	return encoded.Body, "application/json"
}

func umbrellaAnswer(request llmprotocol.Request) llmprotocol.Response {
	formatted := request.OutputFormat.Kind == llmprotocol.OutputJSONSchema || request.OutputFormat.Kind == llmprotocol.OutputJSONObject
	content := llmprotocol.Content{Kind: llmprotocol.ContentText, Text: weatherProse}
	stop := llmprotocol.StopEndTurn
	switch {
	case llmprotocol.StructuredOutputWithTools(request):
		content.Text = madeUpAdvice
	case formatted:
		content.Text = umbrellaAdvice
	case len(request.Tools) > 0 && request.ToolChoice.Mode != llmprotocol.ToolChoiceNone && !holdsToolResult(request):
		content = llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
			ID: "call_weather", Name: "get_weather", Arguments: `{"city":"Paris"}`,
		}}
		stop = llmprotocol.StopToolCall
	}
	input, output := int64(20), int64(5)
	if formatted {
		input, output = 21, 7
	}
	return llmprotocol.Response{
		Generation: 1, ID: "resp_umbrella", CreatedAt: time.Unix(1700000000, 0).UTC(), Model: request.Model,
		Output: []llmprotocol.OutputItem{{
			ID: "item_umbrella", Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{content},
		}},
		StopReason: stop, Usage: umbrellaUsage(input, output),
	}
}

func umbrellaUsage(input, output int64) llmprotocol.Usage {
	count := func(value int64) llmprotocol.TokenCount {
		return llmprotocol.TokenCount{Value: &value, Provenance: llmprotocol.UsageAuthoritative}
	}
	return llmprotocol.Usage{
		State: llmprotocol.UsageAvailable, InputUncached: count(input), InputTotal: count(input),
		OutputOther: count(output), OutputTotal: count(output), Total: count(input + output),
	}
}

func holdsToolResult(request llmprotocol.Request) bool {
	for _, message := range request.Messages {
		for _, content := range message.Content {
			if content.Kind == llmprotocol.ContentToolResult {
				return true
			}
		}
	}
	return false
}

// umbrellaRouter routes format's model to server.
func umbrellaRouter(t *testing.T, format llmprotocol.WireFormat, server *httptest.Server) (*OpenAIRouter, string) {
	t.Helper()
	router, model := routingTestRouterForFormat(format)
	parsed, err := url.Parse(server.URL)
	require.NoError(t, err)
	host, port, err := net.SplitHostPort(parsed.Host)
	require.NoError(t, err)
	cfg := router.Config
	cfg.VLLMEndpoints[0].Address = host
	cfg.VLLMEndpoints[0].Port, err = strconv.Atoi(port)
	require.NoError(t, err)
	profile := cfg.ProviderProfiles["provider"]
	profile.BaseURL = strings.Replace(profile.BaseURL, "127.0.0.1:8000", parsed.Host, 1)
	cfg.ProviderProfiles["provider"] = profile
	router.CredentialResolver = newTestCredentialResolver(cfg)
	router.Cache = cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: false})
	return router, model
}

// umbrellaRequest is format's request for umbrella advice held to the
// schema, with the get_weather tool, before or after the tool answered.
func umbrellaRequest(format llmprotocol.WireFormat, model string, answered, stream bool) map[string]any {
	question := "Do I need an umbrella in Paris today?"
	parameters := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)
	schema := json.RawMessage(umbrellaSchema)
	request := map[string]any{"model": model}
	if stream {
		request["stream"] = true
	}
	switch format {
	case llmprotocol.OpenAIChatV1:
		messages := []any{map[string]any{"role": "user", "content": question}}
		if answered {
			messages = append(messages,
				map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{map[string]any{
					"id": "call_weather", "type": "function",
					"function": map[string]any{"name": "get_weather", "arguments": `{"city":"Paris"}`},
				}}},
				map[string]any{"role": "tool", "tool_call_id": "call_weather", "content": "Sunny, 24 C"})
		}
		request["messages"] = messages
		request["tools"] = []any{map[string]any{"type": "function", "function": map[string]any{
			"name": "get_weather", "description": "Current weather for a city", "parameters": parameters,
		}}}
		request["response_format"] = map[string]any{"type": "json_schema", "json_schema": map[string]any{
			"name": "umbrella_advice", "strict": true, "schema": schema,
		}}
		if stream {
			request["stream_options"] = map[string]any{"include_usage": true}
		}
	case llmprotocol.OpenAIResponsesV1:
		input := []any{map[string]any{"role": "user", "content": question}}
		if answered {
			input = append(input,
				map[string]any{"type": "function_call", "call_id": "call_weather", "name": "get_weather", "arguments": `{"city":"Paris"}`},
				map[string]any{"type": "function_call_output", "call_id": "call_weather", "output": "Sunny, 24 C"})
		}
		request["input"] = input
		request["tools"] = []any{map[string]any{
			"type": "function", "name": "get_weather", "description": "Current weather for a city",
			"parameters": parameters, "strict": false,
		}}
		request["text"] = map[string]any{"format": map[string]any{
			"type": "json_schema", "name": "umbrella_advice", "strict": true, "schema": schema,
		}}
	case llmprotocol.AnthropicMessagesV1:
		messages := []any{map[string]any{"role": "user", "content": question}}
		if answered {
			messages = append(messages,
				map[string]any{"role": "assistant", "content": []any{map[string]any{
					"type": "tool_use", "id": "call_weather", "name": "get_weather", "input": map[string]any{"city": "Paris"},
				}}},
				map[string]any{"role": "user", "content": []any{map[string]any{
					"type": "tool_result", "tool_use_id": "call_weather", "content": "Sunny, 24 C",
				}}})
		}
		request["max_tokens"] = 256
		request["messages"] = messages
		request["tools"] = []any{map[string]any{
			"name": "get_weather", "description": "Current weather for a city", "input_schema": parameters,
		}}
		request["output_config"] = map[string]any{"format": map[string]any{"type": "json_schema", "schema": schema}}
	}
	return request
}

func umbrellaBody(t *testing.T, request map[string]any) []byte {
	t.Helper()
	body, err := json.Marshal(request)
	require.NoError(t, err)
	return body
}

// umbrellaTurn runs body through router's request phase, plays the gateway
// for the dispatched call with backend's answer, and returns the dispatched
// request, what the client received, the replies to the answer's headers
// and body, and the request's context. live drops the pass after dispatch,
// so the answer reaches the client as it would without one.
func umbrellaTurn(
	t *testing.T, router *OpenAIRouter, backend *umbrellaBackend, client llmprotocol.WireFormat, body []byte, live bool,
) (llmprotocol.Request, string, []*ext_proc.ProcessingResponse, *RequestContext) {
	t.Helper()
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: "umbrella-turn", SourceFormat: client,
		TraceContext: t.Context(), StartTime: time.Now(),
	}
	response, err := router.HandleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: body},
	}, ctx)
	require.NoError(t, err)
	if immediate := router.encodeImmediateResponseForClient(response, ctx).GetImmediateResponse(); immediate != nil {
		t.Fatalf("request was refused: %s", immediate.GetBody())
	}
	if live {
		ctx.structuredOutputPass = nil
	}
	dispatched := backend.decode(response.GetRequestBody().GetResponse().GetBodyMutation().GetBody())
	reply, contentType := backend.reply(dispatched)
	headersReply := answerHeaders(t, router, ctx, contentType)
	answer, bodyReplies := answerBody(t, router, ctx, [][]byte{reply})
	return dispatched, answer, append([]*ext_proc.ProcessingResponse{headersReply}, bodyReplies...), ctx
}

// clientAnswer reads what a client of format received.
func clientAnswer(t *testing.T, format llmprotocol.WireFormat, body string, stream bool) llmprotocol.Response {
	t.Helper()
	engine := protocolcodec.NewBuiltinEngine()
	if !stream {
		translated, err := engine.TranslateResponse(format, format, []byte(body), nil)
		require.NoError(t, err, body)
		return translated.Response
	}
	decoder, err := engine.NewStream(format, format, llmprotocol.StreamContext{Context: t.Context(), PublicModel: "client-model"})
	require.NoError(t, err)
	_, events, _, err := decoder.Push([]byte(body))
	require.NoError(t, err, body)
	_, terminal, _, err := decoder.Finalize(nil)
	require.NoError(t, err, body)
	state := &semanticResponseStreamState{
		usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable}, items: map[int]*semanticStreamItem{},
	}
	state.observe(append(events, terminal...))
	response, err := state.response()
	require.NoError(t, err, body)
	return *response
}

func answerText(response llmprotocol.Response) string {
	var text strings.Builder
	for _, item := range response.Output {
		for _, content := range item.Content {
			if content.Kind == llmprotocol.ContentText {
				text.WriteString(content.Text)
			}
		}
	}
	return text.String()
}

func answerToolCall(response llmprotocol.Response) *llmprotocol.ToolCall {
	for _, item := range response.Output {
		for _, content := range item.Content {
			if content.Kind == llmprotocol.ContentToolCall {
				return content.ToolCall
			}
		}
	}
	return nil
}

// requireToolTurn checks that the backend received the turn without its
// format, its tool still callable.
func requireToolTurn(t *testing.T, dispatched llmprotocol.Request) {
	t.Helper()
	require.Empty(t, dispatched.OutputFormat.Schema, "the first call carries no schema")
	require.NotContains(t, []llmprotocol.OutputFormatKind{llmprotocol.OutputJSONSchema, llmprotocol.OutputJSONObject}, dispatched.OutputFormat.Kind)
	require.Len(t, dispatched.Tools, 1)
	require.NotEqual(t, llmprotocol.ToolChoiceNone, dispatched.ToolChoice.Mode)
}

func requireTokens(t *testing.T, usage llmprotocol.Usage, input, output int64) {
	t.Helper()
	require.Equal(t, llmprotocol.UsageAvailable, usage.State)
	require.NotNil(t, usage.InputTotal.Value)
	require.NotNil(t, usage.OutputTotal.Value)
	require.Equal(t, input, *usage.InputTotal.Value, "input tokens")
	require.Equal(t, output, *usage.OutputTotal.Value, "output tokens")
}

// Every client format on every backend format: the model calls the tool on
// the turn that carries tools and a schema, and its final answer is held to
// the schema by a second call, whose answer the client receives with both
// calls' usage.
func TestStructuredOutputPassLetsTheModelCallToolsAndHoldsTheFinalAnswerToTheSchema(t *testing.T) {
	for _, client := range extProcMatrixFormats {
		for _, backendFormat := range extProcMatrixFormats {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s_client_%s_backend_stream_%v", client, backendFormat, stream), func(t *testing.T) {
					backend, server := newUmbrellaBackend(t, backendFormat)
					router, model := umbrellaRouter(t, backendFormat, server)

					question := umbrellaBody(t, umbrellaRequest(client, model, false, stream))
					dispatched, answer, replies, _ := umbrellaTurn(t, router, backend, client, question, false)
					requireToolTurn(t, dispatched)
					require.Empty(t, backend.ownCalls(), "a tool call is the turn's answer as it is")
					if stream {
						require.Equal(t, http_ext.ProcessingMode_BUFFERED, replies[0].GetModeOverride().GetResponseBodyMode(), "the stream is held")
						_, live, _, _ := umbrellaTurn(t, router, backend, client, question, true)
						require.Equal(t, mintedResponseID.ReplaceAllString(live, "resp_"), mintedResponseID.ReplaceAllString(answer, "resp_"),
							"the held stream is released as it would have streamed")
						require.Contains(t, answer, "get_weather")
					} else {
						call := answerToolCall(clientAnswer(t, client, answer, false))
						require.NotNil(t, call, answer)
						require.Equal(t, "get_weather", call.Name)
						require.JSONEq(t, `{"city":"Paris"}`, call.Arguments)
					}

					dispatched, answer, _, ctx := umbrellaTurn(t, router, backend, client, umbrellaBody(t, umbrellaRequest(client, model, true, stream)), false)
					requireToolTurn(t, dispatched)
					calls := backend.ownCalls()
					require.Len(t, calls, 1, "a final answer is asked for again with the schema")
					constrained := calls[0]
					require.Equal(t, llmprotocol.OutputJSONSchema, constrained.OutputFormat.Kind)
					require.JSONEq(t, umbrellaSchema, string(constrained.OutputFormat.Schema))
					require.False(t, constrained.Stream)
					if backendFormat == llmprotocol.OpenAIChatV1 {
						require.Empty(t, constrained.Tools, "a Chat backend gets a tool_choice none request without its tools")
					} else {
						require.Len(t, constrained.Tools, 1)
						require.Equal(t, llmprotocol.ToolChoiceNone, constrained.ToolChoice.Mode)
					}
					require.Len(t, constrained.Messages, len(dispatched.Messages), "the same turn")

					got := clientAnswer(t, client, answer, stream)
					require.JSONEq(t, umbrellaAdvice, answerText(got), answer)
					require.NotContains(t, answer, "sunny in Paris, so", "the first answer is not sent")
					require.Nil(t, answerToolCall(got))
					requireTokens(t, got.Usage, 41, 12)
					if backendFormat == llmprotocol.AnthropicMessagesV1 && client != llmprotocol.AnthropicMessagesV1 {
						require.True(t, hasDiagnosticField(ctx.ProtocolDiagnostics, "output_format.name"),
							"the schema's loss toward the backend is still reported: %+v", ctx.ProtocolDiagnostics)
					}
				})
			}
		}
	}
}

// mintedResponseID is the ID a Responses stream of another backend format
// mints for itself, a new one for every stream.
var mintedResponseID = regexp.MustCompile(`resp_[0-9a-f]{24}`)

func hasDiagnosticField(diagnostics llmprotocol.Diagnostics, field string) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == field {
			return true
		}
	}
	return false
}

// A request whose tools the model cannot call, or that has none, keeps its
// one call with the schema.
func TestStructuredOutputPassKeepsOneCallWithoutCallableTools(t *testing.T) {
	for name, edit := range map[string]func(map[string]any){
		"tool_choice none": func(request map[string]any) { request["tool_choice"] = "none" },
		"no tools":         func(request map[string]any) { delete(request, "tools") },
	} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s_stream_%v", name, stream), func(t *testing.T) {
				backend, server := newUmbrellaBackend(t, llmprotocol.OpenAIChatV1)
				router, model := umbrellaRouter(t, llmprotocol.OpenAIChatV1, server)
				request := umbrellaRequest(llmprotocol.OpenAIChatV1, model, false, stream)
				edit(request)

				dispatched, answer, replies, ctx := umbrellaTurn(t, router, backend, llmprotocol.OpenAIChatV1, umbrellaBody(t, request), false)
				require.Equal(t, llmprotocol.OutputJSONSchema, dispatched.OutputFormat.Kind)
				require.JSONEq(t, umbrellaSchema, string(dispatched.OutputFormat.Schema))
				require.Empty(t, backend.ownCalls())
				require.Nil(t, ctx.structuredOutputPass)
				if stream {
					require.Nil(t, ctx.ResponseGuardHold, "nothing holds the stream")
					require.Equal(t, http_ext.ProcessingMode_STREAMED, replies[0].GetModeOverride().GetResponseBodyMode())
				}
				got := clientAnswer(t, llmprotocol.OpenAIChatV1, answer, stream)
				require.JSONEq(t, umbrellaAdvice, answerText(got))
				requireTokens(t, got.Usage, 21, 7)
			})
		}
	}
}

// A second call that fails answers the client with the backend's error, not
// with the first answer, which the schema does not hold.
func TestStructuredOutputPassAnswersTheSecondCallsFailure(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream_%v", stream), func(t *testing.T) {
			backend, server := newUmbrellaBackend(t, llmprotocol.OpenAIChatV1)
			backend.status = http.StatusServiceUnavailable
			router, model := umbrellaRouter(t, llmprotocol.OpenAIChatV1, server)

			_, answer, replies, ctx := umbrellaTurn(t, router, backend, llmprotocol.OpenAIChatV1,
				umbrellaBody(t, umbrellaRequest(llmprotocol.OpenAIChatV1, model, true, stream)), false)
			require.Len(t, backend.ownCalls(), 1)
			require.NotContains(t, answer, weatherProse)
			require.Contains(t, answer, "slow down", "the backend's error reaches the client")
			if stream {
				require.Contains(t, answer, `"error"`)
			} else {
				immediate := replies[1].GetImmediateResponse()
				require.NotNil(t, immediate, "a buffered request is answered with the error")
				require.EqualValues(t, http.StatusServiceUnavailable, immediate.GetStatus().GetCode())
			}
			requireTokens(t, ctx.SemanticResponse.Usage, 20, 5)
		})
	}
}

// The replay record keeps the final answer with both calls' usage, and an
// outcome for the second call.
func TestStructuredOutputPassRecordsTheSecondCallInRouterReplay(t *testing.T) {
	backend, server := newUmbrellaBackend(t, llmprotocol.OpenAIChatV1)
	router, model := umbrellaRouter(t, llmprotocol.OpenAIChatV1, server)
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	router.ReplayRecorder = recorder
	replayConfig := config.DefaultRouterReplayPluginConfig()
	replayConfig.Enabled = true
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: "umbrella-replay", SourceFormat: llmprotocol.OpenAIChatV1,
		TraceContext: t.Context(), StartTime: time.Now(), RouterReplayPluginConfig: &replayConfig,
	}
	response, err := router.HandleRequestBody(&ext_proc.ProcessingRequest_RequestBody{RequestBody: &ext_proc.HttpBody{
		Body: umbrellaBody(t, umbrellaRequest(llmprotocol.OpenAIChatV1, model, true, false)),
	}}, ctx)
	require.NoError(t, err)
	require.NotEmpty(t, ctx.RouterReplayID)
	reply, contentType := backend.reply(backend.decode(response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()))
	answerHeaders(t, router, ctx, contentType)
	answer, _ := answerBody(t, router, ctx, [][]byte{reply})
	require.JSONEq(t, umbrellaAdvice, answerText(clientAnswer(t, llmprotocol.OpenAIChatV1, answer, false)))

	record, found := recorder.GetRecord(ctx.RouterReplayID)
	require.True(t, found)
	require.NotNil(t, record.PromptTokens)
	require.NotNil(t, record.CompletionTokens)
	require.Equal(t, 41, *record.PromptTokens)
	require.Equal(t, 12, *record.CompletionTokens)
	if record.ResponseBody != "" {
		require.NotContains(t, record.ResponseBody, weatherProse, "the record keeps the answer the client received")
	}
	pass := outcomeFor(t, record.Outcomes, "structured_output", ctx.VSRSelectedModel)
	require.Equal(t, "replaced", pass.Verdict)
	require.Equal(t, "5", pass.Metadata["first_output_tokens"])
	require.Equal(t, "7", pass.Metadata["second_output_tokens"])
}

// A native fallback candidate is sent the request as the client sent it,
// format included, so its answer takes no second call.
func TestNativeFallbackCandidateKeepsTheFormatAndTakesNoSecondCall(t *testing.T) {
	session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
	ctx := session.ctx
	ctx.SemanticRequest.Tools = []llmprotocol.Tool{{Name: "get_weather", InputSchema: []byte(`{"type":"object"}`)}}
	ctx.SemanticRequest.OutputFormat = llmprotocol.OutputFormat{
		Kind: llmprotocol.OutputJSONSchema, Name: "umbrella_advice", Schema: []byte(umbrellaSchema),
	}
	ctx.structuredOutputPass = &structuredOutputPass{dispatch: &providerDispatch{backendAddress: "127.0.0.1:8001"}}

	step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
	require.NoError(t, err)
	require.NotNil(t, step.Call)
	require.Contains(t, string(step.Call.Request.Body), `"response_format"`)
	require.NotNil(t, ctx.structuredOutputPass, "the primary's pass stays until a candidate succeeds")

	_, err = session.ResponseHeaders(routing.Header{{Name: ":status", Value: "200"}, {Name: "content-type", Value: "application/json"}}, false)
	require.NoError(t, err)
	require.Equal(t, "model-fallback-1", ctx.VSRSelectedModel)
	require.Nil(t, ctx.structuredOutputPass)
}

func TestPlanStructuredOutputPassNeedsABackendTheRouterCanCallForOneAnswer(t *testing.T) {
	request := llmprotocol.Request{
		Tools:        []llmprotocol.Tool{{Name: "get_weather", InputSchema: []byte(`{"type":"object"}`)}},
		OutputFormat: llmprotocol.OutputFormat{Kind: llmprotocol.OutputJSONObject},
	}
	candidates := request
	candidates.CandidateCount = llmprotocol.Int64(2)
	dispatch := &providerDispatch{backendAddress: "127.0.0.1:8000"}
	for name, test := range map[string]struct {
		ctx      *RequestContext
		dispatch *providerDispatch
		want     bool
	}{
		"a primary dispatch":          {&RequestContext{SemanticRequest: &request}, dispatch, true},
		"a Looper hop":                {&RequestContext{SemanticRequest: &request, LooperRequest: true}, dispatch, false},
		"an external gateway's model": {&RequestContext{SemanticRequest: &request}, &providerDispatch{}, false},
		"several candidates":          {&RequestContext{SemanticRequest: &candidates}, dispatch, false},
		"no request":                  {&RequestContext{}, dispatch, false},
	} {
		if got := planStructuredOutputPass(test.dispatch, test.ctx) != nil; got != test.want {
			t.Errorf("%s: planned = %v, want %v", name, got, test.want)
		}
	}
}

func TestStructuredOutputTimeoutIsTheShortestBoundTheDecisionAndModelSet(t *testing.T) {
	cfg := &config.RouterConfig{BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
		"umbrella-model": {Reliability: config.ProviderReliability{TotalTimeout: "90s", PerTryTimeout: "120s"}},
	}}}
	router := &OpenAIRouter{Config: cfg}
	dispatch := &providerDispatch{logicalModel: "umbrella-model", backendModel: "umbrella-model"}
	require.Equal(t, 90*time.Second, router.structuredOutputTimeout(&RequestContext{}, dispatch))
	decision := &config.Decision{Reliability: &config.DecisionReliability{PerTryTimeout: "45s"}}
	require.Equal(t, 45*time.Second, router.structuredOutputTimeout(&RequestContext{VSRSelectedDecision: decision}, dispatch))
	unbounded := &OpenAIRouter{Config: &config.RouterConfig{}}
	require.Equal(t, structuredOutputCallTimeout, unbounded.structuredOutputTimeout(&RequestContext{}, dispatch))
}

// A format the backend cannot carry is refused before any call, as it is for
// the same request without tools, although the first call would carry none.
func TestStructuredOutputPassRefusesAFormatTheBackendCannotCarryBeforeEitherCall(t *testing.T) {
	refusal := func(t *testing.T, request map[string]any) error {
		t.Helper()
		backend, server := newUmbrellaBackend(t, llmprotocol.AnthropicMessagesV1)
		router, _ := umbrellaRouter(t, llmprotocol.AnthropicMessagesV1, server)
		request["response_format"] = map[string]any{"type": "json_object"}
		ctx := &RequestContext{Headers: map[string]string{}, RequestID: "umbrella-refused", SourceFormat: llmprotocol.OpenAIChatV1, TraceContext: t.Context()}
		_, err := router.HandleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
			RequestBody: &ext_proc.HttpBody{Body: umbrellaBody(t, request)},
		}, ctx)
		require.Empty(t, backend.ownCalls())
		return err
	}
	_, model := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
	withTools := refusal(t, umbrellaRequest(llmprotocol.OpenAIChatV1, model, false, false))
	withoutTools := umbrellaRequest(llmprotocol.OpenAIChatV1, model, false, false)
	delete(withoutTools, "tools")
	require.Error(t, withTools)
	require.EqualError(t, withTools, refusal(t, withoutTools).Error())
}

// umbrellaModesConfig routes umbrella questions to a Chat model served at
// endpoint.
func umbrellaModesConfig(endpoint string) string {
	return fmt.Sprintf(`
version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 300s
providers:
  defaults:
    model: umbrella-model
  models:
    - name: umbrella-model
      provider_model_id: umbrella-model-instruct
      api_format: openai
      backend_refs:
        - name: umbrella-primary
          endpoint: %s
          protocol: http
          provider: vllm
recipes:
  - name: default
    signals:
      keywords:
        - name: weather_route
          operator: OR
          keywords: ["umbrella"]
    decisions:
      - name: weather
        priority: 10
        rules:
          operator: AND
          conditions:
            - type: keyword
              name: weather_route
        modelRefs:
          - model: umbrella-model
            use_reasoning: false
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
`, endpoint)
}

// Both gateway modes send the turn without the schema, return a tool call
// as it is, and answer a final answer with the second call's.
func TestBothGatewayModesHoldTheFinalAnswerToTheSchema(t *testing.T) {
	backend, server := newUmbrellaBackend(t, llmprotocol.OpenAIChatV1)
	parsed, err := url.Parse(server.URL)
	require.NoError(t, err)
	cfg, err := config.ParseYAMLBytes([]byte(umbrellaModesConfig(parsed.Host)))
	require.NoError(t, err)
	for mode, engineFor := range looperAdapters {
		t.Run(mode, func(t *testing.T) {
			engine := engineFor(newParityRouter(t, cfg))
			for _, stream := range []bool{false, true} {
				for _, answered := range []bool{false, true} {
					out := serveUmbrellaTurn(t, engine, backend, umbrellaRequest(llmprotocol.OpenAIChatV1, "vllm-sr/auto", answered, stream))
					require.Equal(t, http.StatusOK, out.Status)
					got := clientAnswer(t, llmprotocol.OpenAIChatV1, drainResponse(t, out), stream)
					calls := backend.ownCalls()
					if !answered {
						require.Empty(t, calls)
						require.NotNil(t, answerToolCall(got))
						continue
					}
					require.Len(t, calls, 1)
					require.Equal(t, llmprotocol.OutputJSONSchema, calls[0].OutputFormat.Kind)
					require.Empty(t, calls[0].Tools)
					require.JSONEq(t, umbrellaAdvice, answerText(got))
					requireTokens(t, got.Usage, 41, 12)
				}
			}
		})
	}
}

// serveUmbrellaTurn plans request and answers its call with backend's reply.
func serveUmbrellaTurn(t *testing.T, engine routing.Engine, backend *umbrellaBackend, request map[string]any) *routing.Response {
	t.Helper()
	header := routing.Header{
		{Name: ":method", Value: "POST"},
		{Name: ":path", Value: "/v1/chat/completions"},
		{Name: ":authority", Value: "router"},
		{Name: ":scheme", Value: "http"},
		{Name: "content-type", Value: "application/json"},
		{Name: "x-request-id", Value: "umbrella-modes"},
	}
	plan, err := engine.Plan(t.Context(), &routing.Request{Header: header, Body: umbrellaBody(t, request)})
	require.NoError(t, err)
	t.Cleanup(func() { plan.Finish(nil) })
	require.Nil(t, plan.Immediate, "the Router answered the request itself")
	dispatched := backend.decode(plan.Call.Request.Body)
	requireToolTurn(t, dispatched)
	reply, contentType := backend.reply(dispatched)
	out, err := engine.Respond(t.Context(), plan, &routing.UpstreamResponse{
		Status: http.StatusOK, Header: routing.Header{{Name: "content-type", Value: contentType}},
		Body: strings.NewReader(string(reply)),
	})
	require.NoError(t, err)
	return out
}
