package backendinvoker

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestCodecStreamBodySettlesAuthoritativeTerminalOnce(t *testing.T) {
	stream := newTestCodecStream(t)
	finalized := make([]ResponseTerminal, 0, 1)
	body := newCodecStreamBody(io.NopCloser(strings.NewReader(chatStreamFixture())), stream, func(terminal ResponseTerminal) error {
		finalized = append(finalized, terminal)
		return nil
	})
	encoded, err := io.ReadAll(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) == 0 || len(finalized) != 1 {
		t.Fatalf("encoded bytes=%d terminals=%d", len(encoded), len(finalized))
	}
	usage := finalized[0].Usage
	if usage.State != llmprotocol.UsageAvailable || usage.InputTotal.Value == nil ||
		*usage.InputTotal.Value != 2 || usage.InputTotal.Provenance != llmprotocol.UsageAuthoritative ||
		usage.OutputTotal.Value == nil || *usage.OutputTotal.Value != 1 ||
		usage.OutputTotal.Provenance != llmprotocol.UsageAuthoritative {
		t.Fatalf("terminal usage = %+v", usage)
	}
	if err := body.Close(); err != nil {
		t.Fatal(err)
	}
	if len(finalized) != 1 {
		t.Fatalf("Close emitted a duplicate terminal: %d", len(finalized))
	}
}

func TestCodecStreamBodyClosePreservesCancellationEvidence(t *testing.T) {
	stream := newTestCodecStream(t)
	var finalized *ResponseTerminal
	body := newCodecStreamBody(io.NopCloser(strings.NewReader(chatStreamFixture())), stream, func(terminal ResponseTerminal) error {
		finalized = &terminal
		return nil
	})
	if err := body.Close(); err != nil {
		t.Fatal(err)
	}
	if finalized == nil || finalized.Error == nil || finalized.Error.Code != "stream_canceled" ||
		finalized.Usage.State != llmprotocol.UsageUnavailable || finalized.StopReason != llmprotocol.StopError {
		t.Fatalf("cancellation terminal = %+v", finalized)
	}
}

func TestCodecStreamBodyReturnsFinalizerErrorAfterPendingFrames(t *testing.T) {
	stream := newTestCodecStream(t)
	sentinel := errors.New("settlement failed")
	body := newCodecStreamBody(io.NopCloser(strings.NewReader(chatStreamFixture())), stream, func(ResponseTerminal) error {
		return sentinel
	})
	encoded, err := io.ReadAll(body)
	if len(encoded) == 0 {
		t.Fatal("pending encoded frames were discarded")
	}
	if !errors.Is(err, sentinel) {
		t.Fatalf("ReadAll error = %v, want settlement failure", err)
	}
}

func TestCodecStreamBodyRejectsTrailingTransportChunkAfterSemanticTerminal(t *testing.T) {
	stream := newTestCodecStream(t)
	source := &countingChunkReadCloser{chunks: [][]byte{
		[]byte(chatStreamFixture()),
		[]byte("data: {\"trailing\":"),
	}}
	var finalized *ResponseTerminal
	body := newCodecStreamBody(source, stream, func(terminal ResponseTerminal) error {
		finalized = &terminal
		return nil
	})
	encoded, err := io.ReadAll(body)
	if len(encoded) == 0 || err == nil {
		t.Fatalf("ReadAll bytes=%d error=%v, want encoded prefix and trailing-data error", len(encoded), err)
	}
	if source.reads != 2 {
		t.Fatalf("source reads = %d, want trailing chunk to be inspected", source.reads)
	}
	if finalized == nil || finalized.Error == nil || finalized.Usage.State != llmprotocol.UsageUnavailable ||
		finalized.StopReason != llmprotocol.StopError {
		t.Fatalf("trailing-data terminal = %+v", finalized)
	}
}

func TestCodecStreamBodyAppliesCallerBackpressureBeforeReadingMoreSource(t *testing.T) {
	first, second := splitChatStreamFixture()
	source := &countingChunkReadCloser{chunks: [][]byte{first, second}}
	body := newCodecStreamBody(source, newTestCodecStream(t), nil)
	oneByte := make([]byte, 1)
	if n, err := body.Read(oneByte); n != 1 || err != nil {
		t.Fatalf("first Read = %d, %v", n, err)
	}
	if source.reads != 1 {
		t.Fatalf("source reads = %d, want one while encoded output remains buffered", source.reads)
	}
	for streamBody := body.(*codecStreamBody); len(streamBody.pending) > 0; {
		if _, err := body.Read(oneByte); err != nil {
			t.Fatalf("drain pending frame: %v", err)
		}
		if source.reads != 1 {
			t.Fatalf("source was read before pending output drained: %d", source.reads)
		}
	}
	if _, err := io.ReadAll(body); err != nil {
		t.Fatal(err)
	}
	if source.reads < 2 {
		t.Fatalf("remaining source chunk was not consumed: reads=%d", source.reads)
	}
}

func TestTransformResponseEncodesSafeNon2xxAcrossClientFormats(t *testing.T) {
	formats := []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.AnthropicMessagesV1,
	}
	for _, format := range formats {
		t.Run(string(format), func(t *testing.T) {
			var finalized *ResponseTerminal
			invoker := &Invoker{Finalizer: responseFinalizerFunc(func(_ context.Context, _ Plan, _ AttemptResult, terminal ResponseTerminal) error {
				finalized = &terminal
				return nil
			})}
			plan := Plan{RequestID: "request", DispatchID: "dispatch", ModelID: "public-model", SourceFormat: format}
			attempt := AttemptResult{Attempt: Attempt{ID: "attempt", BackendID: "backend"}, State: AttemptResponseStarted}
			response, err := invoker.transformResponse(context.Background(), plan, Backend{WireFormat: llmprotocol.OpenAIChatV1}, attempt, &http.Response{
				StatusCode: http.StatusTooManyRequests,
				Header:     http.Header{"Retry-After": {"7"}, "X-Provider-Secret": {"do-not-forward"}},
				Body: io.NopCloser(strings.NewReader(
					`{"error":{"message":"provider credential sk-secret","type":"provider_error","code":"credential_echo"}}`,
				)),
			}, []string{"sk-secret"})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = response.Body.Close() })
			body, err := io.ReadAll(response.Body)
			if err != nil {
				t.Fatal(err)
			}
			if response.StatusCode != http.StatusTooManyRequests || response.Header.Get("Retry-After") != "7" ||
				response.Header.Get("X-Provider-Secret") != "" || strings.Contains(string(body), "sk-secret") ||
				!strings.Contains(string(body), "rate limited") {
				t.Fatalf("unsafe translated error: status=%d headers=%v body=%s", response.StatusCode, response.Header, body)
			}
			if finalized == nil || finalized.Error == nil || finalized.Error.Category != llmprotocol.ErrorRateLimited ||
				strings.Contains(finalized.Error.Message, "sk-secret") ||
				finalized.Error.Message != "the selected model is rate limited" ||
				finalized.Usage.State != llmprotocol.UsageUnavailable || finalized.StopReason != llmprotocol.StopError {
				t.Fatalf("error terminal = %+v", finalized)
			}
		})
	}
}

func TestTransformResponsePreservesSafeNon2xxAcrossProtocolMatrix(t *testing.T) {
	providerCases := []struct {
		format          llmprotocol.WireFormat
		requestIDHeader string
		body            string
		parameter       string
	}{
		{
			format: llmprotocol.OpenAIChatV1, requestIDHeader: "X-Request-Id",
			body:      `{"error":{"message":"API key is invalid.","type":"authentication_error","param":"model","code":"authentication_error"}}`,
			parameter: "model",
		},
		{
			format: llmprotocol.OpenAIResponsesV1, requestIDHeader: "X-Request-Id",
			body:      `{"error":{"message":"API key is invalid.","type":"authentication_error","param":"model","code":"authentication_error"}}`,
			parameter: "model",
		},
		{
			format: llmprotocol.AnthropicMessagesV1, requestIDHeader: "Request-Id",
			body: `{"type":"error","error":{"type":"authentication_error","message":"API key is invalid."},"request_id":"upstream-request"}`,
		},
	}
	clientFormats := []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.AnthropicMessagesV1,
	}
	for _, providerCase := range providerCases {
		for _, clientFormat := range clientFormats {
			t.Run(string(providerCase.format)+"/"+string(clientFormat), func(t *testing.T) {
				var finalized *ResponseTerminal
				invoker := &Invoker{Finalizer: responseFinalizerFunc(func(_ context.Context, _ Plan, _ AttemptResult, terminal ResponseTerminal) error {
					finalized = &terminal
					return nil
				})}
				plan := Plan{
					RequestID: "request", DispatchID: "dispatch", ModelID: "public-model",
					SourceFormat: clientFormat, Streaming: true,
				}
				attempt := AttemptResult{Attempt: Attempt{ID: "attempt", BackendID: "backend"}, State: AttemptResponseStarted}
				response, err := invoker.transformResponse(
					context.Background(),
					plan,
					Backend{WireFormat: providerCase.format},
					attempt,
					&http.Response{
						StatusCode: http.StatusUnauthorized,
						Header: http.Header{
							"Retry-After":                {"7"},
							providerCase.requestIDHeader: {"provider-request"},
							"X-Provider-Secret":          {"do-not-forward"},
						},
						Body: io.NopCloser(strings.NewReader(providerCase.body)),
					},
					nil,
				)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = response.Body.Close() })
				body, err := io.ReadAll(response.Body)
				if err != nil {
					t.Fatal(err)
				}
				assertPublicTransportErrorWire(
					t, clientFormat, body, "authentication_error", "API key is invalid.",
					providerCase.parameter, "provider-request",
				)
				if response.StatusCode != http.StatusUnauthorized || response.Header.Get("Retry-After") != "7" ||
					response.Header.Get("Request-Id") != "provider-request" ||
					response.Header.Get("X-Request-Id") != "provider-request" ||
					response.Header.Get("X-Provider-Secret") != "" {
					t.Fatalf("public status/headers = %d/%v", response.StatusCode, response.Header)
				}
				if finalized == nil || finalized.Error == nil ||
					finalized.Error.Category != llmprotocol.ErrorAuthentication ||
					finalized.Error.Code != "authentication_error" ||
					finalized.Error.Message != "API key is invalid." ||
					finalized.Error.Parameter != providerCase.parameter ||
					finalized.Error.RetryAfter != 7 {
					t.Fatalf("error terminal = %+v", finalized)
				}
			})
		}
	}
}

func assertPublicTransportErrorWire(
	t *testing.T,
	format llmprotocol.WireFormat,
	body []byte,
	code,
	message,
	parameter,
	requestID string,
) {
	t.Helper()
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		t.Fatalf("public transport error is invalid JSON: %v; body=%s", err, body)
	}
	switch format {
	case llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1:
		if len(object) != 1 || object["error"] == nil {
			t.Fatalf("OpenAI transport error has non-canonical fields: %s", body)
		}
		var detailObject map[string]json.RawMessage
		if err := json.Unmarshal(object["error"], &detailObject); err != nil || len(detailObject) != 4 {
			t.Fatalf("OpenAI transport error detail has non-canonical fields: %v; body=%s", err, body)
		}
		var detail struct {
			Type    string  `json:"type"`
			Code    *string `json:"code"`
			Message string  `json:"message"`
			Param   *string `json:"param"`
		}
		if err := json.Unmarshal(object["error"], &detail); err != nil {
			t.Fatalf("OpenAI transport error detail is invalid: %v; body=%s", err, body)
		}
		if detail.Type != "authentication_error" || detail.Code == nil || *detail.Code != code ||
			detail.Message != message || (parameter == "" && detail.Param != nil) ||
			(parameter != "" && (detail.Param == nil || *detail.Param != parameter)) {
			t.Fatalf("OpenAI transport error detail is not canonical: %+v; body=%s", detail, body)
		}
	case llmprotocol.AnthropicMessagesV1:
		if len(object) != 3 || string(object["type"]) != `"error"` ||
			string(object["request_id"]) != `"`+requestID+`"` || object["error"] == nil {
			t.Fatalf("Anthropic transport error has non-canonical fields: %s", body)
		}
		var detail struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		}
		if err := json.Unmarshal(object["error"], &detail); err != nil ||
			detail.Type != code || detail.Message != message {
			t.Fatalf("Anthropic transport error detail is not canonical: %+v/%v; body=%s", detail, err, body)
		}
	default:
		t.Fatalf("unexpected client format %q", format)
	}
}

func TestTransformResponseSanitizesStreamErrorBeforePublicEncoding(t *testing.T) {
	var finalized *ResponseTerminal
	invoker := &Invoker{Finalizer: responseFinalizerFunc(func(_ context.Context, _ Plan, _ AttemptResult, terminal ResponseTerminal) error {
		finalized = &terminal
		return nil
	})}
	response, err := invoker.transformResponse(
		context.Background(),
		Plan{SourceFormat: llmprotocol.AnthropicMessagesV1, ModelID: "public-model", Streaming: true},
		Backend{WireFormat: llmprotocol.OpenAIChatV1},
		AttemptResult{},
		&http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"text/event-stream"}},
			Body: io.NopCloser(strings.NewReader(
				"data: {\"error\":{\"message\":\"provider credential pw\",\"type\":\"authentication_error\",\"code\":\"authentication_error\"}}\n\n",
			)),
		},
		[]string{"pw"},
	)
	if err != nil {
		t.Fatal(err)
	}
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), "credential pw") || !strings.Contains(string(body), "could not authenticate") {
		t.Fatalf("public stream error = %s", body)
	}
	if finalized == nil || finalized.Error == nil || finalized.Error.Category != llmprotocol.ErrorAuthentication ||
		finalized.Error.Message != "the selected model could not authenticate the request" {
		t.Fatalf("stream terminal = %+v", finalized)
	}
}

func TestTransformResponseDropsUnsafeProviderRequestIDs(t *testing.T) {
	tests := []struct {
		name            string
		requestID       string
		sensitiveValues []string
	}{
		{name: "credential echo", requestID: "pw", sensitiveValues: []string{"pw"}},
		{name: "oversized", requestID: strings.Repeat("r", maximumPublicProviderRequestIDBytes+1)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response, err := (&Invoker{}).transformResponse(
				context.Background(),
				Plan{SourceFormat: llmprotocol.AnthropicMessagesV1},
				Backend{WireFormat: llmprotocol.AnthropicMessagesV1},
				AttemptResult{},
				&http.Response{
					StatusCode: http.StatusUnauthorized,
					Header:     http.Header{"Request-Id": {test.requestID}},
					Body: io.NopCloser(strings.NewReader(
						`{"type":"error","error":{"type":"authentication_error","message":"API key is invalid."},"request_id":"` + test.requestID + `"}`,
					)),
				},
				test.sensitiveValues,
			)
			if err != nil {
				t.Fatal(err)
			}
			body, err := io.ReadAll(response.Body)
			if err != nil {
				t.Fatal(err)
			}
			var object map[string]json.RawMessage
			if err := json.Unmarshal(body, &object); err != nil {
				t.Fatalf("public error body is invalid JSON: %v; body=%s", err, body)
			}
			if object["request_id"] != nil || response.Header.Get("Request-Id") != "" ||
				response.Header.Get("X-Request-Id") != "" {
				t.Fatalf("unsafe request ID escaped: body=%s headers=%v", body, response.Header)
			}
		})
	}
}

func TestTransformResponseBoundsProviderErrorCodeAndParameter(t *testing.T) {
	response, err := (&Invoker{}).transformResponse(
		context.Background(),
		Plan{SourceFormat: llmprotocol.OpenAIResponsesV1},
		Backend{WireFormat: llmprotocol.OpenAIChatV1},
		AttemptResult{},
		&http.Response{
			StatusCode: http.StatusUnauthorized,
			Header:     make(http.Header),
			Body: io.NopCloser(strings.NewReader(
				`{"error":{"message":"API key is invalid.","type":"authentication_error","code":"` +
					strings.Repeat(":", maximumPublicProviderErrorCodeBytes+1) + `","param":"` +
					strings.Repeat(".", maximumPublicProviderErrorParameterBytes+1) + `"}}`,
			)),
		},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	assertPublicTransportErrorWire(
		t, llmprotocol.OpenAIResponsesV1, body,
		"upstream_authentication", "API key is invalid.", "", "",
	)
}

func TestPublicProviderProtocolErrorRejectsExactAndHeuristicCredentials(t *testing.T) {
	fallback := llmprotocol.NewError(
		llmprotocol.ErrorAuthentication,
		"upstream_authentication",
		"the selected model could not authenticate the request",
		nil,
	)
	tests := []struct {
		name            string
		providerError   *llmprotocol.ProtocolError
		sensitiveValues []string
	}{
		{
			name: "exact secret",
			providerError: &llmprotocol.ProtocolError{
				Code: "opaqueValue42", Message: "opaqueValue42", Parameter: "opaqueValue42",
			},
			sensitiveValues: []string{"opaqueValue42"},
		},
		{
			name: "credential shape",
			providerError: &llmprotocol.ProtocolError{
				Code: "Bearer provider-token", Message: "authorization: provider-token",
				Parameter: "api_key=provider-token",
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := publicProviderProtocolError(test.providerError, fallback, test.sensitiveValues)
			if got.Code != fallback.Code || got.Message != fallback.Message || got.Parameter != "" {
				t.Fatalf("unsafe provider error was not replaced: %+v", got)
			}
		})
	}
}

func TestPublicProviderProtocolErrorAcceptsTerminalIdentityBoundary(t *testing.T) {
	code := strings.Repeat(":", maximumPublicProviderErrorCodeBytes)
	parameter := strings.Repeat(".", maximumPublicProviderErrorParameterBytes)
	fallback := llmprotocol.NewError(
		llmprotocol.ErrorAuthentication,
		"upstream_authentication",
		"the selected model could not authenticate the request",
		nil,
	)
	got := publicProviderProtocolError(&llmprotocol.ProtocolError{
		Code: code, Message: "API key is invalid.", Parameter: parameter,
	}, fallback, nil)
	if got.Code != code || got.Parameter != parameter || got.Message != "API key is invalid." {
		t.Fatalf("terminal identity boundary was not preserved: %+v", got)
	}
}

func TestTransformResponseBoundsProviderErrorBody(t *testing.T) {
	response, err := (&Invoker{}).transformResponse(
		context.Background(),
		Plan{SourceFormat: llmprotocol.OpenAIChatV1},
		Backend{WireFormat: llmprotocol.OpenAIChatV1},
		AttemptResult{},
		&http.Response{
			StatusCode: http.StatusUnauthorized,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(strings.Repeat("x", maximumProviderErrorBodyBytes+1))),
		},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), "upstream_authentication") {
		t.Fatalf("oversized provider error was not replaced: %s", body)
	}
}

func TestSafeUpstreamHTTPErrorCategories(t *testing.T) {
	tests := []struct {
		status   int
		category llmprotocol.ErrorCategory
	}{
		{http.StatusBadRequest, llmprotocol.ErrorInvalidRequest},
		{http.StatusUnauthorized, llmprotocol.ErrorAuthentication},
		{http.StatusForbidden, llmprotocol.ErrorPermission},
		{http.StatusNotFound, llmprotocol.ErrorNotFound},
		{http.StatusTooManyRequests, llmprotocol.ErrorRateLimited},
		{http.StatusGatewayTimeout, llmprotocol.ErrorUpstreamTimeout},
		{http.StatusInternalServerError, llmprotocol.ErrorUpstreamUnavailable},
	}
	for _, test := range tests {
		if got := safeUpstreamHTTPError(test.status, nil); got.Category != test.category {
			t.Fatalf("status %d category = %q, want %q", test.status, got.Category, test.category)
		}
	}
}

func newTestCodecStream(t *testing.T) *protocolcodec.StreamEngine {
	t.Helper()
	stream, err := protocolcodec.NewBuiltinEngine().NewStream(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	return stream
}

func chatStreamFixture() string {
	first, second := splitChatStreamFixture()
	return string(first) + string(second)
}

func splitChatStreamFixture() ([]byte, []byte) {
	first := []byte("data: {\"id\":\"response_1\",\"model\":\"source-model\",\"prompt_text\":null,\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"}}]}\n\n")
	second := []byte("data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n" +
		"data: [DONE]\n\n")
	return first, second
}

type countingChunkReadCloser struct {
	chunks [][]byte
	reads  int
	closed bool
}

type responseFinalizerFunc func(context.Context, Plan, AttemptResult, ResponseTerminal) error

func (finalizer responseFinalizerFunc) Finalize(
	ctx context.Context,
	plan Plan,
	attempt AttemptResult,
	terminal ResponseTerminal,
) error {
	return finalizer(ctx, plan, attempt, terminal)
}

func (source *countingChunkReadCloser) Read(target []byte) (int, error) {
	if len(source.chunks) == 0 {
		return 0, io.EOF
	}
	source.reads++
	chunk := source.chunks[0]
	read := copy(target, chunk)
	if read == len(chunk) {
		source.chunks = source.chunks[1:]
	} else {
		source.chunks[0] = chunk[read:]
	}
	return read, nil
}

func (source *countingChunkReadCloser) Close() error {
	source.closed = true
	return nil
}
