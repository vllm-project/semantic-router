package backendinvoker

import (
	"context"
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
			response, err := invoker.transformResponse(context.Background(), plan, Backend{}, attempt, &http.Response{
				StatusCode: http.StatusTooManyRequests,
				Header:     http.Header{"Retry-After": {"7"}, "X-Provider-Secret": {"do-not-forward"}},
				Body:       io.NopCloser(strings.NewReader(`{"error":{"message":"provider credential sk-secret"}}`)),
			})
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
				!strings.Contains(string(body), "upstream_rate_limited") {
				t.Fatalf("unsafe translated error: status=%d headers=%v body=%s", response.StatusCode, response.Header, body)
			}
			if finalized == nil || finalized.Error == nil || finalized.Error.Category != llmprotocol.ErrorRateLimited ||
				finalized.Usage.State != llmprotocol.UsageUnavailable || finalized.StopReason != llmprotocol.StopError {
				t.Fatalf("error terminal = %+v", finalized)
			}
		})
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
	first := []byte("data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"}}]}\n\n")
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
