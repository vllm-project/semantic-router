package extproc

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Providers may send SSE with Content-Length. Headers are released before
// streamed body callbacks, so their original length must not frame translated
// or usage-filtered bytes. Exercise a real HTTP reader after both phases.
func TestStreamingResponseHTTPFramingAcrossProtocols(t *testing.T) {
	for _, internal := range []bool{false, true} {
		for _, client := range extProcMatrixFormats {
			if internal && client != llmprotocol.OpenAIChatV1 {
				continue
			}
			for _, backend := range extProcMatrixFormats {
				t.Run(fmt.Sprintf("internal_%t/%s/%s", internal, client, backend), func(t *testing.T) {
					assertStreamingHTTPFraming(t, internal, client, backend)
				})
			}
		}
	}
}

func assertStreamingHTTPFraming(t *testing.T, internal bool, client, backend llmprotocol.WireFormat) {
	t.Helper()
	upstream := extProcStreamFixture(backend)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		router := &OpenAIRouter{}
		ctx := looperTransportContext(backend, true)
		ctx.LooperRequest = internal
		ctx.SourceFormat = client
		// Public Chat without include_usage exercises same-wire byte changes.
		includeUsage := internal
		ctx.SemanticRequest.StreamOptions.IncludeUsage = &includeUsage
		headers := looperTransportHeaders(200, "text/event-stream")
		headers.ResponseHeaders.Headers.Headers = append(headers.ResponseHeaders.Headers.Headers,
			&core.HeaderValue{Key: "content-length", Value: strconv.Itoa(len(upstream))})
		response, err := router.handleResponseHeaders(headers, ctx)
		require.NoError(t, err)
		mutation := response.GetResponseHeaders().GetResponse().GetHeaderMutation()
		require.Contains(t, mutation.GetRemoveHeaders(), "content-length")
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Content-Length", strconv.Itoa(len(upstream)))
		applyStreamingHTTPHeaders(w.Header(), mutation)
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		for offset := 0; offset < len(upstream); offset += 7 {
			end := min(offset+7, len(upstream))
			body := upstream[offset:end]
			response, err = router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
				ResponseBody: &ext_proc.HttpBody{Body: body, EndOfStream: end == len(upstream)},
			}, ctx)
			require.NoError(t, err)
			require.Nil(t, response.GetImmediateResponse())
			if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
				body = mutation.GetBody()
			}
			_, err = w.Write(body)
			require.NoError(t, err)
			w.(http.Flusher).Flush()
		}
	}))
	defer server.Close()
	response, err := server.Client().Get(server.URL)
	require.NoError(t, err)
	defer response.Body.Close()
	publicBody, err := io.ReadAll(response.Body)
	require.NoError(t, err, "rewritten stream must be readable to EOF without stale length")
	require.Empty(t, response.Header.Get("Content-Length"))
	if client != backend || !internal && client == llmprotocol.OpenAIChatV1 {
		require.NotEqual(t, len(upstream), len(publicBody), "exercise a changed stream length")
	}
	assertStreamingHTTPCompletion(t, client, publicBody)
}

func applyStreamingHTTPHeaders(headers http.Header, mutation *ext_proc.HeaderMutation) {
	for _, name := range mutation.GetRemoveHeaders() {
		headers.Del(name)
	}
	for _, option := range mutation.GetSetHeaders() {
		value := option.GetHeader().GetValue()
		if raw := option.GetHeader().GetRawValue(); len(raw) != 0 {
			value = string(raw)
		}
		headers.Set(option.GetHeader().GetKey(), value)
	}
}

func assertStreamingHTTPCompletion(t *testing.T, format llmprotocol.WireFormat, body []byte) {
	t.Helper()
	stream, err := protocolcodec.NewBuiltinEngine().NewStream(format, format, llmprotocol.StreamContext{Source: format, Target: format})
	require.NoError(t, err)
	_, events, _, err := stream.Push(body)
	require.NoError(t, err)
	_, terminal, _, err := stream.Finalize(nil)
	require.NoError(t, err)
	var content string
	completed := false
	for _, event := range append(events, terminal...) {
		require.NotEqual(t, llmprotocol.EventResponseFailed, event.Type)
		if event.Type == llmprotocol.EventOutputTextDelta {
			content += event.Delta
		}
		completed = completed || event.Type == llmprotocol.EventResponseCompleted
	}
	require.True(t, completed)
	require.Equal(t, "hello", content)
}
