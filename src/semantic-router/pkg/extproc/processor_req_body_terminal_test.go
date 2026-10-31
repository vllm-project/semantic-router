package extproc

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"strings"
	"testing"
	"time"

	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const terminalTestBody = `{"model":"fallback-model","messages":[{"role":"user","content":"hello"}]}`

func TestStreamedBodyRejectionPreventsResumingAtChunkOrTrailers(t *testing.T) {
	for _, end := range []string{"small chunk then EOS", "trailers"} {
		t.Run(end, func(t *testing.T) {
			ctx := &RequestContext{Headers: make(map[string]string), SourceFormat: llmprotocol.OpenAIChatV1, FullDuplexRequestBody: true}
			h := newStreamedBodyHandler(makeTestRouterWithLimits(1024, 0), ctx)
			defer h.Release()
			_, err := h.HandleChunk(&ext_proc.HttpBody{Body: []byte(terminalTestBody)}, ctx)
			require.NoError(t, err)
			response, err := h.HandleChunk(&ext_proc.HttpBody{Body: bytes.Repeat([]byte(" "), 1024)}, ctx)
			require.NoError(t, err)
			require.Equal(t, typev3.StatusCode_PayloadTooLarge, response.GetImmediateResponse().GetStatus().GetCode())
			before := bytes.Clone(h.buf.Bytes())
			beforeCap, beforeCount := h.buf.Cap(), h.chunkCount
			if end == "trailers" {
				response, err = h.finishAtTrailers()
			} else {
				response, err = h.HandleChunk(&ext_proc.HttpBody{Body: []byte(" ")}, ctx)
				require.NoError(t, err)
				assert.Nil(t, response)
				response, err = h.HandleChunk(&ext_proc.HttpBody{EndOfStream: true}, ctx)
			}
			require.NoError(t, err)
			assert.Nil(t, response, "rejected requests must not produce another reply")
			assert.Equal(t, before, h.buf.Bytes())
			assert.Equal(t, beforeCap, h.buf.Cap())
			assert.Equal(t, beforeCount, h.chunkCount)
			assert.Nil(t, ctx.SemanticRequest, "rejected body must never reach the codec or routing pipeline")
		})
	}
}

func TestStreamedBodyDeadlineRejectionPreventsResuming(t *testing.T) {
	ctx := &RequestContext{FullDuplexRequestBody: true}
	h := newStreamedBodyHandler(makeTestRouterWithLimits(1024, 0), ctx)
	defer h.Release()
	h.deadline = time.Now().Add(-time.Second)
	response, err := h.HandleChunk(&ext_proc.HttpBody{Body: []byte(terminalTestBody)}, ctx)
	require.NoError(t, err)
	require.Equal(t, typev3.StatusCode_RequestTimeout, response.GetImmediateResponse().GetStatus().GetCode())
	before := bytes.Clone(h.buf.Bytes())
	// Even removing the timeout cannot resume a request that was rejected.
	h.deadline = time.Time{}
	response, err = h.HandleChunk(&ext_proc.HttpBody{Body: []byte(" "), EndOfStream: true}, ctx)
	require.NoError(t, err)
	assert.Nil(t, response)
	response, err = h.finishAtTrailers()
	require.NoError(t, err)
	assert.Nil(t, response)
	assert.Equal(t, before, h.buf.Bytes())
	assert.Nil(t, ctx.SemanticRequest)
}

func TestStreamedBodyDefensiveSizeCheckRejectsCorruptAccumulation(t *testing.T) {
	ctx := &RequestContext{}
	h := newStreamedBodyHandler(makeTestRouterWithLimits(1, 0), ctx)
	defer h.Release()
	h.buf.WriteString("ab") // Simulate a future caller bypassing the precheck.
	response, err := h.finishAtTrailers()
	require.NoError(t, err)
	require.Equal(t, typev3.StatusCode_PayloadTooLarge, response.GetImmediateResponse().GetStatus().GetCode())
	assert.Nil(t, ctx.SemanticRequest)
}

func TestProcessRequestBodyRejectionReleasesHandlerOnSendFailure(t *testing.T) {
	router := makeTestRouterWithLimits(1024, 0)
	ctx := &RequestContext{Headers: make(map[string]string), FullDuplexRequestBody: true}
	stream := NewMockStream(nil)
	require.NoError(t, router.processRequestBody(stream, bodyRequest(terminalTestBody, false).GetRequest().(*ext_proc.ProcessingRequest_RequestBody), ctx))
	sendErr := errors.New("terminal reply send failed")
	stream.SendError = sendErr
	err := router.processRequestBody(stream, bodyRequest(strings.Repeat(" ", 1024), false).GetRequest().(*ext_proc.ProcessingRequest_RequestBody), ctx)
	require.ErrorIs(t, err, sendErr)
	assert.Nil(t, ctx.StreamedBody)
	assert.Nil(t, ctx.fullDuplexHold)
	assert.Empty(t, stream.Responses)
	// A failed send must not permit the request to resume either.
	stream.SendError = nil
	require.NoError(t, router.processRequestBody(stream, bodyRequest(" ", true).GetRequest().(*ext_proc.ProcessingRequest_RequestBody), ctx))
	assert.Empty(t, stream.Responses)
	assert.Nil(t, ctx.SemanticRequest)
}

func TestRoutingSessionEndsAfterStreamedBodyRejection(t *testing.T) {
	released := 0
	session := makeTestRouterWithLimits(1024, 0).newRoutingSession(t.Context(), func() { released++ })
	defer session.Close(nil)
	_, err := session.RequestBody([]byte(terminalTestBody), false)
	require.NoError(t, err)
	effect, err := session.RequestBody(bytes.Repeat([]byte(" "), 1024), false)
	require.NoError(t, err)
	require.NotNil(t, effect.Immediate)
	assert.Equal(t, 413, effect.Immediate.Status)
	assert.Nil(t, session.ctx.StreamedBody)
	assert.Equal(t, 1, released)
	_, err = session.RequestBody([]byte(" "), true)
	require.ErrorIs(t, err, errSessionEnded)
	_, err = session.ResponseBody(nil, true)
	require.ErrorIs(t, err, errSessionEnded)
	assert.Nil(t, session.ctx.SemanticRequest)
	session.Close(nil)
	assert.Equal(t, 1, released)
}

func TestRoutingSessionCloseReleasesIncompleteStreamedBody(t *testing.T) {
	session := makeTestRouterWithLimits(1024, 0).newRoutingSession(t.Context(), nil)
	_, err := session.RequestBody([]byte(terminalTestBody), false)
	require.NoError(t, err)
	require.NotNil(t, session.ctx.StreamedBody)
	session.Close(context.Canceled)
	assert.Nil(t, session.ctx.StreamedBody)
	session.Close(context.Canceled)
}

func TestProcessRequestBodyRejectionIgnoresQueuedInput(t *testing.T) {
	for _, mode := range []http_ext.ProcessingMode_BodySendMode{http_ext.ProcessingMode_STREAMED, http_ext.ProcessingMode_FULL_DUPLEX_STREAMED} {
		t.Run(mode.String(), func(t *testing.T) {
			router := makeTestRouterWithLimits(1024, 0)
			ctx := &RequestContext{Headers: make(map[string]string)}
			stream := NewMockStream(nil)
			t.Cleanup(func() {
				if ctx.StreamedBody != nil {
					ctx.StreamedBody.Release()
				}
			})
			runProcessRequests(t, router, ctx, stream,
				headersRequest(mode, "POST", "/v1/chat/completions", false),
				bodyRequest(terminalTestBody, false),
				bodyRequest(strings.Repeat(" ", 1024), false),
			)
			responses := len(stream.Responses)
			require.Equal(t, typev3.StatusCode_PayloadTooLarge, stream.Responses[responses-1].GetImmediateResponse().GetStatus().GetCode())
			assert.Nil(t, ctx.StreamedBody, "rejection must release the handler even without EOS")
			runProcessRequests(t, router, ctx, stream,
				bodyRequest(" ", false), bodyRequest("", true), trailersRequest(),
			)
			assert.Len(t, stream.Responses, responses)
			assert.Nil(t, ctx.StreamedBody, "later input must not create a fresh handler")
			assert.Nil(t, ctx.SemanticRequest)
			assert.Nil(t, ctx.fullDuplexHold)
		})
	}
}

func TestProcessReturnsAfterStreamedBodyRejection(t *testing.T) {
	for _, mode := range []http_ext.ProcessingMode_BodySendMode{http_ext.ProcessingMode_STREAMED, http_ext.ProcessingMode_FULL_DUPLEX_STREAMED} {
		t.Run(mode.String(), func(t *testing.T) {
			stream := NewMockStream([]*ext_proc.ProcessingRequest{
				headersRequest(mode, "POST", "/v1/chat/completions", false),
				bodyRequest(terminalTestBody, false), bodyRequest(strings.Repeat(" ", 1024), false),
				bodyRequest(" ", true), trailersRequest(),
			})
			require.NoError(t, makeTestRouterWithLimits(1024, 0).Process(stream))
			assert.Equal(t, 3, stream.RecvIndex, "Process must stop before receiving queued input")
			immediateCount := 0
			for _, response := range stream.Responses {
				if response.GetImmediateResponse() != nil {
					immediateCount++
					assert.Equal(t, typev3.StatusCode_PayloadTooLarge, response.GetImmediateResponse().GetStatus().GetCode())
				}
			}
			assert.Equal(t, 1, immediateCount)
		})
	}
}

// Exercise the real gRPC transport: a terminal reply must close the stream
// without waiting for the client to close its sending side.
func TestGRPCStreamClosesAfterStreamedBodyRejection(t *testing.T) {
	server := grpc.NewServer()
	ext_proc.RegisterExternalProcessorServer(server, makeTestRouterWithLimits(1024, 0))
	listener := bufconn.Listen(1 << 20)
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(server.Stop)
	connection, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = connection.Close() })

	for _, mode := range []http_ext.ProcessingMode_BodySendMode{http_ext.ProcessingMode_STREAMED, http_ext.ProcessingMode_FULL_DUPLEX_STREAMED} {
		t.Run(mode.String(), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
			defer cancel()
			stream, err := ext_proc.NewExternalProcessorClient(connection).Process(ctx)
			require.NoError(t, err)
			require.NoError(t, stream.Send(headersRequest(mode, "POST", "/v1/chat/completions", false)))
			var response *ext_proc.ProcessingResponse
			if mode == http_ext.ProcessingMode_STREAMED {
				response, err = stream.Recv()
				require.NoError(t, err)
				require.NotNil(t, response.GetRequestHeaders())
			}
			require.NoError(t, stream.Send(bodyRequest(terminalTestBody, false)))
			if mode == http_ext.ProcessingMode_STREAMED {
				response, err = stream.Recv()
				require.NoError(t, err)
				assertChunkEaten(t, response)
			}
			require.NoError(t, stream.Send(bodyRequest(strings.Repeat(" ", 1024), false)))
			response, err = stream.Recv()
			require.NoError(t, err)
			require.Equal(t, typev3.StatusCode_PayloadTooLarge, response.GetImmediateResponse().GetStatus().GetCode())
			_, err = stream.Recv()
			require.ErrorIs(t, err, io.EOF, "server must finish after the single 413")
		})
	}
}
