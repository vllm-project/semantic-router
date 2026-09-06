package extproc

import (
	"bytes"
	"fmt"
	"sync"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// StreamedBodyHandler implements semi-streaming request body processing.
//
// In Envoy STREAMED or FULL_DUPLEX_STREAMED mode, body arrives as multiple
// HttpBody messages. The handler treats those chunks as protocol-neutral bytes
// and accumulates them until EOS. Only then does the standard request-body
// pipeline invoke the ingress Codec to decode the complete wire request.
//
// In STREAMED mode, every non-EOS chunk is eaten with a regular empty body
// mutation. In FULL_DUPLEX_STREAMED mode, intermediate replies are deferred
// and the complete body is emitted at EOS with StreamedBodyResponse, as
// required by the ExtProc protocol.
//
// Safety:
//   - MaxBytes: if configured, rejects requests whose accumulated body exceeds
//     the limit (HTTP 413).
//   - Deadline: if configured, rejects requests that take too long to
//     accumulate (HTTP 408).
//   - GC: the "eat chunk" response is pooled so intermediate chunks produce
//     zero allocations on the hot path.
type StreamedBodyHandler struct {
	router *OpenAIRouter
	ctx    *RequestContext
	buf    bytes.Buffer

	// Guards: populated once from config at creation time.
	maxBytes int64
	deadline time.Time // zero value = no deadline
}

var streamedHandlerPool = sync.Pool{
	New: func() interface{} {
		return &StreamedBodyHandler{}
	},
}

// Shared immutable "eat chunk" response. Because the response only contains
// CONTINUE + empty body (no per-request data), a single instance is safe to
// return from every non-EOS chunk across all goroutines. This eliminates ~5
// protobuf allocations per chunk that would otherwise become immediate garbage.
var sharedContinueEmptyBody = &ext_proc.ProcessingResponse{
	Response: &ext_proc.ProcessingResponse_RequestBody{
		RequestBody: &ext_proc.BodyResponse{
			Response: &ext_proc.CommonResponse{
				Status: ext_proc.CommonResponse_CONTINUE,
				BodyMutation: &ext_proc.BodyMutation{
					Mutation: &ext_proc.BodyMutation_Body{
						Body: []byte{},
					},
				},
			},
		},
	},
}

func newStreamedBodyHandler(router *OpenAIRouter, ctx *RequestContext) *StreamedBodyHandler {
	h := streamedHandlerPool.Get().(*StreamedBodyHandler)
	h.router = router
	h.ctx = ctx
	h.buf.Reset()

	h.maxBytes = 0
	h.deadline = time.Time{}
	if router.Config != nil {
		h.maxBytes = router.Config.MaxStreamedBodyBytes
		if sec := router.Config.StreamedBodyTimeoutSec; sec > 0 {
			h.deadline = time.Now().Add(time.Duration(sec) * time.Second)
		}
	}
	return h
}

// Release returns the handler to the pool for reuse.
func (h *StreamedBodyHandler) Release() {
	h.router = nil
	h.ctx = nil
	h.buf.Reset()
	streamedHandlerPool.Put(h)
}

// HandleChunk processes a single body chunk from Envoy STREAMED mode.
func (h *StreamedBodyHandler) HandleChunk(body *ext_proc.HttpBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	chunk := body.GetBody()
	eos := body.GetEndOfStream()

	h.buf.Write(chunk)

	if err := h.checkGuards(); err != nil {
		return nil, err
	}

	if !eos {
		return h.intermediateResponse(), nil
	}

	return h.handleAccumulatedBody()
}

func (h *StreamedBodyHandler) intermediateResponse() *ext_proc.ProcessingResponse {
	if h.ctx != nil && h.ctx.FullDuplexRequestBody {
		return nil
	}
	return sharedContinueEmptyBody
}

// checkGuards enforces max-body and deadline limits. Returning an error causes
// the gRPC stream to close, which makes Envoy apply its failure_mode_allow
// policy (typically returning 500 or passing through).
func (h *StreamedBodyHandler) checkGuards() error {
	if h.maxBytes > 0 && int64(h.buf.Len()) > h.maxBytes {
		logging.Infof("[StreamedBody] Accumulated %d bytes exceeds limit %d — aborting",
			h.buf.Len(), h.maxBytes)
		return fmt.Errorf("streamed body too large: %d > %d bytes", h.buf.Len(), h.maxBytes)
	}
	if !h.deadline.IsZero() && time.Now().After(h.deadline) {
		logging.Infof("[StreamedBody] Accumulation deadline exceeded after %d bytes — aborting",
			h.buf.Len())
		return fmt.Errorf("streamed body accumulation timed out after %d bytes", h.buf.Len())
	}
	return nil
}

// handleAccumulatedBody passes the complete wire request to the standard
// request-body pipeline. The ingress Codec is the only semantic parser.
func (h *StreamedBodyHandler) handleAccumulatedBody() (*ext_proc.ProcessingResponse, error) {
	h.ctx.ProcessingStartTime = time.Now()
	body := bytes.Clone(h.buf.Bytes())

	v := &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{
			Body:        body,
			EndOfStream: true,
		},
	}

	response, err := h.router.handleRequestBody(v, h.ctx)
	return h.finalizeResponse(response), err
}

// finalizeResponse translates the standard request-body pipeline result to
// the response shape required by FULL_DUPLEX_STREAMED. Immediate responses are
// already valid in either mode and pass through unchanged.
func (h *StreamedBodyHandler) finalizeResponse(response *ext_proc.ProcessingResponse) *ext_proc.ProcessingResponse {
	if h.ctx == nil || !h.ctx.FullDuplexRequestBody || response == nil || response.GetImmediateResponse() != nil {
		return response
	}

	bodyResponse := response.GetRequestBody()
	if bodyResponse == nil || bodyResponse.Response == nil {
		return response
	}

	common := bodyResponse.Response
	body := h.buf.Bytes()
	if mutation := common.GetBodyMutation(); mutation != nil {
		switch mutation := mutation.GetMutation().(type) {
		case *ext_proc.BodyMutation_Body:
			body = mutation.Body
		case *ext_proc.BodyMutation_ClearBody:
			if mutation.ClearBody {
				body = nil
			}
		}
	}
	common.BodyMutation = &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_StreamedResponse{
			StreamedResponse: &ext_proc.StreamedBodyResponse{
				Body:        bytes.Clone(body),
				EndOfStream: true,
			},
		},
	}
	return response
}
