package extproc

import (
	"bytes"
	"fmt"
	"sync"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// streamedBodyState tracks the processing phase for STREAMED body mode.
type streamedBodyState int

const (
	stateInit        streamedBodyState = iota // accumulating until model field is detected
	statePassthrough                          // non-auto model: eat chunks, emit full body on EOS
	stateAccumulate                           // auto model: eat chunks, classify on EOS
)

// StreamedBodyHandler implements semi-streaming request body processing.
//
// In Envoy STREAMED mode, body arrives as multiple HttpBody messages. For each
// message the handler MUST return exactly one ProcessingResponse. The handler
// accumulates bytes until the model field can be extracted (via gjson), then
// branches:
//
//   - Passthrough (non-auto model): continue eating chunks (replacing each
//     with an empty body) so that upstream sees nothing until EOS. On EOS the
//     full accumulated body is passed to handleRequestBody which may apply
//     model-alias rewrites and header mutations, then emits the complete
//     (possibly mutated) body as a single response.
//   - Accumulate (auto model): same chunk-eating strategy. On EOS the full
//     classification + body mutation pipeline runs.
//
// Both paths eat every chunk and emit the full body only on EOS, so upstream
// never receives partial/duplicated data regardless of body mutations.
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
	state  streamedBodyState
	buf    bytes.Buffer

	model    string
	isStream bool
	isAuto   bool

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
	h.state = stateInit
	h.buf.Reset()
	h.model = ""
	h.isStream = false
	h.isAuto = false

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

	switch h.state {
	case stateInit:
		return h.handleInit(eos)
	case statePassthrough:
		return h.handlePassthrough(eos)
	case stateAccumulate:
		return h.handleAccumulate(eos)
	default:
		return sharedContinueEmptyBody, nil
	}
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

// handleInit accumulates bytes until the model field can be extracted from the
// partial JSON, then transitions to passthrough or accumulate. Keeps waiting if
// the model key hasn't appeared yet (e.g., because json.Marshal alphabetizes
// keys and "messages" appears before "model").
func (h *StreamedBodyHandler) handleInit(eos bool) (*ext_proc.ProcessingResponse, error) {
	buf := h.buf.Bytes()

	h.model = extractModelFast(buf)

	if h.model == "" && !eos {
		return sharedContinueEmptyBody, nil
	}

	h.isStream = extractStreamParamFast(buf)

	if h.isStream {
		h.ctx.ExpectStreamingResponse = true
	}

	if h.model != "" {
		h.ctx.RequestModel = h.model
	}

	h.isAuto = h.router.Config != nil && h.router.Config.IsAutoModelName(h.model)

	if h.isAuto {
		h.state = stateAccumulate
		logging.Infof("[StreamedBody] Model %q detected as auto — accumulating", h.model)
		return h.handleAccumulate(eos)
	}

	h.state = statePassthrough
	logging.Infof("[StreamedBody] Model %q detected as specified — passthrough", h.model)
	return h.handlePassthrough(eos)
}

// handlePassthrough eats chunks (replacing each with an empty body so upstream
// sees nothing yet). On EOS the full accumulated body is passed through the
// standard pipeline for model-alias rewrites and header mutations, then emitted
// as a single complete body. This avoids the corrupted-body problem where
// forwarded intermediate chunks would be duplicated by an EOS body mutation.
func (h *StreamedBodyHandler) handlePassthrough(eos bool) (*ext_proc.ProcessingResponse, error) {
	if !eos {
		return sharedContinueEmptyBody, nil
	}

	h.ctx.ProcessingStartTime = time.Now()
	h.ctx.OriginalRequestBody = bytes.Clone(h.buf.Bytes())

	v := &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{
			Body:        h.ctx.OriginalRequestBody,
			EndOfStream: true,
		},
	}

	return h.router.handleRequestBody(v, h.ctx)
}

// handleAccumulate eats chunks (replaces with empty body). On EOS the full
// classification + body mutation pipeline runs on the accumulated body.
func (h *StreamedBodyHandler) handleAccumulate(eos bool) (*ext_proc.ProcessingResponse, error) {
	if !eos {
		return sharedContinueEmptyBody, nil
	}

	h.ctx.ProcessingStartTime = time.Now()
	h.ctx.OriginalRequestBody = bytes.Clone(h.buf.Bytes())

	v := &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{
			Body:        h.ctx.OriginalRequestBody,
			EndOfStream: true,
		},
	}

	return h.router.handleRequestBody(v, h.ctx)
}
