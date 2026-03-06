package extproc

import (
	"bytes"
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
type StreamedBodyHandler struct {
	router *OpenAIRouter
	ctx    *RequestContext
	state  streamedBodyState
	buf    bytes.Buffer

	// Fields extracted from first chunk
	model    string
	isStream bool
	isAuto   bool
}

var streamedHandlerPool = sync.Pool{
	New: func() interface{} {
		return &StreamedBodyHandler{}
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

	switch h.state {
	case stateInit:
		return h.handleInit(eos)
	case statePassthrough:
		return h.handlePassthrough(eos)
	case stateAccumulate:
		return h.handleAccumulate(eos)
	default:
		return continueEmptyBody(), nil
	}
}

// handleInit accumulates bytes until the model field can be extracted from the
// partial JSON, then transitions to passthrough or accumulate. Keeps waiting if
// the model key hasn't appeared yet (e.g., because json.Marshal alphabetizes
// keys and "messages" appears before "model").
func (h *StreamedBodyHandler) handleInit(eos bool) (*ext_proc.ProcessingResponse, error) {
	buf := h.buf.Bytes()

	h.model = extractModelFast(buf)

	// If model not found yet and not EOS, keep accumulating
	if h.model == "" && !eos {
		return continueEmptyBody(), nil
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
		return continueEmptyBody(), nil
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
		return continueEmptyBody(), nil
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

// continueEmptyBody returns a CONTINUE response that replaces the chunk with
// an empty body (effectively "eating" it on the Envoy side).
func continueEmptyBody() *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
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
}

