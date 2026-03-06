package extproc

import (
	"bytes"
	"sync"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/promptcompression"
)

// Default chunk size for the semi-streaming model detection boundary.
// The model field is typically within the first 256 bytes of an OpenAI
// request, so 4KB gives plenty of margin while staying small.
const defaultChunkSize = 4096

// streamedBodyState tracks the processing phase for STREAMED body mode.
type streamedBodyState int

const (
	stateInit        streamedBodyState = iota // accumulating first chunk for model detection
	statePassthrough                          // non-auto model: forward chunks unchanged
	stateAccumulate                           // auto model: eat chunks, incremental preprocess
)

// StreamedBodyHandler implements semi-streaming request body processing.
//
// In Envoy STREAMED mode, body arrives as multiple HttpBody messages. For each
// message the handler MUST return exactly one ProcessingResponse. The handler
// accumulates the first ChunkSize bytes to detect the model field (via gjson),
// then branches:
//
//   - Passthrough (non-auto model): forward chunks unchanged to the upstream.
//     Header mutations are applied on the end_of_stream response.
//   - Accumulate (auto model): eat each chunk (replace with empty body) and
//     run incremental sentence splitting. On end_of_stream, perform the full
//     classification + body mutation pipeline.
type StreamedBodyHandler struct {
	router    *OpenAIRouter
	ctx       *RequestContext
	state     streamedBodyState
	buf       bytes.Buffer
	chunkSize int

	// Fields extracted from first chunk
	model    string
	isStream bool
	isAuto   bool

	// Incremental preprocessing state (accumulate path only)
	preproc *incrementalPreprocessor
}

// incrementalPreprocessor accumulates text and performs sentence splitting at
// fixed-size boundaries so that TextRank scoring on end_of_stream has less
// work to do.
type incrementalPreprocessor struct {
	sentences     []string
	tfVectors     []map[string]float64
	processedUpTo int // byte offset already processed
}

var streamedHandlerPool = sync.Pool{
	New: func() interface{} {
		return &StreamedBodyHandler{
			chunkSize: defaultChunkSize,
		}
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
	h.preproc = nil
	return h
}

// Release returns the handler to the pool for reuse.
func (h *StreamedBodyHandler) Release() {
	h.router = nil
	h.ctx = nil
	h.preproc = nil
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
		h.preproc = &incrementalPreprocessor{}
		logging.Infof("[StreamedBody] Model %q detected as auto — accumulating", h.model)
		return h.handleAccumulate(eos)
	}

	h.state = statePassthrough
	logging.Infof("[StreamedBody] Model %q detected as specified — passthrough", h.model)
	return h.handlePassthrough(eos)
}

// handlePassthrough forwards chunks unchanged. On end_of_stream, performs the
// full body processing pipeline for header mutations (endpoint, auth, etc.)
// and deferred classification.
func (h *StreamedBodyHandler) handlePassthrough(eos bool) (*ext_proc.ProcessingResponse, error) {
	if !eos {
		return continueNoMutation(), nil
	}

	h.ctx.ProcessingStartTime = time.Now()
	h.ctx.OriginalRequestBody = h.buf.Bytes()

	requestBody := h.ctx.OriginalRequestBody
	v := &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{
			Body:        requestBody,
			EndOfStream: true,
		},
	}

	return h.router.handleRequestBody(v, h.ctx)
}

// handleAccumulate eats chunks (replaces with empty body) and performs
// incremental sentence preprocessing. On end_of_stream, runs the full
// classification + body mutation pipeline.
func (h *StreamedBodyHandler) handleAccumulate(eos bool) (*ext_proc.ProcessingResponse, error) {
	if !eos {
		h.runIncrementalPreprocess()
		return continueEmptyBody(), nil
	}

	h.ctx.ProcessingStartTime = time.Now()
	h.ctx.OriginalRequestBody = h.buf.Bytes()

	requestBody := h.ctx.OriginalRequestBody
	v := &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{
			Body:        requestBody,
			EndOfStream: true,
		},
	}

	return h.router.handleRequestBody(v, h.ctx)
}

// runIncrementalPreprocess performs sentence splitting on newly accumulated
// bytes and builds per-sentence TF vectors. This work is deducted from the
// TextRank critical path on end_of_stream.
func (h *StreamedBodyHandler) runIncrementalPreprocess() {
	if h.preproc == nil {
		return
	}
	buf := h.buf.Bytes()
	if len(buf) <= h.preproc.processedUpTo {
		return
	}

	fast, err := extractContentFast(buf)
	if err != nil || fast.UserContent == "" {
		return
	}

	sentences := promptcompression.SplitSentences(fast.UserContent)
	if len(sentences) <= len(h.preproc.sentences) {
		return
	}

	newSentences := sentences[len(h.preproc.sentences):]
	for _, s := range newSentences {
		words := promptcompression.TokenizeWords(s)
		tf := make(map[string]float64, len(words))
		for _, w := range words {
			tf[w]++
		}
		h.preproc.tfVectors = append(h.preproc.tfVectors, tf)
	}
	h.preproc.sentences = sentences
	h.preproc.processedUpTo = len(buf)
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

// continueNoMutation returns a CONTINUE response with no body mutation,
// allowing the chunk to pass through to the upstream unchanged.
func continueNoMutation() *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
				},
			},
		},
	}
}
