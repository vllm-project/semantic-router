package extproc

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"sort"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
)

// semanticResponseStreamState is request-scoped reconstruction state for
// telemetry, replay, memory, and cache policy. It consumes neutral events from
// the same codec contract as buffered responses and never inspects client SSE JSON.
type semanticResponseStreamState struct {
	responseID string
	model      string
	// sawContent marks the first event that carries an answer item, so TTFT
	// and the session's latency keep ignoring keepalive frames.
	sawContent bool
	stop       llmprotocol.StopReason
	stopSeq    string
	usage      llmprotocol.Usage
	items      map[int]*semanticStreamItem
	order      []int
	terminal   bool
	failed     *llmprotocol.ProtocolError
}

type semanticStreamItem struct {
	id               string
	role             llmprotocol.Role
	text             string
	refusal          string
	reasoning        string
	reasoningSig     string
	toolCall         *llmprotocol.ToolCall
	image            *llmprotocol.GeneratedImage
	completed        bool
	reasoningDetails map[int]llmprotocol.ReasoningDetail
}

func (r *OpenAIRouter) handleSemanticStreamingResponseBody(
	responseBody []byte,
	endOfStream bool,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if ctx.ResponseGuardHold != nil {
		return r.handleHeldStreamBody(responseBody, endOfStream, ctx)
	}
	r.initializeSemanticResponseStream(ctx)
	buffers := semanticStreamBuffers{}
	responseBody, endOfStream = r.bridgeBufferedBody(&buffers, responseBody, endOfStream, ctx)
	buffers.push(responseBody, ctx)
	if endOfStream {
		buffers.finalize(ctx)
		r.finalizeSemanticStreamingResponse(ctx, buffers.streamErr)
	}
	// TTFT waits for the first event that carries content: a keepalive or
	// comment frame a gateway sends while it queues is not the model's answer.
	recordStreamingTTFT(ctx)
	return buffers.processingResponse(ctx)
}

// bridgeBufferedBody replaces a buffered provider object with stream events.
// Only a provider body that is not SSE can be a buffered answer. Such a reply
// arrives as one body message, whose end_of_stream is false when trailers
// follow, so a complete object ends the read rather than the flag.
func (r *OpenAIRouter) bridgeBufferedBody(
	buffers *semanticStreamBuffers,
	body []byte,
	endOfStream bool,
	ctx *RequestContext,
) ([]byte, bool) {
	bridged, ok := r.bridgeBufferedResponsesBody(body, ctx)
	if !ok {
		return body, endOfStream
	}
	buffers.bridged = true
	return bridged, true
}

func (r *OpenAIRouter) initializeSemanticResponseStream(ctx *RequestContext) {
	if err := r.ensureSemanticResponseStream(ctx); err != nil {
		ctx.StreamingAborted = true
		logging.ComponentErrorEvent("extproc", "neutral_stream_init_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"error":      err.Error(),
		})
	}
}

type semanticStreamBuffers struct {
	translated []byte
	public     []byte
	errorFrame []byte
	streamErr  error
	// bridged marks a body the Router rewrote from a buffered provider object
	// into stream events, so the client keeps the codec's frames instead of the
	// upstream bytes.
	bridged bool
}

func (buffers *semanticStreamBuffers) push(responseBody []byte, ctx *RequestContext) {
	if len(responseBody) == 0 {
		return
	}
	if ctx.PublicChatUsageFilter != nil {
		filtered, err := ctx.PublicChatUsageFilter.Push(responseBody)
		buffers.public = append(buffers.public, filtered...)
		buffers.recordError(ctx, err, true)
	}
	if ctx.ProtocolResponseStream != nil {
		frames, events, diagnostics, err := ctx.ProtocolResponseStream.Push(responseBody)
		buffers.translated = appendProtocolFrames(buffers.translated, frames)
		if frame := failureFrame(frames, events); frame != nil {
			buffers.errorFrame = frame
		}
		observeProtocolStream(ctx, events, diagnostics)
		buffers.recordError(ctx, err, true)
	}
}

func (buffers *semanticStreamBuffers) finalize(ctx *RequestContext) {
	if ctx.ProtocolResponseStream != nil {
		// A failure the Router raised, in this read or an earlier one, is not the
		// provider's error event, so its frame stays on the abort path.
		routerFailed := buffers.streamErr != nil || ctx.StreamingAborted
		frames, events, diagnostics, err := ctx.ProtocolResponseStream.Finalize(buffers.streamErr)
		buffers.translated = appendProtocolFrames(buffers.translated, frames)
		if frame := failureFrame(frames, events); frame != nil && !routerFailed {
			buffers.errorFrame = frame
		}
		observeProtocolStream(ctx, events, diagnostics)
		buffers.recordError(ctx, err, true)
	}
	if ctx.PublicChatUsageFilter != nil {
		filtered, err := ctx.PublicChatUsageFilter.Finalize()
		buffers.public = append(buffers.public, filtered...)
		buffers.recordError(ctx, err, false)
	}
}

func (buffers *semanticStreamBuffers) recordError(ctx *RequestContext, err error, overwrite bool) {
	if err == nil || !overwrite && buffers.streamErr != nil {
		return
	}
	buffers.streamErr = err
	ctx.StreamingAborted = true
}

// bridgeBufferedResponsesBody renders a buffered provider answer as the event
// stream the client asked for. A provider may ignore stream:true and reply with
// one JSON object; that body carries no data: line for the SSE decoder, so the
// forced streaming path would drop every byte and report stream_incomplete. The
// neutral response is decoded and re-emitted through the same encoder synthetic
// producers and cache replay use.
//
// It only applies to a body the provider did not send as SSE, and only once the
// body is complete: the header phase keeps such an answer out of streamed reads
// so the whole object arrives in one message. A gzip reply is decoded here,
// since the streaming path leaves content coding to the client; the client body
// is encoded again under that header.
func (r *OpenAIRouter) bridgeBufferedResponsesBody(body []byte, ctx *RequestContext) ([]byte, bool) {
	if !isResponseAPIStreamRequest(ctx) || ctx.UpstreamBodyStreamed {
		return nil, false
	}
	if ctx.UpstreamResponseGzip {
		decoded, err := gunzipBounded(ctx.TraceContext, body, maxDecodedUpstreamBodyBytes)
		if err != nil {
			return nil, false
		}
		body = decoded
	}
	if !bytes.HasPrefix(bytes.TrimSpace(body), []byte("{")) || !json.Valid(body) {
		return nil, false
	}
	var probe struct {
		Object string `json:"object"`
	}
	if err := json.Unmarshal(body, &probe); err != nil || probe.Object != "response" {
		return nil, false
	}
	engine, err := r.protocolEngineForBackend(ctx)
	if err != nil {
		return nil, false
	}
	source, target := responseWireFormats(ctx)
	response, _, diagnostics, err := engine.DecodeResponse(source, body)
	if err != nil {
		return nil, false
	}
	frames, encoded, err := engine.EncodeResponseStream(target, response, responseStreamContext(ctx, source, target))
	if err != nil || len(frames) == 0 {
		return nil, false
	}
	// The synthesized stream cannot report what decoding already stripped, so
	// the buffered path's diagnostics are recorded here.
	observeProtocolStream(ctx, nil, append(diagnostics, encoded...))
	return frames, true
}

// encodeGzip encodes frames under the coding the client's content-encoding
// header still names.
func encodeGzip(frames []byte) ([]byte, error) {
	var out bytes.Buffer
	writer := gzip.NewWriter(&out)
	if _, err := writer.Write(frames); err != nil {
		return nil, err
	}
	if err := writer.Close(); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

func appendProtocolFrames(body []byte, frames [][]byte) []byte {
	for _, frame := range frames {
		body = append(body, frame...)
	}
	return body
}

func observeProtocolStream(
	ctx *RequestContext,
	events []llmprotocol.Event,
	diagnostics []llmprotocol.Diagnostic,
) {
	ctx.ProtocolDiagnostics = append(ctx.ProtocolDiagnostics, diagnostics...)
	// Streaming headers have already been emitted. Keep late compatibility
	// warnings observable through the same counter and structured diagnostics
	// as buffered responses instead of retaining them only in request state.
	recordProtocolDiagnostics(ctx, diagnostics)
	ctx.protocolDiagnosticsReported = len(ctx.ProtocolDiagnostics)
	ctx.SemanticStreamState.observe(events)
}

// failureFrame returns the frame the encoder wrote for an upstream failure in
// this batch. The Chat encoder writes a failure as the batch's last frame
// (TestChatStreamFailureEndsItsBatch).
func failureFrame(frames [][]byte, events []llmprotocol.Event) []byte {
	for _, event := range events {
		if event.Type == llmprotocol.EventResponseFailed && len(frames) > 0 {
			return frames[len(frames)-1]
		}
	}
	return nil
}

func (buffers *semanticStreamBuffers) processingResponse(ctx *RequestContext) *ext_proc.ProcessingResponse {
	// A Chat client of a Chat upstream gets the public filter's output up to an
	// upstream error event, then the Router's error frame for it, however the
	// upstream body is split into reads. A client that needs a rewrite keeps the
	// codec's frames.
	if ctx.PublicChatUsageFilter.UpstreamFailed() && len(buffers.errorFrame) > 0 && !requiresClientResponseRewrite(ctx) {
		return buildResponseBodyContinueResponse(&ext_proc.BodyMutation{
			Mutation: &ext_proc.BodyMutation_Body{Body: append(buffers.public, buffers.errorFrame...)},
		}, nil)
	}
	if ctx.ProtocolResponseStream != nil &&
		(requiresClientResponseRewrite(ctx) || ctx.StreamingAborted || buffers.bridged) {
		body := buffers.translated
		// A bridged answer replaces a gzip provider body, and the client's
		// content-encoding still names that coding, so the frames carry it too.
		if buffers.bridged && ctx.UpstreamResponseGzip {
			if encoded, err := encodeGzip(body); err == nil {
				body = encoded
			}
		}
		return buildResponseBodyContinueResponse(&ext_proc.BodyMutation{
			Mutation: &ext_proc.BodyMutation_Body{Body: body},
		}, nil)
	}
	if ctx.PublicChatUsageFilter != nil {
		return buildResponseBodyContinueResponse(&ext_proc.BodyMutation{
			Mutation: &ext_proc.BodyMutation_Body{Body: buffers.public},
		}, nil)
	}
	return buildResponseBodyContinueResponse(nil, nil)
}

func recordStreamingTTFT(ctx *RequestContext) {
	if ctx == nil || ctx.TTFTRecorded || ctx.ProcessingStartTime.IsZero() || ctx.RequestModel == "" {
		return
	}
	if state := ctx.SemanticStreamState; state == nil || !state.sawContent {
		return
	}

	ttft := time.Since(ctx.ProcessingStartTime).Seconds()
	if ttft <= 0 {
		return
	}

	metrics.RecordModelFirstResponseObservation(ctx.RequestModel, ttft)
	ctx.TTFTSeconds = ttft
	ctx.TTFTRecorded = true
	latency.UpdateTTFT(ctx.RequestModel, ttft)
	ctx.CacheWarmthEstimate = latency.EstimateCacheProbability(latency.CacheEstimationInput{
		Model:       ctx.RequestModel,
		TTFTSeconds: ttft,
	})
	maybeEmitTransitionEvent(ctx)
	logging.Debugf("Recorded TTFT on first streamed body chunk: model=%q, TTFT=%.4fs", ctx.RequestModel, ttft)
}

func responseStreamContext(
	ctx *RequestContext,
	source, target llmprotocol.WireFormat,
) llmprotocol.StreamContext {
	streamContext := llmprotocol.StreamContext{
		Context: ctx.TraceContext, Source: source, Target: target,
		Options:     clientStreamOptions(ctx),
		PublicModel: ctx.RequestModel, PreviousResponseID: responseObjectPreviousID(ctx),
		// The model the upstream reports, as buffered and same-format replies pass it through.
		PreferProviderModel: true,
	}
	if responseID := responseObjectPublicID(ctx); responseID != "" {
		streamContext.ResponseID = responseID
	}
	return streamContext
}

func (r *OpenAIRouter) ensureSemanticResponseStream(ctx *RequestContext) error {
	if ctx == nil {
		return fmt.Errorf("request context is unavailable")
	}
	if ctx.ProtocolResponseStream != nil {
		return nil
	}
	engine, err := r.protocolEngineForBackend(ctx)
	if err != nil {
		return err
	}
	source, target := responseWireFormats(ctx)
	streamContext := responseStreamContext(ctx, source, target)
	mutation := clientStreamMutation(ctx, source)
	stream, err := engine.NewStreamWithMutation(source, target, streamContext, mutation)
	if err != nil {
		return err
	}
	ctx.ProtocolResponseStream = stream
	if source == llmprotocol.OpenAIChatV1 && target == llmprotocol.OpenAIChatV1 {
		ctx.PublicChatUsageFilter = protocolcodec.NewChatPublicStreamFilter(
			llmprotocol.DefaultPolicy().Limits.SSEFrameBytes, streamUsageRequestedByClient(ctx),
		)
	}
	ctx.SemanticStreamState = &semanticResponseStreamState{
		usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
		items: make(map[int]*semanticStreamItem),
	}
	return nil
}

func streamUsageRequestedByClient(ctx *RequestContext) bool {
	options := clientStreamOptions(ctx)
	return options.IncludeUsage != nil && *options.IncludeUsage
}

//nolint:gocognit,cyclop,funlen // Observation is an exhaustive reducer over the closed neutral event contract.
func (state *semanticResponseStreamState) observe(events []llmprotocol.Event) {
	if state == nil {
		return
	}
	for _, event := range events {
		switch event.Type {
		case llmprotocol.EventOutputItemStarted, llmprotocol.EventOutputTextDelta, llmprotocol.EventReasoningDelta, llmprotocol.EventToolCallDelta:
			state.sawContent = true
		}
		if event.ResponseID != "" {
			state.responseID = event.ResponseID
		}
		if event.Model != "" {
			state.model = event.Model
		}
		if event.Usage != nil {
			state.usage = *event.Usage
		}
		if event.StopReason != "" {
			state.stop = event.StopReason
		}
		switch event.Type {
		case llmprotocol.EventOutputItemStarted:
			item := state.item(event.ItemIndex)
			item.id = event.ItemID
			item.role = event.Role
			if event.ToolCall != nil {
				call := *event.ToolCall
				item.toolCall = &call
			}
			if event.Content != nil && event.Content.Kind == llmprotocol.ContentReasoning {
				item.reasoningSig = event.Content.Signature
			}
		case llmprotocol.EventOutputTextDelta:
			item := state.item(event.ItemIndex)
			if event.Content != nil && event.Content.Kind == llmprotocol.ContentRefusal {
				item.refusal += event.Delta
			} else {
				item.text += event.Delta
			}
		case llmprotocol.EventReasoningDelta:
			item := state.item(event.ItemIndex)
			if event.Content != nil && event.Content.ReasoningDetail != nil {
				if item.reasoningDetails == nil {
					item.reasoningDetails = make(map[int]llmprotocol.ReasoningDetail)
				}
				detail := *event.Content.ReasoningDetail
				if current, found := item.reasoningDetails[event.ContentIndex]; found {
					merged, err := llmprotocol.MergeReasoningDetail(current, detail)
					if err != nil {
						state.failed = llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "reasoning_detail_identity_mismatch", "upstream reasoning detail changed identity", err)
						continue
					}
					detail = merged
				}
				item.reasoningDetails[event.ContentIndex] = detail
				continue
			}
			item.reasoning += event.Delta
			if event.Content != nil && event.Content.Signature != "" {
				item.reasoningSig = event.Content.Signature
			}
		case llmprotocol.EventToolCallDelta:
			item := state.item(event.ItemIndex)
			if item.toolCall == nil {
				item.toolCall = &llmprotocol.ToolCall{}
			}
			if event.ToolCall != nil {
				if event.ToolCall.Kind != "" {
					item.toolCall.Kind = event.ToolCall.Kind
				}
				if event.ToolCall.ID != "" {
					item.toolCall.ID = event.ToolCall.ID
				}
				if event.ToolCall.Name != "" {
					item.toolCall.Name = event.ToolCall.Name
				}
				if event.ToolCall.Namespace != "" {
					item.toolCall.Namespace = event.ToolCall.Namespace
				}
				item.toolCall.Arguments += event.ToolCall.Arguments
			}
		case llmprotocol.EventOutputItemCompleted:
			item := state.item(event.ItemIndex)
			item.completed = true
			if event.ToolCall != nil {
				call := *event.ToolCall
				item.toolCall = &call
			}
			if event.Content != nil && event.Content.GeneratedImage != nil {
				image := *event.Content.GeneratedImage
				item.image = &image
			}
		case llmprotocol.EventResponseCompleted:
			state.terminal = true
			state.stopSeq = event.MatchedStopSequence
		case llmprotocol.EventResponseFailed:
			state.terminal = true
			state.failed = event.Error
		}
	}
}

func (state *semanticResponseStreamState) item(index int) *semanticStreamItem {
	item := state.items[index]
	if item != nil {
		return item
	}
	item = &semanticStreamItem{role: llmprotocol.RoleAssistant}
	state.items[index] = item
	state.order = append(state.order, index)
	return item
}

//nolint:cyclop // Response assembly validates every terminal stream invariant in one place.
func (state *semanticResponseStreamState) response() (*llmprotocol.Response, error) {
	if state == nil || state.failed != nil || !state.terminal {
		return nil, fmt.Errorf("semantic stream did not complete successfully")
	}
	indices := append([]int(nil), state.order...)
	sort.SliceStable(indices, func(left, right int) bool { return indices[left] < indices[right] })
	output := make([]llmprotocol.OutputItem, 0, len(indices))
	for _, index := range indices {
		item := state.items[index]
		if item == nil || !item.completed {
			return nil, fmt.Errorf("semantic stream output item is incomplete")
		}
		contents := make([]llmprotocol.Content, 0, 4)
		detailIndices := make([]int, 0, len(item.reasoningDetails))
		for contentIndex := range item.reasoningDetails {
			detailIndices = append(detailIndices, contentIndex)
		}
		sort.Ints(detailIndices)
		for _, contentIndex := range detailIndices {
			detail := item.reasoningDetails[contentIndex]
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentReasoning, ReasoningDetail: &detail})
		}
		if item.reasoning != "" || item.reasoningSig != "" {
			contents = append(contents, llmprotocol.Content{
				Kind: llmprotocol.ContentReasoning, Text: item.reasoning,
				Signature: item.reasoningSig,
			})
		}
		if item.refusal != "" {
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: item.refusal})
		}
		if item.text != "" {
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentText, Text: item.text})
		}
		if item.toolCall != nil {
			call := *item.toolCall
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: &call})
		}
		if item.image != nil {
			image := *item.image
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentGeneratedImage, GeneratedImage: &image})
		}
		if len(contents) == 0 {
			return nil, fmt.Errorf("semantic stream output item is empty")
		}
		itemID := item.id
		if itemID == "" {
			itemID = fmt.Sprintf("item_%d", index)
		}
		output = append(output, llmprotocol.OutputItem{ID: itemID, Role: item.role, Content: contents})
	}
	if state.stop == "" {
		state.stop = llmprotocol.StopUnknown
	}
	return &llmprotocol.Response{
		Generation: 1, ID: state.responseID, CreatedAt: time.Now().UTC(),
		Model: state.model, Output: output, StopReason: state.stop,
		MatchedStopSequence: state.stopSeq, Usage: state.usage,
	}, nil
}

func (r *OpenAIRouter) finalizeSemanticStreamingResponse(ctx *RequestContext, streamErr error) {
	if semanticResponse := r.completeSemanticStream(ctx, streamErr); semanticResponse != nil {
		r.commitStreamedAnswer(ctx, semanticResponse)
	}
}

// completeSemanticStream settles a stream once it ended: its latency, usage
// and inflight admission. It returns the answer the stream reconstructs, or
// nil when the stream ended without one or was settled before.
func (r *OpenAIRouter) completeSemanticStream(ctx *RequestContext, streamErr error) *llmprotocol.Response {
	if ctx == nil || ctx.StreamingComplete {
		return nil
	}
	ctx.StreamingComplete = true
	if streamErr != nil || ctx.SemanticStreamState == nil || !ctx.SemanticStreamState.terminal {
		ctx.StreamingAborted = true
	}
	semanticResponse, responseErr := ctx.SemanticStreamState.response()
	if responseErr == nil {
		ctx.SemanticResponse = semanticResponse
	} else {
		ctx.StreamingAborted = true
	}
	completionLatency := time.Duration(0)
	if !ctx.StartTime.IsZero() {
		completionLatency = time.Since(ctx.StartTime)
		if ctx.RequestModel != "" {
			metrics.RecordModelCompletionLatency(ctx.RequestModel, completionLatency.Seconds())
		}
	}
	inflight.End(ctx.InflightModel, ctx.InflightToken)
	ctx.InflightToken = 0

	usage := r.takeNeutralResponseUsage(ctx)
	// Decoded stream events carry the client-facing model, not the provider's.
	observeUpstreamResponse(ctx, "", usage)
	r.reportSemanticStreamingUsage(ctx, completionLatency, usage)
	r.calibrateTokenEstimator(ctx, usage.promptTokens)

	if responseErr != nil {
		logging.ComponentWarnEvent("extproc", "neutral_stream_reconstruction_skipped", map[string]interface{}{
			"request_id": ctx.RequestID,
			"error":      responseErr.Error(),
		})
		r.recordUnscheduledResponseMemoryStore(ctx, "skipped", "stream_incomplete", true)
		return nil
	}
	return semanticResponse
}

// commitStreamedAnswer keeps an answer the client received as it streamed:
// the selected decision's response_guard, which neither blocks nor writes
// notes, reads it and records what it found, and the answer reaches the
// cache, memory, the Responses object and Router Replay. An answer the guard
// could not read for want of the request's delegated credential reaches
// Router Replay only.
func (r *OpenAIRouter) commitStreamedAnswer(ctx *RequestContext, semanticResponse *llmprotocol.Response) {
	recordPrimaryOutputDigest(ctx, semanticResponse)
	ctx.ResponseGuard = r.runResponseGuard(ctx, semanticAssistantContent(semanticResponse))
	if ctx.ResponseGuard != nil {
		ctx.ResponseGuard.live = true
	}
	r.recordRouterReplayResponseGuard(ctx)
	encoded, err := r.encodeClientResponse(*semanticResponse, ctx)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "neutral_stream_replay_encode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"error":      err.Error(),
		})
		r.recordUnscheduledResponseMemoryStore(ctx, "skipped", "stream_encode_failed", true)
		return
	}
	r.updateKVAddressRegistry(ctx)
	if principal.Refused(ctx.TraceContext) {
		// The client already has the answer, so it is not refused, but
		// nothing keeps an answer nothing checked.
		logging.ComponentWarnEvent("extproc", "streamed_answer_not_kept", map[string]interface{}{
			"request_id": ctx.RequestID,
			"reason":     delegatedCredentialCode,
		})
		metrics.RecordCacheWriteSkipped(delegatedCredentialCode)
		r.recordUnscheduledResponseMemoryStore(ctx, "policy_blocked", delegatedCredentialCode, false)
	} else {
		r.updateResponseCache(ctx, encoded)
		r.scheduleSemanticResponseMemoryStore(ctx, semanticResponse)
		r.persistResponseObject(ctx)
	}
	r.attachRouterReplayResponse(ctx, encoded, true)
}

func (r *OpenAIRouter) reportSemanticStreamingUsage(
	ctx *RequestContext,
	completionLatency time.Duration,
	usage responseUsageMetrics,
) {
	if ctx == nil {
		return
	}
	recordSessionTurnOutcome(ctx, usage, r.sessionTurnPricing(ctx.RequestModel))
	if usage.invalid {
		r.updateRouterReplayUsageCost(ctx, r.buildReplayUsageCost(ctx, usage))
		return
	}
	totalTokens := responseUsageTotal(usage)
	if r.RateLimiter != nil && ctx.RateLimitCtx != nil && totalTokens > 0 {
		r.RateLimiter.Report(*ctx.RateLimitCtx, ratelimit.TokenUsage{
			InputTokens:  usage.promptTokens,
			OutputTokens: usage.completionTokens,
			TotalTokens:  totalTokens,
		})
	}
	if totalTokens > 0 {
		recordSessionTurn(ctx, usage, r.sessionTurnPricing(ctx.RequestModel))
	}
	if ctx.RequestModel == "" {
		return
	}
	recordModelUsageTokens(ctx.RequestModel, usage)
	metrics.RecordModelWindowedRequest(
		ctx.RequestModel,
		completionLatency.Seconds(),
		int64(usage.promptTokens),
		int64(usage.completionTokens),
	)
	if usage.completionTokens > 0 && completionLatency > 0 {
		timePerToken := completionLatency.Seconds() / float64(usage.completionTokens)
		metrics.RecordModelResponseDurationPerOutputToken(ctx.RequestModel, timePerToken)
		latency.UpdateTPOT(ctx.RequestModel, timePerToken)
	}
	replayUsage := r.recordResponseCost(ctx, completionLatency, usage)
	r.updateRouterReplayUsageCost(ctx, replayUsage)
	r.observeRouterLearningUsageTelemetry(ctx, completionLatency, usage, replayUsage)
}
