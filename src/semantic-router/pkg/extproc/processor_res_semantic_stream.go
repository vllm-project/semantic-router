package extproc

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// semanticResponseStreamState is request-scoped reconstruction state for
// telemetry, replay, memory, and cache policy. It consumes neutral events from
// the same codec contract as buffered responses and never inspects client SSE JSON.
type semanticResponseStreamState struct {
	responseID string
	model      string
	// sawContent marks the first model content, including a choice held by
	// the decoder for ordering. It ignores keepalive and metadata frames.
	sawContent bool
	createdAt  time.Time
	stop       llmprotocol.StopReason
	stopSeq    string
	usage      llmprotocol.Usage
	items      map[int]*semanticStreamItem
	order      []int
	terminal   bool
	failed     *llmprotocol.ProtocolError
	// decoded reports that the stream decoder read at least one event.
	decoded bool
}

type semanticStreamItem struct {
	id               string
	role             llmprotocol.Role
	text             string
	refusal          string
	reasoning        string
	reasoningSig     string
	reasoningScope   llmprotocol.ReasoningScope
	toolCall         *llmprotocol.ToolCall
	image            *llmprotocol.GeneratedImage
	completed        bool
	reasoningDetails map[int]llmprotocol.ReasoningDetail
	// serverTool and citations are what an Anthropic server tool left: its
	// call or result as written, and the citations on this item's text.
	serverTool json.RawMessage
	citations  json.RawMessage
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
	if endOfStream && buffers.unread(ctx) {
		// The client gets what a buffered client gets for such a body.
		return r.invalidUpstreamResponse(ctx, "The selected model returned an invalid response")
	}
	// TTFT waits for the first upstream content: a keepalive or comment
	// frame a gateway sends while it queues is not the model's answer.
	// It is recorded before the stream settles, because the turn's session
	// evidence is written there and takes the latency it has by then.
	recordStreamingTTFT(ctx)
	if endOfStream {
		buffers.finalize(ctx)
		r.finalizeSemanticStreamingResponse(ctx, buffers.streamErr)
	}
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
	if ctx.PublicChatUsageFilter != nil {
		// An empty final read still marks the boundary for a codec failure
		// discovered while finalizing an unterminated frame.
		filtered, err := ctx.PublicChatUsageFilter.Push(responseBody)
		buffers.public = append(buffers.public, filtered...)
		buffers.recordError(ctx, err, true)
	}
	if len(responseBody) != 0 && ctx.ProtocolResponseStream != nil {
		frames, events, diagnostics, err := ctx.ProtocolResponseStream.Push(responseBody)
		buffers.translated = appendProtocolFrames(buffers.translated, frames)
		if frame := failureFrame(frames, events); frame != nil {
			buffers.errorFrame = frame
		}
		observeProtocolStream(ctx, events, diagnostics)
		if state := ctx.SemanticStreamState; err == nil && state != nil && ctx.ProtocolResponseStream.HasPendingContent() {
			state.sawContent = true
		}
		buffers.recordError(ctx, err, true)
	}
}

// unread reports a body the provider answered whole, which its client has not
// seen yet, that neither the bridge nor the stream decoder could read
// anything from.
func (buffers *semanticStreamBuffers) unread(ctx *RequestContext) bool {
	state := ctx.SemanticStreamState
	return !ctx.UpstreamBodyStreamed && !buffers.bridged && ctx.ProtocolResponseStream != nil && state != nil && !state.decoded
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

func estimateAbortedLooperOutputTokens(state *semanticResponseStreamState) int {
	if state == nil || len(state.items) == 0 {
		return 0
	}
	var b strings.Builder
	for _, item := range state.items {
		if item == nil {
			continue
		}
		if item.text != "" {
			b.WriteString(item.text)
		}
		if item.reasoning != "" {
			b.WriteString(item.reasoning)
		}
		if item.refusal != "" {
			b.WriteString(item.refusal)
		}
		if item.toolCall != nil {
			if item.toolCall.Name != "" {
				b.WriteString(item.toolCall.Name)
			}
			if item.toolCall.Arguments != "" {
				b.WriteString(item.toolCall.Arguments)
			}
		}
	}
	combined := b.String()
	if combined == "" {
		return 0
	}
	return contextcompression.EstimateTokens(combined)
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
		// A stream failure switches a same-format client from the public
		// filter's bytes to the codec's frames, which carry only the failing
		// read's output. The tool-argument frames the filter held back in
		// earlier reads must ride along, or the client loses the accepted
		// prefix of the call the provider went on to break.
		if ctx.StreamingAborted && !requiresClientResponseRewrite(ctx) {
			body = append(ctx.PublicChatUsageFilter.ReleaseHeldForAbort(), body...)
		}
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

	ctx.TTFTSeconds = ttft
	ctx.TTFTRecorded = true
	if !ctx.LooperStreamingActive {
		metrics.RecordModelFirstResponseObservation(ctx.RequestModel, ttft)
		latency.UpdateTTFT(ctx.RequestModel, ttft)
	}
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
		PublicModel: ctx.RequestModel, ResponseRender: responseObjectRender(ctx),
		// The model the upstream reports, as buffered and same-format replies pass it through.
		PreferProviderModel: true,
	}
	if responseID := responseObjectPublicID(ctx); responseID != "" {
		streamContext.ResponseID = responseID
	}
	// Chat and Responses providers report usage only at the end, after the
	// message_start a Messages client reads input usage from.
	if target == llmprotocol.AnthropicMessagesV1 && source != llmprotocol.AnthropicMessagesV1 && ctx.SemanticRequest != nil {
		streamContext.EstimatedInputTokens = int64(estimateCountTokens(ctx.SemanticRequest))
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
		ctx.PublicChatUsageFilter = engine.NewChatPublicStreamFilter(streamUsageRequestedByClient(ctx))
		if ctx.LooperStreamingActive && ctx.RequestModel != "" {
			ctx.PublicChatUsageFilter.SetPublicModel(ctx.RequestModel)
		}
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
	state.decoded = state.decoded || len(events) > 0
	for _, event := range events {
		state.sawContent = state.sawContent || streamEventCarriesContent(event)
		if event.ResponseID != "" {
			state.responseID = event.ResponseID
		}
		if event.Model != "" {
			state.model = event.Model
		}
		if state.createdAt.IsZero() {
			state.createdAt = event.CreatedAt
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
				item.reasoningScope = event.Content.Reasoning
			}
			if event.Content != nil && event.Content.Kind == llmprotocol.ContentServerTool {
				item.serverTool = event.Content.Raw
			}
		case llmprotocol.EventOutputTextDelta:
			item := state.item(event.ItemIndex)
			if event.Content != nil {
				item.citations = llmprotocol.AppendCitations(item.citations, event.Content.Raw)
			}
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
			if event.Content != nil {
				if event.Content.Signature != "" {
					item.reasoningSig = event.Content.Signature
				}
				if event.Content.Reasoning != "" {
					item.reasoningScope = event.Content.Reasoning
				}
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

// streamEventCarriesContent recognizes the first model output for TTFT.
func streamEventCarriesContent(event llmprotocol.Event) bool {
	switch {
	case event.Delta != "":
		return true
	case event.ToolCall != nil:
		call := event.ToolCall
		return call.ID != "" || call.Name != "" || call.Arguments != ""
	case event.GeneratedImage != nil:
		return event.GeneratedImage.PartialImage != ""
	default:
		return streamBlockCarriesContent(event.Content)
	}
}

func streamBlockCarriesContent(block *llmprotocol.Content) bool {
	if block == nil {
		return false
	}
	switch block.Kind {
	case llmprotocol.ContentReasoning:
		return block.Text != "" || block.ReasoningDetail != nil && block.ReasoningDetail.PayloadBytes() > 0
	case llmprotocol.ContentServerTool:
		return len(block.Raw) > 0
	case llmprotocol.ContentGeneratedImage:
		image := block.GeneratedImage
		return image != nil && image.Result != nil && *image.Result != ""
	default:
		return block.Text != ""
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

// delivered reports whether the stream reached its terminal event without a
// failure, which makes it a whole answer for its client.
func (state *semanticResponseStreamState) delivered() bool {
	return state != nil && state.failed == nil && state.terminal
}

// streamFailureSummary names a protocol failure by its normalized category and,
// for an error the Router authored, its code. A decoder copies a provider
// frame's code, message and parameter verbatim, and those can echo request
// content, so a provider-framed error contributes only its category. Only a
// Router-authored ErrorDiagnostic is printed from the cause chain: a parser
// error's text can name provider fields. The client keeps the failure it always
// saw.
func streamFailureSummary(err *llmprotocol.ProtocolError) string {
	summary := "category=" + string(err.Category)
	if err.ProviderFramed {
		return summary
	}
	if err.Code != "" {
		summary += ", code=" + err.Code
	}
	var diagnostic llmprotocol.ErrorDiagnostic
	if errors.As(err, &diagnostic) {
		summary += "; " + string(diagnostic)
	}
	return summary
}

//nolint:cyclop // Response assembly validates every terminal stream invariant in one place.
func (state *semanticResponseStreamState) response() (*llmprotocol.Response, error) {
	if !state.delivered() {
		if state != nil && state.failed != nil {
			return nil, fmt.Errorf("semantic stream did not complete successfully: %s", streamFailureSummary(state.failed))
		}
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
				Signature: item.reasoningSig, Reasoning: item.reasoningScope,
			})
		}
		if item.refusal != "" {
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentRefusal, Text: item.refusal})
		}
		if item.text != "" {
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentText, Text: item.text, Raw: item.citations})
		}
		if item.serverTool != nil {
			contents = append(contents, llmprotocol.Content{Kind: llmprotocol.ContentServerTool, Raw: item.serverTool})
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
	createdAt := state.createdAt
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	}
	return &llmprotocol.Response{
		Generation: 1, ID: state.responseID, CreatedAt: createdAt,
		Model: state.model, Output: output, StopReason: state.stop,
		MatchedStopSequence: state.stopSeq, Usage: state.usage,
	}, nil
}

func (r *OpenAIRouter) finalizeSemanticStreamingResponse(ctx *RequestContext, streamErr error) {
	clean := streamErr == nil && !ctx.StreamingAborted
	if semanticResponse := r.completeSemanticStream(ctx, streamErr); semanticResponse != nil {
		if !ctx.StreamingAborted {
			r.finishNativeFallbackResponse(ctx, "")
		}
		r.commitStreamedAnswer(ctx, semanticResponse)
	}
	r.endStreamRouterReplay(ctx, clean)
}

// completeSemanticStream settles a stream once it ended: its latency, usage
// and inflight admission. It returns the answer the stream reconstructs, or
// nil when the stream ended without one or was settled before.
func (r *OpenAIRouter) completeSemanticStream(ctx *RequestContext, streamErr error) *llmprotocol.Response {
	return r.completeSemanticStreamWith(ctx, streamErr, nil)
}

// completeSemanticStreamWith settles a stream as completeSemanticStream does,
// with the answer replace makes of the whole answer the stream reconstructs
// in place of that answer, when replace is set.
func (r *OpenAIRouter) completeSemanticStreamWith(
	ctx *RequestContext, streamErr error, replace func(*llmprotocol.Response) *llmprotocol.Response,
) *llmprotocol.Response {
	if ctx == nil || ctx.StreamingComplete {
		return nil
	}
	ctx.StreamingComplete = true
	if streamErr != nil || ctx.SemanticStreamState == nil || !ctx.SemanticStreamState.terminal {
		ctx.StreamingAborted = true
	}
	semanticResponse, responseErr := ctx.SemanticStreamState.response()
	if responseErr == nil {
		if ctx.SourceFormat == llmprotocol.AnthropicMessagesV1 {
			// The answer the Router keeps carries the identifiers the client read.
			source, _ := responseWireFormats(ctx)
			*semanticResponse = protocolcodec.AnthropicReplyIdentifiers(source, *semanticResponse)
		}
		ctx.SemanticResponse = semanticResponse
		if replace != nil && !ctx.StreamingAborted {
			semanticResponse = replace(semanticResponse)
		}
	} else {
		ctx.StreamingAborted = true
	}
	completionLatency := time.Duration(0)
	if !ctx.StartTime.IsZero() {
		completionLatency = time.Since(ctx.StartTime)
		if !ctx.LooperStreamingActive && ctx.RequestModel != "" {
			metrics.RecordModelCompletionLatency(ctx.RequestModel, completionLatency.Seconds())
		}
	}
	inflight.End(ctx.InflightModel, ctx.InflightToken)
	ctx.InflightToken = 0

	usage := r.takeNeutralResponseUsage(ctx)
	// The buffered path records the response's model; the stream reconstructs
	// the same answer, so the span names the model it observed too. A stream
	// that never initialised (unsupported source format) still settles here.
	responseModel := ""
	if state := ctx.SemanticStreamState; state != nil {
		responseModel = state.model
	}
	observeUpstreamResponse(ctx, responseModel, usage)
	r.reportSemanticStreamingUsage(ctx, completionLatency, usage)
	if ctx.LooperStreamingActive {
		if !usage.invalid {
			ctx.LooperUsageSettled = true
			logging.ComponentEvent("extproc", "looper_streaming_completed", map[string]interface{}{
				"request_id":        ctx.RequestID,
				"decision":          ctx.VSRSelectedDecisionName,
				"algorithm":         ctx.VSRSelectionMethod,
				"models_used":       ctx.LooperModelsUsed,
				"iterations":        ctx.LooperIterations,
				"selected_model":    firstNonEmpty(ctx.LooperTargetModel, ctx.VSRSelectedModel),
				"prompt_tokens":     usage.promptTokens,
				"completion_tokens": usage.completionTokens,
				"total_tokens":      responseUsageTotal(usage),
				"trace_recorded":    ctx.LooperExecutionTrace != nil,
			})
		} else {
			abortErr := streamErr
			if abortErr == nil {
				abortErr = responseErr
			}
			r.settleAbortedLooperStreamingUsage(ctx, abortErr)
		}
	}

	calibrationPromptTokens := usage.promptTokens
	if ctx.LooperStreamingActive {
		calibrationPromptTokens = 0
		if ctx.SemanticResponse != nil {
			synthesisUsage := responseUsageFromSemanticUsage(ctx.SemanticResponse.Usage)
			if !synthesisUsage.invalid {
				calibrationPromptTokens = synthesisUsage.promptTokens
			}
		}
	}
	r.calibrateTokenEstimatorForModel(
		ctx,
		calibrationTargetModel(ctx),
		streamingCalibrationByteLen(ctx),
		calibrationPromptTokens,
	)

	if ctx.LooperStreamingActive && (responseErr != nil || usage.invalid) {
		ctx.StreamingAborted = true
		r.finalizeRouterReplay(ctx, routerreplay.LifecycleFailed, "upstream_stream_incomplete")
	}

	if responseErr != nil {
		observeStreamedApprovalReviewFailure(ctx)
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
	observeApprovalReviewVerdict(ctx, semanticResponse)
	observeSessionTitle(ctx, semanticResponse)
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
		if !ctx.LooperStreamingActive {
			r.updateResponseCache(ctx, encoded)
		}
		r.scheduleSemanticResponseMemoryStore(ctx, semanticResponse)
		r.persistResponseObject(ctx)
	}
	r.attachRouterReplayResponse(ctx, attachLooperTraceToReplaySnapshot(ctx, encoded), true)
}

func streamingCalibrationByteLen(ctx *RequestContext) int {
	if ctx == nil {
		return 0
	}
	if ctx.LooperStreamingActive {
		if ctx.LooperSynthesisTextBytes > 0 {
			return ctx.LooperSynthesisTextBytes
		}
		return 0
	}
	return tokenCalibrationByteLen(ctx)
}

func attachLooperTraceToReplaySnapshot(ctx *RequestContext, encoded []byte) []byte {
	if ctx == nil || !ctx.LooperStreamingActive || ctx.LooperExecutionTrace == nil || len(encoded) == 0 {
		return encoded
	}
	traceJSON, err := json.Marshal(ctx.LooperExecutionTrace)
	if err != nil || len(traceJSON) == 0 || string(traceJSON) == "null" {
		return encoded
	}
	field := "looper_trace"
	switch ctx.VSRSelectionMethod {
	case config.DecisionAlgorithmFusion:
		field = "fusion"
	case config.DecisionAlgorithmWorkflows:
		field = "flow"
	}
	return restoreLooperFieldJSON(encoded, traceJSON, field)
}

// reportStreamingAnswerEstimate debits a stream that carried no usage from its
// rate-limit budget, with the provider-calibrated estimate of the answer the
// stream reconstructed. The metrics stay unreported - nothing measured them -
// but a token budget must not treat the output as free.
func (r *OpenAIRouter) reportStreamingAnswerEstimate(ctx *RequestContext, response *llmprotocol.Response) int {
	if r == nil || ctx == nil || r.RateLimiter == nil || ctx.RateLimitCtx == nil || response == nil {
		return 0
	}
	bytes := 0
	for _, item := range response.Output {
		for _, content := range item.Content {
			bytes += len(content.Text)
		}
	}
	if bytes == 0 {
		return 0
	}
	tokens, _ := r.classifierForRequest(ctx).EstimateTokens(tokenCalibrationCategory(ctx), bytes)
	if tokens <= 0 {
		return 0
	}
	r.RateLimiter.Report(*ctx.RateLimitCtx, ratelimit.TokenUsage{OutputTokens: tokens, TotalTokens: tokens})
	return tokens
}

func (r *OpenAIRouter) reportSemanticStreamingUsage(
	ctx *RequestContext,
	completionLatency time.Duration,
	usage responseUsageMetrics,
) {
	if ctx == nil {
		return
	}
	pricing := r.sessionTurnPricing(ctx.RequestModel)
	if ctx.LooperStreamingActive {
		pricing = sessiontelemetry.TurnPricing{}
	}
	recordSessionTurnOutcome(ctx, usage, pricing)
	if usage.invalid {
		if r.recordInterruptedStreamUsage(ctx) {
			// Request finalization settles the partial charge.
			return
		}
		// The stream carried no usable usage, so the request is settled with
		// what it knows, as the buffered path does: skipping the report would
		// leave its output tokens out of a token budget. What it knows is an
		// estimate of the answer it reconstructed.
		r.updateRouterReplayUsageCost(ctx, r.buildReplayUsageCost(ctx, usage))
		r.reportStreamingAnswerEstimate(ctx, ctx.SemanticResponse)
		usage = responseUsageMetrics{}
	}
	totalTokens := responseUsageTotal(usage)
	if r.RateLimiter != nil && ctx.RateLimitCtx != nil {
		r.RateLimiter.Report(*ctx.RateLimitCtx, ratelimit.TokenUsage{
			InputTokens:  usage.promptTokens,
			OutputTokens: usage.completionTokens,
			TotalTokens:  totalTokens,
		})
	}
	if totalTokens > 0 {
		recordSessionTurn(ctx, usage, pricing)
	}
	if ctx.RequestModel == "" {
		return
	}
	if !ctx.LooperStreamingActive {
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
	}
	replayUsage := r.recordResponseCost(ctx, completionLatency, usage)
	r.settleRequestSpend(ctx, &replayUsage)
	r.updateRouterReplayUsageCost(ctx, replayUsage)
	r.observeRouterLearningUsageTelemetry(ctx, completionLatency, usage, replayUsage)
}
