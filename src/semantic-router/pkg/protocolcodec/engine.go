package protocolcodec

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type (
	RequestMutation  func(*llmprotocol.Request) error
	ResponseMutation func(*llmprotocol.Response) error
)

type RequestResult struct {
	Request     llmprotocol.Request
	Envelope    llmprotocol.Envelope
	Body        []byte
	Diagnostics llmprotocol.Diagnostics
}

type ResponseResult struct {
	Response    llmprotocol.Response
	Envelope    llmprotocol.Envelope
	Body        []byte
	Diagnostics llmprotocol.Diagnostics
}

type Engine struct {
	registry *Registry
	policy   llmprotocol.Policy
}

func NewEngine(registry *Registry, policy llmprotocol.Policy) (*Engine, error) {
	if registry == nil || len(registry.codecs) == 0 {
		return nil, fmt.Errorf("codec registry is required")
	}
	if err := validatePolicy(policy); err != nil {
		return nil, err
	}
	return &Engine{registry: registry, policy: policy}, nil
}

func NewBuiltinEngine() *Engine {
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		panic(err)
	}
	return engine
}

func (engine *Engine) Registry() *Registry {
	if engine == nil {
		return nil
	}
	return engine.registry
}

func (engine *Engine) DecodeRequest(format llmprotocol.WireFormat, body []byte) (llmprotocol.Request, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	pair, err := engine.codec(format)
	if err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	request, envelope, diagnostics, err := pair.buffered.DecodeRequest(body, engine.policy)
	if err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, diagnostics, err
	}
	llmprotocol.MarkDeferredToolLinks(&request)
	if err := llmprotocol.ValidateRequest(request, engine.policy.Limits); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, diagnostics, err
	}
	return request, envelope, diagnostics, nil
}

// EncodeRequest validates and encodes an already-neutral request. Callers that
// mutate semantic state do not need to manufacture an intermediate wire body.
func (engine *Engine) EncodeRequest(format llmprotocol.WireFormat, request llmprotocol.Request, envelope llmprotocol.Envelope) (RequestResult, error) {
	pair, encodeRequestErr := engine.codec(format)
	if encodeRequestErr != nil {
		return RequestResult{Request: request, Envelope: envelope}, encodeRequestErr
	}
	if err := llmprotocol.ValidateRequest(request, engine.policy.Limits); err != nil {
		return RequestResult{Request: request, Envelope: envelope}, err
	}
	if err := llmprotocol.RequireCapabilities(format, pair.buffered.Capabilities(), llmprotocol.RequiredCapabilities(request)); err != nil {
		return RequestResult{Request: request, Envelope: envelope}, err
	}
	body, diagnostics, encodeRequestErr := pair.buffered.EncodeRequest(request, envelope, engine.policy)
	return RequestResult{Request: request, Envelope: envelope, Body: body, Diagnostics: diagnostics}, encodeRequestErr
}

// EncodeResponse validates and encodes an already-neutral response. Settlement
// retains the semantic response independently from client representation.
func (engine *Engine) EncodeResponse(format llmprotocol.WireFormat, response llmprotocol.Response, envelope llmprotocol.Envelope) (ResponseResult, error) {
	pair, encodeResponseErr := engine.codec(format)
	if encodeResponseErr != nil {
		return ResponseResult{Response: response, Envelope: envelope}, encodeResponseErr
	}
	if err := llmprotocol.ValidateResponse(response, engine.policy.Limits); err != nil {
		return ResponseResult{Response: response, Envelope: envelope}, err
	}
	if err := llmprotocol.RequireCapabilities(format, pair.buffered.Capabilities(), llmprotocol.RequiredResponseCapabilities(response)); err != nil {
		return ResponseResult{Response: response, Envelope: envelope}, err
	}
	body, diagnostics, encodeResponseErr := pair.buffered.EncodeResponse(response, envelope, engine.policy)
	return ResponseResult{Response: response, Envelope: envelope, Body: body, Diagnostics: diagnostics}, encodeResponseErr
}

func (engine *Engine) TranslateRequest(source, target llmprotocol.WireFormat, body []byte, mutate RequestMutation) (RequestResult, error) {
	decodePolicy := engine.translationDecodePolicy(source, target, mutate != nil)
	sourcePair, translateRequestErr := engine.codec(source)
	if translateRequestErr != nil {
		return RequestResult{}, translateRequestErr
	}
	request, envelope, diagnostics, translateRequestErr := sourcePair.buffered.DecodeRequest(body, decodePolicy)
	if translateRequestErr != nil {
		return RequestResult{Diagnostics: diagnostics}, translateRequestErr
	}
	llmprotocol.MarkDeferredToolLinks(&request)
	if err := llmprotocol.ValidateRequest(request, engine.policy.Limits); err != nil {
		return RequestResult{Request: request, Envelope: envelope, Diagnostics: diagnostics}, err
	}
	targetPair, translateRequestErr := engine.codec(target)
	if translateRequestErr != nil {
		return RequestResult{Request: request, Envelope: envelope, Diagnostics: diagnostics}, translateRequestErr
	}
	if mutate != nil {
		if err := mutate(&request); err != nil {
			return RequestResult{Request: request, Envelope: envelope, Diagnostics: diagnostics}, err
		}
		request.Generation++
	}
	if err := llmprotocol.ValidateRequest(request, engine.policy.Limits); err != nil {
		return RequestResult{Request: request, Envelope: envelope, Diagnostics: diagnostics}, err
	}
	if err := llmprotocol.RequireCapabilities(target, targetPair.buffered.Capabilities(), llmprotocol.RequiredCapabilities(request)); err != nil {
		return RequestResult{Request: request, Envelope: envelope, Diagnostics: diagnostics}, err
	}
	encoded, encodeDiagnostics, translateRequestErr := targetPair.buffered.EncodeRequest(request, envelope, engine.policy)
	diagnostics = appendDiagnostics(diagnostics, encodeDiagnostics, engine.policy.Limits.Diagnostics)
	return RequestResult{Request: request, Envelope: envelope, Body: encoded, Diagnostics: diagnostics}, translateRequestErr
}

func (engine *Engine) TranslateResponse(source, target llmprotocol.WireFormat, body []byte, mutate ResponseMutation) (ResponseResult, error) {
	sourcePair, translateResponseErr := engine.codec(source)
	if translateResponseErr != nil {
		return ResponseResult{}, translateResponseErr
	}
	decodePolicy := engine.translationDecodePolicy(source, target, mutate != nil)
	response, envelope, diagnostics, translateResponseErr := sourcePair.buffered.DecodeResponse(body, decodePolicy)
	if translateResponseErr != nil {
		return ResponseResult{Diagnostics: diagnostics}, translateResponseErr
	}
	if err := llmprotocol.ValidateResponse(response, engine.policy.Limits); err != nil {
		return ResponseResult{Response: response, Envelope: envelope, Diagnostics: diagnostics}, err
	}
	targetPair, translateResponseErr := engine.codec(target)
	if translateResponseErr != nil {
		return ResponseResult{Response: response, Envelope: envelope, Diagnostics: diagnostics}, translateResponseErr
	}
	if mutate != nil {
		if err := mutate(&response); err != nil {
			return ResponseResult{Response: response, Envelope: envelope, Diagnostics: diagnostics}, err
		}
		response.Generation++
	}
	if err := llmprotocol.ValidateResponse(response, engine.policy.Limits); err != nil {
		return ResponseResult{Response: response, Envelope: envelope, Diagnostics: diagnostics}, err
	}
	if err := llmprotocol.RequireCapabilities(target, targetPair.buffered.Capabilities(), llmprotocol.RequiredResponseCapabilities(response)); err != nil {
		return ResponseResult{Response: response, Envelope: envelope, Diagnostics: diagnostics}, err
	}
	encoded, encodeDiagnostics, translateResponseErr := targetPair.buffered.EncodeResponse(response, envelope, engine.policy)
	diagnostics = appendDiagnostics(diagnostics, encodeDiagnostics, engine.policy.Limits.Diagnostics)
	return ResponseResult{Response: response, Envelope: envelope, Body: encoded, Diagnostics: diagnostics}, translateResponseErr
}

// Preserve-same-format is legal only for byte-identical replay. Once a
// translation crosses formats or a semantic mutation is possible, decoding is
// strict so an unknown field can never disappear without an explicit error.
func (engine *Engine) translationDecodePolicy(source, target llmprotocol.WireFormat, mutated bool) llmprotocol.Policy {
	policy := engine.policy
	if policy.UnknownFields == llmprotocol.UnknownPreserveSameFormat && (source != target || mutated) {
		policy.UnknownFields = llmprotocol.UnknownReject
		policy.SourcePreservation = llmprotocol.SourceDisabled
	}
	return policy
}

func (engine *Engine) EncodeError(format llmprotocol.WireFormat, protocolError *llmprotocol.ProtocolError) ([]byte, error) {
	pair, err := engine.codec(format)
	if err != nil {
		return nil, err
	}
	if protocolError == nil {
		protocolError = llmprotocol.NewError(llmprotocol.ErrorInternal, "internal", "request failed", nil)
	}
	return pair.buffered.EncodeError(protocolError), nil
}

func (engine *Engine) NewStream(source, target llmprotocol.WireFormat, context llmprotocol.StreamContext) (*StreamEngine, error) {
	sourcePair, err := engine.codec(source)
	if err != nil {
		return nil, err
	}
	targetPair, err := engine.codec(target)
	if err != nil {
		return nil, err
	}
	if !sourcePair.buffered.Capabilities().Supports(llmprotocol.CapabilityStreaming) ||
		!targetPair.buffered.Capabilities().Supports(llmprotocol.CapabilityStreaming) {
		return nil, llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "streaming_unsupported", "source or target wire format does not support streaming", nil)
	}
	context.Source = source
	context.Target = target
	streamPolicy := engine.strictStreamPolicy()
	return &StreamEngine{
		decoder:        sourcePair.stream.NewDecoder(context, streamPolicy),
		encoder:        targetPair.stream.NewEncoder(context, streamPolicy),
		maxDiagnostics: engine.policy.Limits.Diagnostics,
	}, nil
}

// Streams are always decoded into neutral events before re-encoding or
// accumulation. Unlike buffered same-format envelopes, they have no complete
// byte-for-byte replay path, so accepting unknown fields would silently drop
// provider semantics.
func (engine *Engine) strictStreamPolicy() llmprotocol.Policy {
	policy := engine.policy
	if policy.UnknownFields == llmprotocol.UnknownPreserveSameFormat {
		policy.UnknownFields = llmprotocol.UnknownReject
		policy.SourcePreservation = llmprotocol.SourceDisabled
	}
	return policy
}

// EventStreamEncoder renders router-produced semantic events directly into a
// public wire format. Synthetic responses never need to manufacture a Chat SSE
// body merely to feed it back through a decoder.
type EventStreamEncoder struct {
	encoder   llmprotocol.StreamEncoder
	terminal  bool
	finalized bool
}

func (engine *Engine) NewEventStreamEncoder(
	format llmprotocol.WireFormat,
	context llmprotocol.StreamContext,
) (*EventStreamEncoder, error) {
	pair, err := engine.codec(format)
	if err != nil {
		return nil, err
	}
	if !pair.buffered.Capabilities().Supports(llmprotocol.CapabilityStreaming) {
		return nil, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature,
			"streaming_unsupported",
			"wire format does not support streaming",
			nil,
		)
	}
	context.Source = format
	context.Target = format
	return &EventStreamEncoder{
		encoder: pair.stream.NewEncoder(context, engine.policy),
	}, nil
}

func (stream *EventStreamEncoder) Push(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if stream == nil || stream.encoder == nil {
		return nil, nil, fmt.Errorf("event stream encoder is unavailable")
	}
	if stream.finalized || stream.terminal {
		return nil, nil, llmprotocol.NewError(
			llmprotocol.ErrorConflict,
			"stream_terminal",
			"stream is already terminal",
			nil,
		)
	}
	frames, diagnostics, err := stream.encoder.Push(event)
	if event.Type == llmprotocol.EventResponseCompleted || event.Type == llmprotocol.EventResponseFailed {
		stream.terminal = err == nil
	}
	return frames, diagnostics, err
}

func (stream *EventStreamEncoder) Finalize(
	reason error,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if stream == nil || stream.encoder == nil {
		return nil, nil, fmt.Errorf("event stream encoder is unavailable")
	}
	if stream.finalized {
		return nil, nil, nil
	}
	stream.finalized = true
	return stream.encoder.Finalize(reason)
}

func (engine *Engine) codec(format llmprotocol.WireFormat) (codecPair, error) {
	if engine == nil || engine.registry == nil {
		return codecPair{}, fmt.Errorf("protocol codec engine is unavailable")
	}
	pair, ok := engine.registry.resolve(format)
	if !ok {
		return codecPair{}, llmprotocol.NewError(
			llmprotocol.ErrorUnsupportedFeature, "wire_format_unavailable",
			fmt.Sprintf("wire format %q is unavailable", format), nil,
		)
	}
	return pair, nil
}

func validatePolicy(policy llmprotocol.Policy) error {
	if policy.UnknownFields != llmprotocol.UnknownReject && policy.UnknownFields != llmprotocol.UnknownPreserveSameFormat {
		return fmt.Errorf("unknown-field policy is invalid")
	}
	if policy.LossyFeatures != llmprotocol.LossyReject && policy.LossyFeatures != llmprotocol.LossyAllowWithDiagnostic {
		return fmt.Errorf("lossy-feature policy is invalid")
	}
	if policy.Limits.BodyBytes <= 0 || policy.Limits.SourceEnvelopeBytes <= 0 ||
		policy.Limits.ModelBytes <= 0 || policy.Limits.Instructions <= 0 ||
		policy.Limits.Messages <= 0 || policy.Limits.Candidates <= 0 || policy.Limits.Alternatives <= 0 ||
		policy.Limits.OutputItems <= 0 || policy.Limits.ContentBlocks <= 0 ||
		policy.Limits.Citations <= 0 || policy.Limits.CitationURLBytes <= 0 || policy.Limits.CitationTitleBytes <= 0 ||
		policy.Limits.JSONDepth <= 0 || policy.Limits.ToolResultDepth <= 0 ||
		policy.Limits.Tools <= 0 || policy.Limits.ToolNameBytes <= 0 ||
		policy.Limits.ToolDescriptionBytes <= 0 || policy.Limits.IdentifierBytes <= 0 ||
		policy.Limits.SchemaBytes <= 0 || policy.Limits.MetadataBytes <= 0 ||
		policy.Limits.MetadataEntries <= 0 || policy.Limits.MetadataKeyBytes <= 0 ||
		policy.Limits.MetadataValueBytes <= 0 || policy.Limits.ToolArgumentsBytes <= 0 ||
		policy.Limits.ReasoningEffortBytes <= 0 || policy.Limits.StopSequences <= 0 ||
		policy.Limits.StopBytes <= 0 || policy.Limits.TextBytes <= 0 ||
		policy.Limits.MediaReferenceBytes <= 0 || policy.Limits.MediaDataBytes <= 0 ||
		policy.Limits.SSEFrameBytes <= 0 || policy.Limits.Diagnostics <= 0 {
		return fmt.Errorf("protocol limits must be positive")
	}
	return nil
}

func appendDiagnostics(left, right llmprotocol.Diagnostics, limit int) llmprotocol.Diagnostics {
	result := append(append(llmprotocol.Diagnostics(nil), left...), right...)
	if limit > 0 && len(result) > limit {
		result = append(result[:limit-1], llmprotocol.Diagnostic{
			Field: "diagnostics", Action: llmprotocol.DiagnosticTruncated,
			Reason: "additional fidelity diagnostics were truncated",
		})
	}
	return result
}

type StreamEngine struct {
	decoder        llmprotocol.StreamDecoder
	encoder        llmprotocol.StreamEncoder
	maxDiagnostics int
	terminal       bool
	finalized      bool
}

func (engine *StreamEngine) Push(frame []byte) ([][]byte, []llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if engine == nil || engine.decoder == nil || engine.encoder == nil {
		return nil, nil, nil, fmt.Errorf("stream codec engine is unavailable")
	}
	if engine.terminal {
		return nil, nil, nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already terminal", nil)
	}
	events, diagnostics, err := engine.decoder.Push(frame)
	if err != nil {
		return nil, events, diagnostics, err
	}
	frames := make([][]byte, 0, len(events))
	for _, event := range events {
		encoded, eventDiagnostics, encodeErr := engine.encoder.Push(event)
		diagnostics = appendDiagnostics(diagnostics, eventDiagnostics, engine.maxDiagnostics)
		frames = append(frames, encoded...)
		if encodeErr != nil {
			return frames, events, diagnostics, encodeErr
		}
		if event.Type == llmprotocol.EventResponseCompleted || event.Type == llmprotocol.EventResponseFailed {
			engine.terminal = true
		}
	}
	return frames, events, diagnostics, nil
}

func (engine *StreamEngine) Finalize(reason error) ([][]byte, []llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if engine == nil || engine.decoder == nil || engine.encoder == nil {
		return nil, nil, nil, fmt.Errorf("stream codec engine is unavailable")
	}
	if engine.finalized {
		return nil, nil, nil, nil
	}
	engine.finalized = true
	events, diagnostics, decodeErr := engine.decoder.Finalize(reason)
	frames := make([][]byte, 0, len(events)+1)
	for _, event := range events {
		encoded, eventDiagnostics, encodeErr := engine.encoder.Push(event)
		diagnostics = appendDiagnostics(diagnostics, eventDiagnostics, engine.maxDiagnostics)
		frames = append(frames, encoded...)
		if encodeErr != nil {
			return frames, events, diagnostics, encodeErr
		}
	}
	encoded, encodeDiagnostics, encodeErr := engine.encoder.Finalize(reason)
	diagnostics = appendDiagnostics(diagnostics, encodeDiagnostics, engine.maxDiagnostics)
	frames = append(frames, encoded...)
	engine.terminal = true
	if decodeErr != nil {
		return frames, events, diagnostics, decodeErr
	}
	return frames, events, diagnostics, encodeErr
}
