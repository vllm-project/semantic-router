package protocolcodec

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type (
	RequestMutation        func(*llmprotocol.Request) error
	ResponseMutation       func(*llmprotocol.Response) error
	TransportErrorMutation func(*llmprotocol.TransportError) error
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

type TransportErrorResult struct {
	TransportError llmprotocol.TransportError
	Body           []byte
	Diagnostics    llmprotocol.Diagnostics
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
	return engine.decodeRequest(format, body, engine.policy)
}

// DecodeRequestForMutation rejects wire fields that cannot survive neutral
// translation. Router ingress uses this contract because model selection and
// policy plugins necessarily mutate the request after decoding; accepting an
// opaque field there would guarantee that it is silently erased on dispatch.
func (engine *Engine) DecodeRequestForMutation(
	format llmprotocol.WireFormat,
	body []byte,
) (llmprotocol.Request, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	return engine.decodeRequest(format, body, engine.translationDecodePolicy(format, format, true))
}

func (engine *Engine) decodeRequest(
	format llmprotocol.WireFormat,
	body []byte,
	policy llmprotocol.Policy,
) (llmprotocol.Request, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	pair, err := engine.codec(format)
	if err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, err
	}
	request, envelope, diagnostics, err := pair.buffered.DecodeRequest(body, policy)
	if err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, diagnostics, err
	}
	applyRequestSemanticDefaults(&request)
	llmprotocol.MarkDeferredToolLinks(&request)
	if err := llmprotocol.ValidateRequest(request, engine.policy.Limits); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, diagnostics, err
	}
	return request, envelope, diagnostics, nil
}

// applyRequestSemanticDefaults makes provider-documented defaults explicit in
// the neutral contract. Tool choice defaults to automatic selection whenever
// tools are present in all built-in request formats. Keeping that semantic fact
// in one place prevents routing plugins from behaving differently based on
// whether a client serialized the default explicitly.
func applyRequestSemanticDefaults(request *llmprotocol.Request) {
	if request != nil && len(request.Tools) > 0 && request.ToolChoice.Mode == "" {
		request.ToolChoice.Mode = llmprotocol.ToolChoiceAuto
	}
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
	applyRequestSemanticDefaults(&request)
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

// TranslateTransportError translates an HTTP non-2xx body through the neutral
// error contract. It intentionally does not use DecodeResponse: a failed model
// Response resource and an API transport error are different wire objects.
func (engine *Engine) TranslateTransportError(
	source,
	target llmprotocol.WireFormat,
	body []byte,
	mutate TransportErrorMutation,
) (TransportErrorResult, error) {
	sourcePair, err := engine.codec(source)
	if err != nil {
		return TransportErrorResult{}, err
	}
	decodePolicy := engine.policy
	decodePolicy.UnknownFields = llmprotocol.UnknownReject
	decodePolicy.SourcePreservation = llmprotocol.SourceDisabled
	transportError, diagnostics, err := sourcePair.buffered.DecodeTransportError(body, decodePolicy)
	if err != nil {
		return TransportErrorResult{Diagnostics: diagnostics}, err
	}
	if validationErr := llmprotocol.ValidateTransportError(transportError, engine.policy.Limits); validationErr != nil {
		return TransportErrorResult{TransportError: transportError, Diagnostics: diagnostics}, validationErr
	}
	targetPair, err := engine.codec(target)
	if err != nil {
		return TransportErrorResult{TransportError: transportError, Diagnostics: diagnostics}, err
	}
	if mutate != nil {
		if mutationErr := mutate(&transportError); mutationErr != nil {
			return TransportErrorResult{TransportError: transportError, Diagnostics: diagnostics}, mutationErr
		}
	}
	if validationErr := llmprotocol.ValidateTransportError(transportError, engine.policy.Limits); validationErr != nil {
		return TransportErrorResult{TransportError: transportError, Diagnostics: diagnostics}, validationErr
	}
	return TransportErrorResult{
		TransportError: transportError,
		Body:           targetPair.buffered.EncodeTransportError(transportError),
		Diagnostics:    diagnostics,
	}, nil
}

// DecodeTransportError decodes a provider-native HTTP error body without
// conflating it with a failed model response resource. It is the symmetric
// ingress operation for EncodeTransportError and uses the strict provider
// boundary regardless of same-format source preservation policy.
func (engine *Engine) DecodeTransportError(
	format llmprotocol.WireFormat,
	body []byte,
) (llmprotocol.TransportError, llmprotocol.Diagnostics, error) {
	pair, err := engine.codec(format)
	if err != nil {
		return llmprotocol.TransportError{}, nil, err
	}
	decodePolicy := engine.policy
	decodePolicy.UnknownFields = llmprotocol.UnknownReject
	decodePolicy.SourcePreservation = llmprotocol.SourceDisabled
	transportError, diagnostics, err := pair.buffered.DecodeTransportError(body, decodePolicy)
	if err != nil {
		return llmprotocol.TransportError{}, diagnostics, err
	}
	if err := llmprotocol.ValidateTransportError(transportError, engine.policy.Limits); err != nil {
		return transportError, diagnostics, err
	}
	return transportError, diagnostics, nil
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
	if protocolError == nil {
		protocolError = llmprotocol.NewError(llmprotocol.ErrorInternal, "internal", "request failed", nil)
	}
	return engine.EncodeTransportError(format, llmprotocol.TransportError{Error: protocolError})
}

func (engine *Engine) EncodeTransportError(
	format llmprotocol.WireFormat,
	transportError llmprotocol.TransportError,
) ([]byte, error) {
	pair, err := engine.codec(format)
	if err != nil {
		return nil, err
	}
	if err := llmprotocol.ValidateTransportError(transportError, engine.policy.Limits); err != nil {
		return nil, err
	}
	return pair.buffered.EncodeTransportError(transportError), nil
}

func (engine *Engine) NewStream(source, target llmprotocol.WireFormat, context llmprotocol.StreamContext) (*StreamEngine, error) {
	return engine.NewStreamWithMutation(source, target, context, nil)
}

// StreamEventMutation applies request-scoped policy to decoded neutral events
// before they reach a public wire encoder. It cannot observe provider bytes or
// HTTP headers and therefore keeps protocol translation provider-neutral.
type StreamEventMutation func(*llmprotocol.Event) error

func (engine *Engine) NewStreamWithMutation(
	source,
	target llmprotocol.WireFormat,
	context llmprotocol.StreamContext,
	mutation StreamEventMutation,
) (*StreamEngine, error) {
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
		decoder:            sourcePair.stream.NewDecoder(context, streamPolicy),
		encoder:            targetPair.stream.NewEncoder(context, streamPolicy),
		mutation:           mutation,
		targetFormat:       target,
		targetCapabilities: targetPair.buffered.Capabilities(),
		maxDiagnostics:     engine.policy.Limits.Diagnostics,
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
	encoder      llmprotocol.StreamEncoder
	format       llmprotocol.WireFormat
	capabilities llmprotocol.CapabilitySet
	terminal     bool
	finalized    bool
	failure      error
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
		format:  format, capabilities: pair.buffered.Capabilities(),
	}, nil
}

func (stream *EventStreamEncoder) Push(
	event llmprotocol.Event,
) ([][]byte, llmprotocol.Diagnostics, error) {
	if stream == nil || stream.encoder == nil {
		return nil, nil, fmt.Errorf("event stream encoder is unavailable")
	}
	if stream.finalized || stream.terminal {
		if stream.failure != nil {
			return nil, nil, stream.failure
		}
		return nil, nil, llmprotocol.NewError(
			llmprotocol.ErrorConflict,
			"stream_terminal",
			"stream is already terminal",
			nil,
		)
	}
	if err := llmprotocol.RequireCapabilities(
		stream.format,
		stream.capabilities,
		llmprotocol.RequiredEventCapabilities(event),
	); err != nil {
		stream.failure = err
		stream.terminal = true
		return nil, nil, err
	}
	frames, diagnostics, err := stream.encoder.Push(event)
	if err != nil {
		stream.failure = err
		stream.terminal = true
		return frames, diagnostics, err
	}
	if event.Type == llmprotocol.EventResponseCompleted || event.Type == llmprotocol.EventResponseFailed {
		stream.terminal = true
	}
	return frames, diagnostics, nil
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
	if stream.failure != nil {
		reason = stream.failure
	}
	if !stream.terminal && reason == nil {
		reason = fmt.Errorf("semantic event stream ended without a terminal event")
	}
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
	if policy.MissingStableIDs != llmprotocol.MissingIDReject && policy.MissingStableIDs != llmprotocol.MissingIDGenerateStable {
		return fmt.Errorf("missing-ID policy is invalid")
	}
	if policy.SourcePreservation != llmprotocol.SourceDisabled && policy.SourcePreservation != llmprotocol.SourceBoundedSameFormat {
		return fmt.Errorf("source-preservation policy is invalid")
	}
	if !positiveProtocolLimits(policy.Limits) {
		return fmt.Errorf("protocol limits must be positive")
	}
	return nil
}

func positiveProtocolLimits(limits llmprotocol.Limits) bool {
	values := []int{
		limits.BodyBytes, limits.SourceEnvelopeBytes, limits.ModelBytes,
		limits.Instructions, limits.Messages, limits.Candidates, limits.Alternatives,
		limits.OutputItems, limits.ContentBlocks, limits.Citations,
		limits.CitationURLBytes, limits.CitationTitleBytes, limits.JSONDepth,
		limits.ToolResultDepth, limits.Tools, limits.ToolNameBytes,
		limits.ToolDescriptionBytes, limits.IdentifierBytes, limits.SchemaBytes,
		limits.MetadataBytes, limits.MetadataEntries, limits.MetadataKeyBytes,
		limits.MetadataValueBytes, limits.ToolArgumentsBytes, limits.ReasoningEffortBytes,
		limits.StopSequences, limits.StopBytes, limits.TextBytes,
		limits.MediaReferenceBytes, limits.MediaDataBytes, limits.SSEFrameBytes,
		limits.UnfinishedArguments, limits.Events, limits.Diagnostics,
	}
	for _, value := range values {
		if value <= 0 {
			return false
		}
	}
	return true
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
	decoder            llmprotocol.StreamDecoder
	encoder            llmprotocol.StreamEncoder
	mutation           StreamEventMutation
	targetFormat       llmprotocol.WireFormat
	targetCapabilities llmprotocol.CapabilitySet
	maxDiagnostics     int
	terminal           bool
	finalized          bool
	failure            error
	pendingCompletion  *llmprotocol.Event
}

func (engine *StreamEngine) Push(frame []byte) ([][]byte, []llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if engine == nil || engine.decoder == nil || engine.encoder == nil {
		return nil, nil, nil, fmt.Errorf("stream codec engine is unavailable")
	}
	if engine.terminal {
		if engine.failure != nil {
			return nil, nil, nil, engine.failure
		}
		return nil, nil, nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already terminal", nil)
	}
	events, diagnostics, decodeErr := engine.decoder.Push(frame)
	if decodeErr != nil {
		events = suppressSuccessfulTerminal(events)
		engine.pendingCompletion = nil
	} else {
		events = engine.deferSuccessfulTerminal(events)
	}
	frames := make([][]byte, 0, len(events))
	acceptedEvents := make([]llmprotocol.Event, 0, len(events))
	for index := range events {
		if engine.mutation != nil {
			if mutationErr := engine.mutation(&events[index]); mutationErr != nil {
				engine.poison(mutationErr)
				return frames, acceptedEvents, diagnostics, mutationErr
			}
		}
		event := events[index]
		if capabilityErr := llmprotocol.RequireCapabilities(
			engine.targetFormat,
			engine.targetCapabilities,
			llmprotocol.RequiredEventCapabilities(event),
		); capabilityErr != nil {
			engine.poison(capabilityErr)
			return frames, acceptedEvents, diagnostics, capabilityErr
		}
		encoded, eventDiagnostics, encodeErr := engine.encoder.Push(event)
		diagnostics = appendDiagnostics(diagnostics, eventDiagnostics, engine.maxDiagnostics)
		frames = append(frames, encoded...)
		if encodeErr != nil {
			engine.poison(encodeErr)
			return frames, acceptedEvents, diagnostics, encodeErr
		}
		acceptedEvents = append(acceptedEvents, event)
		if event.Type == llmprotocol.EventResponseCompleted || event.Type == llmprotocol.EventResponseFailed {
			engine.terminal = true
		}
	}
	// A transport read may contain several complete SSE frames followed by a
	// malformed one. The decoder deliberately returns the valid semantic prefix
	// together with the trailing error; encode that prefix before surfacing the
	// error so network chunk coalescing cannot erase already accepted output.
	if decodeErr != nil {
		engine.poison(decodeErr)
		return frames, acceptedEvents, diagnostics, decodeErr
	}
	return frames, acceptedEvents, diagnostics, nil
}

func suppressSuccessfulTerminal(events []llmprotocol.Event) []llmprotocol.Event {
	result := events[:0]
	for _, event := range events {
		if event.Type != llmprotocol.EventResponseCompleted {
			result = append(result, event)
		}
	}
	return result
}

// A provider's semantic success event is held until the HTTP body reaches a
// clean end-of-stream. Otherwise a terminal frame followed by a malformed
// unterminated fragment in a later transport read could publish success before
// the codec has had an opportunity to validate the trailing bytes.
func (engine *StreamEngine) deferSuccessfulTerminal(events []llmprotocol.Event) []llmprotocol.Event {
	result := events[:0]
	for index := range events {
		if events[index].Type != llmprotocol.EventResponseCompleted {
			result = append(result, events[index])
			continue
		}
		completion := events[index]
		engine.pendingCompletion = &completion
	}
	return result
}

func (engine *StreamEngine) poison(err error) {
	if err == nil {
		return
	}
	if engine.failure == nil {
		engine.failure = err
	}
	engine.pendingCompletion = nil
	engine.terminal = true
}

func (engine *StreamEngine) Finalize(reason error) ([][]byte, []llmprotocol.Event, llmprotocol.Diagnostics, error) {
	if engine == nil || engine.decoder == nil || engine.encoder == nil {
		return nil, nil, nil, fmt.Errorf("stream codec engine is unavailable")
	}
	if engine.finalized {
		return nil, nil, nil, nil
	}
	engine.finalized = true
	firstFailure := engine.failure
	if firstFailure != nil {
		reason = firstFailure
	}
	events, diagnostics, decodeErr := engine.decoder.Finalize(reason)
	terminalReason := reason
	if decodeErr != nil {
		events = suppressSuccessfulTerminal(events)
		engine.pendingCompletion = nil
		terminalReason = decodeErr
		if firstFailure != nil {
			// Finalizing a poisoned decoder may surface a secondary state or
			// resource-limit error while it tries to synthesize its terminal
			// event. The first failure is the stream contract: use it for both
			// the public terminal frame and the caller-visible error.
			terminalReason = firstFailure
		}
	} else if terminalReason != nil {
		// A provider success observed before HTTP EOS is provisional. Transport
		// cancellation or deadline expiry wins even when the source decoder has
		// already seen its semantic success event.
		events = suppressSuccessfulTerminal(events)
		engine.pendingCompletion = nil
		if !containsFailedTerminal(events) {
			events = append(events, llmprotocol.Event{
				Type:       llmprotocol.EventResponseFailed,
				StopReason: llmprotocol.StopError,
				Error:      streamFinalizationError(terminalReason, "stream ended before completion"),
				Failure:    llmprotocol.FailureTransport,
			})
		}
	} else if engine.pendingCompletion != nil {
		events = append(events, *engine.pendingCompletion)
		engine.pendingCompletion = nil
	}
	frames := make([][]byte, 0, len(events)+1)
	acceptedEvents := make([]llmprotocol.Event, 0, len(events))
	for index := range events {
		if engine.mutation != nil {
			if mutationErr := engine.mutation(&events[index]); mutationErr != nil {
				return engine.finalizeFailure(frames, acceptedEvents, diagnostics, mutationErr)
			}
		}
		event := events[index]
		if capabilityErr := llmprotocol.RequireCapabilities(
			engine.targetFormat,
			engine.targetCapabilities,
			llmprotocol.RequiredEventCapabilities(event),
		); capabilityErr != nil {
			return engine.finalizeFailure(frames, acceptedEvents, diagnostics, capabilityErr)
		}
		encoded, eventDiagnostics, encodeErr := engine.encoder.Push(event)
		diagnostics = appendDiagnostics(diagnostics, eventDiagnostics, engine.maxDiagnostics)
		frames = append(frames, encoded...)
		if encodeErr != nil {
			return engine.finalizeFailure(frames, acceptedEvents, diagnostics, encodeErr)
		}
		acceptedEvents = append(acceptedEvents, event)
	}
	encoded, encodeDiagnostics, encodeErr := engine.encoder.Finalize(terminalReason)
	diagnostics = appendDiagnostics(diagnostics, encodeDiagnostics, engine.maxDiagnostics)
	frames = append(frames, encoded...)
	engine.terminal = true
	if decodeErr != nil {
		if firstFailure != nil {
			return frames, acceptedEvents, diagnostics, firstFailure
		}
		return frames, acceptedEvents, diagnostics, decodeErr
	}
	return frames, acceptedEvents, diagnostics, encodeErr
}

func containsFailedTerminal(events []llmprotocol.Event) bool {
	for _, event := range events {
		if event.Type == llmprotocol.EventResponseFailed {
			return true
		}
	}
	return false
}

func (engine *StreamEngine) finalizeFailure(
	frames [][]byte,
	events []llmprotocol.Event,
	diagnostics llmprotocol.Diagnostics,
	cause error,
) ([][]byte, []llmprotocol.Event, llmprotocol.Diagnostics, error) {
	engine.poison(cause)
	encoded, finalDiagnostics, finalizeErr := engine.encoder.Finalize(cause)
	diagnostics = appendDiagnostics(diagnostics, finalDiagnostics, engine.maxDiagnostics)
	frames = append(frames, encoded...)
	if finalizeErr != nil {
		return frames, events, diagnostics, finalizeErr
	}
	return frames, events, diagnostics, cause
}
