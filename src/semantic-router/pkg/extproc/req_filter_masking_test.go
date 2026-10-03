package extproc

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const (
	maskingRawEmail   = "alice@example.com"
	maskingChatBody   = `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"email me at alice@example.com please"}]}`
	maskingMaskedText = "email me at [EMAIL_ADDRESS_0] please"
)

// stubPIIDetector reports every occurrence of each value as an EMAIL_ADDRESS
// span, so the dispatch path is tested without loading a PII model.
type stubPIIDetector struct {
	values  []string
	partial bool
	err     error
	calls   int
}

func (s *stubPIIDetector) ClassifyPIIWithDetailsAndThreshold(
	_ context.Context, text string, _ float32,
) ([]classification.PIIDetection, error) {
	s.calls++
	if s.err != nil {
		return nil, s.err
	}
	var detections []classification.PIIDetection
	for _, value := range s.values {
		for offset := 0; ; {
			index := strings.Index(text[offset:], value)
			if index < 0 {
				break
			}
			start := offset + index
			detections = append(detections, classification.PIIDetection{
				EntityType: "EMAIL_ADDRESS", Start: start, End: start + len(value), Text: value, Confidence: 1,
			})
			offset = start + len(value)
		}
	}
	if s.partial {
		// The real classifier returns the spans it saw plus this error when
		// the provider truncated its input.
		return detections, classification.ErrTokenSpansTruncated
	}
	return detections, nil
}

func useMaskingDetector(t *testing.T, detector piiDetector) {
	t.Helper()
	previous := maskingDetectorFor
	maskingDetectorFor = func(*OpenAIRouter, *RequestContext) (piiDetector, float32, bool) {
		return detector, 0, true
	}
	t.Cleanup(func() { maskingDetectorFor = previous })
}

func maskingDecision(enabled bool) *config.Decision {
	return &config.Decision{Plugins: []config.DecisionPlugin{{
		Type:          config.DecisionPluginMasking,
		Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": enabled}),
	}}}
}

func newMaskingContext(
	t *testing.T, router *OpenAIRouter, target llmprotocol.WireFormat, decision *config.Decision,
) *RequestContext {
	t.Helper()
	ctx := &RequestContext{
		SourceFormat:        llmprotocol.OpenAIChatV1,
		TargetFormat:        target,
		RequestID:           "request_masking",
		TraceContext:        t.Context(),
		VSRSelectedDecision: decision,
	}
	request, immediate := router.prepareProtocolRequest([]byte(maskingChatBody), ctx)
	if immediate != nil || request == nil {
		t.Fatalf("request was rejected: request=%+v immediate=%+v", request, immediate)
	}
	return ctx
}

// finalizeMaskingDispatch returns the bytes that would reach the provider.
func finalizeMaskingDispatch(router *OpenAIRouter, ctx *RequestContext) ([]byte, error) {
	dispatch := &providerDispatch{logicalModel: "m", upstreamModel: "m", targetFormat: ctx.TargetFormat}
	response := buildRequestBodyContinueResponse(&routeHeaderState{}, nil, false)
	finalized, err := router.finalizeProviderDispatchResponse(dispatch, response, ctx)
	if err != nil {
		return nil, err
	}
	return finalized.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), nil
}

// assertMaskingFailsClosed checks the failure becomes an immediate 503, not a
// stream error that failure_mode_allow would answer by forwarding the
// original body, and that no masked content was produced for dispatch.
func assertMaskingFailsClosed(t *testing.T, router *OpenAIRouter, ctx *RequestContext, err error, generation uint64) {
	t.Helper()
	var maskingErr *maskingDispatchError
	if !errors.As(err, &maskingErr) {
		t.Fatalf("expected a masking dispatch error, got %v", err)
	}
	response, handled := router.processBodyRoutingError(err, ctx)
	if !handled {
		t.Fatal("masking failure would end the ext_proc stream instead of answering the client")
	}
	if code := response.GetImmediateResponse().GetStatus().GetCode(); code != typev3.StatusCode_ServiceUnavailable {
		t.Fatalf("status = %v, want 503", code)
	}
	if strings.Contains(string(response.GetImmediateResponse().GetBody()), maskingRawEmail) {
		t.Fatal("the 503 body echoes request content")
	}
	if ctx.SemanticRequest.Generation != generation {
		t.Fatalf("Generation changed on a failed mask: %d -> %d", generation, ctx.SemanticRequest.Generation)
	}
}

// The most important test: it fails if Generation++ is removed, because the
// codec then replays the client's original bytes (plan §0.5). A test on
// ctx.SemanticRequest alone would pass while the raw value is dispatched.
func TestMaskingDispatchSameFormatSendsNoRawValue(t *testing.T) {
	router := &OpenAIRouter{}
	useMaskingDetector(t, &stubPIIDetector{values: []string{maskingRawEmail}})
	ctx := newMaskingContext(t, router, llmprotocol.OpenAIChatV1, maskingDecision(true))

	body, err := finalizeMaskingDispatch(router, ctx)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	if bytes.Contains(body, []byte(maskingRawEmail)) {
		t.Fatalf("dispatched bytes contain the raw value:\n%s", body)
	}
	if !bytes.Contains(body, []byte("[EMAIL_ADDRESS_0]")) {
		t.Fatalf("dispatched bytes lack the placeholder:\n%s", body)
	}
	if _, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1, body); err != nil {
		t.Fatalf("dispatched bytes are not a valid Chat request: %v\n%s", err, body)
	}
}

func TestMaskingDispatchCrossFormatSendsNoRawValue(t *testing.T) {
	router := &OpenAIRouter{}
	useMaskingDetector(t, &stubPIIDetector{values: []string{maskingRawEmail}})
	ctx := newMaskingContext(t, router, llmprotocol.AnthropicMessagesV1, maskingDecision(true))

	body, err := finalizeMaskingDispatch(router, ctx)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	if bytes.Contains(body, []byte(maskingRawEmail)) {
		t.Fatalf("dispatched bytes contain the raw value:\n%s", body)
	}
	if !bytes.Contains(body, []byte("[EMAIL_ADDRESS_0]")) {
		t.Fatalf("dispatched bytes lack the placeholder:\n%s", body)
	}
	if _, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, body); err != nil {
		t.Fatalf("dispatched bytes are not a valid Anthropic request: %v\n%s", err, body)
	}
}

func TestMaskingDispatchPluginAbsentIsByteIdentical(t *testing.T) {
	router := &OpenAIRouter{}
	detector := &stubPIIDetector{values: []string{maskingRawEmail}}
	useMaskingDetector(t, detector)
	ctx := newMaskingContext(t, router, llmprotocol.OpenAIChatV1, &config.Decision{})

	body, err := finalizeMaskingDispatch(router, ctx)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	if !bytes.Equal(body, []byte(maskingChatBody)) {
		t.Fatalf("dispatch changed without the plugin:\n got=%s\nwant=%s", body, maskingChatBody)
	}
	if detector.calls != 0 || ctx.Masking != nil {
		t.Fatalf("masking ran without the plugin: calls=%d allocator=%v", detector.calls, ctx.Masking != nil)
	}
}

func TestMaskingDispatchPluginDisabledIsByteIdentical(t *testing.T) {
	router := &OpenAIRouter{}
	detector := &stubPIIDetector{values: []string{maskingRawEmail}}
	useMaskingDetector(t, detector)
	ctx := newMaskingContext(t, router, llmprotocol.OpenAIChatV1, maskingDecision(false))

	body, err := finalizeMaskingDispatch(router, ctx)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	if !bytes.Equal(body, []byte(maskingChatBody)) {
		t.Fatalf("dispatch changed with the plugin disabled:\n got=%s\nwant=%s", body, maskingChatBody)
	}
	if detector.calls != 0 || ctx.Masking != nil {
		t.Fatalf("masking ran while disabled: calls=%d allocator=%v", detector.calls, ctx.Masking != nil)
	}
}

func TestMaskingDispatchBumpsGenerationWhenChanged(t *testing.T) {
	router := &OpenAIRouter{}
	useMaskingDetector(t, &stubPIIDetector{values: []string{maskingRawEmail}})
	ctx := newMaskingContext(t, router, llmprotocol.OpenAIChatV1, maskingDecision(true))
	before := ctx.SemanticRequest.Generation

	if _, err := finalizeMaskingDispatch(router, ctx); err != nil {
		t.Fatalf("finalize: %v", err)
	}
	if ctx.SemanticRequest.Generation <= before {
		t.Fatalf("Generation not bumped after masking: before=%d after=%d", before, ctx.SemanticRequest.Generation)
	}
}

// Bumping when nothing changed would retire byte replay for every request on
// the route, a latency regression with no privacy gain.
func TestMaskingDispatchKeepsGenerationWhenNothingFound(t *testing.T) {
	router := &OpenAIRouter{}
	useMaskingDetector(t, &stubPIIDetector{values: []string{"nobody@example.org"}})
	ctx := newMaskingContext(t, router, llmprotocol.OpenAIChatV1, maskingDecision(true))
	before := ctx.SemanticRequest.Generation

	body, err := finalizeMaskingDispatch(router, ctx)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	if ctx.SemanticRequest.Generation != before {
		t.Fatalf("Generation bumped with nothing masked: before=%d after=%d", before, ctx.SemanticRequest.Generation)
	}
	if !bytes.Equal(body, []byte(maskingChatBody)) {
		t.Fatalf("byte replay lost with nothing masked:\n got=%s\nwant=%s", body, maskingChatBody)
	}
}

// No stub: the real resolver finds no classifier on a bare router.
func TestMaskingDispatchFailsClosedWithoutClassifier(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := newMaskingContext(t, router, llmprotocol.OpenAIChatV1, maskingDecision(true))
	before := ctx.SemanticRequest.Generation

	body, err := finalizeMaskingDispatch(router, ctx)
	if body != nil {
		t.Fatalf("a body was produced without a classifier:\n%s", body)
	}
	assertMaskingFailsClosed(t, router, ctx, err, before)
	if got := ctx.SemanticRequest.Messages[0].Content[0].Text; !strings.Contains(got, maskingRawEmail) {
		t.Fatalf("request was mutated without a classifier: %q", got)
	}
}

func TestMaskingDispatchFailsClosedOnMaskingError(t *testing.T) {
	cases := map[string]*stubPIIDetector{
		"scan error": {err: errors.New("classifier unavailable")},
		// Spans for only the prefix the model saw: the rest is unscanned.
		"truncated scan": {values: []string{maskingRawEmail}, partial: true},
	}
	for name, detector := range cases {
		t.Run(name, func(t *testing.T) {
			router := &OpenAIRouter{}
			useMaskingDetector(t, detector)
			ctx := newMaskingContext(t, router, llmprotocol.OpenAIChatV1, maskingDecision(true))
			before := ctx.SemanticRequest.Generation

			body, err := finalizeMaskingDispatch(router, ctx)
			if body != nil {
				t.Fatalf("a body was produced after a masking error:\n%s", body)
			}
			assertMaskingFailsClosed(t, router, ctx, err, before)
		})
	}
}

// A second dispatch on the same request adds a different email. It must get
// _1 from the request-scoped allocator, not a colliding _0 (D5), and the
// already-masked text must be left alone.
func TestMaskingDispatchLooperRedispatchContinuesIndices(t *testing.T) {
	router := &OpenAIRouter{}
	detector := &stubPIIDetector{values: []string{maskingRawEmail}}
	useMaskingDetector(t, detector)
	ctx := newMaskingContext(t, router, llmprotocol.OpenAIChatV1, maskingDecision(true))

	if _, err := finalizeMaskingDispatch(router, ctx); err != nil {
		t.Fatalf("first dispatch: %v", err)
	}
	allocator := ctx.Masking
	if allocator == nil {
		t.Fatal("the first dispatch did not create the request allocator")
	}

	ctx.SemanticRequest.Messages = append(ctx.SemanticRequest.Messages, llmprotocol.Message{
		Role:    llmprotocol.RoleUser,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "also cc bob@example.com"}},
	})
	detector.values = []string{maskingRawEmail, "bob@example.com"}

	body, err := finalizeMaskingDispatch(router, ctx)
	if err != nil {
		t.Fatalf("second dispatch: %v", err)
	}
	if ctx.Masking != allocator {
		t.Fatal("the allocator was recreated for the second dispatch")
	}
	if bytes.Contains(body, []byte(maskingRawEmail)) || bytes.Contains(body, []byte("bob@example.com")) {
		t.Fatalf("second dispatch contains a raw value:\n%s", body)
	}
	if !bytes.Contains(body, []byte("[EMAIL_ADDRESS_1]")) {
		t.Fatalf("new value did not get _1:\n%s", body)
	}
	if got := ctx.SemanticRequest.Messages[0].Content[0].Text; got != maskingMaskedText {
		t.Fatalf("already-masked text changed on re-dispatch: %q", got)
	}
}

// Concrete models bypass decision plugins, and this path clears the decision
// before finalize, so no masking config can apply here today. The hook is on
// this path regardless, so it will cover the gateway if that bypass changes.
func TestMaskingExternalGatewayPathCarriesNoDecision(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	detector := &stubPIIDetector{values: []string{maskingRawEmail}}
	useMaskingDetector(t, detector)
	ctx := newMaskingContext(t, router, "", maskingDecision(true))

	response, err := router.handleExternalGatewayModelRouting(ctx.SemanticRequest, "m", ctx)
	if err != nil || response == nil {
		t.Fatalf("gateway dispatch failed: response=%v err=%v", response, err)
	}
	if ctx.VSRSelectedDecision != nil {
		t.Fatal("gateway path kept a decision; masking now applies here and this test should assert masked bytes")
	}
	if detector.calls != 0 {
		t.Fatalf("masking scanned on a path with no decision: calls=%d", detector.calls)
	}
}
